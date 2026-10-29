use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use rig_core::providers::ollama::OllamaConfig;
use serde_json::{Value, json};
use taquba::EnqueueEntry;
use taquba::object_store::memory::InMemory;
use taquba_research::ResearchStepRunner;
use taquba_research::search::{SearchBackend, SearchError, SearchResult};
use taquba_research::store::RunDisplayStatus;
use taquba_workflow::{CancelCommand, Command, MemoStore, RunId};
use tokio::sync::oneshot;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use super::{
    Cli, ReplyOutcome, StoreCtx, WORKFLOW_CONTROL_QUEUE_NAME, WORKFLOW_MEMO_PREFIX, cmd_reply,
    cmd_resume, cmd_run, ellipsize, gather_rows, inbox_sender, open_queue, open_reader,
    pending_cancellations, resolve_store, store_options, with_reader, writer_is_live,
};

const QUERY: &str = "test query";
const QUESTION: &str = "test question?";
const REPLY: &str = "test reply";
/// The bound on each wait for a state of a run.
const DEADLINE: Duration = Duration::from_secs(60);

/// A search backend without results, after which a run proceeds from searching
/// to the investigating step.
struct NoResults;

#[async_trait::async_trait]
impl SearchBackend for NoResults {
    async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<SearchResult>, SearchError> {
        Ok(Vec::new())
    }
}

/// A fake of the `/api/chat` of Ollama that replies to each phase by a phrase
/// of its prompt. The investigating agent calls `ask_user` until the request
/// contains its result.
struct FakeOllama;

impl Respond for FakeOllama {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body: Value = serde_json::from_slice(&request.body).expect("a JSON request");
        let prompt: String = messages(&body)
            .iter()
            .filter_map(|message| message["content"].as_str())
            .collect();
        let answered = !tool_results(&body).is_empty();
        let message = if prompt.contains("research planner") {
            text(json!({ "sub_questions": ["test sub-question"] }).to_string())
        } else if prompt.contains("research investigator") && !answered {
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": { "name": "ask_user", "arguments": { "question": QUESTION } }
                }]
            })
        } else if prompt.contains("research investigator") {
            text(json!({ "findings": [] }).to_string())
        } else {
            text("test answer".to_string())
        };
        ResponseTemplate::new(200).set_body_json(json!({
            "model": "llama3.2",
            "created_at": "2026-10-29T00:00:00Z",
            "done": true,
            "done_reason": "stop",
            "message": message,
        }))
    }
}

fn text(content: String) -> Value {
    json!({ "role": "assistant", "content": content })
}

fn messages(body: &Value) -> Vec<Value> {
    body["messages"].as_array().cloned().unwrap_or_default()
}

/// The contents of the tool results that a chat request sends.
fn tool_results(body: &Value) -> Vec<String> {
    messages(body)
        .iter()
        .filter(|message| message["role"] == "tool")
        .filter_map(|message| message["content"].as_str().map(str::to_string))
        .collect()
}

/// A store in memory and the fake provider of its runs.
struct Harness {
    server: MockServer,
    ctx: StoreCtx,
}

impl Harness {
    async fn start() -> Self {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(FakeOllama)
            .mount(&server)
            .await;
        let ctx = StoreCtx {
            object_store: Arc::new(InMemory::new()),
            source: "memory".to_string(),
        };
        Self { server, ctx }
    }

    fn runner(&self) -> ResearchStepRunner {
        let client = OllamaConfig::new()
            .with_base_url(self.server.uri())
            .client();
        ResearchStepRunner::new_ollama(client, Arc::new(NoResults))
    }

    /// A `run` of [`QUERY`] that may wait up to 600 seconds for a reply, until
    /// `interrupt` resolves.
    fn spawn_run(
        &self,
        interrupt: impl Future<Output = ()> + Send + 'static,
    ) -> tokio::task::JoinHandle<anyhow::Result<()>> {
        let cli = cli(&["--clarification-wait", "600", QUERY]);
        let (ctx, runner) = (self.ctx.clone(), self.runner());
        tokio::spawn(async move { cmd_run(&cli, &ctx, runner, QUERY.to_string(), interrupt).await })
    }

    /// The id of the run once `list` shows it as waiting for a reply.
    async fn await_waiting(&self) -> String {
        tokio::time::timeout(DEADLINE, async {
            loop {
                let rows = gather_rows(&self.ctx).await.unwrap_or_default();
                let waiting = rows
                    .iter()
                    .find(|row| row.status == RunDisplayStatus::Waiting);
                if let Some(row) = waiting {
                    return row.run.status.run_id.to_string();
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        })
        .await
        .expect("the run waits for a reply")
    }

    /// The tool results that the fake provider received, in request order.
    async fn received_tool_results(&self) -> Vec<String> {
        let requests = self.server.received_requests().await.unwrap_or_default();
        requests
            .iter()
            .filter_map(|request| serde_json::from_slice::<Value>(&request.body).ok())
            .flat_map(|body| tool_results(&body))
            .collect()
    }
}

fn cli(args: &[&str]) -> Cli {
    let base = ["taquba-research", "-q", "--provider", "ollama"];
    Cli::parse_from(base.iter().chain(args))
}

async fn join(task: tokio::task::JoinHandle<anyhow::Result<()>>) {
    tokio::time::timeout(DEADLINE, task)
        .await
        .expect("the command returns")
        .expect("the command does not panic")
        .expect("the command succeeds");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_reaches_the_waiting_run_of_a_live_writer() {
    let harness = Harness::start().await;
    let run = harness.spawn_run(std::future::pending());
    let run_id = harness.await_waiting().await;

    let outcome = cmd_reply(&harness.ctx, run_id.clone(), REPLY).await;
    let question = QUESTION.to_string();
    assert_eq!(outcome.unwrap(), ReplyOutcome::Delivered { question });
    join(run).await;
    assert_eq!(harness.received_tool_results().await, [REPLY]);
    let late = cmd_reply(&harness.ctx, run_id, "late reply").await;
    assert!(
        late.unwrap_err()
            .to_string()
            .contains("is not waiting for a reply")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_reply_without_a_writer_takes_effect_at_resume() {
    let harness = Harness::start().await;
    let (interrupt, interrupted) = oneshot::channel::<()>();
    let run = harness.spawn_run(async move {
        let _ = interrupted.await;
    });
    let run_id = harness.await_waiting().await;
    interrupt.send(()).unwrap();
    join(run).await;
    let live = with_reader(&harness.ctx, async |reader| writer_is_live(reader).await).await;
    assert!(!live.unwrap(), "the interrupted run closes its writer");

    let outcome = cmd_reply(&harness.ctx, run_id.clone(), REPLY).await;
    let question = QUESTION.to_string();
    assert_eq!(outcome.unwrap(), ReplyOutcome::Sent { question });
    let resume = cli(&["resume", &run_id]);
    let resumed = cmd_resume(
        &resume,
        &harness.ctx,
        harness.runner(),
        run_id.clone(),
        false,
        std::future::pending(),
    );
    tokio::time::timeout(DEADLINE, resumed)
        .await
        .expect("resume returns")
        .expect("resume succeeds");
    assert_eq!(harness.received_tool_results().await, [REPLY]);
}

#[tokio::test]
async fn pending_cancellations_include_inbox_entries_and_control_jobs() {
    let ctx = StoreCtx {
        object_store: Arc::new(InMemory::new()),
        source: "memory".to_string(),
    };
    let cancel = |run_id: &str| {
        EnqueueEntry::from(Command::Cancel(CancelCommand::new(
            RunId::new(run_id).unwrap(),
        )))
    };
    let queue = open_queue(&ctx).await.unwrap();
    let sender = inbox_sender(&ctx);
    sender
        .send(WORKFLOW_CONTROL_QUEUE_NAME, cancel("01APPLIED"))
        .await
        .unwrap();
    // The pass turns the first entry into a job of the control queue, and no
    // runtime applies the job. The second entry follows the close, so it stays
    // in the inbox.
    queue.apply_inbox_now().await.unwrap();
    Arc::into_inner(queue).unwrap().close().await.unwrap();
    sender
        .send(WORKFLOW_CONTROL_QUEUE_NAME, cancel("01SENT"))
        .await
        .unwrap();

    let reader = open_reader(&ctx).await.unwrap();
    let pending = pending_cancellations(&ctx, &reader).await.unwrap();
    let mut run_ids: Vec<&str> = pending.keys().map(String::as_str).collect();
    run_ids.sort_unstable();
    assert_eq!(run_ids, ["01APPLIED", "01SENT"]);
}

#[tokio::test]
async fn a_url_store_keeps_the_memos_under_its_path() {
    let dir = std::env::temp_dir().join(format!("taquba-research-{}", ulid::Ulid::new()));
    let ctx = resolve_store(Some(&format!("file://{}/sub", dir.display())))
        .await
        .unwrap();
    MemoStore::new(ctx.object_store.clone(), WORKFLOW_MEMO_PREFIX)
        .new_run_memo(&RunId::new("01RUN").unwrap())
        .put("key", b"value")
        .await
        .unwrap();
    let under_path = dir.join("sub").join(WORKFLOW_MEMO_PREFIX).exists();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(under_path);
}

#[test]
fn store_options_are_the_provider_variables_in_lowercase() {
    let vars = [
        ("AWS_ENDPOINT", "http://127.0.0.1:9000"),
        ("ENDPOINT", "other"),
        ("GOOGLE_SERVICE_ACCOUNT", "sa.json"),
        ("AZURE_STORAGE_ACCOUNT_NAME", "account"),
        ("HOME", "/home/u"),
    ]
    .map(|(name, value)| (name.to_string(), value.to_string()));
    let options: Vec<(String, String)> = store_options(vars.into_iter()).collect();
    assert_eq!(
        options,
        [
            ("aws_endpoint", "http://127.0.0.1:9000"),
            ("google_service_account", "sa.json"),
            ("azure_storage_account_name", "account"),
        ]
        .map(|(name, value)| (name.to_string(), value.to_string()))
    );
}

#[test]
fn ellipsize_keeps_strings_within_the_limit() {
    assert_eq!(ellipsize("short", 50), "short");
    let exact = "a".repeat(50);
    assert_eq!(ellipsize(&exact, 50), exact);
}

#[test]
fn ellipsize_truncates_on_char_boundaries() {
    let ascii = "a".repeat(60);
    assert_eq!(ellipsize(&ascii, 50), format!("{}...", "a".repeat(47)));
    let multibyte = "é".repeat(30);
    assert_eq!(ellipsize(&multibyte, 25), format!("{}...", "é".repeat(22)));
}
