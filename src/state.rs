//! Per-run state that lives inside the step payload between transitions.
//!
//! The runner is stateless across calls. Every value that the next step reads
//! is serialized into [`ResearchState`] and returned via
//! [`taquba_workflow::StepOutcome::Continue`].

use std::collections::{BTreeMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::report::Citation;
use crate::search::SearchResult;

/// Configuration the user passes to a research run. Build with
/// [`ResearchConfig::new`], passing the provider-specific model identifier
/// explicitly. The other fields take their defaults from `new` and can be
/// overridden field by field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    /// Number of sub-questions the planning step decomposes the query into. The
    /// planner can return fewer when the query does not warrant `depth`
    /// distinct sub-questions.
    pub depth: usize,
    /// Hard cap on URLs queued for fetching. Search results beyond this cap are
    /// dropped before the fetch phase starts.
    pub max_sources: usize,
    /// Model identifier passed to Rig. Provider-specific, such as
    /// `"gpt-5-nano"` for OpenAI or `"claude-haiku-4-5"` for Anthropic. Must be
    /// a valid identifier for whichever provider the runner was built against
    /// (see [`crate::ResearchStepRunner::new_openai`] /
    /// [`crate::ResearchStepRunner::new_anthropic`]).
    pub model: String,
    /// Maximum tokens per single LLM call. When `None`, no explicit limit is
    /// sent and the provider's or model's default applies.
    pub max_tokens_per_call: Option<u64>,
    /// Per-page text limit fed to the summarising step (UTF-8 chars). Larger
    /// pages are truncated.
    pub max_page_chars: usize,
    /// Model-call budget of the investigating step. With 0 the run skips the
    /// step. A state persisted without the field decodes as 0.
    #[serde(default)]
    pub investigation_turns: usize,
    /// How long a run waits for the reply of the user to a question of the
    /// investigating agent, in seconds. With `None` the agent cannot ask. After
    /// the wait the agent proceeds on a stated assumption.
    #[serde(default)]
    pub clarification_wait_secs: Option<u64>,
}

impl ResearchConfig {
    /// Build a `ResearchConfig` with the given model identifier and the
    /// standard defaults for every other field. The model string must match the
    /// provider of the runner.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            depth: 6,
            max_sources: 30,
            model: model.into(),
            max_tokens_per_call: None,
            max_page_chars: 16_000,
            investigation_turns: 8,
            clarification_wait_secs: None,
        }
    }
}

/// Aggregate token usage across every LLM call in a run. Mirrors the fields of
/// `rig_core::completion::Usage`, and the on-disk JSON layout of the persisted
/// run state does not depend on Rig's struct definition. All zeros means either
/// "no calls yet" or "the provider did not report usage."
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    /// Sum of input ("prompt") tokens reported across all calls, cache reads
    /// and writes included.
    pub input_tokens: u64,
    /// Sum of output ("completion") tokens reported across all calls.
    pub output_tokens: u64,
    /// Sum of `total_tokens` across all calls. A call reports a total only with
    /// both its input and its output count.
    pub total_tokens: u64,
    /// Input tokens read from a provider-managed prompt cache.
    pub cached_input_tokens: u64,
    /// Input tokens written into a provider-managed prompt cache.
    pub cache_creation_input_tokens: u64,
    /// Input tokens consumed by provider tool-use prompts. Zero for this
    /// crate's tool-less agents.
    #[serde(default)]
    pub tool_use_prompt_tokens: u64,
    /// Tokens consumed by internal reasoning / "thinking" by reasoning-capable
    /// models.
    pub reasoning_tokens: u64,
}

impl TokenUsage {
    /// `true` when all fields are zero: either no LLM call is recorded yet or
    /// the provider did not report usage for any of them.
    pub fn is_zero(&self) -> bool {
        *self == Self::default()
    }
}

/// Lifecycle phase of a research run. Each step advances the state machine
/// through these phases. Some phases iterate: `Searching` pops one item off
/// `search_queue` per step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Phase {
    /// Decomposing the user query into sub-questions via the LLM.
    Planning,
    /// Running the search backend once per sub-question.
    Searching,
    /// Fetching each unique URL via HTTP.
    Fetching,
    /// Summarising each fetched page via the LLM.
    Summarizing,
    /// Searching and fetching pages for the gaps of the summaries, through a
    /// Rig agent with tools.
    Investigating,
    /// Combining per-page summaries into a single narrative via the LLM.
    Synthesizing,
    /// Writing the final markdown report via the LLM.
    Writing,
}

impl Phase {
    /// Stable lowercase identifier used in CLI output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::Searching => "searching",
            Self::Fetching => "fetching",
            Self::Summarizing => "summarizing",
            Self::Investigating => "investigating",
            Self::Synthesizing => "synthesizing",
            Self::Writing => "writing",
        }
    }
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Progress snapshot of an in-flight run, decoded from a step job's payload by
/// inspection commands.
#[derive(Debug, Clone)]
pub struct StateSummary {
    /// Phase the next (or current) step will run.
    pub phase: Phase,
    /// Steps the runner completed so far.
    pub steps_completed: u32,
    /// The question of the investigating agent while the run waits for the
    /// reply of the user.
    pub pending_question: Option<String>,
}

/// Decode the progress-relevant fields of a step-job payload. Returns `None`
/// when the payload is not a serialized research state.
pub fn summarize_state(payload: &[u8]) -> Option<StateSummary> {
    let state = ResearchState::from_bytes(payload).ok()?;
    Some(StateSummary {
        phase: state.phase,
        steps_completed: state.steps_completed,
        pending_question: state
            .clarification
            .filter(|c| c.reply.is_none())
            .map(|c| c.question),
    })
}

/// Per-page fetched text, plus the title that citations use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchedPage {
    /// Best-effort page title (from `<title>` or the original search result).
    pub title: String,
    /// Extracted plain text, truncated to [`ResearchConfig::max_page_chars`].
    pub text: String,
}

/// Per-page summary produced by the summarising step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    /// Title copied from the fetched page.
    pub title: String,
    /// One-paragraph summary keyed to the user's query.
    pub text: String,
    /// LLM-assigned relevance score, 0.0–1.0.
    pub relevance: f32,
}

/// A verbatim span quoted from a cited source, keyed to the [`Citation`] it
/// backs. Only ever populated from provider-returned citation metadata
/// (currently Anthropic document citations), and empty otherwise.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SourceQuote {
    /// 1-based index of the [`Citation`] this quote supports.
    pub citation_index: usize,
    /// Verbatim span quoted from the source.
    pub excerpt: String,
}

/// Product of the synthesizing step: the narrative, the numbered source list it
/// references, and any provider-returned excerpts backing those references.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SynthesisOutput {
    /// Synthesized narrative produced by the synthesizing step.
    pub narrative: String,
    /// Numbered sources referenced by the narrative's `[N]` markers.
    pub citations: Vec<Citation>,
    /// Verbatim quotes from cited sources, each keyed to a citation index.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<SourceQuote>,
}

/// The entire bytes-in / bytes-out state the runner threads between steps.
/// Serialized as JSON so a `taquba-research show` against a future version can
/// still decode old runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchState {
    /// User query that started the run.
    pub query: String,
    /// Configuration captured at submission time.
    pub config: ResearchConfig,
    /// Current phase of the state machine.
    pub phase: Phase,
    /// Wall-clock instant the run was submitted (for `RunStats`).
    pub started_at: DateTime<Utc>,
    /// Monotonically counted steps the runner executed, separate from the
    /// workflow runtime's `step_number` (which counts queue transitions
    /// including ones that returned `ContinueAfter`).
    pub steps_completed: u32,
    /// Sub-questions produced by the planning step.
    pub sub_questions: Vec<String>,
    /// Queue of sub-question indices yet to be searched.
    pub search_queue: VecDeque<usize>,
    /// Search results per sub-question (key = sub-question index).
    pub search_results: BTreeMap<usize, Vec<SearchResult>>,
    /// Queue of URLs yet to be fetched. Deduplicated against `fetched.keys()`
    /// before insertion.
    pub fetch_queue: VecDeque<Url>,
    /// Successfully fetched pages, with the URL as the key.
    pub fetched: BTreeMap<Url, FetchedPage>,
    /// Queue of URLs yet to be summarised.
    pub summarize_queue: VecDeque<Url>,
    /// Per-page summaries, with the URL as the key.
    pub summaries: BTreeMap<Url, Summary>,
    /// Synthesized narrative and citation evidence produced by the synthesizing
    /// step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthesis: Option<SynthesisOutput>,
    /// Aggregate token usage across every LLM call made by this run.
    pub token_usage: TokenUsage,
    /// The question of the investigating agent to the user, and its reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clarification: Option<Clarification>,
}

/// The question that the investigating agent of a run asks the user, a single
/// question per run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clarification {
    /// The journal key of the `ask_user` call.
    pub key: String,
    /// The question.
    pub question: String,
    /// When the agent asked.
    pub asked_at: DateTime<Utc>,
    /// The reply, or `None` while the run waits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply: Option<Reply>,
}

/// How a question of the investigating agent was resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "text", rename_all = "snake_case")]
pub enum Reply {
    /// The user replied with this text.
    Given(String),
    /// The wait ran out, and the agent proceeds on an assumption.
    Assumed,
}

impl ResearchState {
    /// Build the initial state for a fresh run.
    pub fn new(query: impl Into<String>, config: ResearchConfig) -> Self {
        Self {
            query: query.into(),
            config,
            phase: Phase::Planning,
            started_at: Utc::now(),
            steps_completed: 0,
            sub_questions: Vec::new(),
            search_queue: VecDeque::new(),
            search_results: BTreeMap::new(),
            fetch_queue: VecDeque::new(),
            fetched: BTreeMap::new(),
            summarize_queue: VecDeque::new(),
            summaries: BTreeMap::new(),
            synthesis: None,
            token_usage: TokenUsage::default(),
            clarification: None,
        }
    }

    /// Encode the state into the bytes of the step payload.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ResearchState is serde-derivable")
    }

    /// Decode a state from a step payload.
    pub fn from_bytes(bytes: &[u8]) -> serde_json::Result<Self> {
        serde_json::from_slice(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_serde() {
        let mut s = ResearchState::new("a query", ResearchConfig::new("gpt-4o-mini"));
        s.config.max_tokens_per_call = Some(2048);
        s.config.investigation_turns = 3;
        s.config.clarification_wait_secs = Some(600);
        s.clarification = Some(Clarification {
            key: "tool/0/call_1".to_string(),
            question: "Which edition?".to_string(),
            asked_at: Utc::now(),
            reply: Some(Reply::Given("2024".to_string())),
        });
        let url: Url = "https://example.com/page".parse().unwrap();
        s.fetched.insert(
            url.clone(),
            FetchedPage {
                title: "Example".to_string(),
                text: "page text".to_string(),
            },
        );
        s.summaries.insert(
            url.clone(),
            Summary {
                title: "Example".to_string(),
                text: "summary text".to_string(),
                relevance: 0.5,
            },
        );
        s.token_usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 20,
            total_tokens: 120,
            cached_input_tokens: 30,
            cache_creation_input_tokens: 40,
            tool_use_prompt_tokens: 5,
            reasoning_tokens: 10,
        };
        s.synthesis = Some(SynthesisOutput {
            narrative: "synthesized answer".to_string(),
            citations: vec![Citation {
                index: 1,
                url: "https://example.com".parse().unwrap(),
                title: "Example".to_string(),
            }],
            evidence: vec![SourceQuote {
                citation_index: 1,
                excerpt: "quoted source text".to_string(),
            }],
        });
        let bytes = s.to_bytes();
        let back = ResearchState::from_bytes(&bytes).unwrap();
        assert_eq!(back.query, "a query");
        assert_eq!(back.phase, Phase::Planning);
        assert_eq!(back.steps_completed, 0);
        assert_eq!(
            back.fetched.get(&url).map(|p| p.text.as_str()),
            Some("page text")
        );
        assert_eq!(
            back.summaries.get(&url).map(|m| m.text.as_str()),
            Some("summary text")
        );
        assert_eq!(back.synthesis, s.synthesis);
        assert_eq!(back.token_usage, s.token_usage);
        assert_eq!(back.config.max_tokens_per_call, Some(2048));
        assert_eq!(back.config.investigation_turns, 3);
        assert_eq!(back.config.clarification_wait_secs, Some(600));
        assert_eq!(back.clarification, s.clarification);
    }

    #[test]
    fn summarize_state_reports_the_question_only_while_the_run_waits() {
        let mut s = ResearchState::new("a query", ResearchConfig::new("m"));
        s.clarification = Some(Clarification {
            key: "tool/0/call_1".to_string(),
            question: "Which edition?".to_string(),
            asked_at: Utc::now(),
            reply: None,
        });
        let waiting = summarize_state(&s.to_bytes()).unwrap();
        assert_eq!(waiting.pending_question.as_deref(), Some("Which edition?"));

        s.clarification.as_mut().unwrap().reply = Some(Reply::Assumed);
        let resolved = summarize_state(&s.to_bytes()).unwrap();
        assert_eq!(resolved.pending_question, None);
    }

    #[test]
    fn research_config_decodes_without_investigation_turns() {
        let json = r#"{"depth":6,"max_sources":30,"model":"m",
            "max_tokens_per_call":null,"max_page_chars":100}"#;
        let config: ResearchConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.investigation_turns, 0);
    }

    #[test]
    fn token_usage_decodes_without_tool_use_prompt_tokens() {
        let json = r#"{"input_tokens":1,"output_tokens":2,"total_tokens":3,
            "cached_input_tokens":0,"cache_creation_input_tokens":0,
            "reasoning_tokens":0}"#;
        let usage: TokenUsage = serde_json::from_str(json).unwrap();
        assert_eq!(usage.tool_use_prompt_tokens, 0);
        assert_eq!(usage.input_tokens, 1);
    }
}
