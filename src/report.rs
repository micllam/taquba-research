//! The [`Report`] type returned at the end of a successful research run,
//! plus markdown rendering with numeric citations.

use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use url::Url;

/// Final result of a successful research run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Run identifier produced by the workflow runtime.
    pub run_id: String,
    /// Original user query that started the run.
    pub query: String,
    /// Rendered markdown report including inline `[N]` numeric citations
    /// and a citations list at the end.
    pub markdown: String,
    /// Citations referenced from `markdown`. Indices match the inline
    /// `[N]` markers.
    pub citations: Vec<Citation>,
    /// Aggregated run-level statistics.
    pub stats: RunStats,
}

/// A single citation referenced in [`Report::markdown`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// 1-based index matching the inline `[N]` marker in the markdown body.
    pub index: usize,
    /// Source URL.
    pub url: Url,
    /// Page title (best-effort, taken from the fetched page or the search
    /// result snippet).
    pub title: String,
}

/// Aggregated stats for a finished run. Surfaced both via the library
/// [`Report`] and the CLI's end-of-run summary line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStats {
    /// Number of steps the runner completed.
    pub steps_completed: u32,
    /// Wall-clock time from submission to terminal hook.
    #[serde(with = "duration_secs")]
    pub wall_time: Duration,
    /// UTC instant the run was submitted.
    pub started_at: DateTime<Utc>,
    /// UTC instant the terminal hook fired.
    pub finished_at: DateTime<Utc>,
}

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        d.as_secs_f64().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = f64::deserialize(d)?;
        Ok(Duration::from_secs_f64(secs.max(0.0)))
    }
}

/// Render the metadata block, body, and citations list into a single
/// markdown document.
pub(crate) fn render_markdown(
    query: &str,
    run_id: &str,
    body: &str,
    citations: &[Citation],
    stats: &RunStats,
) -> String {
    let mut out = String::with_capacity(body.len() + 1024);
    out.push_str("# Research report\n\n");
    out.push_str(&format!("**Query:** {query}\n\n"));
    out.push_str(&format!("**Run:** `{run_id}`  \n"));
    out.push_str(&format!(
        "**Generated:** {}  \n",
        stats.finished_at.to_rfc3339()
    ));
    out.push_str(&format!(
        "**Stats:** {} steps · {}\n\n",
        stats.steps_completed,
        format_duration(stats.wall_time),
    ));
    out.push_str("---\n\n");
    out.push_str(body.trim());
    out.push_str("\n\n## Citations\n\n");
    if citations.is_empty() {
        out.push_str("_No sources cited._\n");
    } else {
        for c in citations {
            out.push_str(&format!("[{}] [{}]({})\n", c.index, c.title, c.url));
        }
    }
    out
}

fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    let m = total / 60;
    let s = total % 60;
    if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}
