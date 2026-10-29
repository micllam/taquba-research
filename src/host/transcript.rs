//! The transcript of a host run: its messages in a form of this crate, which a
//! release of Rig does not change.

use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, Message, ToolCall, ToolFunction, ToolName,
    ToolResult, ToolResultContent, UserContent,
};
use serde::{Deserialize, Serialize};
use taquba_workflow::StepError;

/// A message of a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Record {
    /// The prompt of the run.
    User {
        /// The text of the prompt.
        text: String,
    },
    /// A reply of the model.
    Assistant {
        /// The text of the reply.
        text: String,
        /// The tool calls of the reply, in the order of the model.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        calls: Vec<RecordedCall>,
    },
    /// The results of the tool calls of the preceding reply.
    Results {
        /// The results, in the order of the calls.
        results: Vec<RecordedResult>,
    },
}

/// A tool call of a reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedCall {
    /// The call id of the provider.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The arguments of the call.
    pub arguments: serde_json::Value,
}

/// The result of a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedResult {
    /// The call id of the provider.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// The text that the model reads.
    pub text: String,
    /// Whether the call failed or never ran.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

/// The records of `messages`. The records omit a system message, reasoning and
/// a provider item.
pub(crate) fn records(messages: &[Message]) -> Vec<Record> {
    let mut records = Vec::new();
    for message in messages {
        match message {
            Message::System { .. } => {}
            Message::User { content } => {
                let results: Vec<RecordedResult> = content
                    .iter()
                    .filter_map(|item| match item {
                        UserContent::ToolResult(result) => Some(recorded_result(result)),
                        _ => None,
                    })
                    .collect();
                if results.is_empty() {
                    let text = content
                        .iter()
                        .filter_map(|item| match item {
                            UserContent::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        })
                        .collect();
                    records.push(Record::User { text });
                } else {
                    records.push(Record::Results { results });
                }
            }
            Message::Assistant(reply) => records.push(Record::Assistant {
                text: reply.text(),
                calls: reply
                    .tool_calls()
                    .map(|call| RecordedCall {
                        id: call.id.wire().into_owned(),
                        name: call.function.name.as_str().to_string(),
                        arguments: serde_json::Value::Object(call.function.arguments.clone()),
                    })
                    .collect(),
            }),
        }
    }
    records
}

fn recorded_result(result: &ToolResult) -> RecordedResult {
    let text = result
        .content
        .iter()
        .map(|content| match content {
            ToolResultContent::Text(text) => text.text.clone(),
            ToolResultContent::Json { value, .. } => value.to_string(),
            ToolResultContent::Image(_) => "[image]".to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    RecordedResult {
        id: result.call.wire().into_owned(),
        name: result.name.as_str().to_string(),
        text,
        is_error: result.is_error,
    }
}

/// Remove the result of the call `id` from `records`, and a results record that
/// it leaves empty.
pub(crate) fn remove_result(records: &mut Vec<Record>, id: &str) {
    for record in records.iter_mut() {
        if let Record::Results { results } = record {
            results.retain(|result| result.id != id);
        }
    }
    records.retain(|record| !matches!(record, Record::Results { results } if results.is_empty()));
}

/// The Rig messages of `records`. A refused tool name is a permanent error.
pub(crate) fn messages(records: &[Record]) -> Result<Vec<Message>, StepError> {
    records.iter().map(message).collect()
}

fn message(record: &Record) -> Result<Message, StepError> {
    Ok(match record {
        Record::User { text } => Message::user(text.clone()),
        Record::Assistant { text, calls } => {
            let mut content = Vec::with_capacity(calls.len() + 1);
            if !text.is_empty() {
                content.push(AssistantContent::text(text.clone()));
            }
            for call in calls {
                let function = ToolFunction::new(tool_name(&call.name)?, call.arguments.clone());
                content.push(AssistantContent::ToolCall(ToolCall::from_wire(
                    call.id.clone(),
                    function,
                )));
            }
            Message::Assistant(AssistantMessage::new(content))
        }
        Record::Results { results } => {
            let content = results
                .iter()
                .map(|result| {
                    Ok(UserContent::ToolResult(ToolResult {
                        call: CallId::from_wire(result.id.clone()),
                        name: tool_name(&result.name)?,
                        content: vec![ToolResultContent::text(result.text.clone())],
                        is_error: result.is_error,
                    }))
                })
                .collect::<Result<_, StepError>>()?;
            Message::User { content }
        }
    })
}

fn tool_name(name: &str) -> Result<ToolName, StepError> {
    ToolName::new(name).map_err(|e| StepError::permanent(format!("transcript tool name: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_of_the_messages_of_records_are_the_records() {
        let recorded = vec![
            Record::User {
                text: "query".to_string(),
            },
            Record::Assistant {
                text: "searching".to_string(),
                calls: vec![RecordedCall {
                    id: "call_1".to_string(),
                    name: "web_search".to_string(),
                    arguments: serde_json::json!({ "query": "gaps" }),
                }],
            },
            Record::Results {
                results: vec![RecordedResult {
                    id: "call_1".to_string(),
                    name: "web_search".to_string(),
                    text: "no results".to_string(),
                    is_error: true,
                }],
            },
            Record::Assistant {
                text: "done".to_string(),
                calls: Vec::new(),
            },
        ];
        assert_eq!(records(&messages(&recorded).unwrap()), recorded);
    }
}
