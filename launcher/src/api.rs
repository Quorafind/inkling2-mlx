//! Shared pieces of the OpenAI-compatible API: app state, errors, request parameters.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::worker::Worker;

pub struct Defaults {
    pub effort: Option<String>,
    pub max_tokens: u64,
    pub temperature: f64,
}

/// A stored Responses API response: what GET returns, and the conversation that a
/// later `previous_response_id` continues (chat-format messages, without the
/// per-request instructions).
#[derive(Clone)]
pub struct Stored {
    pub response: Value,
    pub history: Vec<Value>,
}

pub struct ResponseStore {
    map: HashMap<String, Stored>,
    order: VecDeque<String>,
    cap: usize,
}

impl ResponseStore {
    pub fn new(cap: usize) -> Self {
        Self { map: HashMap::new(), order: VecDeque::new(), cap }
    }
    pub fn get(&self, id: &str) -> Option<Stored> {
        self.map.get(id).cloned()
    }
    /// Returns the ids evicted to stay within the cap.
    pub fn put(&mut self, id: String, s: Stored) -> Vec<String> {
        if self.map.insert(id.clone(), s).is_none() {
            self.order.push_back(id);
        }
        let mut evicted = vec![];
        while self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
                evicted.push(old);
            }
        }
        evicted
    }
    pub fn remove(&mut self, id: &str) -> bool {
        self.order.retain(|x| x != id);
        self.map.remove(id).is_some()
    }
}

pub struct AppState {
    pub worker: Arc<Worker>,
    pub model_id: String,
    pub api_key: Option<String>,
    pub defaults: Defaults,
    pub responses: Mutex<ResponseStore>,
}

pub type St = Arc<AppState>;

pub struct ApiError(pub StatusCode, pub String, pub Option<&'static str>);

impl ApiError {
    pub fn bad(msg: impl Into<String>) -> Self {
        ApiError(StatusCode::BAD_REQUEST, msg.into(), None)
    }
    pub fn param(msg: impl Into<String>, param: &'static str) -> Self {
        ApiError(StatusCode::BAD_REQUEST, msg.into(), Some(param))
    }
    pub fn not_found(msg: impl Into<String>, param: Option<&'static str>) -> Self {
        ApiError(StatusCode::NOT_FOUND, msg.into(), param)
    }
    pub fn unavailable(msg: impl Into<String>) -> Self {
        ApiError(StatusCode::SERVICE_UNAVAILABLE, msg.into(), None)
    }
    /// An error event from the worker: ValueError means a bad request.
    pub fn from_worker(msg: &str) -> Self {
        if msg.starts_with("ValueError") || msg.starts_with("TemplateError") || msg.starts_with("KeyError") {
            ApiError::bad(msg)
        } else {
            ApiError(StatusCode::INTERNAL_SERVER_ERROR, msg.to_string(), None)
        }
    }
    pub fn body(&self) -> Value {
        let kind = match self.0 {
            StatusCode::BAD_REQUEST | StatusCode::NOT_FOUND => "invalid_request_error",
            StatusCode::UNAUTHORIZED => "authentication_error",
            _ => "server_error",
        };
        json!({"error": {"message": self.1, "type": kind, "param": self.2, "code": null}})
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(self.body())).into_response()
    }
}

pub fn check_auth(st: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(key) = &st.api_key else { return Ok(()) };
    let got = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("");
    if got.strip_prefix("Bearer ").map(str::trim) == Some(key.as_str()) {
        Ok(())
    } else {
        Err(ApiError(StatusCode::UNAUTHORIZED, "invalid API key".into(), None))
    }
}

pub fn parse_body(bytes: &[u8]) -> Result<Value, ApiError> {
    let v: Value = serde_json::from_slice(bytes).map_err(|e| ApiError::bad(format!("invalid JSON body: {e}")))?;
    if !v.is_object() {
        return Err(ApiError::bad("the request body must be a JSON object"));
    }
    Ok(v)
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

/// Sampling parameters shared by both APIs, in the worker's request format.
pub fn sampling(st: &AppState, body: &Value, max_tokens: Option<u64>, effort: Option<&Value>) -> Value {
    let mut req = json!({
        "max_tokens": max_tokens.unwrap_or(st.defaults.max_tokens),
        "temperature": body.get("temperature").and_then(Value::as_f64).unwrap_or(st.defaults.temperature),
        "top_p": body.get("top_p").and_then(Value::as_f64).unwrap_or(1.0),
        "top_k": body.get("top_k").and_then(Value::as_u64).unwrap_or(0),
    });
    if let Some(seed) = body.get("seed").and_then(Value::as_i64) {
        req["seed"] = json!(seed);
    }
    // Loop passes for variable-loop checkpoints (an extension; OpenAI SDKs send it via extra_body).
    if let Some(passes) = body.get("passes").and_then(Value::as_u64) {
        req["passes"] = json!(passes);
    }
    let stop = match body.get("stop") {
        Some(Value::String(s)) => vec![json!(s)],
        Some(Value::Array(a)) => a.iter().filter(|s| s.is_string()).cloned().collect(),
        _ => vec![],
    };
    req["stop"] = json!(stop);
    let effort = match effort {
        Some(Value::String(s)) if !s.is_empty() => Some(json!(s)),
        Some(Value::Number(n)) => Some(json!(n)),
        _ => st.defaults.effort.as_ref().map(|s| json!(s)),
    };
    if let Some(e) = effort {
        req["effort"] = e;
    }
    req
}

/// The worker's tool calls -> OpenAI arguments strings.
pub fn arguments_string(args: &Value) -> String {
    match args {
        Value::String(s) => s.clone(),
        Value::Null => "{}".into(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".into()),
    }
}

pub fn usage_chat(u: &Value) -> Value {
    let p = u["prompt_tokens"].as_u64().unwrap_or(0);
    let c = u["completion_tokens"].as_u64().unwrap_or(0);
    json!({
        "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
        "prompt_tokens_details": {"cached_tokens": u["cached_tokens"].as_u64().unwrap_or(0)},
        "completion_tokens_details": {"reasoning_tokens": u["reasoning_tokens"].as_u64().unwrap_or(0)},
    })
}

pub fn usage_responses(u: &Value) -> Value {
    let p = u["prompt_tokens"].as_u64().unwrap_or(0);
    let c = u["completion_tokens"].as_u64().unwrap_or(0);
    json!({
        "input_tokens": p, "output_tokens": c, "total_tokens": p + c,
        "input_tokens_details": {"cached_tokens": u["cached_tokens"].as_u64().unwrap_or(0)},
        "output_tokens_details": {"reasoning_tokens": u["reasoning_tokens"].as_u64().unwrap_or(0)},
    })
}

/// Responses-style or chat-style function tools -> the chat format the template renders.
/// Non-function tools (web search, file search, ...) are not supported and are dropped.
pub fn function_tools(tools: Option<&Value>, tool_choice: Option<&Value>) -> Value {
    if tool_choice.and_then(Value::as_str) == Some("none") {
        return Value::Null;
    }
    let Some(Value::Array(list)) = tools else { return Value::Null };
    let out: Vec<Value> = list
        .iter()
        .filter(|t| t.get("type").and_then(Value::as_str).unwrap_or("function") == "function")
        .map(|t| {
            if let Some(f) = t.get("function") {
                json!({"type": "function", "function": f})
            } else {
                let mut f = serde_json::Map::new();
                for k in ["name", "description", "parameters"] {
                    if let Some(v) = t.get(k) {
                        f.insert(k.into(), v.clone());
                    }
                }
                json!({"type": "function", "function": f})
            }
        })
        .collect();
    if out.is_empty() {
        Value::Null
    } else {
        Value::Array(out)
    }
}
