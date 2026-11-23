//! The Responses API: POST /v1/responses, GET / DELETE /v1/responses/{id}.
//!
//! `previous_response_id` continues a stored response: the launcher keeps the
//! conversation, the worker keeps the KV state under the response id, so a follow-up
//! turn prefills only its new input (not the history, not the previous reply).
//! Clients that resend everything (store = false) still hit the worker's prefix cache.

use std::convert::Infallible;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::api::{arguments_string, check_auth, function_tools, new_id, now, parse_body, sampling, usage_responses, ApiError, St, Stored};

/// Responses input (a string or a list of items) -> chat-format messages.
fn input_to_messages(input: &Value) -> Result<Vec<Value>, ApiError> {
    let items = match input {
        Value::String(s) => return Ok(vec![json!({"role": "user", "content": s})]),
        Value::Array(a) => a,
        _ => return Err(ApiError::param("`input` must be a string or an array of items", "input")),
    };
    let mut out: Vec<Value> = vec![];
    let mut reasoning: Option<String> = None;
    for item in items {
        let kind = item.get("type").and_then(Value::as_str).unwrap_or("message");
        match kind {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let content = convert_content(item.get("content"))?;
                let mut msg = json!({"role": role, "content": content});
                if role == "assistant" {
                    if let Some(r) = reasoning.take() {
                        msg["reasoning_content"] = json!(r);
                    }
                }
                out.push(msg);
            }
            "reasoning" => {
                let text: String = item.get("content").and_then(Value::as_array).into_iter().flatten()
                    .filter_map(|p| p.get("text").and_then(Value::as_str)).collect();
                if !text.is_empty() {
                    reasoning = Some(text);
                }
            }
            "function_call" => {
                let call = json!({
                    "id": item.get("call_id").or(item.get("id")),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments").cloned().unwrap_or(json!("{}"))},
                });
                let merge = matches!(out.last(), Some(m) if m["role"] == "assistant");
                if !merge {
                    let mut msg = json!({"role": "assistant", "content": ""});
                    if let Some(r) = reasoning.take() {
                        msg["reasoning_content"] = json!(r);
                    }
                    out.push(msg);
                }
                let last = out.last_mut().unwrap();
                if last.get("tool_calls").is_none() {
                    last["tool_calls"] = json!([]);
                }
                last["tool_calls"].as_array_mut().unwrap().push(call);
            }
            "function_call_output" => {
                let output = match item.get("output") {
                    Some(Value::String(s)) => json!(s),
                    Some(v @ Value::Array(_)) => convert_content(Some(v))?,
                    Some(v) => json!(v.to_string()),
                    None => json!(""),
                };
                out.push(json!({"role": "tool", "tool_call_id": item.get("call_id"), "content": output}));
            }
            "item_reference" => {}
            other => return Err(ApiError::param(format!("unsupported input item type {other:?}"), "input")),
        }
    }
    Ok(out)
}

fn convert_content(c: Option<&Value>) -> Result<Value, ApiError> {
    match c {
        None | Some(Value::Null) => Ok(json!("")),
        Some(Value::String(s)) => Ok(json!(s)),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("input_text") | Some("output_text") | Some("text") | Some("refusal") => {
                    Ok(json!({"type": "text", "text": p.get("text").or(p.get("refusal")).cloned().unwrap_or(json!(""))}))
                }
                Some("input_image") => {
                    let url = p.get("image_url").cloned().unwrap_or(Value::Null);
                    if url.is_null() {
                        Err(ApiError::param("input_image needs an image_url (data: URL or local path)", "input"))
                    } else {
                        Ok(json!({"type": "image_url", "image_url": {"url": url}}))
                    }
                }
                other => Err(ApiError::param(format!("unsupported content part {other:?}"), "input")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Some(_) => Err(ApiError::param("message content must be a string or a list of parts", "input")),
    }
}

enum Open {
    None,
    Reasoning { id: String, text: String },
    Message { id: String, text: String },
}

/// Builds the output items and, for streaming, the event sequence.
struct Builder {
    out: Vec<Value>,
    open: Open,
    events: Vec<(&'static str, Value)>,
}

impl Builder {
    fn new() -> Self {
        Self { out: vec![], open: Open::None, events: vec![] }
    }
    fn ev(&mut self, kind: &'static str, data: Value) {
        self.events.push((kind, data));
    }
    fn idx(&self) -> usize {
        self.out.len()
    }
    fn close(&mut self) {
        let i = self.idx();
        match std::mem::replace(&mut self.open, Open::None) {
            Open::None => {}
            Open::Reasoning { id, text } => {
                self.ev("response.reasoning_text.done", json!({"item_id": id, "output_index": i, "content_index": 0, "text": text}));
                let item = json!({"type": "reasoning", "id": id, "summary": [], "content": [{"type": "reasoning_text", "text": text}]});
                self.ev("response.output_item.done", json!({"output_index": i, "item": item}));
                self.out.push(item);
            }
            Open::Message { id, text } => {
                let part = json!({"type": "output_text", "text": text, "annotations": []});
                self.ev("response.output_text.done", json!({"item_id": id, "output_index": i, "content_index": 0, "text": text, "logprobs": []}));
                self.ev("response.content_part.done", json!({"item_id": id, "output_index": i, "content_index": 0, "part": part}));
                let item = json!({"type": "message", "id": id, "status": "completed", "role": "assistant", "content": [part]});
                self.ev("response.output_item.done", json!({"output_index": i, "item": item}));
                self.out.push(item);
            }
        }
    }
    fn reasoning(&mut self, t: &str) {
        if !matches!(self.open, Open::Reasoning { .. }) {
            self.close();
            let id = new_id("rs_");
            self.ev("response.output_item.added", json!({"output_index": self.idx(),
                "item": {"type": "reasoning", "id": id, "summary": [], "content": []}}));
            self.open = Open::Reasoning { id, text: String::new() };
        }
        let i = self.idx();
        if let Open::Reasoning { id, text } = &mut self.open {
            text.push_str(t);
            let data = json!({"item_id": id, "output_index": i, "content_index": 0, "delta": t});
            self.events.push(("response.reasoning_text.delta", data));
        }
    }
    fn text(&mut self, t: &str) {
        if !matches!(self.open, Open::Message { .. }) {
            self.close();
            let id = new_id("msg_");
            let i = self.idx();
            self.ev("response.output_item.added", json!({"output_index": i,
                "item": {"type": "message", "id": id, "status": "in_progress", "role": "assistant", "content": []}}));
            self.ev("response.content_part.added", json!({"item_id": id, "output_index": i, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}}));
            self.open = Open::Message { id, text: String::new() };
        }
        let i = self.idx();
        if let Open::Message { id, text } = &mut self.open {
            text.push_str(t);
            let data = json!({"item_id": id, "output_index": i, "content_index": 0, "delta": t, "logprobs": []});
            self.events.push(("response.output_text.delta", data));
        }
    }
    fn tool_call(&mut self, name: &Value, args: &Value) {
        self.close();
        let i = self.idx();
        let (id, call_id, a) = (new_id("fc_"), new_id("call_"), arguments_string(args));
        self.ev("response.output_item.added", json!({"output_index": i, "item":
            {"type": "function_call", "id": id, "call_id": call_id, "name": name, "arguments": "", "status": "in_progress"}}));
        self.ev("response.function_call_arguments.delta", json!({"item_id": id, "output_index": i, "delta": a}));
        self.ev("response.function_call_arguments.done", json!({"item_id": id, "output_index": i, "arguments": a, "name": name}));
        let item = json!({"type": "function_call", "id": id, "call_id": call_id, "name": name, "arguments": a, "status": "completed"});
        self.ev("response.output_item.done", json!({"output_index": i, "item": item}));
        self.out.push(item);
    }
    fn feed(&mut self, ev: &Value) {
        match ev["type"].as_str() {
            Some("delta") if ev["kind"] == "reasoning" => self.reasoning(ev["text"].as_str().unwrap_or("")),
            Some("delta") => self.text(ev["text"].as_str().unwrap_or("")),
            Some("tool_call") => self.tool_call(&ev["name"], &ev["arguments"]),
            _ => {}
        }
    }
    /// The finished reply as a chat-format assistant message (for the stored history).
    fn assistant_message(&self) -> Value {
        let mut msg = json!({"role": "assistant", "content": ""});
        let mut calls = vec![];
        for item in &self.out {
            match item["type"].as_str() {
                Some("reasoning") => msg["reasoning_content"] = item["content"][0]["text"].clone(),
                Some("message") => msg["content"] = item["content"][0]["text"].clone(),
                Some("function_call") => calls.push(json!({"id": item["call_id"], "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]}})),
                _ => {}
            }
        }
        if !calls.is_empty() {
            msg["tool_calls"] = json!(calls);
        }
        msg
    }
}

pub async fn create(State(st): State<St>, headers: HeaderMap, body: Bytes) -> Response {
    match run(st, headers, body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn run(st: St, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    check_auth(&st, &headers)?;
    let body = parse_body(&body)?;
    let prev = body.get("previous_response_id").and_then(Value::as_str).map(str::to_string);
    let mut history = match &prev {
        Some(p) => st.responses.lock().unwrap().get(p)
            .ok_or_else(|| ApiError::not_found(format!("Previous response with id '{p}' not found."), Some("previous_response_id")))?
            .history,
        None => vec![],
    };
    let input = body.get("input").ok_or_else(|| ApiError::param("`input` is required", "input"))?;
    history.extend(input_to_messages(input)?);
    let instructions = body.get("instructions").and_then(Value::as_str).filter(|s| !s.is_empty());
    let mut messages: Vec<Value> = instructions.map(|i| json!({"role": "system", "content": i})).into_iter().collect();
    messages.extend(history.iter().cloned());

    let effort = body.pointer("/reasoning/effort");
    let mut req = sampling(&st, &body, body.get("max_output_tokens").and_then(Value::as_u64), effort);
    let effort_used = req.get("effort").cloned().unwrap_or(Value::Null);
    req["messages"] = Value::Array(messages);
    req["tools"] = function_tools(body.get("tools"), body.get("tool_choice"));
    let id = new_id("resp_");
    let store = body.get("store").and_then(Value::as_bool).unwrap_or(true);
    if store {
        req["store_as"] = json!(id);
    }
    if let Some(p) = &prev {
        req["continue_from"] = json!(p);
    }
    let model = body.get("model").and_then(Value::as_str).unwrap_or(&st.model_id).to_string();
    let base = json!({
        "id": id, "object": "response", "created_at": now(), "status": "in_progress", "model": model,
        "output": [], "previous_response_id": prev, "instructions": instructions,
        "temperature": req["temperature"], "top_p": req["top_p"], "max_output_tokens": req["max_tokens"],
        "tools": body.get("tools").cloned().unwrap_or(json!([])),
        "tool_choice": body.get("tool_choice").cloned().unwrap_or(json!("auto")),
        "parallel_tool_calls": true, "store": store, "reasoning": {"effort": effort_used, "summary": null},
        "text": {"format": {"type": "text"}}, "truncation": "disabled", "usage": null, "error": null,
        "incomplete_details": null, "metadata": body.get("metadata").cloned().unwrap_or(json!({})),
    });
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let mut gen = st.worker.generate(req).await.map_err(ApiError::unavailable)?;

    if !stream {
        let mut b = Builder::new();
        loop {
            let ev = gen.events.recv().await.ok_or_else(|| ApiError::unavailable("the model worker exited"))?;
            match ev["type"].as_str() {
                Some("error") => {
                    gen.finish();
                    return Err(ApiError::from_worker(ev["message"].as_str().unwrap_or("worker error")));
                }
                Some("done") => {
                    gen.finish();
                    b.close();
                    let resp = finalize(&st, base, &b, &ev, history, store).await;
                    return Ok(Json(resp).into_response());
                }
                _ => b.feed(&ev),
            }
        }
    }

    let first = gen.events.recv().await.ok_or_else(|| ApiError::unavailable("the model worker exited"))?;
    if first["type"] == "error" {
        gen.finish();
        return Err(ApiError::from_worker(first["message"].as_str().unwrap_or("worker error")));
    }
    let s = async_stream::stream! {
        let mut seq = 0u64;
        let mut emit = |kind: &str, mut data: Value| {
            data["type"] = json!(kind);
            data["sequence_number"] = json!(seq);
            seq += 1;
            Ok::<Event, Infallible>(Event::default().event(kind).data(data.to_string()))
        };
        yield emit("response.created", json!({"response": base}));
        yield emit("response.in_progress", json!({"response": base}));
        let mut b = Builder::new();
        let mut pending = Some(first);
        loop {
            let ev = match pending.take() {
                Some(e) => e,
                None => match gen.events.recv().await {
                    Some(e) => e,
                    None => break,
                },
            };
            match ev["type"].as_str() {
                Some("error") => {
                    gen.finish();
                    let mut failed = base.clone();
                    failed["status"] = json!("failed");
                    failed["error"] = json!({"code": "server_error", "message": ev["message"]});
                    yield emit("response.failed", json!({"response": failed}));
                    break;
                }
                Some("done") => {
                    gen.finish();
                    b.close();
                    for (kind, data) in std::mem::take(&mut b.events) {
                        yield emit(kind, data);
                    }
                    let resp = finalize(&st, base.clone(), &b, &ev, history.clone(), store).await;
                    let kind = if resp["status"] == "incomplete" { "response.incomplete" } else { "response.completed" };
                    yield emit(kind, json!({"response": resp}));
                    break;
                }
                _ => {
                    b.feed(&ev);
                    for (kind, data) in std::mem::take(&mut b.events) {
                        yield emit(kind, data);
                    }
                }
            }
        }
    };
    Ok(Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))).into_response())
}

async fn finalize(st: &St, mut resp: Value, b: &Builder, done: &Value, mut history: Vec<Value>, store: bool) -> Value {
    let length = done["finish_reason"] == "length";
    resp["status"] = json!(if length { "incomplete" } else { "completed" });
    if length {
        resp["incomplete_details"] = json!({"reason": "max_output_tokens"});
    }
    resp["output"] = json!(b.out);
    resp["usage"] = usage_responses(&done["usage"]);
    if store {
        history.push(b.assistant_message());
        let id = resp["id"].as_str().unwrap_or_default().to_string();
        let evicted = st.responses.lock().unwrap().put(id, Stored { response: resp.clone(), history });
        for old in evicted {
            st.worker.forget(&old).await;
        }
    }
    resp
}

pub async fn get(State(st): State<St>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Err(e) = check_auth(&st, &headers) {
        return e.into_response();
    }
    match st.responses.lock().unwrap().get(&id) {
        Some(s) => Json(s.response).into_response(),
        None => ApiError::not_found(format!("Response with id '{id}' not found."), None).into_response(),
    }
}

pub async fn delete(State(st): State<St>, headers: HeaderMap, Path(id): Path<String>) -> Response {
    if let Err(e) = check_auth(&st, &headers) {
        return e.into_response();
    }
    let removed = st.responses.lock().unwrap().remove(&id);
    if !removed {
        return ApiError::not_found(format!("Response with id '{id}' not found."), None).into_response();
    }
    st.worker.forget(&id).await;
    Json(json!({"id": id, "object": "response", "deleted": true})).into_response()
}
