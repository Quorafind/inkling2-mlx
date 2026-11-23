//! POST /v1/chat/completions (streaming and not). Thinking goes to `reasoning_content`
//! (the field DeepSeek, vLLM and most chat front-ends read), tool calls to `tool_calls`.
//! Clients resend the whole history; the worker reuses the KV of the longest stored
//! prefix, so only the new messages are prefilled.

use std::convert::Infallible;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::api::{arguments_string, check_auth, function_tools, new_id, now, parse_body, sampling, usage_chat, ApiError, St};

pub async fn completions(State(st): State<St>, headers: HeaderMap, body: Bytes) -> Response {
    match run(st, headers, body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn run(st: St, headers: HeaderMap, body: Bytes) -> Result<Response, ApiError> {
    check_auth(&st, &headers)?;
    let body = parse_body(&body)?;
    let messages = body.get("messages").and_then(Value::as_array).filter(|m| !m.is_empty())
        .ok_or_else(|| ApiError::param("`messages` must be a non-empty array", "messages"))?;
    if body.get("n").and_then(Value::as_u64).unwrap_or(1) != 1 {
        return Err(ApiError::param("only n = 1 is supported", "n"));
    }
    let max = body.get("max_completion_tokens").or(body.get("max_tokens")).and_then(Value::as_u64);
    let mut req = sampling(&st, &body, max, body.get("reasoning_effort"));
    req["messages"] = Value::Array(messages.clone());
    req["tools"] = function_tools(body.get("tools"), body.get("tool_choice"));
    let model = body.get("model").and_then(Value::as_str).unwrap_or(&st.model_id).to_string();
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);

    let mut gen = st.worker.generate(req).await.map_err(ApiError::unavailable)?;
    let id = new_id("chatcmpl-");
    let created = now();

    if !stream {
        loop {
            let ev = gen.events.recv().await.ok_or_else(|| ApiError::unavailable("the model worker exited"))?;
            match ev["type"].as_str() {
                Some("error") => {
                    gen.finish();
                    return Err(ApiError::from_worker(ev["message"].as_str().unwrap_or("worker error")));
                }
                Some("done") => {
                    gen.finish();
                    let calls: Vec<Value> = ev["tool_calls"].as_array().cloned().unwrap_or_default().iter()
                        .map(|c| json!({"id": new_id("call_"), "type": "function",
                                        "function": {"name": c["name"], "arguments": arguments_string(&c["arguments"])}}))
                        .collect();
                    let text = ev["text"].as_str().unwrap_or("");
                    let mut message = json!({"role": "assistant", "content": if text.is_empty() && !calls.is_empty() { Value::Null } else { json!(text) }});
                    if let Some(r) = ev["reasoning"].as_str().filter(|r| !r.is_empty()) {
                        message["reasoning_content"] = json!(r);
                    }
                    if !calls.is_empty() {
                        message["tool_calls"] = json!(calls);
                    }
                    return Ok(Json(json!({
                        "id": id, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish(&ev), "logprobs": null}],
                        "usage": usage_chat(&ev["usage"]),
                    })).into_response());
                }
                _ => {}
            }
        }
    }

    // Streaming: wait for the first event so request errors still get a proper status.
    let first = gen.events.recv().await.ok_or_else(|| ApiError::unavailable("the model worker exited"))?;
    if first["type"] == "error" {
        gen.finish();
        return Err(ApiError::from_worker(first["message"].as_str().unwrap_or("worker error")));
    }
    let chunk = move |delta: Value, finish: Value| {
        json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}]})
    };
    let s = async_stream::stream! {
        let data = |v: Value| Ok::<Event, Infallible>(Event::default().data(v.to_string()));
        yield data(chunk(json!({"role": "assistant", "content": ""}), Value::Null));
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
                Some("delta") => {
                    let field = if ev["kind"] == "reasoning" { "reasoning_content" } else { "content" };
                    yield data(chunk(json!({field: ev["text"]}), Value::Null));
                }
                Some("tool_call") => {
                    yield data(chunk(json!({"tool_calls": [{
                        "index": ev["index"], "id": new_id("call_"), "type": "function",
                        "function": {"name": ev["name"], "arguments": arguments_string(&ev["arguments"])},
                    }]}), Value::Null));
                }
                Some("error") => {
                    gen.finish();
                    yield data(json!({"error": {"message": ev["message"], "type": "server_error"}}));
                    break;
                }
                Some("done") => {
                    gen.finish();
                    yield data(chunk(json!({}), json!(finish(&ev))));
                    if include_usage {
                        let mut u = chunk(json!({}), Value::Null);
                        u["choices"] = json!([]);
                        u["usage"] = usage_chat(&ev["usage"]);
                        yield data(u);
                    }
                    break;
                }
                _ => {}
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Ok(Sse::new(s).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))).into_response())
}

fn finish(done: &Value) -> &'static str {
    match done["finish_reason"].as_str() {
        Some("length") => "length",
        Some("tool_calls") => "tool_calls",
        _ => "stop",
    }
}
