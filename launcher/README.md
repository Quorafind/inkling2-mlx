# inkling2 launcher

A Rust launcher that serves Inkling 2 Mini on a local OpenAI-compatible API. It starts
one model worker (`inkling2_mini/server_worker.py` in the project's `.venv`) and handles
HTTP, the OpenAI wire formats, queueing and cancellation itself.

```bash
./inkling2                          # from the project root: builds on first use, serves 127.0.0.1:8000
./inkling2 serve --port 8080 --api-key secret
./inkling2 chat                     # the interactive terminal chat (chat.py)
```

Point any OpenAI-compatible client at it:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=local
curl $OPENAI_BASE_URL/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "inkling-2-mini", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

## Endpoints

| Endpoint | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Streaming and not. Thinking is returned in `reasoning_content`, function tools in `tool_calls`, and `stream_options.include_usage` is honoured. Also accepts `reasoning_effort`, `stop`, `seed`, `top_k`, and images as `image_url` data URLs. |
| `POST /v1/responses` | Streaming with the typed event sequence, or not. Supports `previous_response_id`, `instructions`, function tools with `function_call` / `function_call_output` items, `reasoning.effort`, and `store`. |
| `GET` / `DELETE /v1/responses/{id}` | Stored responses (the last 1,000). |
| `GET /v1/models`, `GET /health` | |

Not supported: `n > 1`, logprobs, hosted tools (web search, file search), and the Anthropic Messages API.

## KV reuse

Prefill dominates multi-turn latency, so the worker keeps the KV state of finished replies
(4 GB by default, `--cache-gb`):

- **Chat Completions.** The client resends the whole history. The worker finds the stored
  state whose rendered conversation is the longest prefix of the request and prefills only
  the rest. Clients drop `reasoning_content` from echoed assistant messages, so the worker
  re-attaches it from its own replies. Without that, the rendering would not match.
- **Responses with `previous_response_id`.** The launcher keeps the conversation, and the
  worker keeps the state under the response id. A follow-up prefills only its new input.
- **States are cloned on use.** Retries, branches and side requests do not evict the main
  conversation.
- **Shared prefixes are snapshotted.** A long system prompt + tool list (512 tokens or more)
  is stored on a miss, so new conversations with the same prefix start from it.

Measured on turn 2 of an 8k-token conversation (`tests/check_launcher.py`), time to first
token:

| Turn 2 | Time to first token |
| --- | --- |
| Cold prefill | 9.25 s |
| Chat Completions, history resent | 0.32 s |
| Responses, `previous_response_id` | 0.29 s |

Decode speed through the server matches the runtime called directly (29.5 vs 29.6 tok/s,
measured in the same session).

## Options

`./inkling2 serve --help` lists them all. The main ones:

| Option | Default |
| --- | --- |
| `--host`, `--port` | `127.0.0.1`, `8000` |
| `--model` | `models/Inkling-2-Mini` |
| `--dense-bits` | 6 |
| `--kv-bits` | 8 |
| `--no-spec` | DSpark stays on |
| `--max-context` | 65,536 tokens |
| `--effort` | `low`, used when a request sets none |
| `--max-tokens` | 8,192, used when a request sets none |
| `--api-key` | none |

Environment: `INKLING2_HOME`, `INKLING2_PYTHON`, `INKLING2_HOST`, `INKLING2_PORT`,
`INKLING2_MODEL`, `INKLING2_API_KEY`.

## Build and tests

`cargo build --release` (`--offline` works from a warm cargo cache). Tests, run from the
project root:

- `tests/check_server_worker.py MODEL_DIR` tests the worker protocol: KV reuse, independent
  clones, the prefix snapshot, tool calls, stop sequences and cancel.
- `tests/check_launcher.py` runs every endpoint end to end over HTTP, then the turn-2
  latency table above.
- `tests/check_launcher_sdk.py PORT` runs the official `openai` Python SDK against a running
  server.
