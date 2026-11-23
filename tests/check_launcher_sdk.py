"""The launcher against the official `openai` Python SDK (pip install openai): the
SDK's own parsing of chat completions, chat streaming, responses, responses
streaming events and previous_response_id. Expects a server on PORT (default 8123),
e.g. `launcher/target/release/inkling2 serve --port 8123`.

    python tests/check_launcher_sdk.py [PORT]
"""

import sys

from openai import OpenAI

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local")
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(': ' + detail) if detail else ''}", flush=True)


models = client.models.list()
check("models.list", models.data[0].id == "inkling-2-mini")

r = client.chat.completions.create(model="inkling-2-mini", messages=[{"role": "user", "content": "Say hello in French."}],
                                   temperature=0, max_tokens=100, reasoning_effort="low")
check("chat.completions.create", r.choices[0].message.content, repr(r.choices[0].message.content))

parts = []
with client.chat.completions.stream(model="inkling-2-mini", messages=[{"role": "user", "content": "Count from 1 to 5."}],
                                    temperature=0, max_tokens=100, reasoning_effort="none") as s:
    for ev in s:
        if ev.type == "content.delta":
            parts.append(ev.delta)
    final = s.get_final_completion()
check("chat.completions.stream", "".join(parts) == final.choices[0].message.content and parts, repr("".join(parts)))

tools = [{"type": "function", "name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
r1 = client.responses.create(model="inkling-2-mini", input="What's the weather in Tokyo? Use the tool.", tools=tools,
                             temperature=0, reasoning={"effort": "none"})
calls = [o for o in r1.output if o.type == "function_call"]
check("responses.create -> function_call", calls and calls[0].name == "get_weather", calls[0].arguments if calls else "")
if calls:
    r2 = client.responses.create(model="inkling-2-mini", previous_response_id=r1.id, tools=tools, temperature=0,
                                 reasoning={"effort": "none"},
                                 input=[{"type": "function_call_output", "call_id": calls[0].call_id, "output": "25°C, sunny"}])
    check("responses.create(previous_response_id) after a tool result", "25" in r2.output_text,
          f"{r2.output_text!r}, cached {r2.usage.input_tokens_details.cached_tokens}")

deltas, kinds = [], set()
with client.responses.stream(model="inkling-2-mini", input="Name two planets.", temperature=0,
                             reasoning={"effort": "low"}, max_output_tokens=300) as s:
    for ev in s:
        kinds.add(ev.type)
        if ev.type == "response.output_text.delta":
            deltas.append(ev.delta)
    final = s.get_final_response()
check("responses.stream", "".join(deltas) == final.output_text and "response.completed" in kinds,
      f"{final.output_text!r}, {len(kinds)} event types")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
