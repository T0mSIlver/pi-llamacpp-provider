# pi-llamacpp-provider

[![npm](https://img.shields.io/npm/v/pi-llamacpp-provider)](https://www.npmjs.com/package/pi-llamacpp-provider)
[![CI](https://github.com/T0mSIlver/pi-llamacpp-provider/actions/workflows/ci.yml/badge.svg)](https://github.com/T0mSIlver/pi-llamacpp-provider/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A [pi](https://pi.dev) extension that registers your [llama.cpp](https://github.com/ggml-org/llama.cpp)
server as a provider and **discovers its models automatically**. Add a preset to
the server, restart pi, and it is in `/model` — no `models.json` entry, no code
change.

```
$ pi --list-models llamacpp
provider  model            context  max-out  thinking  images
llamacpp  fastcontext      80K      8.2K     no        no
llamacpp  gemma4-12b       131.1K   8.2K     no        yes
llamacpp  gemma4-31b       100K     8.2K     no        no
llamacpp  ornith-35b       262.1K   8.2K     yes       yes
llamacpp  qwen36a3b-35b    262.1K   8.2K     yes       yes
```

## Install

```sh
pi install npm:pi-llamacpp-provider
```

Or straight from git, or pinned, or just for one run:

```sh
pi install git:github.com/T0mSIlver/pi-llamacpp-provider
pi install npm:pi-llamacpp-provider@0.1.0   # pinned, skipped by pi update
pi -e npm:pi-llamacpp-provider              # try without installing
```

If your server is not on `localhost:8080`, point the extension at it:

```sh
export LLAMACPP_BASE_URL=http://gpu-box.lan:8080
```

Then select a model:

```sh
pi --model 'llamacpp/gemma4-31b'
pi --models 'llamacpp/*'    # cycle every llama.cpp model with Ctrl+P
/model                      # or pick one inside the TUI
```

## What it reads from your server

At startup the extension calls `GET /v1/models` and maps each entry to a pi
model. Two server shapes are supported, because their payloads differ:

| Server | What the payload gives us |
|---|---|
| **Router** (`llama-server --router`, multi-preset) | `status.args` — the full `llama-server` argv per preset — plus `preset` and `architecture.input_modalities` |
| **Plain `llama-server`** (single model) | a `meta` block with `n_ctx` / `n_ctx_train` |

From that, per model:

| pi field | Derived from |
|---|---|
| `contextWindow` | `--ctx-size ÷ --parallel` — llama.cpp splits the context evenly across slots, so this is what one conversation actually gets. Falls back to `meta.n_ctx`, then `meta.n_ctx_train`, then llama.cpp's own 4096 default. Floored at 2048. |
| `input` | `["text", "image"]` when the model reports an `image` modality (i.e. it was started with `--mmproj`), else `["text"]`. Audio modalities are dropped — pi models declare text and image only. |
| `reasoning` | True when the model id matches `/qwen/i`, or the preset/argv contains `preserve_thinking` or `enable_thinking`. Thinking models also get `compat.thinkingFormat = "qwen-chat-template"`, so pi toggles thinking through `chat_template_kwargs` rather than OpenAI's `reasoning_effort`, which llama.cpp does not implement. |
| `maxTokens` | The context window, capped at 8192. |
| `cost` | Zero — local inference is free, and pi's session cost readout should say so. |
| `compat` | `supportsDeveloperRole: false` (llama.cpp chat templates expect a `system` role) and `maxTokensField: "max_tokens"` (it rejects `max_completion_tokens`). |

## When the server is down

Discovery never blocks pi from starting. On a connection refusal, timeout,
non-2xx response, malformed body, or empty model list, the extension registers
the provider anyway with a single `default` model and warns once at session
start:

```
[pi-llamacpp-provider] Model discovery from http://127.0.0.1:8080/v1 failed
(fetch failed). Registered provider "llamacpp" with a single "default" model.
Start the server and run /reload, or set LLAMACPP_BASE_URL.
```

Start the server and `/reload` — no restart needed.

## Environment overrides

| Variable | Default | Effect |
|---|---|---|
| `LLAMACPP_BASE_URL` | `http://127.0.0.1:8080` | Server address. `/v1` is appended if you leave it off. |
| `LLAMACPP_API_KEY` | `sk-llamacpp-local` | Sent as `Authorization: Bearer …`. llama.cpp ignores it unless started with `--api-key`, but pi needs a non-empty key to treat the provider as authenticated. |
| `LLAMACPP_TIMEOUT_MS` | `4000` | How long discovery may take before falling back. |
| `LLAMACPP_PROVIDER` | `llamacpp` | Provider name, i.e. the `…/model-id` prefix. Change it to avoid clashing with a `llamacpp` provider defined in `models.json`. |

All four are read on every load, so `/reload` picks up a change without a
reinstall.

## Test evidence

Verified against a **live llama.cpp router serving 9 presets**: all 9 appear in
`pi --list-models` with correct context windows, vision flags and thinking
flags; text and thinking models both complete a real prompt end-to-end; killing
the server degrades to the `default` model instead of breaking startup. Full
transcripts: [docs/test-evidence.md](./docs/test-evidence.md).

## Development

```sh
npm install
npm run verify        # typecheck + test suite (needs Node ≥ 22.18)
pi install .          # smoke-test the package manifest against a local pi
```

`test/provider.mjs` runs the extension against throwaway HTTP servers — a
router payload captured from a real llama.cpp router, a plain `llama-server`
payload, and every failure mode (500, timeout, refused connection, malformed
JSON, empty list) — asserting the fallback and the once-per-process warning.

## License

[MIT](./LICENSE)
