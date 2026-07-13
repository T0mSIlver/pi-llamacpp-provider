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
llamacpp  fastcontext      80K      16.4K    no        no
llamacpp  gemma4-12b       131.1K   16.4K    yes       yes
llamacpp  gemma4-31b       100K     16.4K    yes       no
llamacpp  ornith-35b       262.1K   16.4K    yes       yes
llamacpp  qwen36a3b-35b    262.1K   16.4K    yes       yes
```

`thinking` is read from each model's **chat template** — the same thing llama.cpp
itself uses to decide whether a model reasons — not guessed from its name.

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

At startup the extension calls `GET /v1/models`, then probes each model once at
`GET /props`, and maps every entry to a pi model. Two server shapes are
supported, because their payloads differ:

| Server | What the payload gives us |
|---|---|
| **Router** (`llama-server` with no model, multi-preset) | `status.args` — the full `llama-server` argv per preset — plus `status.preset` and `architecture.input_modalities`. A `meta` block appears only for a preset that is currently loaded. |
| **Plain `llama-server`** (single model) | a `meta` block with `n_ctx` / `n_ctx_train`, and everything else from `/props`. |

From that, per model:

| pi field | Derived from |
|---|---|
| `contextWindow` | What one conversation actually gets: the per-slot `n_ctx` from `/props` or `meta` when the model is loaded (already divided by `--parallel`), else `--ctx-size ÷ --parallel`, else `meta.n_ctx_train ÷ --parallel`, else llama.cpp's own 4096 default. Reported honestly, even when it is small. |
| `maxTokens` | **Max output tokens for one turn** — not the context window. The preset's `--predict` / `-n` / `--n-predict` if it pins one, else `LLAMACPP_MAX_OUTPUT_TOKENS` (16384 by default), clamped to the context window. llama.cpp's own default is -1, "generate until the context is full", which is not a number pi can budget against. |
| `reasoning` | The model's **chat template** — see below. |
| `input` | `["text", "image"]` when the model reports an `image` modality (a router says so in `architecture`, a plain server in `/props` `modalities`), else `["text"]`. Audio is dropped — pi models declare text and image only. |
| `cost` | Zero — local inference is free, and pi's session cost readout should say so. |
| `compat` | `supportsDeveloperRole: false` (llama.cpp chat templates expect a `system` role) and `maxTokensField: "max_tokens"` (it rejects `max_completion_tokens`). |

## How thinking support is detected

llama.cpp decides whether a model reasons by *rendering its chat template* and
looking at what comes out ([`common_chat_templates_support_enable_thinking`](https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp)).
That template is the only honest answer, and it is published at `GET /props`. So
that is what this extension reads, in this order:

1. **`LLAMACPP_THINKING_MODELS` / `LLAMACPP_NON_THINKING_MODELS`** — comma-separated
   globs (`gemma4-*,qwen3?-*`). Your word is final.
2. **Flags that make thinking impossible**: `--reasoning off` and `--no-jinja`. Those two
   are the only ones llama.cpp consults —
   `enable_thinking = enable_reasoning != 0 && template_supports_thinking` — so either
   settles it. `--reasoning-budget 0` is *not* one of them: it cuts the thinking block
   short at the sampler, it does not remove the model's ability to think.
3. **The chat template**, fetched live from `GET /props?model=<id>&autoload=0`. The
   `autoload=0` matters: without it a router would *load* the model — spawning a
   `llama-server` and filling VRAM — just to answer us. With it, a router answers for
   what it already has in memory and replies "not loaded" for the rest, which we take
   as "no answer", never as a failure. The template is matched against the markers
   llama.cpp itself keys on: `enable_thinking`, `reasoning_content`, `<think>` and its
   variants, `[THINK]`, `<|channel|>analysis` (gpt-oss), `<|channel>thought` (Gemma 4).
   This is validated against all 59 chat templates shipped in llama.cpp's
   `models/templates`, and it is why Gemma 4 is a thinking model here and Qwen3-Coder
   is not.
4. **A cached verdict.** A router only answers `/props` for what it has loaded, so a
   template we read once is remembered under `$XDG_CACHE_HOME/pi-llamacpp-provider/`,
   keyed by the preset's argv — edit the preset and the entry is dropped. A model the
   router has since unloaded keeps the answer its own template gave.
5. **The preset**, finally: a `--chat-template-kwargs` carrying a thinking switch
   (`enable_thinking`, `preserve_thinking`, …), a positive `--reasoning-budget`, or
   `--reasoning on`. A preset only configures a thinking switch for a model that has one.
6. **Otherwise: no.** No evidence, no claim — the model id is never consulted, because
   a name is not a capability.

Thinking models get `compat.thinkingFormat = "chat-template"` with
`chat_template_kwargs: { enable_thinking, preserve_reasoning, preserve_thinking }`, so
pi toggles thinking the way llama.cpp expects rather than through OpenAI's
`reasoning_effort`, which llama.cpp does not implement.

Set `LLAMACPP_PROBE=0` to skip probing altogether; detection then falls back to steps
1, 2, 5 and 6.

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
| `LLAMACPP_MAX_OUTPUT_TOKENS` | `16384` | `maxTokens` for models whose preset does not pin a `--predict`. |
| `LLAMACPP_THINKING_MODELS` | — | Comma-separated globs forced to `reasoning: true`, e.g. `gemma4-*,my-model`. |
| `LLAMACPP_NON_THINKING_MODELS` | — | The same, forced to `reasoning: false`. |
| `LLAMACPP_PROBE` | `1` | Set to `0` to skip the `/props` probe entirely. |

All of them are read on every load, so `/reload` picks up a change without a
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
npm run verify        # typecheck + test suite (needs Node ≥ 22.19)
pi install .          # smoke-test the package manifest against a local pi
```

`test/provider.mjs` runs the extension against throwaway HTTP servers — a router
payload captured from a real llama.cpp router, a plain `llama-server` payload,
and every failure mode (500, timeout, refused connection, malformed JSON, empty
list) — asserting the fallback and the once-per-process warning. Thinking
detection is checked against real chat templates lifted from llama.cpp's
`models/templates` (`test/fixtures/templates/`): Gemma 4, Qwen3, gpt-oss and
Ministral-Reasoning must all reason; Qwen3-Coder and Llama 3.3 must not.

## License

[MIT](./LICENSE)
