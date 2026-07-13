# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Changed

- **Thinking support is now read from the model's chat template**, which is what
  llama.cpp itself decides on, instead of being guessed from the model id. Each
  model is probed once at `GET /props?model=<id>&autoload=0` — side-effect free,
  so a router never loads a model to answer us — and its `chat_template` is
  matched against the markers llama.cpp keys on (`enable_thinking`,
  `reasoning_content`, `<think>`, `[THINK]`, `<|channel|>analysis`,
  `<|channel>thought`, …). Validated against all 59 templates in llama.cpp's
  `models/templates` (six of them ship as test fixtures). Gemma 4, gpt-oss,
  DeepSeek-R1 and GLM are now recognised as thinking models; Qwen3-Coder and
  other non-thinking Qwens no longer are.
  Verdicts are cached under `$XDG_CACHE_HOME/pi-llamacpp-provider/`, keyed by the
  preset's argv, so a model the router has since unloaded keeps its answer.
  Set `LLAMACPP_PROBE=0` to skip probing entirely.
- **`maxTokens` is the output cap it is meant to be**, not a copy of the context
  window. It comes from the preset's `--predict` / `-n` / `--n-predict` when set,
  otherwise from `LLAMACPP_MAX_OUTPUT_TOKENS` (default 16384), clamped to the
  context window. It was previously `min(contextWindow, 8192)`, which both
  under-served long reasoning turns and misreported what the server would do.
- Thinking models now use `compat.thinkingFormat: "chat-template"` with explicit
  `chat_template_kwargs` (`enable_thinking`, plus `preserve_reasoning` — the
  kwarg llama.cpp itself reads to keep reasoning traces in history). The old
  `qwen-chat-template` format assumed every thinking model was a Qwen.
- The advertised context window is no longer floored at 2048. A server started
  with `-c 512` is reported as 512, because that is what it will accept.

### Fixed

- The preset INI was read from `m.preset`, but a llama.cpp router reports it at
  `m.status.preset`, so it was always `undefined` and never contributed to
  detection.
- `meta.n_ctx` is already the *per-slot* context (`slot_n_ctx`) and was being
  divided by `--parallel` a second time. Only `--ctx-size`, which is the total,
  is divided now.
- `--ctx-size`/`--parallel`/`--predict` are now also read from their short forms
  (`-c`, `-np`, `-n`), and a repeated flag resolves to its last occurrence, the
  way llama.cpp resolves it.
- A plain `llama-server` now reports vision support, read from `/props`
  `modalities` — it has no `architecture` block, so it previously always looked
  text-only.

### Added

- `LLAMACPP_THINKING_MODELS` / `LLAMACPP_NON_THINKING_MODELS`: comma-separated
  globs that force `reasoning` on or off, for the cases no detection can win.
- `LLAMACPP_MAX_OUTPUT_TOKENS` and `LLAMACPP_PROBE`.

## [0.1.0] - 2026-07-09

### Added

- Initial release. Registers a llama.cpp server as the `llamacpp` pi provider
  and turns every entry of its `GET /v1/models` response into a pi model, so
  new server presets appear in `/model` without a `models.json` edit.
- Supports both llama.cpp server shapes: the multi-preset **router** (context,
  slot count and vision read from each preset's `status.args` and
  `architecture.input_modalities`) and a **plain `llama-server`** (context read
  from `meta.n_ctx`, falling back to `meta.n_ctx_train`).
- Per-model derivation: context window as `--ctx-size ÷ --parallel` floored at
  2048; `image` input when the model advertises a vision modality, with audio
  modalities dropped; `reasoning` plus
  `compat.thinkingFormat = "qwen-chat-template"` for Qwen ids and for presets
  carrying `preserve_thinking`/`enable_thinking`; zero cost; `max_tokens`
  capped at 8192; `supportsDeveloperRole: false`.
- Graceful degradation: a refused connection, timeout, non-2xx response,
  malformed body, or empty model list registers the provider with a single
  `default` model and warns once at `session_start` via `ctx.ui.notify`
  (`console.warn` when headless), so pi always starts and `/reload` recovers.
- `LLAMACPP_BASE_URL` (default `http://127.0.0.1:8080`, `/v1` appended if
  omitted), `LLAMACPP_API_KEY`, `LLAMACPP_TIMEOUT_MS`, and `LLAMACPP_PROVIDER`
  overrides, all read at load time so `/reload` picks them up.
- Verified against a live 9-preset llama.cpp router: discovery, context/vision/
  thinking derivation, end-to-end completions on both a text and a thinking
  model, and degraded startup. See `docs/test-evidence.md`.

[0.1.0]: https://github.com/T0mSIlver/pi-llamacpp-provider/releases/tag/v0.1.0
