# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

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
