# Test evidence

> **Recorded against v0.1.0.** The transcripts below are verbatim and unedited,
> so they still show that release's derivations: thinking guessed from the model
> id (`gemma4-*` reported as `no`, which is what prompted the rewrite),
> `max-out` pinned at 8.2K, and `thinkingFormat: "qwen-chat-template"`. Since
> then thinking is read from each model's chat template and `maxTokens` is a real
> output cap — see the README. Everything else here (discovery, context windows,
> vision, degraded startup) still holds. These will be re-recorded on the next
> live run against the router.

Collected 2026-07-09 on Debian 13, Node v24.18.0, against pi v0.80.3 and a
**live llama.cpp router** serving 9 presets on the LAN (its address is written
below as `$LLAMA_HOST`; the extension was pointed at it with
`LLAMACPP_BASE_URL`). `test/fixtures/models.json` is that server's real
`GET /v1/models` response, captured verbatim and used by the unit suite.

Every `pi` invocation below passes `-ne` (no extension discovery) and
`-e ./src/index.ts`, so the extension under test is the only one loaded.

## Discovery against the live router

```
$ LLAMACPP_BASE_URL=$LLAMA_HOST pi -ne -e ./src/index.ts --list-models llamacpp
provider  model            context  max-out  thinking  images
llamacpp  default          4.1K     4.1K     no        no
llamacpp  fastcontext      80K      8.2K     no        no
llamacpp  gemma4-12b       131.1K   8.2K     no        yes
llamacpp  gemma4-26b-a4b   262.1K   8.2K     no        yes
llamacpp  gemma4-31b       100K     8.2K     no        no
llamacpp  laguna-xs2       262.1K   8.2K     no        no
llamacpp  ornith-35b       262.1K   8.2K     yes       yes
llamacpp  qwen36a3b-35b    262.1K   8.2K     yes       yes
llamacpp  qwen36dense-27b  140K     8.2K     yes       yes
```

All 9 presets registered, no `models.json` entry for any of them. Spot-checking
the derived columns against what the server reports in `status.args`:

| Model | Server argv | pi shows | Why |
|---|---|---|---|
| `fastcontext` | `--ctx-size 160000 --parallel 2` | 80K | context split across 2 slots |
| `gemma4-31b` | `--ctx-size 100000 --parallel 1` | 100K | one slot, full context |
| `default` | no `--ctx-size`, no `meta` | 4.1K | llama.cpp's own 4096 default |
| `gemma4-12b` | `input_modalities: [text, image, audio]` | images ✓ | audio dropped, image kept |
| `laguna-xs2` | no `--mmproj`, text-only | images ✗ | no vision advertised |
| `ornith-35b` | preset contains `preserve_thinking` | thinking ✓ | detected from the preset, **not** the id |
| `qwen36a3b-35b` | qwen id | thinking ✓ | detected from the id |

`ornith-35b` is the interesting one: a non-Qwen id whose preset enables
thinking. Id-matching alone would have missed it.

## End-to-end inference

Not just registration — real completions through the registered provider.

```
$ LLAMACPP_BASE_URL=$LLAMA_HOST pi -ne -e ./src/index.ts --no-session -nt \
    --model llamacpp/fastcontext -p "Reply with exactly one word: pong"
pong
```

The thinking path exercises `compat.thinkingFormat = "qwen-chat-template"` —
pi sends `chat_template_kwargs.enable_thinking` instead of `reasoning_effort`,
which llama.cpp would reject:

```
$ LLAMACPP_BASE_URL=$LLAMA_HOST pi -ne -e ./src/index.ts --no-session -nt \
    --model llamacpp/qwen36a3b-35b --thinking low -p "What is 17*23? Reply with just the number."
391
```

Both `maxTokensField: "max_tokens"` and `supportsDeveloperRole: false` are
implicitly confirmed here: llama.cpp 400s on `max_completion_tokens` and on a
`developer` role message, and neither request errored.

## Degraded mode: server unreachable

Port 9 (discard) refuses the connection; discovery fails and pi still starts.

```
$ LLAMACPP_BASE_URL=http://127.0.0.1:9 LLAMACPP_TIMEOUT_MS=800 \
    pi -ne -e ./src/index.ts --list-models llamacpp
provider  model    context  max-out  thinking  images
llamacpp  default  4.1K     4.1K     no        no
```

The provider exists, so `/reload` recovers the full list once the server is
back. In a TUI session the fallback also raises a one-time `session_start`
warning naming the URL and the cause (asserted in the unit suite).

## Package manifest

The `pi.extensions` manifest entry resolves through a real install, not just
`-e`:

```
$ pi install .
Installed .

$ pi list
  ../../work/pi-llamacpp-provider
    /home/dev/work/pi-llamacpp-provider

$ LLAMACPP_PROVIDER=llamacpp-pkg LLAMACPP_BASE_URL=$LLAMA_HOST pi --list-models llamacpp-pkg
provider      model            context  max-out  thinking  images
llamacpp-pkg  default          4.1K     4.1K     no        no
llamacpp-pkg  fastcontext      80K      8.2K     no        no
…9 models…

$ pi remove .
Removed .
```

Loaded from `settings.json` with no `-e` flag, and `LLAMACPP_PROVIDER` renamed
the provider — which also proves config is read at load time, so `/reload`
picks up env changes.

## Unit suite

`test/provider.mjs` — 56 assertions, run against throwaway `node:http` servers
rather than a stubbed `fetch`, so the request path, bearer header, timeout and
connection-refused behaviour are all real.

```sh
npm run verify   # tsc --noEmit && node test/provider.mjs
```

Covered:

- **URL handling** — `/v1` appended when missing, trailing slashes stripped,
  whitespace trimmed, existing `/v1` preserved.
- **argv parsing** — value after a flag, missing flag, flag in trailing
  position (no out-of-bounds read).
- **Defaults** — `readConfig({})` resolves to `http://127.0.0.1:8080/v1`, not
  to any LAN address.
- **Context math** — slot division, the 2048 floor (before *and* after
  division), `--ctx-size` beating `meta.n_ctx`, the 8192 output cap, and
  `maxTokens` never exceeding a small context window.
- **Router payload** (the captured fixture) — all 9 models, per-model context /
  vision / thinking assertions, zero cost, `max_tokens`, no developer role, and
  the request landing on `/v1/models` with `Authorization: Bearer …`.
- **Plain-server payload** — `meta.n_ctx` preferred over `meta.n_ctx_train`,
  `n_ctx_train` used when `n_ctx` is absent, no `architecture` block → text
  only.
- **Malformed entries** — `null`, missing `id`, empty `id` are skipped; a
  partially bad payload is *not* treated as a discovery failure.
- **Every failure mode → fallback** — HTTP 500, malformed JSON, empty `data`,
  no usable entries, connection refused, timeout. Each still registers the
  provider with exactly one `default` model.
- **Warning wiring** — fires once per process even though `session_start` fires
  on every reload and session switch; goes to `ctx.ui.notify(…, "warning")`
  when there is a UI and to `console.warn` when there is not.
