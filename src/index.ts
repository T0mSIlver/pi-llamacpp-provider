/**
 * llama.cpp provider for pi
 * ─────────────────────────
 * Registers a llama.cpp server (OpenAI-compatible `/v1` API) as a pi provider
 * and turns every model the server advertises at `GET /v1/models` into a pi
 * model. Start a new preset on the server, restart pi, and it is there — no
 * `models.json` edits, no code changes here.
 *
 * Two server shapes are supported, because their `/v1/models` payloads differ:
 *
 *   - **llama.cpp router** (`llama-server` started with no model, or the
 *     `llama-swap`-style multi-preset server): each entry carries `status.args`
 *     — the full argv of the `llama-server` process for that preset — plus
 *     `status.preset` and `architecture.input_modalities`.
 *   - **plain `llama-server`**: a single entry with a `meta` block whose `n_ctx`
 *     is the per-slot context and `n_ctx_train` the model's trained maximum.
 *
 * On top of the model list, each model is probed once at `GET /props`, which is
 * where llama.cpp publishes the thing that actually decides whether a model
 * thinks: its **chat template**. See `detectsThinking()`. The probe carries
 * `autoload=0`, so a router answers from memory and never spawns a
 * `llama-server` to satisfy us.
 *
 * Select a model with any of:
 *   pi --model 'llamacpp/gemma4-31b'
 *   pi --models 'llamacpp/*'          # cycle every llama.cpp model with Ctrl+P
 *   /model                            # inside the TUI, pick a llamacpp/… entry
 *
 * Discovery never blocks pi from starting: if the server is down, unreachable,
 * or slow, the provider is still registered with a single `default` model and a
 * one-time warning, so it works again as soon as the server is back.
 *
 * Zero runtime dependencies. Install: `pi install npm:pi-llamacpp-provider`
 */

import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

/** llama.cpp's own default `--ctx-size` when a preset does not set one. */
const DEFAULT_CTX = 4096;
/**
 * Output cap used when the server does not set one. llama.cpp defaults `--predict`
 * to -1 (generate until the context is full), which is not a number pi can budget
 * against, so we advertise a bounded default instead — matching pi's own default
 * for models that declare no `maxTokens`.
 */
const DEFAULT_MAX_OUTPUT_TOKENS = 16384;

/** The subset of a `/v1/models` entry this extension reads. Everything is optional but `id`. */
export interface LlamaModel {
	id: string;
	/** Router only: `input_modalities` may include "text", "image", "audio". */
	architecture?: { input_modalities?: string[] };
	/** Router only: `args` is the llama-server argv for this preset, `preset` its INI text. */
	status?: { value?: string; args?: string[]; preset?: string };
	/** Present for a plain llama-server, and for a *loaded* model on a router. */
	meta?: { n_ctx?: number; n_ctx_train?: number };
}

/**
 * The subset of `GET /props` this extension reads. Served by a plain llama-server
 * for its one model, and by a router (via `?model=<id>`) for a *loaded* model.
 */
export interface LlamaProps {
	/** "router" when a router answered about itself rather than about a model. */
	role?: string;
	/** The model's `--alias`, i.e. the id it is listed under. */
	model_alias?: string;
	/** The Jinja chat template source — the ground truth for thinking support. */
	chat_template?: string;
	/** `n_ctx` here is the *per-slot* context, already divided by `--parallel`. */
	default_generation_settings?: { n_ctx?: number; params?: { n_predict?: number } };
	modalities?: { vision?: boolean; audio?: boolean };
}

export interface LlamaConfig {
	baseUrl: string;
	apiKey: string;
	timeoutMs: number;
	provider: string;
	/** Probe `GET /props` per model. Set LLAMACPP_PROBE=0 to skip it. */
	probe: boolean;
	/** Output cap for models whose server sets no `--predict`. */
	maxOutputTokens: number;
	/** Globs forcing `reasoning: true` / `false`, highest precedence. */
	thinkingModels: string[];
	nonThinkingModels: string[];
}

/**
 * Read configuration from the environment. Called on every extension load (and
 * `/reload`), so an env change takes effect without reinstalling.
 */
export function readConfig(env: NodeJS.ProcessEnv = process.env): LlamaConfig {
	return {
		baseUrl: normalizeBaseUrl(env.LLAMACPP_BASE_URL || "http://127.0.0.1:8080"),
		// llama.cpp ignores the key unless started with --api-key, but pi requires
		// a non-empty one to consider the provider authenticated.
		apiKey: env.LLAMACPP_API_KEY || "sk-llamacpp-local",
		// A nonsense value falls back to the default rather than through it: a negative
		// timeout would make AbortSignal.timeout throw and degrade every model to the
		// fallback, and a negative maxTokens is a value pi rejects outright.
		timeoutMs: positiveInt(env.LLAMACPP_TIMEOUT_MS) ?? 4000,
		provider: env.LLAMACPP_PROVIDER || "llamacpp",
		probe: !/^(0|false|off|no)$/i.test((env.LLAMACPP_PROBE ?? "").trim()),
		maxOutputTokens: positiveInt(env.LLAMACPP_MAX_OUTPUT_TOKENS) ?? DEFAULT_MAX_OUTPUT_TOKENS,
		thinkingModels: splitPatterns(env.LLAMACPP_THINKING_MODELS),
		nonThinkingModels: splitPatterns(env.LLAMACPP_NON_THINKING_MODELS),
	};
}

function splitPatterns(raw: string | undefined): string[] {
	return (raw ?? "")
		.split(",")
		.map((p) => p.trim())
		.filter(Boolean);
}

/** Strip trailing slashes and append `/v1` unless the URL already ends in it. */
export function normalizeBaseUrl(raw: string): string {
	const trimmed = raw.trim().replace(/\/+$/, "");
	return /\/v1$/.test(trimmed) ? trimmed : `${trimmed}/v1`;
}

/** `…/v1` → `…`. `/props` lives at the server root, not under `/v1`. */
export function rootUrl(baseUrl: string): string {
	return baseUrl.replace(/\/v1$/, "");
}

/** Read the value following `flag` (or any of its aliases) in a llama-server argv array. */
export function argValue(args: string[], ...flags: string[]): string | undefined {
	for (const flag of flags) {
		const i = args.lastIndexOf(flag); // a repeated flag: llama.cpp keeps the last
		if (i >= 0 && i + 1 < args.length) return args[i + 1];
	}
	return undefined;
}

/** Positive integer or undefined — `--ctx-size 0` means "take it from the model", not "zero". */
function positiveInt(raw: string | number | undefined): number | undefined {
	const n = Number(raw);
	return Number.isFinite(n) && n > 0 ? Math.floor(n) : undefined;
}

/** Case-insensitive `*`/`?` glob over a model id. */
export function globMatch(pattern: string, id: string): boolean {
	const rx = pattern
		.split("")
		.map((c) => (c === "*" ? ".*" : c === "?" ? "." : c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")))
		.join("");
	return new RegExp(`^${rx}$`, "i").test(id);
}

/**
 * Does this chat template make the model think?
 *
 * llama.cpp answers this question by *rendering* the template and looking at what
 * comes out (`common_chat_templates_support_enable_thinking`, common/chat.cpp).
 * We cannot render Jinja here, so we look for the markers that rendering would
 * turn up. Every marker below is one that llama.cpp itself keys on, and the set
 * was validated against all 59 templates in llama.cpp's `models/templates`:
 *
 *   - `enable_thinking`   the toggle variable (Qwen3, GLM, DeepSeek, Gemma 4, Nemotron, …)
 *   - `reasoning_content` the field a template reads back from assistant history
 *   - `<think>` and friends: `<|think|>`, `<thinking>`, `<seed:think>`, `<|START_THINKING|>`
 *   - `[THINK]`           Mistral / Magistral / Ministral-Reasoning
 *   - `<|channel|>analysis`  gpt-oss
 *   - `<|channel>thought`    Gemma 4  (COMMON_CHAT_FORMAT_PEG_GEMMA4, supports_thinking = true)
 *
 * A model whose template shows none of these — Qwen2.5, Qwen3-Coder, Llama 3.x,
 * Gemma 2, Mistral-Nemo, Kimi-K2-Instruct — does not think, and llama.cpp will
 * not extract reasoning from it either. That is the contract we want to mirror.
 */
export function detectsThinking(chatTemplate: string): boolean {
	return [
		/\benable_thinking\b/,
		/\breasoning_content\b/,
		/<\|?\/?(?:seed:)?think(?:ing)?[|:]?>/i,
		/_THINKING\|?>/i,
		/\[\/?THINK\]/,
		/<\|channel\|>\s*analysis/i,
		/<\|channel>\s*thought/i,
	].some((marker) => marker.test(chatTemplate));
}

/** Why a model was (not) marked as reasoning — surfaced in the cache and in tests. */
export type ThinkingSource = "env" | "flag" | "template" | "cache" | "kwargs" | "unknown";

/**
 * Resolve thinking support, most authoritative signal first.
 *
 *  1. `LLAMACPP_THINKING_MODELS` / `LLAMACPP_NON_THINKING_MODELS` — the user's word is final.
 *  2. Server flags that make thinking *impossible*: `--reasoning off` and `--no-jinja`. Those
 *     are the only two llama.cpp consults:
 *     `enable_thinking = enable_reasoning != 0 && template_supports_thinking`.
 *     `--reasoning-budget 0` is deliberately *not* one of them: it is a sampler knob that cuts
 *     the thinking block short, not a template toggle, so the model still thinks — and reporting
 *     it as non-thinking would leave pi unable to ask the template to suppress thinking cleanly.
 *  3. The chat template, read live from `GET /props` — ground truth.
 *  4. The template verdict cached from a previous run, for models the router has
 *     since unloaded.
 *  5. `--chat-template-kwargs {"enable_thinking":…}` / `preserve_thinking` in the preset:
 *     a preset only bothers with these for a model that thinks.
 *  6. Otherwise: no evidence, so `false`.
 *
 * Note `--reasoning on` is *not* proof: llama.cpp still requires the template to
 * support thinking, so it is treated as a hint at step 5, not a decision.
 */
export function resolveThinking(
	m: LlamaModel,
	config: LlamaConfig,
	props?: LlamaProps,
	cached?: boolean,
): { reasoning: boolean; source: ThinkingSource } {
	if (config.thinkingModels.some((p) => globMatch(p, m.id))) return { reasoning: true, source: "env" };
	if (config.nonThinkingModels.some((p) => globMatch(p, m.id))) return { reasoning: false, source: "env" };

	const args = m.status?.args ?? [];
	const reasoningFlag = argValue(args, "--reasoning", "-rea")?.toLowerCase();
	const budget = argValue(args, "--reasoning-budget");
	if (reasoningFlag === "off" || args.includes("--no-jinja")) {
		return { reasoning: false, source: "flag" };
	}

	if (props?.chat_template) return { reasoning: detectsThinking(props.chat_template), source: "template" };
	if (cached !== undefined) return { reasoning: cached, source: "cache" };

	// A preset only bothers with a thinking kwarg, a thinking budget, or `--reasoning on`
	// for a model whose template has a thinking switch to begin with — so the *presence*
	// of the switch is the signal. Its value is not: `{"enable_thinking": false}` says
	// "start with thinking off", and pi flips it back on per request.
	if (hasThinkingKwarg(m) || positiveInt(budget) !== undefined || reasoningFlag === "on") {
		return { reasoning: true, source: "kwargs" };
	}
	return { reasoning: false, source: "unknown" };
}

/**
 * Does the preset pass a thinking-related `--chat-template-kwargs`?
 *
 * These are JSON, so read them as JSON rather than grepping the argv: a preset that
 * mentions `enable_thinking` inside some unrelated string is not making a claim about
 * thinking, and a bare substring match cannot tell the difference.
 */
function hasThinkingKwarg(m: LlamaModel): boolean {
	const raw =
		argValue(m.status?.args ?? [], "--chat-template-kwargs") ??
		// The preset INI spells the same setting `chat-template-kwargs = {…}`.
		/^\s*chat-template-kwargs\s*=\s*(\{.*\})\s*$/m.exec(m.status?.preset ?? "")?.[1];
	if (!raw) return false;
	try {
		const kwargs = JSON.parse(raw) as Record<string, unknown>;
		return ["enable_thinking", "preserve_thinking", "preserve_reasoning", "thinking"].some(
			(key) => typeof kwargs[key] === "boolean",
		);
	} catch {
		return false; // not JSON we understand — no opinion, rather than a wrong one
	}
}

/**
 * Usable context for one conversation.
 *
 * `--ctx-size` is the *total* KV cache, which llama.cpp splits evenly across
 * `--parallel` slots, so a conversation gets `--ctx-size ÷ --parallel`. The
 * `n_ctx` reported by `/props` and by a loaded model's `meta` block is already
 * the per-slot figure (`slot_n_ctx`), so it must *not* be divided again.
 */
export function contextWindow(m: LlamaModel, props?: LlamaProps): number {
	const args = m.status?.args ?? [];
	const parallel = positiveInt(argValue(args, "--parallel", "-np")) ?? 1;

	// Report what the server will actually give a conversation, even when that is
	// small: rounding a 512-token server up to a comfortable-looking number only
	// buys a "prompt is too long" error at the first turn.
	return (
		positiveInt(props?.default_generation_settings?.n_ctx) ??
		positiveInt(m.meta?.n_ctx) ??
		divide(positiveInt(argValue(args, "--ctx-size", "-c")), parallel) ??
		divide(positiveInt(m.meta?.n_ctx_train), parallel) ??
		divide(DEFAULT_CTX, parallel) ??
		DEFAULT_CTX
	);
}

function divide(total: number | undefined, parallel: number): number | undefined {
	return total === undefined ? undefined : Math.max(1, Math.floor(total / parallel));
}

/**
 * Max output tokens for one assistant turn — *not* the context window.
 *
 * llama.cpp calls this `--predict` / `-n` / `--n-predict` and defaults it to -1,
 * meaning "generate until the context is full". When a preset pins a positive
 * value the server will stop there no matter what pi asks for, so that is the
 * honest ceiling; otherwise we advertise `LLAMACPP_MAX_OUTPUT_TOKENS` (16384 by
 * default). Either way the answer is clamped to the context window, since output
 * has to fit inside it.
 */
export function maxOutputTokens(m: LlamaModel, config: LlamaConfig, ctx: number, props?: LlamaProps): number {
	const args = m.status?.args ?? [];
	const pinned =
		positiveInt(argValue(args, "--predict", "-n", "--n-predict")) ??
		positiveInt(props?.default_generation_settings?.params?.n_predict);
	return Math.max(1, Math.min(pinned ?? config.maxOutputTokens, ctx)); // pi rejects maxTokens <= 0
}

/** Map one `/v1/models` entry — plus its `/props` probe, when we got one — to a pi model. */
export function toModel(m: LlamaModel, config: LlamaConfig = readConfig({}), props?: LlamaProps, cached?: boolean): ProviderModelConfig {
	const ctx = contextWindow(m, props);
	const { reasoning } = resolveThinking(m, config, props, cached);

	// pi models declare "text" and "image" inputs only; audio is dropped. A router
	// reports modalities in the model list, a plain llama-server only in /props.
	const modalities = m.architecture?.input_modalities ?? [];
	const vision = modalities.includes("image") || props?.modalities?.vision === true;
	const input: ("text" | "image")[] = vision ? ["text", "image"] : ["text"];

	const compat = {
		supportsDeveloperRole: false, // llama.cpp chat templates expect "system", not "developer"
		maxTokensField: "max_tokens" as const, // llama.cpp does not accept max_completion_tokens
	};

	return {
		id: m.id,
		name: m.id,
		reasoning,
		input,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, // local inference is free
		contextWindow: ctx,
		maxTokens: maxOutputTokens(m, config, ctx, props),
		compat: reasoning
			? {
					...compat,
					// llama.cpp implements no `reasoning_effort`; thinking is toggled through the
					// chat template. `enable_thinking` is the template variable, `preserve_reasoning`
					// the kwarg llama.cpp itself reads to keep reasoning traces in history, and
					// `preserve_thinking` the same idea for templates that spell it that way.
					thinkingFormat: "chat-template" as const,
					chatTemplateKwargs: {
						enable_thinking: { $var: "thinking.enabled" as const },
						preserve_reasoning: true,
						preserve_thinking: true,
					},
				}
			: compat,
	};
}

/** The model registered when discovery fails, so the provider exists once the server returns. */
export function fallbackModels(config: LlamaConfig = readConfig({})): ProviderModelConfig[] {
	return [toModel({ id: "default" }, config)];
}

/**
 * Ask `GET /props` about one model.
 *
 * `?autoload=0` is the important part: on a router, a bare `/props?model=<id>`
 * would *load* the model (spawning a llama-server and filling VRAM) just to
 * answer us. With `autoload=0` the router replies 400 "model is not loaded" for
 * anything it is not already serving, which we take as "no answer" — never as a
 * failure. A plain llama-server ignores the query entirely and describes its one
 * model.
 */
export async function fetchProps(
	config: LlamaConfig,
	id: string,
	fetchImpl: typeof fetch = fetch,
	single = false,
): Promise<LlamaProps | undefined> {
	try {
		const url = `${rootUrl(config.baseUrl)}/props?model=${encodeURIComponent(id)}&autoload=0`;
		const res = await fetchImpl(url, {
			headers: { Authorization: `Bearer ${config.apiKey}` },
			signal: AbortSignal.timeout(config.timeoutMs),
		});
		if (!res.ok) return undefined; // 400 = not loaded, 404/501 = /props disabled
		const props = (await res.json()) as LlamaProps;
		if (!props || typeof props !== "object" || props.role === "router") return undefined;
		// A router proxies to the child that owns the id, so the alias must match. A
		// plain llama-server ignores ?model= and always answers about its one model,
		// which is only safe to attribute when the list has exactly one entry.
		if (props.model_alias && props.model_alias !== id && !single) return undefined;
		return props;
	} catch {
		return undefined; // a probe is best-effort; discovery must survive it
	}
}

// ── learned-template cache ────────────────────────────────────────────────────
//
// A router only answers /props for models it currently has loaded, so a cold
// start sees no template for most presets. Rather than guess from the model's
// name — the thing that got Gemma 4 wrong in the first place — we remember what
// the template said the last time the model *was* loaded, keyed by the preset's
// argv so that editing the preset invalidates the entry.

interface CacheEntry {
	sig: string;
	reasoning: boolean;
}
type Cache = Record<string, CacheEntry>;

/** FNV-1a over the argv — a preset change must not keep an old verdict alive. */
export function argsSignature(m: LlamaModel): string {
	const input = (m.status?.args ?? []).join(" ");
	let hash = 0x811c9dc5;
	for (let i = 0; i < input.length; i++) {
		hash ^= input.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(16);
}

export function cachePath(env: NodeJS.ProcessEnv = process.env): string {
	const base = env.XDG_CACHE_HOME || join(env.HOME || homedir(), ".cache");
	return join(base, "pi-llamacpp-provider", "templates.json");
}

function readCache(path: string): Cache {
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8")) as { version?: number; entries?: Cache };
		return parsed?.version === 1 && parsed.entries ? parsed.entries : {};
	} catch {
		return {}; // absent, unreadable or from a future version — start over
	}
}

function writeCache(path: string, entries: Cache): void {
	try {
		mkdirSync(dirname(path), { recursive: true });
		// Write-then-rename: two pi processes starting at once must never leave a
		// half-written file behind for the next one to read.
		const tmp = `${path}.${process.pid}.tmp`;
		writeFileSync(tmp, JSON.stringify({ version: 1, entries }, null, "\t"));
		renameSync(tmp, path);
	} catch {
		// A cache we cannot write — read-only FS, no HOME — is a cache we do without.
	}
}

/** `http://host:port/v1` → `host:port`, for the provider's display name and cache keys. */
function displayHost(baseUrl: string): string {
	return baseUrl.replace(/^https?:\/\//, "").replace(/\/v1$/, "");
}

/**
 * Fetch and map the server's model list. Resolves to `{ models, error }` rather
 * than rejecting: a discovery failure degrades to `fallbackModels()`, it never
 * takes pi's startup down with it.
 */
export async function discoverModels(
	config: LlamaConfig,
	fetchImpl: typeof fetch = fetch,
	env: NodeJS.ProcessEnv = process.env,
): Promise<{ models: ProviderModelConfig[]; error?: string }> {
	let entries: LlamaModel[];
	try {
		const res = await fetchImpl(`${config.baseUrl}/models`, {
			headers: { Authorization: `Bearer ${config.apiKey}` },
			signal: AbortSignal.timeout(config.timeoutMs),
		});
		if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

		const payload = (await res.json()) as { data?: LlamaModel[] };
		entries = (payload?.data ?? []).filter((m): m is LlamaModel => typeof m?.id === "string" && m.id.length > 0);
		if (entries.length === 0) throw new Error("server returned no models");
	} catch (err) {
		const reason = err instanceof Error ? err.message : String(err);
		return { models: fallbackModels(config), error: reason };
	}

	// From here on nothing may fail the discovery: we already have the model list.
	const path = cachePath(env);
	const cache = config.probe ? readCache(path) : {};
	const host = displayHost(config.baseUrl);
	const key = (m: LlamaModel) => `${encodeURIComponent(host)}|${encodeURIComponent(m.id)}`;

	const probes = config.probe
		? await Promise.all(entries.map((m) => fetchProps(config, m.id, fetchImpl, entries.length === 1)))
		: entries.map(() => undefined);

	const models = entries.map((m, i) => {
		const props = probes[i];
		const entry = cache[key(m)];
		const cached = entry?.sig === argsSignature(m) ? entry.reasoning : undefined;
		return toModel(m, config, props, cached);
	});

	if (config.probe) {
		// Remember every template we actually saw, so the next cold start knows about
		// models the router has unloaded by then.
		let dirty = false;
		entries.forEach((m, i) => {
			const template = probes[i]?.chat_template;
			if (!template) return;
			const next = { sig: argsSignature(m), reasoning: detectsThinking(template) };
			const prev = cache[key(m)];
			if (prev?.sig !== next.sig || prev?.reasoning !== next.reasoning) {
				cache[key(m)] = next;
				dirty = true;
			}
		});
		if (dirty) writeCache(path, cache);
	}

	return { models };
}

export default async function (pi: ExtensionAPI) {
	const config = readConfig();
	const { models, error } = await discoverModels(config);

	pi.registerProvider(config.provider, {
		name: `llama.cpp (${displayHost(config.baseUrl)})`,
		baseUrl: config.baseUrl,
		apiKey: config.apiKey,
		api: "openai-completions",
		models,
	});

	if (!error) return;

	const message =
		`Model discovery from ${config.baseUrl} failed (${error}). ` +
		`Registered provider "${config.provider}" with a single "default" model. ` +
		`Start the server and run /reload, or set LLAMACPP_BASE_URL.`;

	let warned = false;
	pi.on("session_start", async (_event, ctx) => {
		if (warned) return; // session_start also fires on reload/switch — warn once per process
		warned = true;
		if (ctx.hasUI) {
			ctx.ui.notify(`[pi-llamacpp-provider] ${message}`, "warning");
		} else {
			console.warn(`[pi-llamacpp-provider] ${message}`);
		}
	});
}
