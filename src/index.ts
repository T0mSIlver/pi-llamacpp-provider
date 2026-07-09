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
 *   - **llama.cpp router** (`llama-server --router`, or the `llama-swap`-style
 *     multi-preset server): each entry carries `status.args` — the full argv of
 *     the `llama-server` process for that preset — plus `preset` and
 *     `architecture.input_modalities`. Context size, parallel slot count, and
 *     vision support are all read from there.
 *   - **plain `llama-server`**: a single entry with a `meta` block. Context
 *     size falls back to `meta.n_ctx`, then `meta.n_ctx_train`.
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

import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

/** llama.cpp default `--ctx-size` when a preset does not set one. */
const DEFAULT_CTX = 4096;
/** Never advertise a context window below this — pi needs room for a system prompt. */
const MIN_CTX = 2048;
/** Output-token cap. llama.cpp has no `max_tokens` ceiling of its own; this keeps a runaway generation bounded. */
const MAX_OUTPUT_TOKENS = 8192;

/** The subset of a `/v1/models` entry this extension reads. Everything is optional but `id`. */
export interface LlamaModel {
	id: string;
	/** Router only: `input_modalities` may include "text", "image", "audio". */
	architecture?: { input_modalities?: string[] };
	/** Router only: `args` is the llama-server argv for this preset. */
	status?: { args?: string[] };
	/** Router only: the preset's INI text. Searched for thinking flags. */
	preset?: string;
	/** Plain llama-server: `n_ctx` is the runtime context, `n_ctx_train` the model's trained maximum. */
	meta?: { n_ctx?: number; n_ctx_train?: number };
}

export interface LlamaConfig {
	baseUrl: string;
	apiKey: string;
	timeoutMs: number;
	provider: string;
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
		timeoutMs: Number(env.LLAMACPP_TIMEOUT_MS) || 4000,
		provider: env.LLAMACPP_PROVIDER || "llamacpp",
	};
}

/** Strip trailing slashes and append `/v1` unless the URL already ends in it. */
export function normalizeBaseUrl(raw: string): string {
	const trimmed = raw.trim().replace(/\/+$/, "");
	return /\/v1$/.test(trimmed) ? trimmed : `${trimmed}/v1`;
}

/** Read the value following `flag` in a llama-server argv array. */
export function argValue(args: string[], flag: string): string | undefined {
	const i = args.indexOf(flag);
	return i >= 0 && i + 1 < args.length ? args[i + 1] : undefined;
}

/** Map one `/v1/models` entry to a pi model definition. */
export function toModel(m: LlamaModel): ProviderModelConfig {
	const args = m.status?.args ?? [];

	// Usable context per conversation = total context / parallel slots, since
	// llama.cpp splits --ctx-size evenly across --parallel slots.
	const ctxSize = Number(argValue(args, "--ctx-size")) || m.meta?.n_ctx || m.meta?.n_ctx_train || DEFAULT_CTX;
	const parallel = Number(argValue(args, "--parallel")) || 1;
	const contextWindow = Math.max(MIN_CTX, Math.floor(ctxSize / parallel));

	// pi models declare "text" and "image" inputs only; drop anything else (audio).
	const modalities = m.architecture?.input_modalities ?? ["text"];
	const input: ("text" | "image")[] = modalities.includes("image") ? ["text", "image"] : ["text"];

	// Qwen presets run with preserve_thinking; drive them through llama.cpp's
	// chat-template thinking toggle rather than OpenAI's reasoning_effort, which
	// llama.cpp does not implement.
	const reasoning = isThinkingModel(m);

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
		contextWindow,
		maxTokens: Math.max(1024, Math.min(MAX_OUTPUT_TOKENS, contextWindow)),
		compat: reasoning ? { ...compat, thinkingFormat: "qwen-chat-template" as const } : compat,
	};
}

/** A model reasons if its id says Qwen, or its preset/argv turns thinking on. */
function isThinkingModel(m: LlamaModel): boolean {
	if (/qwen/i.test(m.id)) return true;
	const config = `${m.preset ?? ""} ${(m.status?.args ?? []).join(" ")}`;
	return /preserve_thinking|enable_thinking/i.test(config);
}

/** The model registered when discovery fails, so the provider exists once the server returns. */
export function fallbackModels(): ProviderModelConfig[] {
	return [toModel({ id: "default" })];
}

/**
 * Fetch and map the server's model list. Resolves to `{ models, error }` rather
 * than rejecting: a discovery failure degrades to `fallbackModels()`, it never
 * takes pi's startup down with it.
 */
export async function discoverModels(
	config: LlamaConfig,
	fetchImpl: typeof fetch = fetch,
): Promise<{ models: ProviderModelConfig[]; error?: string }> {
	try {
		const res = await fetchImpl(`${config.baseUrl}/models`, {
			headers: { Authorization: `Bearer ${config.apiKey}` },
			signal: AbortSignal.timeout(config.timeoutMs),
		});
		if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);

		const payload = (await res.json()) as { data?: LlamaModel[] };
		const entries = (payload?.data ?? []).filter((m): m is LlamaModel => typeof m?.id === "string" && m.id.length > 0);
		if (entries.length === 0) throw new Error("server returned no models");

		return { models: entries.map(toModel) };
	} catch (err) {
		const reason = err instanceof Error ? err.message : String(err);
		return { models: fallbackModels(), error: reason };
	}
}

/** `http://host:port/v1` → `host:port`, for the provider's display name. */
function displayHost(baseUrl: string): string {
	return baseUrl.replace(/^https?:\/\//, "").replace(/\/v1$/, "");
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
