// Test harness for pi-llamacpp-provider (run with node >= 22.19 for native .ts import).
//
// Exercises the extension against real HTTP servers rather than a stubbed fetch:
// a llama.cpp *router* payload captured from a live server, a plain
// `llama-server` payload, real chat templates taken from llama.cpp's own
// `models/templates`, and every failure mode that must degrade to the fallback
// model instead of taking pi's startup down.
import http from "node:http";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ext = await import(new URL("../src/index.ts", import.meta.url));
const { default: extension, normalizeBaseUrl, rootUrl, argValue, toModel, readConfig, detectsThinking, globMatch } = ext;

const ROUTER = readFileSync(new URL("./fixtures/models.json", import.meta.url), "utf8");
const PLAIN = readFileSync(new URL("./fixtures/plain-server.json", import.meta.url), "utf8");
const template = (name) => readFileSync(new URL(`./fixtures/templates/${name}.jinja`, import.meta.url), "utf8");

const GEMMA4 = template("google-gemma-4-31B-it");
const QWEN3 = template("Qwen-Qwen3-0.6B");
const QWEN3_CODER = template("Qwen3-Coder");
const GPT_OSS = template("openai-gpt-oss-120b");
const LLAMA3 = template("meta-llama-Llama-3.3-70B-Instruct");
const MINISTRAL_REASONING = template("mistralai-Ministral-3-14B-Reasoning-2512");

let failures = 0;
function check(name, cond, detail = "") {
	console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
	if (!cond) failures++;
}

/** Start a throwaway HTTP server on a random loopback port. */
function serve(handler) {
	const sockets = new Set();
	const server = http.createServer(handler);
	server.on("connection", (s) => {
		sockets.add(s);
		s.on("close", () => sockets.delete(s));
	});
	return new Promise((resolve) => {
		server.listen(0, "127.0.0.1", () => {
			resolve({
				url: `http://127.0.0.1:${server.address().port}`,
				requests: [],
				stop: () => {
					for (const s of sockets) s.destroy(); // the timeout server never finishes a response
					return new Promise((r) => server.close(r));
				},
			});
		});
	});
}

/**
 * A llama.cpp stand-in: serves `GET /v1/models` from `body`, and `GET /props` from
 * `props` — a map of model id → props payload. Ids missing from the map answer the
 * way a router answers for a model it has not loaded: 400, no template.
 */
async function serveLlama(body, { status = 200, props = {}, propsStatus = 200 } = {}) {
	let ctx;
	ctx = await serve((req, res) => {
		ctx.requests.push({ url: req.url, auth: req.headers.authorization });
		const url = new URL(req.url, "http://x");
		if (url.pathname === "/props") {
			const id = url.searchParams.get("model");
			const found = props[id];
			if (propsStatus !== 200 || !found) {
				res.writeHead(propsStatus === 200 ? 400 : propsStatus, { "content-type": "application/json" });
				res.end(JSON.stringify({ error: { message: "model is not loaded" } }));
				return;
			}
			res.writeHead(200, { "content-type": "application/json" });
			res.end(JSON.stringify(found));
			return;
		}
		res.writeHead(status, { "content-type": "application/json" });
		res.end(body);
	});
	return ctx;
}

/** Every env var the extension reads — cleared between runs so tests do not leak into each other. */
const ENV_KEYS = [
	"LLAMACPP_BASE_URL",
	"LLAMACPP_API_KEY",
	"LLAMACPP_TIMEOUT_MS",
	"LLAMACPP_PROVIDER",
	"LLAMACPP_PROBE",
	"LLAMACPP_MAX_OUTPUT_TOKENS",
	"LLAMACPP_THINKING_MODELS",
	"LLAMACPP_NON_THINKING_MODELS",
	"XDG_CACHE_HOME",
];

/** A cache directory per test, so the learned-template cache never touches the real one. */
const CACHE_ROOT = mkdtempSync(join(tmpdir(), "pi-llamacpp-test-"));
let cacheSeq = 0;
const freshCache = () => join(CACHE_ROOT, `c${cacheSeq++}`);

/** Minimal stand-in for ExtensionAPI: records registerProvider + event handlers. */
function fakePi() {
	const providers = [];
	const handlers = {};
	return {
		providers,
		handlers,
		registerProvider: (name, config) => providers.push({ name, config }),
		on: (event, handler) => (handlers[event] ??= []).push(handler),
	};
}

/** Run the extension's default export with a temporary environment. */
async function run(env) {
	const saved = Object.fromEntries(ENV_KEYS.map((k) => [k, process.env[k]]));
	for (const k of ENV_KEYS) delete process.env[k];
	Object.assign(process.env, { XDG_CACHE_HOME: freshCache(), ...env });
	const pi = fakePi();
	try {
		await extension(pi);
	} finally {
		for (const k of ENV_KEYS) {
			if (saved[k] === undefined) delete process.env[k];
			else process.env[k] = saved[k];
		}
	}
	return pi;
}

const byId = (models, id) => models.find((m) => m.id === id);
const propsRequests = (srv) => srv.requests.filter((r) => r.url.startsWith("/props"));

// ── URL handling ──────────────────────────────────────────────────────────────
check("normalizeBaseUrl appends /v1", normalizeBaseUrl("http://h:8080") === "http://h:8080/v1");
check("normalizeBaseUrl strips trailing slashes", normalizeBaseUrl("http://h:8080///") === "http://h:8080/v1");
check("normalizeBaseUrl keeps an existing /v1", normalizeBaseUrl("http://h:8080/v1") === "http://h:8080/v1");
check("normalizeBaseUrl trims whitespace", normalizeBaseUrl("  http://h:8080/v1/  ") === "http://h:8080/v1");
check("rootUrl drops /v1 — /props lives at the server root", rootUrl("http://h:8080/v1") === "http://h:8080");

// ── argv parsing ──────────────────────────────────────────────────────────────
check("argValue reads the value after a flag", argValue(["--ctx-size", "4096"], "--ctx-size") === "4096");
check("argValue returns undefined when absent", argValue(["--parallel", "2"], "--ctx-size") === undefined);
check("argValue returns undefined for a trailing flag", argValue(["--slots"], "--slots") === undefined);
check("argValue accepts flag aliases", argValue(["-c", "8192"], "--ctx-size", "-c") === "8192");
check("argValue takes the last of a repeated flag", argValue(["-n", "10", "-n", "20"], "-n") === "20");

// ── glob overrides ────────────────────────────────────────────────────────────
check("globMatch is case-insensitive and honours *", globMatch("gemma4-*", "GEMMA4-31b") && !globMatch("gemma4-*", "qwen3"));
check("globMatch anchors both ends", !globMatch("qwen", "qwen36a3b-35b") && globMatch("qwen*", "qwen36a3b-35b"));

// ── thinking detection, against real llama.cpp chat templates ─────────────────
check("detectsThinking: Gemma 4 thinks (<|channel>thought)", detectsThinking(GEMMA4) === true);
check("detectsThinking: Qwen3 thinks (enable_thinking)", detectsThinking(QWEN3) === true);
check("detectsThinking: gpt-oss thinks (<|channel|>analysis)", detectsThinking(GPT_OSS) === true);
check("detectsThinking: Ministral-Reasoning thinks ([THINK])", detectsThinking(MINISTRAL_REASONING) === true);
check("detectsThinking: Qwen3-Coder does NOT think, despite the qwen name", detectsThinking(QWEN3_CODER) === false);
check("detectsThinking: Llama 3.3 does not think", detectsThinking(LLAMA3) === false);
check("detectsThinking: an empty template is not a thinking model", detectsThinking("") === false);

// ── defaults ──────────────────────────────────────────────────────────────────
const defaults = readConfig({});
check(
	"readConfig defaults to loopback, probes /props, and caps output at 16384",
	defaults.baseUrl === "http://127.0.0.1:8080/v1" &&
		defaults.provider === "llamacpp" &&
		defaults.timeoutMs === 4000 &&
		defaults.probe === true &&
		defaults.maxOutputTokens === 16384,
	JSON.stringify(defaults),
);
check("readConfig: LLAMACPP_PROBE=0 turns probing off", readConfig({ LLAMACPP_PROBE: "0" }).probe === false);
check("readConfig: LLAMACPP_PROBE=1 keeps probing on", readConfig({ LLAMACPP_PROBE: "1" }).probe === true);
check(
	"readConfig: a nonsense env value falls back to the default rather than through it",
	(() => {
		const bad = readConfig({ LLAMACPP_TIMEOUT_MS: "-1", LLAMACPP_MAX_OUTPUT_TOKENS: "-5" });
		// A negative timeout makes AbortSignal.timeout throw (degrading every model to the
		// fallback); a negative maxTokens is a value pi rejects outright.
		return bad.timeoutMs === 4000 && bad.maxOutputTokens === 16384;
	})(),
);
check(
	"maxTokens is never <= 0, whatever the environment says",
	toModel({ id: "x" }, readConfig({ LLAMACPP_MAX_OUTPUT_TOKENS: "0" })).maxTokens > 0,
);

// ── context-window math ───────────────────────────────────────────────────────
const cfg = readConfig({});
check("ctx defaults to 4096 with no args and no meta", toModel({ id: "x" }, cfg).contextWindow === 4096);
check(
	"ctx is divided across --parallel slots",
	toModel({ id: "x", status: { args: ["--ctx-size", "160000", "--parallel", "2"] } }, cfg).contextWindow === 80000,
);
check(
	"a small context is reported honestly, not rounded up to something comfortable",
	toModel({ id: "x", meta: { n_ctx: 512 } }, cfg).contextWindow === 512,
);
check(
	"ctx is divided even when the result is small",
	toModel({ id: "x", status: { args: ["--ctx-size", "4096", "--parallel", "8"] } }, cfg).contextWindow === 512,
);
check(
	"meta.n_ctx is already per-slot, so it is NOT divided by --parallel again",
	toModel({ id: "x", status: { args: ["--parallel", "4"] }, meta: { n_ctx: 32768 } }, cfg).contextWindow === 32768,
);
check(
	"props n_ctx (per-slot, from the live server) wins over --ctx-size",
	toModel({ id: "x", status: { args: ["--ctx-size", "8192"] } }, cfg, { default_generation_settings: { n_ctx: 65536 } })
		.contextWindow === 65536,
);
check(
	"--ctx-size 0 means 'take it from the model', not zero",
	toModel({ id: "x", status: { args: ["--ctx-size", "0"] }, meta: { n_ctx_train: 8192 } }, cfg).contextWindow === 8192,
);

// ── maxTokens is an OUTPUT cap, not the context window ────────────────────────
check("maxTokens defaults to 16384, not the context window", toModel({ id: "x", meta: { n_ctx: 262144 } }, cfg).maxTokens === 16384);
check("maxTokens cannot exceed the context window", toModel({ id: "x" }, cfg).maxTokens === 4096);
check(
	"maxTokens honours a preset's --predict",
	toModel({ id: "x", status: { args: ["--ctx-size", "65536", "--predict", "2048"] } }, cfg).maxTokens === 2048,
);
check(
	"maxTokens honours -n and --n-predict too",
	toModel({ id: "x", meta: { n_ctx: 65536 }, status: { args: ["-n", "512"] } }, cfg).maxTokens === 512 &&
		toModel({ id: "x", meta: { n_ctx: 65536 }, status: { args: ["--n-predict", "700"] } }, cfg).maxTokens === 700,
);
check(
	"maxTokens ignores llama.cpp's -1 ('generate until the context is full')",
	toModel({ id: "x", meta: { n_ctx: 65536 }, status: { args: ["--predict", "-1"] } }, cfg).maxTokens === 16384,
);
check(
	"maxTokens reads n_predict from a live /props",
	toModel({ id: "x", meta: { n_ctx: 65536 } }, cfg, { default_generation_settings: { params: { n_predict: 4096 } } }).maxTokens ===
		4096,
);
check(
	"LLAMACPP_MAX_OUTPUT_TOKENS overrides the default cap",
	toModel({ id: "x", meta: { n_ctx: 262144 } }, readConfig({ LLAMACPP_MAX_OUTPUT_TOKENS: "65536" })).maxTokens === 65536,
);

// ── thinking resolution precedence (unit) ─────────────────────────────────────
check(
	"a thinking model gets llama.cpp's chat-template toggle, not reasoning_effort",
	(() => {
		const m = toModel({ id: "x" }, cfg, { chat_template: GEMMA4 });
		return (
			m.reasoning === true &&
			m.compat.thinkingFormat === "chat-template" &&
			m.compat.chatTemplateKwargs.enable_thinking.$var === "thinking.enabled" &&
			m.compat.chatTemplateKwargs.preserve_reasoning === true
		);
	})(),
);
check(
	"a non-thinking model gets no thinkingFormat at all",
	(() => {
		const m = toModel({ id: "x" }, cfg, { chat_template: LLAMA3 });
		return m.reasoning === false && m.compat.thinkingFormat === undefined && m.compat.chatTemplateKwargs === undefined;
	})(),
);
check(
	"--reasoning off beats a thinking template",
	toModel({ id: "x", status: { args: ["--reasoning", "off"] } }, cfg, { chat_template: QWEN3 }).reasoning === false,
);
check(
	// --reasoning-budget is a sampler knob that cuts the thinking block short, not a
	// template toggle: llama.cpp's enable_thinking depends only on --reasoning and on
	// template support. A model with budget 0 still thinks, and saying otherwise would
	// leave pi unable to ask the template to suppress thinking cleanly.
	"--reasoning-budget 0 does NOT make a thinking model non-thinking",
	toModel({ id: "x", status: { args: ["--reasoning-budget", "0"] } }, cfg, { chat_template: QWEN3 }).reasoning === true,
);
check(
	"-rea off is the same flag as --reasoning off",
	toModel({ id: "x", status: { args: ["-rea", "off"] } }, cfg, { chat_template: QWEN3 }).reasoning === false,
);
check(
	"--no-jinja beats a thinking template (llama.cpp cannot think without it)",
	toModel({ id: "x", status: { args: ["--no-jinja"] } }, cfg, { chat_template: QWEN3 }).reasoning === false,
);
check(
	"the live template beats the cached verdict",
	toModel({ id: "x" }, cfg, { chat_template: LLAMA3 }, true).reasoning === false,
);
check("the cached verdict is used when there is no template", toModel({ id: "x" }, cfg, undefined, true).reasoning === true);
check(
	"a preset's preserve_thinking is a fallback hint, below the template",
	toModel({ id: "x", status: { preset: 'chat-template-kwargs = {"preserve_thinking":true}' } }, cfg).reasoning === true &&
		toModel({ id: "x", status: { preset: 'chat-template-kwargs = {"preserve_thinking":true}' } }, cfg, { chat_template: LLAMA3 })
			.reasoning === false,
);
check(
	"the thinking kwarg is read as JSON: a switch set to false still means the model has one",
	toModel({ id: "x", status: { args: ["--chat-template-kwargs", '{"enable_thinking":false}'] } }, cfg).reasoning === true,
);
check(
	"kwargs unrelated to thinking are not a thinking signal",
	toModel({ id: "x", status: { args: ["--chat-template-kwargs", '{"custom_tool_prompt":"enable_thinking"}'] } }, cfg).reasoning ===
		false,
);
check(
	"a positive --reasoning-budget implies a model that thinks",
	toModel({ id: "x", status: { args: ["--reasoning-budget", "2048"] } }, cfg).reasoning === true,
);
check(
	"no evidence at all → not a thinking model (the id is never consulted)",
	toModel({ id: "qwen3-coder-30b" }, cfg).reasoning === false,
);
check(
	"LLAMACPP_THINKING_MODELS / _NON_THINKING_MODELS beat every other signal",
	toModel({ id: "gemma4-31b" }, readConfig({ LLAMACPP_THINKING_MODELS: "gemma4-*" })).reasoning === true &&
		toModel({ id: "qwen3-8b" }, readConfig({ LLAMACPP_NON_THINKING_MODELS: "qwen3-*" }), { chat_template: QWEN3 }).reasoning ===
			false,
);

// ── router payload (captured from a live llama.cpp router) ────────────────────
{
	// The router has two presets loaded; the other seven are unloaded, so /props
	// answers 400 for them exactly as a real router does with autoload=0.
	const srv = await serveLlama(ROUTER, {
		props: {
			"gemma4-31b": { model_alias: "gemma4-31b", chat_template: GEMMA4, default_generation_settings: { n_ctx: 100000 } },
			"laguna-xs2": { model_alias: "laguna-xs2", chat_template: LLAMA3 },
		},
	});
	const pi = await run({ LLAMACPP_BASE_URL: srv.url, LLAMACPP_API_KEY: "sk-test" });
	const { name, config } = pi.providers[0] ?? {};
	const models = config?.models ?? [];

	check("registers exactly one provider", pi.providers.length === 1 && name === "llamacpp");
	check(
		"provider config points at the server's /v1 with the OpenAI completions API",
		config?.api === "openai-completions" && config?.baseUrl === `${srv.url}/v1` && config?.apiKey === "sk-test",
		`${config?.api} ${config?.baseUrl}`,
	);
	check("provider display name carries the host", /^llama\.cpp \(127\.0\.0\.1:\d+\)$/.test(config?.name ?? ""), config?.name);
	check(
		"discovery requests /v1/models with a bearer token",
		srv.requests[0].url === "/v1/models" && srv.requests[0].auth === "Bearer sk-test",
		JSON.stringify(srv.requests[0]),
	);
	check(
		"every model is probed at /props with autoload=0, so the router never loads one for us",
		propsRequests(srv).length === 9 && propsRequests(srv).every((r) => /[?&]autoload=0(&|$)/.test(r.url)),
		propsRequests(srv)
			.map((r) => r.url)
			.join(" "),
	);
	check("every server model is registered", models.length === 9, models.map((m) => m.id).join(","));
	check("no session_start warning on success", (pi.handlers.session_start ?? []).length === 0);

	check("fastcontext: 160000 ctx over 2 slots → 80000", byId(models, "fastcontext")?.contextWindow === 80000);
	check("gemma4-31b: ctx 100000", byId(models, "gemma4-31b")?.contextWindow === 100000);
	check("default preset: no ctx-size → 4096", byId(models, "default")?.contextWindow === 4096);
	check("maxTokens is the output cap, not the context window", models.every((m) => m.maxTokens === Math.min(16384, m.contextWindow)));

	const gemma12b = byId(models, "gemma4-12b");
	check(
		"gemma4-12b: audio modality dropped, image kept",
		JSON.stringify(gemma12b?.input) === '["text","image"]',
		JSON.stringify(gemma12b?.input),
	);
	check("laguna-xs2: text-only model has no image input", JSON.stringify(byId(models, "laguna-xs2")?.input) === '["text"]');

	// The headline fix: gemma4-31b reasons, and the server's own chat template says so.
	const gemma31b = byId(models, "gemma4-31b");
	check(
		"gemma4-31b: thinking read from the live chat template, not from its name",
		gemma31b?.reasoning === true && gemma31b?.compat?.thinkingFormat === "chat-template",
		JSON.stringify(gemma31b?.compat),
	);
	check(
		"laguna-xs2: its template has no thinking markers → no thinking",
		byId(models, "laguna-xs2")?.reasoning === false && byId(models, "laguna-xs2")?.compat?.thinkingFormat === undefined,
	);
	const ornith = byId(models, "ornith-35b");
	check(
		"ornith-35b: unloaded, but its preset passes preserve_thinking → thinking",
		ornith?.reasoning === true && ornith?.compat?.thinkingFormat === "chat-template",
	);
	check("qwen36a3b-35b: unloaded, preserve_thinking in the preset → thinking", byId(models, "qwen36a3b-35b")?.reasoning === true);
	check(
		"fastcontext: unloaded, no thinking evidence anywhere → not a thinking model",
		byId(models, "fastcontext")?.reasoning === false,
	);

	check(
		"every model is free, uses max_tokens, and avoids the developer role",
		models.every(
			(m) =>
				m.cost.input === 0 &&
				m.cost.output === 0 &&
				m.cost.cacheRead === 0 &&
				m.cost.cacheWrite === 0 &&
				m.compat.maxTokensField === "max_tokens" &&
				m.compat.supportsDeveloperRole === false,
		),
	);
	check("model name mirrors the server-side id", models.every((m) => m.name === m.id));

	await srv.stop();
}

// ── the learned-template cache ────────────────────────────────────────────────
{
	// First run: gemma4-31b happens to be loaded, so we read its template.
	const cache = freshCache();
	const props = { "gemma4-31b": { model_alias: "gemma4-31b", chat_template: GEMMA4 } };
	const srv = await serveLlama(ROUTER, { props });
	const first = await run({ LLAMACPP_BASE_URL: srv.url, XDG_CACHE_HOME: cache });
	check(
		"cache: first run reads the template of the loaded model",
		byId(first.providers[0].config.models, "gemma4-31b").reasoning === true,
	);

	// The router unloads it (idle eviction, a restart, whatever). /props now says
	// "not loaded" — but we still know what its template said last time.
	delete props["gemma4-31b"];
	const second = await run({ LLAMACPP_BASE_URL: srv.url, XDG_CACHE_HOME: cache });
	check(
		"cache: an unloaded model keeps the verdict learned from its template",
		byId(second.providers[0].config.models, "gemma4-31b")?.reasoning === true,
	);

	// A cache entry is keyed by the preset argv, so editing the preset drops it.
	const edited = JSON.parse(ROUTER);
	const entry = edited.data.find((m) => m.id === "gemma4-31b");
	entry.status.args = [...entry.status.args, "--reasoning", "off"];
	await srv.stop();
	const changed = await serveLlama(JSON.stringify(edited), { props: {} });
	const third = await run({ LLAMACPP_BASE_URL: changed.url, XDG_CACHE_HOME: cache });
	check(
		"cache: a stale entry never overrides a newer explicit flag",
		byId(third.providers[0].config.models, "gemma4-31b")?.reasoning === false,
	);
	await changed.stop();
}

// ── probing can be turned off ─────────────────────────────────────────────────
{
	const srv = await serveLlama(ROUTER, { props: { "gemma4-31b": { model_alias: "gemma4-31b", chat_template: GEMMA4 } } });
	const pi = await run({ LLAMACPP_BASE_URL: srv.url, LLAMACPP_PROBE: "0" });
	check("LLAMACPP_PROBE=0: no /props request is made", propsRequests(srv).length === 0);
	check(
		"LLAMACPP_PROBE=0: detection falls back to the preset, and gemma4 goes unrecognised",
		byId(pi.providers[0].config.models, "gemma4-31b")?.reasoning === false &&
			byId(pi.providers[0].config.models, "ornith-35b")?.reasoning === true,
	);
	await srv.stop();
}

// ── a broken /props must never break discovery ────────────────────────────────
{
	const srv = await serveLlama(ROUTER, { propsStatus: 500 });
	const pi = await run({ LLAMACPP_BASE_URL: srv.url });
	const models = pi.providers[0]?.config?.models ?? [];
	check("a failing /props degrades to static detection, models still register", models.length === 9);
	check("a failing /props is not a discovery failure", (pi.handlers.session_start ?? []).length === 0);
	await srv.stop();
}

// ── plain llama-server payload ────────────────────────────────────────────────
{
	// A plain llama-server ignores ?model= and always describes its one model; it is
	// also the only place a plain server reports vision support.
	const srv = await serveLlama(PLAIN, {
		props: {
			"qwen3-8b-instruct": { chat_template: QWEN3_CODER, modalities: { vision: true } },
			"mistral-7b-instruct": { chat_template: LLAMA3 },
		},
	});
	const pi = await run({ LLAMACPP_BASE_URL: `${srv.url}/v1/`, LLAMACPP_PROVIDER: "local" });
	const models = pi.providers[0]?.config?.models ?? [];

	check("provider name is overridable", pi.providers[0]?.name === "local");
	check("plain server: both models registered", models.length === 2);
	check(
		"plain server: runtime n_ctx wins over n_ctx_train",
		byId(models, "qwen3-8b-instruct")?.contextWindow === 32768,
		String(byId(models, "qwen3-8b-instruct")?.contextWindow),
	);
	check("plain server: falls back to n_ctx_train", byId(models, "mistral-7b-instruct")?.contextWindow === 32768);
	check(
		"plain server: a qwen-named model with a non-thinking template does NOT reason",
		byId(models, "qwen3-8b-instruct")?.reasoning === false,
	);
	check("plain server: mistral does not reason", byId(models, "mistral-7b-instruct")?.reasoning === false);
	check(
		"plain server: vision comes from /props, since there is no architecture block",
		JSON.stringify(byId(models, "qwen3-8b-instruct")?.input) === '["text","image"]',
		JSON.stringify(byId(models, "qwen3-8b-instruct")?.input),
	);

	await srv.stop();
}

// ── a single-model server may answer /props for any id ────────────────────────
{
	const srv = await serveLlama(JSON.stringify({ data: [{ id: "solo", meta: { n_ctx: 8192 } }] }), {
		props: { solo: { model_alias: "some-other-alias", chat_template: QWEN3 } },
	});
	const pi = await run({ LLAMACPP_BASE_URL: srv.url });
	check(
		"single-model server: props are trusted even when the alias differs from the listed id",
		byId(pi.providers[0].config.models, "solo")?.reasoning === true,
	);
	await srv.stop();
}

// ── a mismatched alias on a multi-model server is ignored ─────────────────────
{
	const srv = await serveLlama(JSON.stringify({ data: [{ id: "a" }, { id: "b" }] }), {
		props: { a: { model_alias: "somebody-else", chat_template: QWEN3 } },
	});
	const pi = await run({ LLAMACPP_BASE_URL: srv.url });
	check(
		"multi-model server: props whose alias names a different model are discarded",
		byId(pi.providers[0].config.models, "a")?.reasoning === false,
	);
	await srv.stop();
}

// ── partially malformed payload ───────────────────────────────────────────────
{
	const srv = await serveLlama(JSON.stringify({ data: [{ id: "good" }, { name: "no-id" }, null, { id: "" }] }));
	const pi = await run({ LLAMACPP_BASE_URL: srv.url });
	const models = pi.providers[0]?.config?.models ?? [];
	check("entries without a usable id are skipped", models.length === 1 && models[0].id === "good", models.map((m) => m.id).join(","));
	check("a partially malformed payload is not a failure", (pi.handlers.session_start ?? []).length === 0);
	await srv.stop();
}

// ── failure modes all degrade to the fallback model ───────────────────────────
async function expectFallback(label, { env, srv }) {
	const pi = await run(env);
	const { name, config } = pi.providers[0] ?? {};
	const models = config?.models ?? [];
	check(
		`fallback (${label}): provider still registered with a single "default" model`,
		pi.providers.length === 1 && name === "llamacpp" && models.length === 1 && models[0].id === "default" && models[0].contextWindow === 4096,
		models.map((m) => m.id).join(","),
	);
	check(`fallback (${label}): warns at session_start`, (pi.handlers.session_start ?? []).length === 1);
	await srv?.stop();
	return pi;
}

{
	const srv = await serveLlama('{"error":"boom"}', { status: 500 });
	await expectFallback("HTTP 500", { env: { LLAMACPP_BASE_URL: srv.url }, srv });
}
{
	const srv = await serveLlama("this is not json");
	await expectFallback("malformed JSON", { env: { LLAMACPP_BASE_URL: srv.url }, srv });
}
{
	const srv = await serveLlama(JSON.stringify({ data: [] }));
	await expectFallback("empty model list", { env: { LLAMACPP_BASE_URL: srv.url }, srv });
}
{
	const srv = await serveLlama(JSON.stringify({ data: [{ name: "no-id" }] }));
	await expectFallback("no usable entries", { env: { LLAMACPP_BASE_URL: srv.url }, srv });
}
{
	// Bind a port, then release it: nothing is listening when the extension connects.
	const dead = await serveLlama("{}");
	const url = dead.url;
	await dead.stop();
	await expectFallback("connection refused", { env: { LLAMACPP_BASE_URL: url }, srv: null });
}
{
	const srv = await serve(() => {}); // accepts the request, never responds
	await expectFallback("timeout", { env: { LLAMACPP_BASE_URL: srv.url, LLAMACPP_TIMEOUT_MS: "150" }, srv });
}

// ── the degraded warning fires exactly once, through the UI when there is one ─
{
	const srv = await serveLlama("{}", { status: 503 });
	const pi = await expectFallback("warning wiring", { env: { LLAMACPP_BASE_URL: srv.url }, srv: null });
	const handler = pi.handlers.session_start[0];

	const notices = [];
	const uiCtx = { hasUI: true, ui: { notify: (msg, level) => notices.push({ msg, level }) } };
	await handler({}, uiCtx);
	await handler({}, uiCtx); // session_start also fires on /reload and session switch
	check("warns once per process, not once per session_start", notices.length === 1, JSON.stringify(notices.map((n) => n.level)));
	check(
		"warning is a UI notification naming the package, the URL and the cause",
		notices[0]?.level === "warning" &&
			notices[0]?.msg.startsWith("[pi-llamacpp-provider]") &&
			notices[0].msg.includes(srv.url) &&
			notices[0].msg.includes("503"),
		notices[0]?.msg,
	);

	// A second run, headless: the warning goes to stderr instead of the TUI.
	const pi2 = await run({ LLAMACPP_BASE_URL: srv.url });
	const logged = [];
	const origWarn = console.warn;
	console.warn = (...a) => logged.push(a.join(" "));
	try {
		await pi2.handlers.session_start[0]({}, { hasUI: false });
	} finally {
		console.warn = origWarn;
	}
	check("headless mode logs the warning to console", logged.length === 1 && logged[0].includes("[pi-llamacpp-provider]"), logged[0]);

	await srv.stop();
}

rmSync(CACHE_ROOT, { recursive: true, force: true });

console.log(`\n${failures ? `${failures} failing` : "all checks passed"}`);
process.exit(failures ? 1 : 0);
