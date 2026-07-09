// Test harness for pi-llamacpp-provider (run with node >= 22.18 for native .ts import).
//
// Exercises the extension against real HTTP servers rather than a stubbed fetch:
// a llama.cpp *router* payload captured from a live server, a plain
// `llama-server` payload, and every failure mode that must degrade to the
// fallback model instead of taking pi's startup down.
import http from "node:http";
import { readFileSync } from "node:fs";

const ext = await import(new URL("../src/index.ts", import.meta.url));
const { default: extension, normalizeBaseUrl, argValue, toModel, readConfig } = ext;

const ROUTER = readFileSync(new URL("./fixtures/models.json", import.meta.url), "utf8");
const PLAIN = readFileSync(new URL("./fixtures/plain-server.json", import.meta.url), "utf8");

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

/** A JSON server that records what it was asked for. */
async function serveJson(body, status = 200) {
	let ctx;
	ctx = await serve((req, res) => {
		ctx.requests.push({ url: req.url, auth: req.headers.authorization });
		res.writeHead(status, { "content-type": "application/json" });
		res.end(body);
	});
	return ctx;
}

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
	const keys = ["LLAMACPP_BASE_URL", "LLAMACPP_API_KEY", "LLAMACPP_TIMEOUT_MS", "LLAMACPP_PROVIDER"];
	const saved = Object.fromEntries(keys.map((k) => [k, process.env[k]]));
	for (const k of keys) delete process.env[k];
	Object.assign(process.env, env);
	const pi = fakePi();
	try {
		await extension(pi);
	} finally {
		for (const k of keys) {
			if (saved[k] === undefined) delete process.env[k];
			else process.env[k] = saved[k];
		}
	}
	return pi;
}

const byId = (models, id) => models.find((m) => m.id === id);

// ── URL normalization ─────────────────────────────────────────────────────────
check("normalizeBaseUrl appends /v1", normalizeBaseUrl("http://h:8080") === "http://h:8080/v1");
check("normalizeBaseUrl strips trailing slashes", normalizeBaseUrl("http://h:8080///") === "http://h:8080/v1");
check("normalizeBaseUrl keeps an existing /v1", normalizeBaseUrl("http://h:8080/v1") === "http://h:8080/v1");
check("normalizeBaseUrl trims whitespace", normalizeBaseUrl("  http://h:8080/v1/  ") === "http://h:8080/v1");

// ── argv parsing ──────────────────────────────────────────────────────────────
check("argValue reads the value after a flag", argValue(["--ctx-size", "4096"], "--ctx-size") === "4096");
check("argValue returns undefined when absent", argValue(["--parallel", "2"], "--ctx-size") === undefined);
check("argValue returns undefined for a trailing flag", argValue(["--slots"], "--slots") === undefined);

// ── defaults ──────────────────────────────────────────────────────────────────
const defaults = readConfig({});
check(
	"readConfig defaults to loopback, not a LAN address",
	defaults.baseUrl === "http://127.0.0.1:8080/v1" && defaults.provider === "llamacpp" && defaults.timeoutMs === 4000,
	JSON.stringify(defaults),
);

// ── context-window math ───────────────────────────────────────────────────────
check("ctx defaults to 4096 with no args and no meta", toModel({ id: "x" }).contextWindow === 4096);
check(
	"ctx is divided across --parallel slots",
	toModel({ id: "x", status: { args: ["--ctx-size", "160000", "--parallel", "2"] } }).contextWindow === 80000,
);
check("ctx is clamped up to MIN_CTX", toModel({ id: "x", meta: { n_ctx: 512 } }).contextWindow === 2048);
check(
	"ctx clamps after slot division too",
	toModel({ id: "x", status: { args: ["--ctx-size", "4096", "--parallel", "8"] } }).contextWindow === 2048,
);
check(
	"--ctx-size beats meta.n_ctx",
	toModel({ id: "x", status: { args: ["--ctx-size", "8192"] }, meta: { n_ctx: 262144 } }).contextWindow === 8192,
);
check("maxTokens is capped at 8192", toModel({ id: "x", meta: { n_ctx: 262144 } }).maxTokens === 8192);
check("maxTokens never exceeds the context window", toModel({ id: "x" }).maxTokens === 4096);

// ── router payload (captured from a live llama.cpp router) ────────────────────
{
	const srv = await serveJson(ROUTER);
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
		srv.requests.length === 1 && srv.requests[0].url === "/v1/models" && srv.requests[0].auth === "Bearer sk-test",
		JSON.stringify(srv.requests),
	);
	check("every server model is registered", models.length === 9, models.map((m) => m.id).join(","));
	check("no session_start warning on success", (pi.handlers.session_start ?? []).length === 0);

	check("fastcontext: 160000 ctx over 2 slots → 80000", byId(models, "fastcontext")?.contextWindow === 80000);
	check("gemma4-31b: ctx 100000, maxTokens capped", byId(models, "gemma4-31b")?.contextWindow === 100000 && byId(models, "gemma4-31b")?.maxTokens === 8192);
	check("default preset: no ctx-size → 4096", byId(models, "default")?.contextWindow === 4096);

	const gemma12b = byId(models, "gemma4-12b");
	check(
		"gemma4-12b: audio modality dropped, image kept",
		JSON.stringify(gemma12b?.input) === '["text","image"]',
		JSON.stringify(gemma12b?.input),
	);
	check("laguna-xs2: text-only model has no image input", JSON.stringify(byId(models, "laguna-xs2")?.input) === '["text"]');

	const ornith = byId(models, "ornith-35b");
	check(
		"ornith-35b: thinking detected from the preset, not the id",
		ornith?.reasoning === true && ornith?.compat?.thinkingFormat === "qwen-chat-template",
		JSON.stringify(ornith?.compat),
	);
	const qwen = byId(models, "qwen36a3b-35b");
	check("qwen36a3b-35b: thinking detected from the id", qwen?.reasoning === true && qwen?.compat?.thinkingFormat === "qwen-chat-template");
	const laguna = byId(models, "laguna-xs2");
	check(
		"laguna-xs2: non-thinking model gets no thinkingFormat",
		laguna?.reasoning === false && laguna?.compat?.thinkingFormat === undefined,
		JSON.stringify(laguna?.compat),
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

// ── plain llama-server payload ────────────────────────────────────────────────
{
	const srv = await serveJson(PLAIN);
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
		"plain server: no architecture block → text-only, qwen id still reasons",
		JSON.stringify(byId(models, "qwen3-8b-instruct")?.input) === '["text"]' && byId(models, "qwen3-8b-instruct")?.reasoning === true,
	);
	check("plain server: mistral does not reason", byId(models, "mistral-7b-instruct")?.reasoning === false);

	await srv.stop();
}

// ── partially malformed payload ───────────────────────────────────────────────
{
	const srv = await serveJson(JSON.stringify({ data: [{ id: "good" }, { name: "no-id" }, null, { id: "" }] }));
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
	const srv = await serveJson('{"error":"boom"}', 500);
	await expectFallback("HTTP 500", { env: { LLAMACPP_BASE_URL: srv.url }, srv });
}
{
	const srv = await serveJson("this is not json");
	await expectFallback("malformed JSON", { env: { LLAMACPP_BASE_URL: srv.url }, srv });
}
{
	const srv = await serveJson(JSON.stringify({ data: [] }));
	await expectFallback("empty model list", { env: { LLAMACPP_BASE_URL: srv.url }, srv });
}
{
	const srv = await serveJson(JSON.stringify({ data: [{ name: "no-id" }] }));
	await expectFallback("no usable entries", { env: { LLAMACPP_BASE_URL: srv.url }, srv });
}
{
	// Bind a port, then release it: nothing is listening when the extension connects.
	const dead = await serveJson("{}");
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
	const srv = await serveJson("{}", 503);
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

console.log(`\n${failures ? `${failures} failing` : "all checks passed"}`);
process.exit(failures ? 1 : 0);
