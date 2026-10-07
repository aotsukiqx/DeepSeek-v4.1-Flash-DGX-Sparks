#!/usr/bin/env node
/**
 * sparkDash DecodeBench protocol re-implementation (v1.8.9 == 1.8.8 prompts/math).
 * Protocol: chat completions, stream, temperature 0, top_p 1, thinking off,
 * min_tokens = max_tokens = 256, ignore_eos, stop: [], warmup 32 tokens per type,
 * per-stream tps = (completion_tokens-1)/(tLast-tFirst)*1000,
 * aggregate = sum(completion_tokens-1)/(max(tLast)-min(tFirst))*1000.
 */
const BASE = process.env.BASE || "http://192.168.2.23:8000";
const MODEL = process.env.MODEL || "deepseek-v4.1-flash";
const MAXT = Number(process.env.MAXT || 256);
const TYPES = (process.env.TYPES || "prose,code").split(",");
const LEVELS = (process.env.LEVELS || "1,16").split(",").map(Number);
const REPS = Number(process.env.REPS || 3);

const DECODE_PROSE_PROMPT =
  "Write a detailed step-by-step explanation of how a hash map works, " +
  "including collision handling, resizing, and time complexity. Be thorough.";
const CODE_TASK_TAIL =
  "Output only Python source. No comments, no docstrings, no markdown fences. " +
  "Then add tests and the helpers this needs. Keep writing code.";
const codeTask = (name, spec) => `${name}\n${spec}\n${CODE_TASK_TAIL}`;
const CODE_TASKS = [
  ["binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."],
  ["merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."],
  ["lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."],
  ["token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."],
  ["ring_buffer", "class RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow."],
  ["dijkstra", "def dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph."],
  ["edit_distance", "def edit_distance(a, b) -> int: Levenshtein distance between two strings."],
  ["semver_cmp", "def semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b."],
  ["url_parse", "def url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries."],
  ["json_pointer", "def json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing."],
  ["glob_match", "def glob_match(pattern, text) -> bool: * and ? wildcards, no character classes."],
  ["csv_parse", "def csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes."],
  ["rle", "def rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs."],
  ["top_k", "def top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap."],
  ["interval_merge", "def merge_intervals(spans) -> list: merge overlapping [start, end] pairs."],
  ["topo_sort", "def topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle."],
].map(([n, s]) => codeTask(n, s));
const CODE_WARMUP = codeTask("warmup_noop", "def warmup_noop(x): return x unchanged.");

function promptsFor(count, type) {
  if (type === "code") return CODE_TASKS.slice(0, count);
  if (count <= 1) return [DECODE_PROSE_PROMPT];
  return Array.from({ length: count }, (_, i) => `${DECODE_PROSE_PROMPT} (stream ${i + 1}/${count})`);
}

function body(prompt, maxTokens, { fill = false } = {}) {
  const b = {
    model: MODEL,
    messages: [{ role: "user", content: prompt }],
    max_tokens: maxTokens,
    temperature: 0,
    top_p: 1,
    stream: true,
    stream_options: { include_usage: true },
    chat_template_kwargs: { enable_thinking: false, thinking: false, thinking_mode: "disabled" },
  };
  if (fill) { b.min_tokens = maxTokens; b.ignore_eos = true; b.stop = []; }
  return b;
}

async function oneStream(prompt, maxTokens, fill) {
  const t0 = Date.now();
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body(prompt, maxTokens, { fill })),
    signal: AbortSignal.timeout(360_000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", tFirst = null, tLast = null, ct = 0, finish = null, textLen = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") continue;
      let j; try { j = JSON.parse(data); } catch { continue; }
      if (j.usage?.completion_tokens != null) ct = Number(j.usage.completion_tokens);
      const ch = j.choices?.[0];
      if (ch?.delta?.content) {
        textLen += ch.delta.content.length;
        const now = Date.now();
        if (tFirst == null) tFirst = now;
        tLast = now;
      }
      if (ch?.finish_reason) finish = ch.finish_reason;
    }
  }
  if (tFirst == null) tFirst = t0;
  if (tLast == null) tLast = t0;
  return { t0, tFirst, tLast, completionTokens: ct, textLen, finish };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[(s.length >> 1)]; };
const r2 = (x) => Math.round(x * 100) / 100;

async function wave(type, c) {
  const prompts = promptsFor(c, type);
  const rs = await Promise.all(prompts.map((p) => oneStream(p, MAXT, true)));
  const decodeTokens = rs.reduce((s, r) => s + Math.max(0, r.completionTokens - 1), 0);
  const firsts = rs.map((r) => r.tFirst), lasts = rs.map((r) => r.tLast);
  const winMs = Math.max(...lasts) - Math.min(...firsts);
  const agg = winMs > 0 ? (decodeTokens / winMs) * 1000 : 0;
  const per = rs.map((r) => (r.tLast > r.tFirst ? ((r.completionTokens - 1) / (r.tLast - r.tFirst)) * 1000 : 0));
  const ttft = rs.map((r) => r.tFirst - r.t0);
  return {
    type, concurrency: c, aggregateTps: r2(agg),
    perStreamTps: per.map(r2), perStreamMedian: r2(median(per)),
    completionTokens: rs.map((r) => r.completionTokens), finishes: rs.map((r) => r.finish),
    ttftMsMin: Math.min(...ttft), ttftMsMax: Math.max(...ttft), windowMs: winMs,
  };
}

const results = [];
const t = new Date().toISOString();
console.log(`# sparkDash-protocol decode bench  base=${BASE} model=${MODEL} max_tokens=${MAXT} reps=${REPS} start=${t}`);

for (const type of TYPES) {
  const warmPrompt = type === "code" ? CODE_WARMUP : DECODE_PROSE_PROMPT;
  try { await oneStream(warmPrompt, 32, false); console.log(`[${type}] warmup ok`); } catch (e) { console.log(`[${type}] warmup FAILED: ${e.message}`); }
  for (const c of LEVELS) {
    for (let rep = 1; rep <= REPS; rep++) {
      try {
        const w = await wave(type, c);
        results.push(w);
        console.log(`[${type} c${c} #${rep}] aggregate=${w.aggregateTps} tok/s  per-stream median=${w.perStreamMedian} (min ${Math.min(...w.perStreamTps)} / max ${Math.max(...w.perStreamTps)})  ttft ${w.ttftMsMin}-${w.ttftMsMax} ms  ct=${w.completionTokens.join(",")} finish=${[...new Set(w.finishes)].join("/")}`);
      } catch (e) {
        console.log(`[${type} c${c} #${rep}] FAILED: ${e.message}`);
      }
      await sleep(3000);
    }
  }
}

const summary = {};
for (const w of results) {
  const k = `${w.type} c${w.concurrency}`;
  (summary[k] ??= []).push(w.aggregateTps);
}
console.log("\n== summary (aggregate tok/s per rep | median) ==");
for (const [k, v] of Object.entries(summary)) console.log(`${k}: ${v.join(" | ")}  median=${r2(median(v))}`);
const { writeFileSync } = await import("fs");
writeFileSync("/tmp/sparkbench-results.json", JSON.stringify({ base: BASE, model: MODEL, maxTokens: MAXT, started: t, results }, null, 2));
console.log("saved /tmp/sparkbench-results.json");
