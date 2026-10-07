#!/usr/bin/env node
/** sparkDash PrefillBench protocol re-implementation: prompt_tokens / TTFT, " the" filler. */
import { randomUUID } from "crypto";
const BASE = process.env.BASE || "http://192.168.2.23:8000";
const MODEL = process.env.MODEL || "deepseek-v4.1-flash";
const SIZES = (process.env.SIZES || "4096,16384,32768,65536,131072").split(",").map(Number);
const PASSES = Number(process.env.PASSES || 2);
const FILLER_UNIT = " the";
const estimateTokenCount = (t) => Math.max(1, Math.round(t.length / 4));

function buildPrompt(targetTokens, salt) {
  const n = Math.max(8, Math.round(targetTokens));
  const header = `[prefill-bench ${salt}]\nIgnore the filler below. Reply with the single word OK.\n`;
  const footer = "\nReply OK.";
  const reserved = estimateTokenCount(header + footer);
  return header + FILLER_UNIT.repeat(Math.max(1, n - reserved)) + footer;
}

async function runSize(target) {
  const salt = randomUUID();
  const prompt = buildPrompt(target, salt);
  const t0 = performance.now();
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 8, temperature: 0, top_p: 1,
      stream: true, stream_options: { include_usage: true },
      chat_template_kwargs: { enable_thinking: false, thinking: false, thinking_mode: "disabled" },
    }),
    signal: AbortSignal.timeout(2_700_000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", tFirst = null, pt = 0, ct = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
      if (!line.startsWith("data:")) continue;
      const d = line.slice(5).trim();
      if (d === "[DONE]") continue;
      let j; try { j = JSON.parse(d); } catch { continue; }
      if (j.usage) { pt = Number(j.usage.prompt_tokens) || pt; ct = Number(j.usage.completion_tokens) || ct; }
      if (j.choices?.[0]?.delta?.content && tFirst == null) tFirst = performance.now();
    }
  }
  const ttftMs = (tFirst ?? performance.now()) - t0;
  const tps = (pt / ttftMs) * 1000;
  return { target, promptTokens: pt, completionTokens: ct, ttftMs: Math.round(ttftMs), prefillTps: Math.round(tps * 10) / 10 };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
console.log(`# sparkDash-protocol prefill bench base=${BASE} model=${MODEL} sizes=${SIZES} passes=${PASSES}`);
await runSize(512).then((w) => console.log(`warmup(512): ${w.promptTokens} tok, ttft ${w.ttftMs} ms, ${w.prefillTps} tok/s`)).catch((e) => console.log(`warmup FAILED: ${e.message}`));
const out = [];
for (const s of SIZES) {
  for (let p = 1; p <= PASSES; p++) {
    try {
      const r = await runSize(s);
      out.push(r);
      console.log(`[${s} #${p}] prompt_tokens=${r.promptTokens} ttft=${r.ttftMs} ms  prefill=${r.prefillTps} tok/s  (completion=${r.completionTokens})`);
    } catch (e) { console.log(`[${s} #${p}] FAILED: ${e.message}`); }
    await sleep(2000);
  }
}
const { writeFileSync } = await import("fs");
writeFileSync("/tmp/sparkprefill-results.json", JSON.stringify({ base: BASE, model: MODEL, results: out }, null, 2));
console.log("saved /tmp/sparkprefill-results.json");
