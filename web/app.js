// In-browser English -> Amharic translation with the exported ONNX models.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { normalizeEn, detokenizeAm, SentencePiece } from "./tokenizer.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1;

const BOS = 2, EOS = 3;
const $ = (id) => document.getElementById(id);
let vocab, spEn, spAm;
const sessions = {};

async function loadSessions(kind) {
  if (!sessions[kind]) {
    sessions[kind] = (async () => {
      const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      const [enc, dec] = await Promise.all([
        ort.InferenceSession.create(`models/${kind}_encoder.onnx`, opts),
        ort.InferenceSession.create(`models/${kind}_decoder.onnx`, opts),
      ]);
      return { enc, dec };
    })();
  }
  return sessions[kind];
}

async function translate(kind, text) {
  const t0 = performance.now();
  const norm = normalizeEn(text);
  let ids = spEn.encode(norm).slice(0, vocab.max_tokens);
  if (!ids.length) ids = [EOS];
  const { enc, dec } = await loadSessions(kind);
  const S = ids.length;
  const src = new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, S]);
  const e = await enc.run({ src });
  let h = e.h, c = e.c;
  let tok = BOS;
  const out = [], attn = [];
  const maxLen = 2 * S + 10;
  for (let step = 0; step < maxLen; step++) {
    const feeds = { tok: new ort.Tensor("int64", BigInt64Array.from([BigInt(tok)]), [1]),
                    h_in: h, c_in: c, enc_out: e.enc_out, keys: e.keys };
    for (const k of Object.keys(feeds)) if (!dec.inputNames.includes(k)) delete feeds[k];
    const r = await dec.run(feeds);
    const logits = r.logits.data;
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    attn.push(Array.from(r.attn.data));
    h = r.h_out; c = r.c_out; tok = best;
    if (best === EOS) break;
    out.push(best);
  }
  return {
    translation: detokenizeAm(spAm.decode(out)),
    normalized: norm,
    srcTokens: ids.map((i) => spEn.idToPiece(i)),
    tgtTokens: [...out.map((i) => spAm.idToPiece(i)), "</s>"],
    attention: attn,
    ms: Math.round(performance.now() - t0),
  };
}

function renderHeatmap(r) {
  const el = $("heatmap");
  const cell = 26, left = 110, top = 96;
  const W = left + r.srcTokens.length * cell + 10, H = top + r.tgtTokens.length * cell + 10;
  const dpr = window.devicePixelRatio || 1;
  el.width = W * dpr; el.height = H * dpr;
  el.style.width = W + "px"; el.style.height = H + "px";
  const g = el.getContext("2d");
  g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  g.fillStyle = css.getPropertyValue("--surface");
  g.fillRect(0, 0, W, H);
  const accent = css.getPropertyValue("--heat").trim().split(",").map(Number);
  r.attention.forEach((row, i) => row.forEach((v, j) => {
    g.fillStyle = `rgba(${accent[0]},${accent[1]},${accent[2]},${0.06 + 0.94 * v})`;
    g.fillRect(left + j * cell + 1, top + i * cell + 1, cell - 2, cell - 2);
  }));
  g.fillStyle = css.getPropertyValue("--text");
  g.font = "13px 'Noto Sans Ethiopic', system-ui, sans-serif";
  g.textAlign = "right"; g.textBaseline = "middle";
  r.tgtTokens.forEach((t, i) => g.fillText(t.replace(/▁/g, "_"), left - 6, top + i * cell + cell / 2));
  r.srcTokens.forEach((t, j) => {
    g.save();
    g.translate(left + j * cell + cell / 2, top - 6);
    g.rotate(-Math.PI / 3);
    g.textAlign = "left";
    g.fillText(t.replace(/▁/g, "_"), 0, 0);
    g.restore();
  });
}

async function run() {
  const text = $("input").value.trim();
  if (!text) return;
  $("go").disabled = true;
  $("status").textContent = "Translating…";
  try {
    const att = await translate("attention", text);
    $("out-attention").textContent = att.translation || "—";
    $("ms-attention").textContent = `${att.ms} ms`;
    $("normalized").textContent = att.normalized;
    $("src-tokens").textContent = att.srcTokens.join("  ");
    renderHeatmap(att);
    $("attn-wrap").hidden = false;
    const base = await translate("seq2seq", text);
    $("out-seq2seq").textContent = base.translation || "—";
    $("ms-seq2seq").textContent = `${base.ms} ms`;
    $("status").textContent = "Ran entirely in your browser (ONNX Runtime Web, CPU).";
  } catch (err) {
    console.error(err);
    $("status").textContent = "Error: " + err.message;
  } finally {
    $("go").disabled = false;
  }
}

async function init() {
  $("status").textContent = "Loading models (~25 MB, first visit only)…";
  vocab = await (await fetch("models/vocab.json")).json();
  spEn = new SentencePiece(vocab.en);
  spAm = new SentencePiece(vocab.am);
  await Promise.all([loadSessions("attention"), loadSessions("seq2seq")]);
  $("status").textContent = "Models ready.";
  $("go").disabled = false;
  $("go").addEventListener("click", run);
  $("input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(); } });
  document.querySelectorAll(".example").forEach((b) =>
    b.addEventListener("click", () => { $("input").value = b.textContent; run(); }));
  try {
    const m = await (await fetch("metrics.json")).json();
    const rows = [["BLEU", "bleu"], ["chrF", "chrf"], ["Test loss", "test_loss"], ["Parameters", "parameters"],
                  ["Training time (min)", "training_time_min"], ["Inference (ms/sentence)", "inference_ms_per_sentence_single"]];
    $("metrics").innerHTML = rows.map(([l, k]) =>
      `<tr><th>${l}</th><td>${m.metrics.seq2seq[k].toLocaleString()}</td><td>${m.metrics.attention[k].toLocaleString()}</td></tr>`).join("");
    $("metrics-wrap").hidden = false;
  } catch { /* metrics are optional */ }
  if ($("input").value.trim()) run();
}

init().catch((e) => { $("status").textContent = "Failed to load models: " + e.message; });
