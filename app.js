// In-browser English -> Amharic translation with the exported ONNX models.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { normalizeEn, detokenizeAm, SentencePiece } from "./tokenizer.js";
import { beamSearch } from "./beam.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1;

const BOS = 2, EOS = 3;
const $ = (id) => document.getElementById(id);
let vocab, spEn, spAm;
const sessions = {};
const progress = {}; // url -> [loaded, total]

function showProgress(label) {
  let done = 0, total = 0;
  for (const [l, t, lab] of Object.values(progress)) if (lab === label) { done += l; total += t; }
  if (!total) return;
  const pct = Math.floor((100 * done) / total);
  if (label.includes("baseline")) {  // background download: shown in the baseline's box
    $("ms-seq2seq").textContent = pct < 100 ? `baseline model loading… ${pct}%` : "";
  } else {
    $("status").textContent = `${label} ${pct}% (${(done / 2 ** 20).toFixed(1)} of ${(total / 2 ** 20).toFixed(1)} MB, first visit only)…`;
  }
}

// Download a model file with progress reporting; retried once on network errors.
// The server may gzip the response, so Content-Length (compressed size) is not the
// number of bytes we receive: collect chunks and use the true size from vocab.json.
async function fetchBytes(url, label, attempt = 1) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const name = url.split("/").pop();
    const total = (vocab.files && vocab.files[name]) || Number(res.headers.get("content-length")) || 0;
    if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      progress[url] = [Math.min(got, total), total, label];
      showProgress(label);
    }
    const buf = new Uint8Array(got);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    progress[url] = [total, total, label];
    return buf;
  } catch (e) {
    delete progress[url];
    if (attempt < 2) return fetchBytes(url, label, attempt + 1);
    throw e;
  }
}

async function loadSessions(kind, label = "Downloading models") {
  if (!sessions[kind]) {
    sessions[kind] = (async () => {
      const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
      const [encBytes, decBytes] = await Promise.all([
        fetchBytes(`models/${kind}_encoder.onnx`, label),
        fetchBytes(`models/${kind}_decoder.onnx`, label),
      ]);
      if (kind === "attention") $("status").textContent = "Starting the translation engine…";
      const [enc, dec] = await Promise.all([
        ort.InferenceSession.create(encBytes, opts),
        ort.InferenceSession.create(decBytes, opts),
      ]);
      return { enc, dec };
    })();
    sessions[kind].catch(() => { delete sessions[kind]; }); // allow a retry after a failure
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
  const step = async (tok, [h, c]) => {
    const feeds = { tok: new ort.Tensor("int64", BigInt64Array.from([BigInt(tok)]), [1]),
                    h_in: h, c_in: c, enc_out: e.enc_out, keys: e.keys };
    for (const k of Object.keys(feeds)) if (!dec.inputNames.includes(k)) delete feeds[k];
    const r = await dec.run(feeds);
    return { logits: r.logits.data, state: [r.h_out, r.c_out], attn: Array.from(r.attn.data) };
  };
  const best = await beamSearch(step, [e.h, e.c], S, vocab.beam);
  const out = best.seq, attn = best.atts;
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
    $("status").textContent = "Ran entirely in your browser (beam search, ONNX Runtime Web).";
    $("out-seq2seq").textContent = "…";
    try {
      const base = await translate("seq2seq", text);
      $("out-seq2seq").textContent = base.translation || "—";
      $("ms-seq2seq").textContent = `${base.ms} ms`;
    } catch (err) {
      console.error(err);
      $("out-seq2seq").textContent = "—";
      $("ms-seq2seq").textContent = "baseline model could not be loaded: " + err.message;
    }
  } catch (err) {
    console.error(err);
    $("status").textContent = "Error: " + err.message;
  } finally {
    $("go").disabled = false;
  }
}

async function init() {
  if (typeof WebAssembly !== "object") {
    throw new Error("this browser does not support WebAssembly; please use a recent Chrome, Edge, Firefox or Safari");
  }
  $("status").textContent = "Downloading vocabulary…";
  const vr = await fetch("models/vocab.json");
  if (!vr.ok) throw new Error(`vocabulary: HTTP ${vr.status}`);
  vocab = await vr.json();
  spEn = new SentencePiece(vocab.en);
  spAm = new SentencePiece(vocab.am);
  // The best model first, so the page becomes usable as early as possible
  await loadSessions("attention", "Downloading the translation model");
  $("status").textContent = "Ready. Type an English sentence and press Translate.";
  $("go").disabled = false;
  loadSessions("seq2seq", "Downloading the baseline model").catch((e) => console.error(e)); // background
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

init().catch((e) => {
  console.error(e);
  $("status").textContent = "Could not load the translator: " + e.message + ". Check your internet connection and reload the page.";
});
