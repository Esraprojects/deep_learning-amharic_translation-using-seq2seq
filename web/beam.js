// Beam search, a line-by-line port of beam_search() in src/models.py.
// `step(tok, state)` runs one decoder step for ONE hypothesis and resolves to
// { logits: Float32Array, state, attn: number[] }.
const PAD = 0, BOS = 2, EOS = 3, NEG = -1e9;

function logSoftmax(x) {
  let m = -Infinity;
  for (const v of x) if (v > m) m = v;
  let s = 0;
  for (const v of x) s += Math.exp(v - m);
  const lse = m + Math.log(s);
  return Float64Array.from(x, (v) => v - lse);
}

function blocked(seq, n) {
  const out = new Set();
  if (n <= 0 || seq.length < n - 1) return out;
  const prefix = seq.slice(seq.length - n + 1);
  for (let i = 0; i + n - 1 < seq.length; i++) {
    let ok = true;
    for (let j = 0; j < n - 1; j++) if (seq[i + j] !== prefix[j]) { ok = false; break; }
    if (ok) out.add(seq[i + n - 1]);
  }
  return out;
}

const lenPen = (len, alpha) => Math.pow((5 + len) / 6, alpha);

export async function beamSearch(step, initState, srcLen, { beam = 5, alpha = 0.7, no_repeat_ngram = 3 } = {}) {
  const maxLen = 2 * srcLen + 10;
  let hyps = [{ seq: [], score: 0, atts: [], state: initState }];
  const finished = [];
  for (let t = 0; t < maxLen; t++) {
    const cands = [];
    const outs = [];
    for (let i = 0; i < hyps.length; i++) {
      const h = hyps[i];
      const r = await step(h.seq.length ? h.seq[h.seq.length - 1] : BOS, h.state);
      const lp = logSoftmax(r.logits);
      lp[PAD] = NEG; lp[BOS] = NEG;
      for (const b of blocked(h.seq, no_repeat_ngram)) lp[b] = NEG;
      outs.push(r);
      // best 2*beam continuations of this hypothesis are enough for the global top 2*beam
      const idx = Array.from(lp.keys()).sort((a, b) => lp[b] - lp[a] || a - b).slice(0, 2 * beam);
      for (const w of idx) cands.push({ i, w, sc: h.score + lp[w], flat: i * lp.length + w });
    }
    cands.sort((a, b) => b.sc - a.sc || a.flat - b.flat);
    const next = [];
    for (const { i, w, sc } of cands.slice(0, 2 * beam)) {
      const h = hyps[i], r = outs[i];
      if (w === EOS) {
        finished.push({ score: sc / lenPen(h.seq.length + 1, alpha), seq: h.seq, atts: [...h.atts, r.attn] });
      } else {
        next.push({ seq: [...h.seq, w], score: sc, atts: [...h.atts, r.attn], state: r.state });
      }
      if (next.length === beam) break;
    }
    if (finished.length) {
      const bestDone = Math.max(...finished.map((f) => f.score));
      if (!next.length || bestDone > Math.max(...next.map((n) => n.score)) / lenPen(maxLen, alpha)) break;
    }
    if (!next.length) break;
    hyps = next;
  }
  const pool = finished.length ? finished
    : hyps.map((h) => ({ score: h.score / lenPen(h.seq.length, alpha), seq: h.seq, atts: h.atts }));
  return pool.reduce((a, b) => (b.score > a.score ? b : a));
}
