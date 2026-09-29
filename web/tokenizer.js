// JavaScript port of src/text.py + SentencePiece (unigram) encode/decode.
// Used by the in-browser demo; verified against the Python pipeline by
// tests/test_web_parity.py.

const EN_QUOTES = {
  "‘": "'", "’": "'", "‚": "'", "′": "'", "`": "'",
  "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
  "–": "-", "—": "-", "‒": "-", "−": "-",
  "…": "...",
};

function replaceQuotes(text) {
  let out = "";
  for (const ch of text) out += EN_QUOTES[ch] ?? ch;
  return out;
}

export function normalizeEn(text) {
  text = replaceQuotes(text.normalize("NFKC"));
  text = text.replace(/_/g, " ").toLowerCase();
  text = text.replace(/([^\p{L}\p{N}_\s])/gu, " $1 ");
  return text.replace(/\s+/g, " ").trim();
}

const AM_NO_SPACE_BEFORE = /\s+([።፣፤፥፦፧!?.,;:)\]])/g;
const AM_NO_SPACE_AFTER = /([(\[])\s+/g;

export function detokenizeAm(text) {
  return text.replace(AM_NO_SPACE_BEFORE, "$1").replace(AM_NO_SPACE_AFTER, "$1").trim();
}

const SPACE = "▁";
const UNK_ID = 1;
const UNK_PENALTY = 10.0;

export class SentencePiece {
  // pieces: [[piece, score], ...] in id order (from spm_xx.vocab)
  constructor(pieces) {
    this.pieces = pieces.map((p) => p[0]);
    this.trie = new Map(); // piece -> [id, score]
    let minScore = Infinity;
    this.maxLen = 0;
    pieces.forEach(([p, s], id) => {
      if (id < 4) return; // <pad> <unk> <s> </s> are control symbols
      this.trie.set(p, [id, s]);
      minScore = Math.min(minScore, s);
      this.maxLen = Math.max(this.maxLen, Array.from(p).length);
    });
    this.unkScore = minScore - UNK_PENALTY;
  }

  encode(text) {
    if (!text) return [];
    const chars = Array.from(SPACE + text.replace(/ /g, SPACE));
    const n = chars.length;
    const best = new Array(n + 1).fill(-Infinity);
    const back = new Array(n + 1);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let sub = "";
      let single = false;
      for (let l = 1; l <= this.maxLen && i + l <= n; l++) {
        sub += chars[i + l - 1];
        const hit = this.trie.get(sub);
        if (!hit) continue;
        if (l === 1) single = true;
        const sc = best[i] + hit[1];
        if (sc > best[i + l]) { best[i + l] = sc; back[i + l] = [i, hit[0]]; }
      }
      if (!single) {
        const sc = best[i] + this.unkScore;
        if (sc > best[i + 1]) { best[i + 1] = sc; back[i + 1] = [i, UNK_ID]; }
      }
    }
    const ids = [];
    for (let j = n; j > 0; j = back[j][0]) ids.push(back[j][1]);
    ids.reverse();
    // SentencePiece merges consecutive unknown pieces into one
    return ids.filter((id, k) => !(id === UNK_ID && ids[k - 1] === UNK_ID));
  }

  idToPiece(id) {
    return this.pieces[id];
  }

  decode(ids) {
    let s = "";
    for (const id of ids) {
      if (id === UNK_ID) s += " ⁇ ";
      else if (id > 3) s += this.pieces[id];
    }
    return s.replace(/▁/g, " ").replace(/\s+/g, " ").trim();
  }
}
