"""Checks that web/tokenizer.js reproduces the Python preprocessing exactly.

    python tests/test_web_parity.py      (requires node >= 18)
"""
import json
import os
import random
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import Tokenizers  # noqa: E402
from text import detokenize_am, normalize_en  # noqa: E402

NODE = r"""
import { readFileSync } from "fs";
import { normalizeEn, detokenizeAm, SentencePiece } from "%s";
const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const vocab = (lang) => readFileSync("%s/models/spm_" + lang + ".vocab", "utf8").trim().split("\n")
  .map((l) => { const [p, s] = l.split("\t"); return [p, parseFloat(s)]; });
const en = new SentencePiece(vocab("en")), am = new SentencePiece(vocab("am"));
const out = {
  norm: cases.en.map(normalizeEn),
  ids: cases.en.map((s) => en.encode(normalizeEn(s))),
  dec: cases.am_ids.map((ids) => detokenizeAm(am.decode(ids))),
};
console.log(JSON.stringify(out));
"""


def main():
    tok = Tokenizers()
    raw = pd.read_parquet(os.path.join(ROOT, "data/raw/mt560.parquet")).eng.tolist()
    rng = random.Random(0)
    en = rng.sample(raw, 3000) + [
        "I am going to the university.", "Don't worry — it's “fine”… 100%!",
        "Café naïve résumé ½ ①", "email me at a_b@c.com #hashtag", "  spaces\tand\nnewlines  ",
        "Jehovah's Witnesses", "What?! No way...", "漢字 and emoji 😀 test", "",
    ]
    _, am_ref = zip(*[(0, x) for x in pd.read_csv(os.path.join(ROOT, "data/processed/test.tsv"),
                                                  sep="\t", keep_default_na=False).am[:1000]])
    am_ids = tok.am.encode(list(am_ref)) + [[rng.randrange(1, 8000) for _ in range(12)] for _ in range(200)]
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "cases.json")
        json.dump({"en": en, "am_ids": am_ids}, open(cp, "w"), ensure_ascii=False)
        script = os.path.join(d, "run.mjs")
        open(script, "w").write(NODE % ("file://" + os.path.join(ROOT, "web", "tokenizer.js"), ROOT))
        js = json.loads(subprocess.check_output(["node", script, cp]))
    bad_norm = [(s, js["norm"][i]) for i, s in enumerate(en) if normalize_en(s) != js["norm"][i]]
    bad_ids = [s for i, s in enumerate(en) if tok.en.encode(normalize_en(s)) != js["ids"][i]]
    bad_dec = [i for i, ids in enumerate(am_ids) if detokenize_am(tok.am.decode(ids)) != js["dec"][i]]
    print(f"normalization mismatches: {len(bad_norm)}/{len(en)}")
    print(f"sentencepiece id mismatches: {len(bad_ids)}/{len(en)}")
    print(f"amharic decode mismatches: {len(bad_dec)}/{len(am_ids)}")
    for s, j in bad_norm[:5]:
        print("  NORM", repr(normalize_en(s)), "!=", repr(j))
    for s in bad_ids[:5]:
        print("  IDS ", repr(normalize_en(s)), tok.en.encode(normalize_en(s), out_type=str))
    for i in bad_dec[:5]:
        print("  DEC ", repr(detokenize_am(tok.am.decode(am_ids[i]))), "!=", repr(js["dec"][i]))
    assert not bad_norm and not bad_ids and not bad_dec


if __name__ == "__main__":
    main()
