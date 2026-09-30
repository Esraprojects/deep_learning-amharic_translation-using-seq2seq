"""Dataset download, cleaning, normalization, de-duplication, splitting and
SentencePiece tokenizer training.

Dataset: OPUS MT560 English-Amharic sentence pairs, as packaged on the
Hugging Face Hub (michsethowusu/english-amharic_sentence-pairs_mt560),
license CC-BY-4.0.

Usage:
    python src/preprocess.py            # writes data/processed/* and models/spm_*.model
"""
import argparse
import json
import os
import re
import sys

import pandas as pd
import sentencepiece as spm

sys.path.insert(0, os.path.dirname(__file__))
from text import ETHIOPIC_RE, LATIN_RE, normalize_am, normalize_en  # noqa: E402

HF_ID = "michsethowusu/english-amharic_sentence-pairs_mt560"
HF_URL = ("https://huggingface.co/datasets/" + HF_ID +
          "/resolve/main/data/train-00000-of-00001.parquet")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Footnote markers such as "* ፍ1 *" and trailing footnote text "ፍ1 ..." (Bible).
_FOOTNOTE_MARK = re.compile(r"\*\s*ፍ\d+\s*\*")
_FOOTNOTE_TAIL = re.compile(r"\s+ፍ\d+\s.*$")


def load_raw(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            from datasets import load_dataset
            ds = load_dataset(HF_ID, split="train")
            ds.to_pandas().to_parquet(path)
        except Exception:  # plain HTTP fallback
            import urllib.request
            urllib.request.urlretrieve(HF_URL, path)
    df = pd.read_parquet(path)
    return df.rename(columns={"eng": "en", "amh": "am"})[["en", "am"]]


def ethiopic_ratio(s):
    letters = len(ETHIOPIC_RE.findall(s)) + len(LATIN_RE.findall(s))
    return len(ETHIOPIC_RE.findall(s)) / letters if letters else 0.0


def describe(df, name):
    en_len = df.en.str.split().str.len()
    am_len = df.am.str.split().str.len()
    return {
        "name": name,
        "pairs": int(len(df)),
        "en_tokens_mean": round(float(en_len.mean()), 2),
        "en_tokens_median": float(en_len.median()),
        "en_tokens_max": int(en_len.max()),
        "am_tokens_mean": round(float(am_len.mean()), 2),
        "am_tokens_median": float(am_len.median()),
        "am_tokens_max": int(am_len.max()),
        "en_word_types": int(len(set(" ".join(df.en).split()))),
        "am_word_types": int(len(set(" ".join(df.am).split()))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "data/raw/mt560.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data/processed"))
    ap.add_argument("--models", default=os.path.join(ROOT, "models"))
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=25, help="max words per side after normalization")
    ap.add_argument("--val_size", type=int, default=3000)
    ap.add_argument("--test_size", type=int, default=5000)
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.models, exist_ok=True)

    log = []
    df = load_raw(args.raw)
    stats = {"raw": describe(df.astype(str), "raw")}
    log.append(("raw pairs", len(df)))

    # 1. missing / empty
    df = df.dropna()
    df["en"] = df.en.astype(str).str.strip()
    df["am"] = df.am.astype(str).str.strip()
    df = df[(df.en != "") & (df.am != "")]
    log.append(("after dropping missing/empty", len(df)))

    # 2. exact duplicate pairs
    df = df.drop_duplicates()
    log.append(("after dropping exact duplicate pairs", len(df)))

    # 3. remove footnote artefacts, wrong-script pairs
    df["am"] = df.am.str.replace(_FOOTNOTE_MARK, " ", regex=True).str.replace(_FOOTNOTE_TAIL, "", regex=True)
    df = df[df.en.map(lambda s: not ETHIOPIC_RE.search(s))]
    df = df[df.am.map(ethiopic_ratio) >= 0.9]
    log.append(("after script filter (EN has no Ge'ez, AM >= 90% Ge'ez letters)", len(df)))

    # 4. normalize
    df["en"] = df.en.map(normalize_en)
    df["am"] = df.am.map(normalize_am)
    df = df[(df.en != "") & (df.am != "")]

    # 5. length and length-ratio filters (removes misaligned pairs)
    en_len = df.en.str.split().str.len()
    am_len = df.am.str.split().str.len()
    ratio = am_len / en_len
    df = df[(en_len >= args.min_len) & (am_len >= args.min_len) &
            (en_len <= args.max_len) & (am_len <= args.max_len) &
            (ratio >= 0.3) & (ratio <= 1.6)]
    log.append((f"after length filter ({args.min_len}-{args.max_len} tokens, AM/EN ratio 0.3-1.6)", len(df)))

    # 6. duplicates after normalization; one translation per English source
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="en", keep="first")
    log.append(("after de-duplicating normalized pairs and repeated English sources", len(df)))

    # 7. split (sources are unique, so no train/test leakage of identical inputs)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test = df.iloc[:args.test_size]
    val = df.iloc[args.test_size:args.test_size + args.val_size]
    train = df.iloc[args.test_size + args.val_size:]
    for name, part in [("train", train), ("valid", val), ("test", test)]:
        part.to_csv(os.path.join(args.out, f"{name}.tsv"), sep="\t", index=False)
        stats[name] = describe(part, name)

    # 8. SentencePiece (unigram) subword tokenizers, trained on train split only
    for lang in ["en", "am"]:
        txt = os.path.join(args.out, f"train.{lang}.txt")
        train[lang].to_csv(txt, index=False, header=False)
        spm.SentencePieceTrainer.train(
            input=txt, model_prefix=os.path.join(args.models, f"spm_{lang}"),
            vocab_size=args.vocab_size, model_type="unigram",
            character_coverage=1.0 if lang == "am" else 0.9999,
            normalization_rule_name="identity", split_digits=True,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            input_sentence_size=2_000_000, shuffle_input_sentence=True,
            num_threads=4, minloglevel=2)
        os.remove(txt)

    stats["cleaning_log"] = [{"step": s, "pairs": int(n)} for s, n in log]
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    for s, n in log:
        print(f"{n:>8}  {s}")
    print(json.dumps({k: stats[k] for k in ["train", "valid", "test"]}, indent=2))


if __name__ == "__main__":
    main()
