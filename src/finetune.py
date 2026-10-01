"""Fine-tune a trained model on the curated pairs (data/curated/*.tsv) mixed with
original training data, so it learns the corrected patterns without forgetting.

    python src/augment.py
    python src/finetune.py --model attention
    python src/finetune.py --model seq2seq

15 % of the curated pairs are held out to check that the pattern generalises to
sentences the model has not seen. Results: results/finetune_<model>.json
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import pandas as pd
import sacrebleu
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data import ROOT, Tokenizers, batches, encode_split  # noqa: E402
from models import PAD, beam_search, build_model  # noqa: E402
from text import normalize_am, normalize_en  # noqa: E402
from train import evaluate_loss  # noqa: E402


def load_curated(seed=0, holdout=0.15):
    df = pd.concat([pd.read_csv(f, sep="\t", keep_default_na=False)
                    for f in sorted(glob.glob(os.path.join(ROOT, "data/curated/*.tsv")))])
    pairs = [(normalize_en(e), normalize_am(a)) for e, a in zip(df.en, df.am)]
    random.Random(seed).shuffle(pairs)
    n = int(len(pairs) * holdout)
    return pairs[n:], pairs[:n]


def check(model, tok, pairs):
    hyps = [tok.decode_tgt(beam_search(model, tok.en.encode(e), beam=5, alpha=1.0, no_repeat_ngram=3)[0])
            for e, _ in pairs]
    return sacrebleu.corpus_bleu(hyps, [[a for _, a in pairs]]).score, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["seq2seq", "attention"], required=True)
    ap.add_argument("--upsample", type=int, default=20, help="repeat each curated pair this many times")
    ap.add_argument("--original", type=int, default=40000, help="original training pairs mixed in")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)

    tok = Tokenizers()
    path = os.path.join(ROOT, "models", f"{args.model}.pt")
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["kind"], **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])

    train_cur, held = load_curated()
    cur_ids = list(zip(tok.encode_src([e for e, _ in train_cur]), tok.encode_tgt([a for _, a in train_cur])))
    orig = encode_split(tok, "train")
    rng = random.Random(1)
    data = cur_ids * args.upsample + rng.sample(orig, args.original)
    valid = encode_split(tok, "valid")

    before = {"valid_loss": round(evaluate_loss(model, valid), 4),
              "heldout_bleu": round(check(model, tok, held)[0], 2)}
    print("before", before, flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for src, src_len, tgt_in, tgt_out in batches(data, args.batch_size, seed=100 + epoch):
            logits = model(src, src_len, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"epoch {epoch} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    bleu, hyps = check(model, tok, held)
    after = {"valid_loss": round(evaluate_loss(model, valid), 4), "heldout_bleu": round(bleu, 2)}
    print("after", after, flush=True)
    for (e, a), h in list(zip(held, hyps))[:8]:
        print(f"  {e}\n    ref: {a}\n    hyp: {h}")

    torch.save({**ckpt, "state_dict": model.state_dict(), "finetuned_on": "data/curated"}, path)
    with open(os.path.join(ROOT, "results", f"finetune_{args.model}.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "curated_train": len(train_cur), "curated_heldout": len(held),
                   "settings": vars(args), "before": before, "after": after,
                   "minutes": round((time.time() - t0) / 60, 1),
                   "heldout_examples": [{"en": e, "ref": a, "hyp": h} for (e, a), h in zip(held, hyps)]},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
