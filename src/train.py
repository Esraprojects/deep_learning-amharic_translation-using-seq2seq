"""Train a Seq2Seq model.

    python src/train.py --model seq2seq
    python src/train.py --model attention
"""
import argparse
import json
import math
import os
import sys
import time

import sacrebleu
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data import ROOT, Tokenizers, batches, encode_split, load_split  # noqa: E402
from models import PAD, build_model, count_parameters, greedy_decode  # noqa: E402


@torch.no_grad()
def evaluate_loss(model, data, batch_size=128):
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    total, n = 0.0, 0
    for src, src_len, tgt_in, tgt_out in batches(data, batch_size, shuffle=False):
        logits = model(src, src_len, tgt_in)
        total += crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
        n += (tgt_out != PAD).sum().item()
    return total / n


@torch.no_grad()
def quick_bleu(model, tok, data, refs, n=500, batch_size=100):
    hyps = []
    for i in range(0, n, batch_size):
        chunk = data[i:i + batch_size]
        src = [s for s, _ in chunk]
        L = torch.tensor([len(s) for s in src])
        S = torch.zeros(len(src), int(L.max()), dtype=torch.long)
        for j, s in enumerate(src):
            S[j, :len(s)] = torch.tensor(s)
        ids, _ = greedy_decode(model, S, L)
        hyps += [tok.decode_tgt(x) for x in ids]
    return sacrebleu.corpus_bleu(hyps, [refs[:n]]).score, hyps[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["seq2seq", "attention"], required=True)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--hid", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max_train", type=int, default=0, help="use only the first N training pairs (0 = all)")
    ap.add_argument("--time_budget", type=float, default=0, help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tok = Tokenizers()
    train = encode_split(tok, "train")
    if args.max_train:
        train = train[:args.max_train]
    valid = encode_split(tok, "valid")
    _, valid_refs = load_split("valid")

    model = build_model(args.model, src_vocab=tok.en.get_piece_size(), tgt_vocab=tok.am.get_piece_size(),
                        emb=args.emb, hid=args.hid, layers=args.layers, dropout=args.dropout)
    n_params = count_parameters(model)
    print(f"{args.model}: {n_params:,} parameters, {len(train):,} training pairs", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0)
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=args.label_smoothing)

    ckpt = os.path.join(args.out, f"{args.model}.pt")
    last = os.path.join(args.out, f"{args.model}.last.pt")  # resumable training state
    history, best = [], float("inf")
    start_epoch, skip, run, n, elapsed = 1, 0, 0.0, 0, 0.0
    if os.path.exists(last):
        st = torch.load(last, weights_only=False)
        model.load_state_dict(st["state_dict"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["rng"])
        history, best, start_epoch, skip = st["history"], st["best"], st["epoch"], st["step"]
        run, n, elapsed = st["run"], st["n"], st["elapsed"]
        print(f"resumed from epoch {start_epoch} step {skip} ({elapsed / 60:.1f} min trained)", flush=True)

    def save_state(epoch, step):
        torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "rng": torch.get_rng_state(), "history": history,
                    "best": best, "epoch": epoch, "step": step, "run": run, "n": n,
                    "elapsed": elapsed + time.time() - t_resume}, last + ".tmp")
        os.replace(last + ".tmp", last)

    t_resume = time.time()
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    stop = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        ep_t = time.time()
        for step, (src, src_len, tgt_in, tgt_out) in enumerate(batches(train, args.batch_size, seed=epoch), 1):
            if step <= skip:
                continue
            logits = model(src, src_len, tgt_in)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            run += loss.item()
            n += 1
            if step % 500 == 0:
                el = time.time() - ep_t
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {run / n:.3f} "
                      f"({el / (step - skip):.3f}s/step)", flush=True)
            if step % 250 == 0:
                save_state(epoch, step)
            if args.time_budget and (elapsed + time.time() - t_resume) / 60 > args.time_budget:
                stop = True
                break
        skip = 0
        val_loss = evaluate_loss(model, valid)
        bleu, samples = quick_bleu(model, tok, valid, valid_refs)
        sched.step(val_loss)
        rec = dict(epoch=epoch, train_loss=round(run / max(n, 1), 4), val_loss=round(val_loss, 4),
                   val_ppl=round(math.exp(val_loss), 2), val_bleu_500=round(bleu, 2),
                   lr=opt.param_groups[0]["lr"], steps=n, partial_epoch=stop)
        history.append(rec)
        print(json.dumps(rec), flush=True)
        for s in samples:
            print("   ", s, flush=True)
        if val_loss < best:
            best = val_loss
            torch.save({"kind": args.model, "config": model.config, "state_dict": model.state_dict(),
                        "epoch": epoch, "val_loss": val_loss}, ckpt)
        run, n = 0.0, 0
        save_state(epoch + 1, 0)
        if stop:
            break

    train_time = elapsed + time.time() - t_resume
    info = dict(model=args.model, parameters=n_params, training_pairs=len(train),
                training_time_sec=round(train_time, 1), best_val_loss=round(best, 4),
                hyperparameters=vars(args) | {"optimizer": "Adam", "loss": "CrossEntropy (label smoothing)",
                                               "scheduler": "ReduceLROnPlateau(factor=0.5, patience=0)"},
                history=history)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"train_{args.model}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"done in {train_time / 60:.1f} min, best val loss {best:.4f}")


if __name__ == "__main__":
    main()
