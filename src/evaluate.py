"""Evaluate and compare both models on the held-out test set.

Outputs (results/):
    metrics.json, comparison.md        BLEU, chrF, test loss, timing, #params
    examples.md                        Source -> Reference -> Seq2Seq -> Attention
    test_predictions.tsv               all test-set outputs
    error_analysis.md / .json          automatic error statistics + examples
    figures/*.png                      training curves, BLEU by length, attention maps

    python src/evaluate.py
"""
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter

import matplotlib
import pandas as pd
import sacrebleu
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data import ROOT, Tokenizers, batches, encode_split, load_split, pad  # noqa: E402
from models import count_parameters, greedy_decode  # noqa: E402
from train import evaluate_loss  # noqa: E402
from translate import Translator  # noqa: E402
from text import detokenize_am, normalize_am  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
FONT = os.path.join(ROOT, "assets", "NotoSansEthiopic.ttf")
KINDS = ["seq2seq", "attention"]
NAMES = {"seq2seq": "Seq2Seq + LSTM", "attention": "Attention Seq2Seq + LSTM"}

# English name -> (normalized) Amharic stem, used for the named-entity check.
NAMED_ENTITIES = {
    "jehovah": "ይሖዋ", "jesus": "ኢየሱስ", "god": "አምላክ", "moses": "ሙሴ", "israel": "እስራኤል",
    "paul": "ጳውሎስ", "david": "ዳዊት", "abraham": "አብርሀም", "egypt": "ግብጽ", "jerusalem": "ኢየሩሳሌም",
    "peter": "ጴጥሮስ", "satan": "ሰይጣን", "allah": "አላህ", "christ": "ክርስቶስ", "bible": "መጽሀፍ ቅዱስ",
    "john": "ዮሀንስ", "solomon": "ሰሎሞን", "noah": "ኖህ", "adam": "አዳም", "babylon": "ባቢሎን",
    "ethiopia": "ኢትዮጵያ", "africa": "አፍሪካ", "joseph": "ዮሴፍ", "jacob": "ያዕቆብ", "pharaoh": "ፈርኦን",
}
NAMED_ENTITIES = {k: normalize_am(v) for k, v in NAMED_ENTITIES.items()}


def setup_font():
    if os.path.exists(FONT):
        font_manager.fontManager.addfont(FONT)
        name = font_manager.FontProperties(fname=FONT).get_name()
        plt.rcParams["font.family"] = [name, "DejaVu Sans"]


def decode_all(model, tok, data, batch_size=100):
    hyps, t0 = [], time.perf_counter()
    for i in range(0, len(data), batch_size):
        src = [s for s, _ in data[i:i + batch_size]]
        L = torch.tensor([len(s) for s in src])
        ids, _ = greedy_decode(model, pad(src), L)
        hyps += [tok.decode_tgt(x) for x in ids]
    return hyps, time.perf_counter() - t0


def single_latency(tr, sentences):
    tr.translate(sentences[0])  # warm-up
    ts = []
    for s in sentences:
        t = time.perf_counter()
        tr.translate(s)
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.mean(ts)


# ---------------------------------------------------------------------------
# automatic error analysis
# ---------------------------------------------------------------------------
def has_repetition(h):
    w = h.split()
    if any(a == b for a, b in zip(w, w[1:])):
        return True
    bigrams = list(zip(w, w[1:]))
    return any(c > 1 for c in Counter(bigrams).values())


def order_agreement(h, r):
    """Kendall-tau style agreement of the order of words shared (once) by hyp and ref."""
    hw, rw = h.split(), r.split()
    hc, rc = Counter(hw), Counter(rw)
    shared = [w for w in hw if hc[w] == 1 and rc[w] == 1 and w not in "።፣፤?!.,:;\"'()"]
    if len(shared) < 3:
        return None
    pos = [rw.index(w) for w in shared]
    pairs = [(i, j) for i in range(len(pos)) for j in range(i + 1, len(pos))]
    return sum(pos[i] < pos[j] for i, j in pairs) / len(pairs)


def morphology_errors(h, r):
    """Hypothesis words not in the reference that share a >=3-syllable prefix
    (same stem) with a reference word: right lexeme, wrong inflection."""
    rset = set(r.split())
    wrong, total = 0, 0
    for w in h.split():
        if len(w) < 2 or not re.search(r"[ሀ-፿]", w):
            continue
        total += 1
        if w not in rset and any(len(x) >= 3 and w[:3] == x[:3] for x in rset):
            wrong += 1
    return wrong, total


def error_analysis(src, refs, hyps, train_counts):
    res, ex = {}, {}
    n = len(refs)
    rep = [i for i in range(n) if has_repetition(hyps[i])]
    ratios = [len(hyps[i].split()) / max(1, len(refs[i].split())) for i in range(n)]
    short = [i for i in range(n) if ratios[i] < 0.7]
    long_ = [i for i in range(n) if ratios[i] > 1.3]
    orders = [order_agreement(hyps[i], refs[i]) for i in range(n)]
    ord_vals = [o for o in orders if o is not None]
    bad_order = [i for i in range(n) if orders[i] is not None and orders[i] < 0.6]
    m_wrong = m_total = 0
    for i in range(n):
        a, b = morphology_errors(hyps[i], refs[i])
        m_wrong += a
        m_total += b
    # named entities
    ne_total = ne_ok = 0
    ne_bad = []
    for i in range(n):
        sw = set(src[i].split())
        for en, am in NAMED_ENTITIES.items():
            if en in sw and am in refs[i]:
                ne_total += 1
                if am in hyps[i]:
                    ne_ok += 1
                else:
                    ne_bad.append(i)
    # numbers copied?
    num_total = num_ok = 0
    for i in range(n):
        nums = re.findall(r"\d+", src[i])
        if nums and all(x in refs[i] for x in nums):
            num_total += 1
            num_ok += all(x in hyps[i] for x in nums)
    # rare words: sentences containing a source word seen < 5 times in training
    rare = [i for i in range(n) if any(train_counts[w] < 5 for w in src[i].split() if w.isalpha())]
    common = [i for i in range(n) if i not in set(rare)]

    def sub_bleu(idx):
        return round(sacrebleu.corpus_bleu([hyps[i] for i in idx], [[refs[i] for i in idx]]).score, 2) if idx else None

    res["repeated_words_pct"] = round(100 * len(rep) / n, 2)
    res["too_short_missing_words_pct"] = round(100 * len(short) / n, 2)
    res["too_long_additional_words_pct"] = round(100 * len(long_) / n, 2)
    res["mean_length_ratio_hyp_ref"] = round(sum(ratios) / n, 3)
    res["word_order_agreement"] = round(sum(ord_vals) / max(1, len(ord_vals)), 3)
    res["word_order_errors_pct"] = round(100 * len(bad_order) / max(1, len(ord_vals)), 2)
    res["morphology_error_rate_pct"] = round(100 * m_wrong / max(1, m_total), 2)
    res["named_entity_accuracy_pct"] = round(100 * ne_ok / max(1, ne_total), 2)
    res["named_entity_cases"] = ne_total
    res["number_copy_accuracy_pct"] = round(100 * num_ok / max(1, num_total), 2)
    res["rare_word_sentences"] = len(rare)
    res["bleu_rare_word_sentences"] = sub_bleu(rare)
    res["bleu_common_word_sentences"] = sub_bleu(common)
    ex = {"repeated": rep[:5], "missing": short[:5], "additional": long_[:5],
          "word_order": bad_order[:5], "named_entity": ne_bad[:5], "rare": rare[:5]}
    return res, ex


def length_buckets(src, refs, hyps):
    edges = [(1, 10), (11, 15), (16, 20), (21, 25)]
    out = []
    for lo, hi in edges:
        idx = [i for i in range(len(src)) if lo <= len(src[i].split()) <= hi]
        b = sacrebleu.corpus_bleu([hyps[i] for i in idx], [[refs[i] for i in idx]]).score
        out.append({"bucket": f"{lo}-{hi}", "n": len(idx), "bleu": round(b, 2)})
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_training_curves(train_info):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for k in KINDS:
        h = train_info[k]["history"]
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], marker="o", label=NAMES[k])
        axes[1].plot(ep, [r["val_loss"] for r in h], marker="o", label=NAMES[k])
        axes[2].plot(ep, [r["val_bleu_500"] for r in h], marker="o", label=NAMES[k])
    for ax, t in zip(axes, ["Training loss (label-smoothed CE)", "Validation loss (CE)", "Validation BLEU (500 sent.)"]):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curves.png"), dpi=130)
    plt.close(fig)


def plot_length(buckets):
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    xs = range(len(buckets["seq2seq"]))
    for j, k in enumerate(KINDS):
        ax.bar([x + (j - .5) * w for x in xs], [b["bleu"] for b in buckets[k]], w, label=NAMES[k])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f'{b["bucket"]}\n(n={b["n"]})' for b in buckets["seq2seq"]])
    ax.set_xlabel("source sentence length (words)")
    ax.set_ylabel("BLEU")
    ax.set_title("BLEU by source sentence length")
    ax.legend()
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "bleu_by_length.png"), dpi=130)
    plt.close(fig)


def plot_attention(d, path, title):
    A = d["attention"]
    fig, ax = plt.subplots(figsize=(max(5, .45 * len(d["source_tokens"]) + 2), max(4, .38 * len(d["target_tokens"]) + 2)))
    ax.imshow(A, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(d["source_tokens"])))
    ax.set_xticklabels(d["source_tokens"], rotation=70, ha="right", fontsize=9)
    ax.set_yticks(range(len(d["target_tokens"])))
    ax.set_yticklabels(d["target_tokens"], fontsize=10)
    ax.set_xlabel("source (English subwords)")
    ax.set_ylabel("output (Amharic subwords)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    setup_font()
    torch.set_num_threads(4)
    tok = Tokenizers()
    test = encode_split(tok, "test")
    src, refs = load_split("test")
    train_src, _ = load_split("train")
    train_counts = Counter(w for s in train_src for w in s.split())

    metrics, preds, analysis, examples_idx, buckets, train_info = {}, {}, {}, {}, {}, {}
    translators = {}
    for k in KINDS:
        tr = Translator(k)
        translators[k] = tr
        with open(os.path.join(RES, f"train_{k}.json")) as f:
            train_info[k] = json.load(f)
        loss = evaluate_loss(tr.model, test)
        hyps, dt = decode_all(tr.model, tok, test)
        preds[k] = hyps
        lat = single_latency(tr, src[:200])
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        chrf = sacrebleu.corpus_chrf(hyps, [refs])
        size_mb = os.path.getsize(os.path.join(ROOT, "models", f"{k}.pt")) / 2**20
        metrics[k] = {
            "model": NAMES[k],
            "bleu": round(bleu.score, 2),
            "chrf": round(chrf.score, 2),
            "test_loss": round(loss, 4),
            "test_perplexity": round(math.exp(loss), 2),
            "parameters": count_parameters(tr.model),
            "model_size_mb": round(size_mb, 1),
            "epochs_trained": len(train_info[k]["history"]),
            "training_time_min": round(train_info[k]["training_time_sec"] / 60, 1),
            "inference_time_test_set_sec": round(dt, 1),
            "inference_ms_per_sentence_batched": round(1000 * dt / len(test), 2),
            "inference_ms_per_sentence_single": round(lat, 1),
        }
        analysis[k], examples_idx[k] = error_analysis(src, refs, hyps, train_counts)
        buckets[k] = length_buckets(src, refs, hyps)
        analysis[k]["bleu_by_length"] = buckets[k]
        print(json.dumps(metrics[k]), flush=True)

    best = max(KINDS, key=lambda k: metrics[k]["bleu"])
    with open(os.path.join(RES, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "best_model": best, "error_analysis": analysis,
                   "test_sentences": len(test)}, f, indent=2, ensure_ascii=False)

    # comparison table
    rows = [("BLEU ↑", "bleu"), ("chrF ↑", "chrf"), ("Test loss (CE) ↓", "test_loss"),
            ("Test perplexity ↓", "test_perplexity"), ("Parameters", "parameters"),
            ("Model size (MB)", "model_size_mb"), ("Epochs", "epochs_trained"),
            ("Training time (min)", "training_time_min"),
            ("Inference, whole test set (s, batch 100)", "inference_time_test_set_sec"),
            ("Inference per sentence, batched (ms)", "inference_ms_per_sentence_batched"),
            ("Inference per sentence, single (ms)", "inference_ms_per_sentence_single")]
    md = [f"# Model comparison (test set, {len(test)} sentences, greedy decoding)\n",
          "| Metric | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |", "|---|---:|---:|"]
    for label, key in rows:
        a, b = metrics["seq2seq"][key], metrics["attention"][key]
        fmt = (lambda v: f"{v:,}") if key == "parameters" else str
        md.append(f"| {label} | {fmt(a)} | {fmt(b)} |")
    md.append(f"\n**Better model: {NAMES[best]}** "
              f"(+{abs(metrics['attention']['bleu'] - metrics['seq2seq']['bleu']):.2f} BLEU, "
              f"+{abs(metrics['attention']['chrf'] - metrics['seq2seq']['chrf']):.2f} chrF).\n")
    md.append(f"BLEU/chrF: sacrebleu {sacrebleu.__version__}, corpus-level, "
              "on normalized, punctuation-tokenized text.\n")
    md.append("\n## Error analysis (automatic)\n")
    md.append("| Indicator | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |")
    md.append("|---|---:|---:|")
    labels = {
        "repeated_words_pct": "Outputs with repeated words/bigrams (%) ↓",
        "too_short_missing_words_pct": "Outputs < 70% of reference length – missing words (%) ↓",
        "too_long_additional_words_pct": "Outputs > 130% of reference length – additional words (%) ↓",
        "mean_length_ratio_hyp_ref": "Mean length ratio hyp/ref (1.0 ideal)",
        "word_order_agreement": "Word-order agreement of shared words (1.0 ideal) ↑",
        "word_order_errors_pct": "Sentences with word-order agreement < 0.6 (%) ↓",
        "morphology_error_rate_pct": "Right stem, wrong inflection – morphology errors (% of words) ↓",
        "named_entity_accuracy_pct": "Named entities translated correctly (%) ↑",
        "number_copy_accuracy_pct": "Numbers copied correctly (%) ↑",
        "bleu_rare_word_sentences": "BLEU on sentences with rare source words (<5 in train)",
        "bleu_common_word_sentences": "BLEU on sentences without rare words",
    }
    for key, label in labels.items():
        md.append(f"| {label} | {analysis['seq2seq'][key]} | {analysis['attention'][key]} |")
    md.append("\n### BLEU by source length (long-sentence errors)\n")
    md.append("| Source length | n | Seq2Seq | Attention |")
    md.append("|---|---:|---:|---:|")
    for a, b in zip(buckets["seq2seq"], buckets["attention"]):
        md.append(f"| {a['bucket']} words | {a['n']} | {a['bleu']} | {b['bleu']} |")
    with open(os.path.join(RES, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # predictions + example table
    pd.DataFrame({"source": src, "reference": refs, "seq2seq": preds["seq2seq"],
                  "attention": preds["attention"]}).to_csv(os.path.join(RES, "test_predictions.tsv"),
                                                           sep="\t", index=False)
    ex = ["# Translation examples (test set)\n",
          "| # | Source (EN) | Reference (AM) | Seq2Seq + LSTM | Attention-LSTM |", "|---|---|---|---|---|"]
    for n, i in enumerate(range(0, 5000, 200)[:25], 1):
        cells = [src[i], detokenize_am(refs[i]), detokenize_am(preds["seq2seq"][i]),
                 detokenize_am(preds["attention"][i])]
        ex.append(f"| {n} | " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    with open(os.path.join(RES, "examples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(ex) + "\n")

    # error examples
    em = ["# Error examples (selected automatically)\n"]
    for k in KINDS:
        em.append(f"\n## {NAMES[k]}\n")
        for cat, idx in examples_idx[k].items():
            em.append(f"\n### {cat.replace('_', ' ')}\n")
            for i in idx[:3]:
                em.append(f"- **SRC:** {src[i]}  \n  **REF:** {detokenize_am(refs[i])}  \n"
                          f"  **HYP:** {detokenize_am(preds[k][i])}")
    with open(os.path.join(RES, "error_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(em) + "\n")

    # figures
    plot_training_curves(train_info)
    plot_length(buckets)
    demo = ["I am going to the university.",
            "Jesus taught his disciples to love one another.",
            "The children are playing in the garden.",
            "God created the heavens and the earth.",
            "We must read the Bible every day.",
            "My mother is cooking dinner for the family."]
    demo += [src[i] for i in (10, 400, 1300)]
    att = []
    for j, s in enumerate(demo):
        d = translators["attention"].translate(s, return_details=True)
        d["seq2seq"] = translators["seq2seq"].translate(s)
        d["source"] = s
        plot_attention(d, os.path.join(FIG, f"attention_{j + 1}.png"), f"{s}\n→ {d['translation']}")
        att.append({k: d[k] for k in ["source", "translation", "seq2seq", "source_tokens", "target_tokens"]})
    with open(os.path.join(RES, "attention_examples.json"), "w", encoding="utf-8") as f:
        json.dump(att, f, indent=2, ensure_ascii=False)
    print(open(os.path.join(RES, "comparison.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
