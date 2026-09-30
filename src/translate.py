"""Inference pipeline: English text -> Amharic text.

    python src/translate.py "I am going to the university."
    python src/translate.py --model seq2seq "Where is the church?"
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import MAX_TOKENS, ROOT, Tokenizers  # noqa: E402
from models import EOS, beam_search, build_model, greedy_decode  # noqa: E402
from text import detokenize_am, normalize_en  # noqa: E402


# Beam-search settings: tuned on the validation set by src/evaluate.py and stored in
# results/decoding_tuning.json (falls back to common defaults if it is missing).
BEAM = dict(beam=5, alpha=0.7, no_repeat_ngram=3)
_TUNED = os.path.join(ROOT, "results", "decoding_tuning.json")
if os.path.exists(_TUNED):
    with open(_TUNED) as _f:
        BEAM.update(json.load(_f)["chosen"])


class Translator:
    def __init__(self, kind="attention", model_dir=os.path.join(ROOT, "models")):
        self.tok = Tokenizers(model_dir)
        ckpt = torch.load(os.path.join(model_dir, f"{kind}.pt"), map_location="cpu")
        self.model = build_model(ckpt["kind"], **ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.kind = kind

    def preprocess(self, text):
        norm = normalize_en(text)
        ids = self.tok.en.encode(norm)[:MAX_TOKENS] or [EOS]
        return norm, ids

    def decode(self, ids, decoding="beam", return_attention=False):
        if decoding == "greedy":
            out, attn = greedy_decode(self.model, torch.tensor([ids]), torch.tensor([len(ids)]),
                                      return_attention=return_attention)
            return out[0], (attn[0] if attn else None)
        return beam_search(self.model, ids, return_attention=return_attention, **BEAM)

    def translate(self, text, return_details=False, decoding="beam"):
        t = time.perf_counter()
        norm, ids = self.preprocess(text)
        out, attn = self.decode(ids, decoding, return_attention=return_details)
        translation = detokenize_am(self.tok.decode_tgt(out))
        if not return_details:
            return translation
        return {
            "translation": translation,
            "normalized_input": norm,
            "source_tokens": self.tok.en.id_to_piece(ids),
            "target_tokens": self.tok.am.id_to_piece(out) + ["</s>"],
            "attention": attn.tolist() if attn is not None else None,
            "decoding": decoding,
            "latency_ms": round((time.perf_counter() - t) * 1000, 1),
            "model": self.kind,
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--model", default="attention", choices=["seq2seq", "attention"])
    ap.add_argument("--decoding", default="beam", choices=["beam", "greedy"])
    a = ap.parse_args()
    print(Translator(a.model).translate(" ".join(a.text), decoding=a.decoding))
