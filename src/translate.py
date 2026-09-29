"""Inference pipeline: English text -> Amharic text.

    python src/translate.py "I am going to the university."
    python src/translate.py --model seq2seq "Where is the church?"
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import MAX_TOKENS, ROOT, Tokenizers  # noqa: E402
from models import EOS, build_model, greedy_decode  # noqa: E402
from text import detokenize_am, normalize_en  # noqa: E402


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

    def translate(self, text, return_details=False):
        t = time.perf_counter()
        norm, ids = self.preprocess(text)
        src = torch.tensor([ids])
        out, attn = greedy_decode(self.model, src, torch.tensor([len(ids)]),
                                  return_attention=return_details)
        translation = detokenize_am(self.tok.decode_tgt(out[0]))
        if not return_details:
            return translation
        return {
            "translation": translation,
            "normalized_input": norm,
            "source_tokens": self.tok.en.id_to_piece(ids),
            "target_tokens": self.tok.am.id_to_piece(out[0]) + ["</s>"],
            "attention": attn[0].tolist() if attn is not None else None,
            "latency_ms": round((time.perf_counter() - t) * 1000, 1),
            "model": self.kind,
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--model", default="attention", choices=["seq2seq", "attention"])
    a = ap.parse_args()
    print(Translator(a.model).translate(" ".join(a.text)))
