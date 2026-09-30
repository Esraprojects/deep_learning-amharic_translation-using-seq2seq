"""Export trained models + tokenizers for the in-browser demo (web/).

For each model two ONNX graphs are written:
    <kind>_encoder.onnx   src ids (1,S)                 -> enc_out, keys, h, c
    <kind>_decoder.onnx   token, h, c, enc_out, keys    -> logits, h, c, attention
The large embedding / output matrices are int8-quantized (MatMul, Gather) to
shrink the download; LSTM weights stay float32.

    python src/export_onnx.py
"""
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data import MAX_TOKENS, ROOT  # noqa: E402
from models import build_model  # noqa: E402

WEB = os.path.join(ROOT, "web", "models")


class EncoderExport(nn.Module):
    """Encoder without packing (batch of one sentence => no padding)."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, src):
        enc = self.m.encoder
        out, (h, c) = enc.rnn(enc.emb(src))
        h, c = enc._merge(h), enc._merge(c)
        keys = self.m.W_a(out) if self.m.has_attention else out
        return out, keys, h, c


class DecoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, tok, h, c, enc_out, keys):
        m = self.m
        o, (h, c) = m.rnn(m.emb(tok).unsqueeze(1), (h, c))
        if m.has_attention:
            scores = torch.bmm(o, keys.transpose(1, 2))
            a = torch.softmax(scores, dim=-1)
            ctx = torch.bmm(a, enc_out)
            o = torch.tanh(m.W_c(torch.cat([o, ctx], dim=-1)))
            a = a.squeeze(1)
        else:
            # dummy attention; keeps the decoder signature identical for both models
            a = torch.zeros(1, enc_out.size(1)) + (enc_out.sum() + keys.sum()) * 0
        return m.out(o.squeeze(1)), h, c, a


def export(kind, quantize=True):
    ckpt = torch.load(os.path.join(ROOT, "models", f"{kind}.pt"), map_location="cpu")
    model = build_model(ckpt["kind"], **ckpt["config"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    cfg = ckpt["config"]
    S = 7
    src = torch.randint(4, 100, (1, S))
    enc_path = os.path.join(WEB, f"{kind}_encoder.onnx")
    dec_path = os.path.join(WEB, f"{kind}_decoder.onnx")
    torch.onnx.export(EncoderExport(model), (src,), enc_path, input_names=["src"],
                      output_names=["enc_out", "keys", "h", "c"],
                      dynamic_axes={"src": {1: "S"}, "enc_out": {1: "S"}, "keys": {1: "S"}},
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        out, keys, h, c = EncoderExport(model)(src)
    tok = torch.tensor([2])
    torch.onnx.export(DecoderExport(model), (tok, h, c, out, keys), dec_path,
                      input_names=["tok", "h_in", "c_in", "enc_out", "keys"],
                      output_names=["logits", "h_out", "c_out", "attn"],
                      dynamic_axes={"enc_out": {1: "S"}, "keys": {1: "S"}, "attn": {1: "S"}},
                      opset_version=17, dynamo=False)
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for p in (enc_path, dec_path):
            tmp = p + ".tmp"
            os.replace(p, tmp)
            quantize_dynamic(tmp, p, op_types_to_quantize=["MatMul", "Gather"], weight_type=QuantType.QInt8)
            os.remove(tmp)
    return cfg


def export_vocab(lang):
    pieces = []
    with open(os.path.join(ROOT, "models", f"spm_{lang}.vocab"), encoding="utf-8") as f:
        for line in f:
            p, s = line.rstrip("\n").split("\t")
            pieces.append([p, float(s)])
    return pieces


def main():
    os.makedirs(WEB, exist_ok=True)
    from translate import BEAM
    meta = {"max_tokens": MAX_TOKENS, "beam": BEAM, "models": {}}
    for kind in ("attention", "seq2seq"):
        meta["models"][kind] = export(kind)
        for part in ("encoder", "decoder"):
            p = os.path.join(WEB, f"{kind}_{part}.onnx")
            print(p, round(os.path.getsize(p) / 2**20, 1), "MB")
    with open(os.path.join(WEB, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"en": export_vocab("en"), "am": export_vocab("am"), **meta}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
