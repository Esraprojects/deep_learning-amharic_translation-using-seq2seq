"""Tokenization (SentencePiece), vocabulary handling and batching."""
import os
import random

import pandas as pd
import sentencepiece as spm
import torch

from models import BOS, EOS, PAD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_TOKENS = 60  # subword cap per side


class Tokenizers:
    def __init__(self, model_dir=os.path.join(ROOT, "models")):
        self.en = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "spm_en.model"))
        self.am = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "spm_am.model"))

    def encode_src(self, texts):
        return [ids[:MAX_TOKENS] for ids in self.en.encode(list(texts))]

    def encode_tgt(self, texts):
        return [ids[:MAX_TOKENS - 1] for ids in self.am.encode(list(texts))]

    def decode_tgt(self, ids):
        return self.am.decode(ids)


def load_split(name, data_dir=os.path.join(ROOT, "data/processed")):
    df = pd.read_csv(os.path.join(data_dir, f"{name}.tsv"), sep="\t", keep_default_na=False)
    return df.en.tolist(), df.am.tolist()


def encode_split(tok, name, cache_dir=os.path.join(ROOT, "data/processed")):
    cache = os.path.join(cache_dir, f"{name}.ids.pt")
    if os.path.exists(cache):
        return torch.load(cache)
    en, am = load_split(name)
    data = list(zip(tok.encode_src(en), tok.encode_tgt(am)))
    tmp = f"{cache}.{os.getpid()}.tmp"  # atomic: parallel training jobs may build it at once
    torch.save(data, tmp)
    os.replace(tmp, cache)
    return data


def pad(seqs, value=PAD):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (L - len(s)) for s in seqs], dtype=torch.long)


def collate(pairs):
    src = [s if s else [EOS] for s, _ in pairs]
    src_len = torch.tensor([len(s) for s in src])
    tgt_in = [[BOS] + t for _, t in pairs]
    tgt_out = [t + [EOS] for _, t in pairs]
    return pad(src), src_len, pad(tgt_in), pad(tgt_out)


def batches(data, batch_size, shuffle=True, seed=0):
    """Length-bucketed batches: sort within large pools to minimise padding."""
    idx = list(range(len(data)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    pool = batch_size * 100
    out = []
    for i in range(0, len(idx), pool):
        chunk = sorted(idx[i:i + pool], key=lambda j: (len(data[j][0]), len(data[j][1])))
        out += [chunk[k:k + batch_size] for k in range(0, len(chunk), batch_size)]
    if shuffle:
        rng.shuffle(out)
    for b in out:
        yield collate([data[j] for j in b])
