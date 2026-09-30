"""Seq2Seq models.

* Seq2SeqLSTM      - basic encoder-decoder (Sutskever et al., 2014): the final
                     encoder states are the only information the decoder gets.
* AttnSeq2SeqLSTM  - the same encoder/decoder plus Luong "general" global
                     attention over all encoder states (Luong et al., 2015).

Both share the same bidirectional LSTM encoder so that the comparison isolates
the effect of attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid, layers, dropout):
        super().__init__()
        assert hid % 2 == 0
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.rnn = nn.LSTM(emb, hid // 2, layers, batch_first=True, bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.layers = layers

    def forward(self, src, src_len):
        x = self.drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        # (layers*2, B, H/2) -> (layers, B, H): concatenate forward/backward states
        h = self._merge(h)
        c = self._merge(c)
        return out, (h, c)

    def _merge(self, s):
        L2, B, H = s.shape
        return s.view(self.layers, 2, B, H).permute(0, 2, 1, 3).reshape(self.layers, B, 2 * H)


class Seq2SeqLSTM(nn.Module):
    has_attention = False

    def __init__(self, src_vocab, tgt_vocab, emb=256, hid=512, layers=2, dropout=0.3):
        super().__init__()
        self.config = dict(src_vocab=src_vocab, tgt_vocab=tgt_vocab, emb=emb, hid=hid,
                           layers=layers, dropout=dropout)
        self.encoder = Encoder(src_vocab, emb, hid, layers, dropout)
        self.emb = nn.Embedding(tgt_vocab, emb, padding_idx=PAD)
        self.rnn = nn.LSTM(emb, hid, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hid, tgt_vocab)

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        mask = src != PAD
        return enc_out, mask, state

    def init_extra(self, enc_out):
        return None

    def step(self, tok, state, extra, enc_out, mask):
        """One decoder step. tok: (B,) -> logits (B,V)."""
        x = self.drop(self.emb(tok)).unsqueeze(1)
        o, state = self.rnn(x, state)
        return self.out(self.drop(o.squeeze(1))), state, extra, None

    def forward(self, src, src_len, tgt_in):
        """Teacher-forced forward pass. Returns logits (B, T, V)."""
        _, _, state = self.encode(src, src_len)
        x = self.drop(self.emb(tgt_in))
        o, _ = self.rnn(x, state)
        return self.out(self.drop(o))


class AttnSeq2SeqLSTM(Seq2SeqLSTM):
    """Luong et al. (2015) global attention, "general" score, without input
    feeding: h_t = LSTM(y_{t-1}, h_{t-1});  a_t = softmax(h_t^T W_a H_enc);
    c_t = a_t H_enc;  ~h_t = tanh(W_c [h_t; c_t]);  p(y_t) = softmax(W_o ~h_t)."""
    has_attention = True

    def __init__(self, src_vocab, tgt_vocab, emb=256, hid=512, layers=2, dropout=0.3):
        super().__init__(src_vocab, tgt_vocab, emb, hid, layers, dropout)
        self.W_a = nn.Linear(hid, hid, bias=False)       # general score: h_t^T W_a h_s
        self.W_c = nn.Linear(2 * hid, hid, bias=False)   # attentional hidden state

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        mask = src != PAD
        keys = self.W_a(enc_out)  # projected keys, computed once per sentence
        return (enc_out, keys), mask, state

    def attend(self, h, enc, mask):
        """h: (B,T,H) decoder states -> attentional states (B,T,H), weights (B,T,S)."""
        enc_out, keys = enc
        scores = torch.bmm(h, keys.transpose(1, 2))                  # (B,T,S)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        a = F.softmax(scores, dim=-1)
        ctx = torch.bmm(a, enc_out)                                   # (B,T,H)
        return torch.tanh(self.W_c(torch.cat([h, ctx], dim=-1))), a

    def step(self, tok, state, extra, enc, mask):
        x = self.drop(self.emb(tok)).unsqueeze(1)
        o, state = self.rnn(x, state)
        att_h, a = self.attend(o, enc, mask)
        return self.out(self.drop(att_h.squeeze(1))), state, extra, a.squeeze(1)

    def forward(self, src, src_len, tgt_in):
        enc, mask, state = self.encode(src, src_len)
        o, _ = self.rnn(self.drop(self.emb(tgt_in)), state)
        att_h, _ = self.attend(o, enc, mask)
        return self.out(self.drop(att_h))


MODELS = {"seq2seq": Seq2SeqLSTM, "attention": AttnSeq2SeqLSTM}


def build_model(kind, **cfg):
    return MODELS[kind](**cfg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def greedy_decode(model, src, src_len, max_len=None, return_attention=False):
    """Batched greedy decoding. Returns list of token-id lists (without BOS/EOS)
    and, for the attention model, per-sentence attention matrices (T x S)."""
    model.eval()
    B = src.size(0)
    if max_len is None:
        max_len = int(src_len.max().item() * 2 + 10)
    enc, mask, state = model.encode(src, src_len)
    extra = model.init_extra(enc)
    tok = torch.full((B,), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    out, atts = [], []
    for _ in range(max_len):
        logits, state, extra, a = model.step(tok, state, extra, enc, mask)
        tok = logits.argmax(-1)
        tok = tok.masked_fill(done, PAD)
        out.append(tok)
        if a is not None and return_attention:
            atts.append(a)
        done = done | (tok == EOS)
        if done.all():
            break
    out = torch.stack(out, 1).tolist()
    results = []
    for seq in out:
        ids = []
        for t in seq:
            if t in (EOS, PAD):
                break
            ids.append(t)
        results.append(ids)
    if return_attention and atts:
        A = torch.stack(atts, 1)  # (B,T,S)
        attn = [A[i, :len(results[i]) + 1, :int(src_len[i])].cpu().numpy() for i in range(B)]
        return results, attn
    return results, None
