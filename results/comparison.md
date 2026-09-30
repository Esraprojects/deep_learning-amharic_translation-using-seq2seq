# Model comparison (test set, 5000 sentences, greedy decoding)

| Metric | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |
|---|---:|---:|
| BLEU ↑ | 2.97 | 8.57 |
| chrF ↑ | 12.16 | 19.46 |
| Test loss (CE) ↓ | 3.9742 | 3.5477 |
| Test perplexity ↓ | 53.21 | 34.73 |
| Parameters | 7,995,200 | 8,191,808 |
| Model size (MB) | 30.5 | 31.3 |
| Epochs | 4 | 4 |
| Training time (min) | 150.1 | 150.2 |
| Inference, whole test set (s, batch 100) | 9.7 | 16.8 |
| Inference per sentence, batched (ms) | 1.93 | 3.37 |
| Inference per sentence, single (ms) | 19.5 | 22.1 |

**Better model: Attention Seq2Seq + LSTM** (+5.60 BLEU, +7.30 chrF).

BLEU/chrF: sacrebleu 2.6.0, corpus-level, on normalized, punctuation-tokenized text.


## Error analysis (automatic)

| Indicator | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |
|---|---:|---:|
| Outputs with repeated words/bigrams (%) ↓ | 18.94 | 24.12 |
| Outputs < 70% of reference length – missing words (%) ↓ | 26.64 | 17.16 |
| Outputs > 130% of reference length – additional words (%) ↓ | 10.06 | 13.72 |
| Mean length ratio hyp/ref (1.0 ideal) | 0.902 | 1.006 |
| Word-order agreement of shared words (1.0 ideal) ↑ | 0.94 | 0.942 |
| Sentences with word-order agreement < 0.6 (%) ↓ | 4.96 | 4.41 |
| Right stem, wrong inflection – morphology errors (% of words) ↓ | 3.71 | 5.66 |
| Named entities translated correctly (%) ↑ | 85.08 | 89.42 |
| Numbers copied correctly (%) ↑ | 38.28 | 93.55 |
| BLEU on sentences with rare source words (<5 in train) | 0.73 | 3.22 |
| BLEU on sentences without rare words | 3.24 | 9.28 |

### BLEU by source length (long-sentence errors)

| Source length | n | Seq2Seq | Attention |
|---|---:|---:|---:|
| 1-10 words | 1066 | 7.02 | 15.52 |
| 11-15 words | 1291 | 3.26 | 9.67 |
| 16-20 words | 1416 | 2.57 | 7.28 |
| 21-25 words | 1227 | 1.91 | 7.04 |
