# Model comparison (test set, 5000 sentences, beam search {'beam': 5, 'alpha': 1.0, 'no_repeat_ngram': 3})

| Metric | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |
|---|---:|---:|
| BLEU (beam search) ↑ | 5.21 | 11.27 |
| chrF (beam search) ↑ | 13.52 | 22.84 |
| BLEU (greedy) ↑ | 4.77 | 10.23 |
| chrF (greedy) ↑ | 13.23 | 21.62 |
| Outputs with repeated words, greedy (%) ↓ | 18.34 | 15.98 |
| Test loss (CE) ↓ | 3.7009 | 3.2015 |
| Test perplexity ↓ | 40.48 | 24.57 |
| Parameters | 7,995,200 | 8,191,808 |
| Model size (MB) | 30.5 | 31.3 |
| Epochs | 6 | 6 |
| Training time (min) | 420.2 | 420.2 |
| Inference, whole test set, beam search (s) | 406.2 | 394.2 |
| Inference, whole test set, greedy batch 100 (s) | 20.2 | 23.3 |
| Inference per sentence, greedy batched (ms) | 4.04 | 4.66 |
| Inference per sentence, beam, single request (ms) | 63.7 | 81.3 |

**Better model: Attention Seq2Seq + LSTM** (+6.06 BLEU, +9.32 chrF).

BLEU/chrF: sacrebleu 2.6.0, corpus-level, on normalized, punctuation-tokenized text.


### BLEU by data source (beam search)

| Source | n | Seq2Seq | Attention |
|---|---:|---:|---:|
| habtew | 771 | 8.2 | 15.75 |
| mt560 | 4229 | 4.67 | 10.5 |

## Error analysis (automatic)

| Indicator | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |
|---|---:|---:|
| Outputs with repeated words/bigrams (%) ↓ | 5.88 | 8.34 |
| Outputs < 70% of reference length – missing words (%) ↓ | 40.22 | 26.12 |
| Outputs > 130% of reference length – additional words (%) ↓ | 5.86 | 6.38 |
| Mean length ratio hyp/ref (1.0 ideal) | 0.796 | 0.875 |
| Word-order agreement of shared words (1.0 ideal) ↑ | 0.953 | 0.95 |
| Sentences with word-order agreement < 0.6 (%) ↓ | 4.02 | 3.66 |
| Right stem, wrong inflection – morphology errors (% of words) ↓ | 3.82 | 5.62 |
| Named entities translated correctly (%) ↑ | 81.85 | 88.23 |
| Numbers copied correctly (%) ↑ | 70.77 | 96.58 |
| BLEU on sentences with rare source words (<5 in train) | 1.87 | 4.19 |
| BLEU on sentences without rare words | 5.61 | 12.11 |

### BLEU by source length (long-sentence errors)

| Source length | n | Seq2Seq | Attention |
|---|---:|---:|---:|
| 1-10 words | 1164 | 11.94 | 19.52 |
| 11-15 words | 1312 | 5.85 | 11.76 |
| 16-20 words | 1413 | 4.4 | 9.84 |
| 21-25 words | 1111 | 3.22 | 9.55 |
