# Model comparison (test set, 5000 sentences, beam search {'beam': 5, 'alpha': 1.0, 'no_repeat_ngram': 3})

| Metric | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |
|---|---:|---:|
| BLEU (beam search) ↑ | 5.26 | 11.35 |
| chrF (beam search) ↑ | 13.68 | 23.13 |
| BLEU (greedy) ↑ | 4.91 | 10.36 |
| chrF (greedy) ↑ | 13.5 | 22.04 |
| Outputs with repeated words, greedy (%) ↓ | 22.86 | 16.38 |
| Test loss (CE) ↓ | 3.6906 | 3.1865 |
| Test perplexity ↓ | 40.07 | 24.2 |
| Parameters | 7,995,200 | 8,191,808 |
| Model size (MB) | 30.5 | 31.3 |
| Epochs | 6 | 6 |
| Training time (min) | 420.2 | 420.2 |
| Inference, whole test set, beam search (s) | 174.8 | 185.5 |
| Inference, whole test set, greedy batch 100 (s) | 10.0 | 11.1 |
| Inference per sentence, greedy batched (ms) | 2.0 | 2.22 |
| Inference per sentence, beam, single request (ms) | 33.0 | 35.6 |

**Better model: Attention Seq2Seq + LSTM** (+6.09 BLEU, +9.45 chrF).

BLEU/chrF: sacrebleu 2.6.0, corpus-level, on normalized, punctuation-tokenized text.


### BLEU by data source (beam search)

| Source | n | Seq2Seq | Attention |
|---|---:|---:|---:|
| habtew | 771 | 8.07 | 16.3 |
| mt560 | 4229 | 4.75 | 10.49 |

## Error analysis (automatic)

| Indicator | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |
|---|---:|---:|
| Outputs with repeated words/bigrams (%) ↓ | 7.74 | 8.14 |
| Outputs < 70% of reference length – missing words (%) ↓ | 35.34 | 22.56 |
| Outputs > 130% of reference length – additional words (%) ↓ | 7.42 | 7.7 |
| Mean length ratio hyp/ref (1.0 ideal) | 0.833 | 0.906 |
| Word-order agreement of shared words (1.0 ideal) ↑ | 0.948 | 0.95 |
| Sentences with word-order agreement < 0.6 (%) ↓ | 4.72 | 3.7 |
| Right stem, wrong inflection – morphology errors (% of words) ↓ | 3.81 | 5.56 |
| Named entities translated correctly (%) ↑ | 82.54 | 90.81 |
| Numbers copied correctly (%) ↑ | 70.94 | 97.95 |
| BLEU on sentences with rare source words (<5 in train) | 1.9 | 3.88 |
| BLEU on sentences without rare words | 5.65 | 12.21 |

### BLEU by source length (long-sentence errors)

| Source length | n | Seq2Seq | Attention |
|---|---:|---:|---:|
| 1-10 words | 1164 | 12.26 | 19.2 |
| 11-15 words | 1312 | 5.79 | 11.79 |
| 16-20 words | 1413 | 4.4 | 10.14 |
| 21-25 words | 1111 | 3.26 | 9.58 |
