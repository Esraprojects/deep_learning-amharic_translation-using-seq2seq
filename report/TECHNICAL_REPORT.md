# English → Amharic Neural Machine Translation: Seq2Seq-LSTM vs Attention-LSTM

**Deep Learning Course Project — Technical Report**
Instructor: Fantahun Bogale Gereme · Group members: _see README_

---

## Abstract

We build, evaluate, compare and deploy an English→Amharic neural machine translation system.
Two recurrent models are trained on the same data under the same compute budget: a **basic
Seq2Seq + LSTM** encoder–decoder (the only link between encoder and decoder is the final
hidden state) and an **Attention-based Seq2Seq + LSTM** (Luong global attention). On a held-out
test set of 5,000 sentences, the attention model reaches **8.57 BLEU / 19.46 chrF** against
**2.97 BLEU / 12.16 chrF** for the baseline. Its test perplexity is also lower (34.7 vs 53.2), with
2.5 % more parameters. It wins in every sentence-length bucket, copies numbers correctly 94 % of the
time (baseline 38 %) and translates named entities more reliably. The best model is served through a
FastAPI `POST /translate` endpoint with a Gradio UI, and through a public browser demo that runs the
model client-side with ONNX Runtime Web.

---

## 1. Dataset & preprocessing

### 1.1 Source and license
* **Dataset:** OPUS **MT560** English–Amharic parallel corpus, as packaged on the Hugging Face Hub:
  [`michsethowusu/english-amharic_sentence-pairs_mt560`](https://huggingface.co/datasets/michsethowusu/english-amharic_sentence-pairs_mt560).
  Original source: [OPUS MT560](https://opus.nlpl.eu/MT560).
* **License:** Creative Commons Attribution 4.0 (CC-BY-4.0).
* We chose it over `habtew/english-amharic-translation` because its license is documented and it is
  about 3× larger (669k vs 237k pairs). Both corpora are dominated by the same religious sources.

### 1.2 Size and characteristics (raw)
| Property | Value |
|---|---|
| Sentence pairs | 669,145 |
| Mean / median / max English tokens | 21.4 / 19 / 120 |
| Mean / median / max Amharic tokens | 14.6 / 13 / 118 |
| English word types | 91,739 |
| Amharic word types | 361,906 |

Observations:
* **Domain:** mostly religious text (Bible translations, the Qur'an, Watchtower/Awake! publications),
  plus some software-localization strings. About 24 % of test sentences contain *jehovah / god / bible / jesus*.
* **Morphology:** Amharic has ~4× more word types than English for the same content. It is highly
  inflected: subject/object agreement, prepositions and possessives attach to the verb or noun
  (e.g. *አስተምሯቸዋል* = "he taught them"). A word-level vocabulary would therefore be very sparse.
* **Word order:** English is SVO; Amharic is **SOV** with the verb at the end of the sentence.
* **Noise:** the data is already whitespace-tokenized ("God 's"). It has Ge'ez punctuation variants
  (`፡፡` vs `።`), Bible footnote markers (`* ፍ1 *`), some misaligned pairs, and pairs in the wrong script.

### 1.3 Cleaning & normalization (`src/preprocess.py`, `src/text.py`)
| Step | Pairs left |
|---|---:|
| Raw | 669,145 |
| Drop missing / empty | 669,145 |
| Drop exact duplicate pairs | 669,094 |
| Remove footnote markers; drop wrong-script pairs (EN must have no Ge'ez, AM must be ≥ 90 % Ge'ez letters) | 663,293 |
| Normalize; keep 2–25 tokens per side and an AM/EN length ratio in [0.3, 1.6] (removes misalignments) | 442,746 |
| De-duplicate normalized pairs; keep one translation per English source | **435,665** |

**English normalization:** Unicode NFKC, unified quotes and dashes, lower-casing, and every
punctuation character split off as its own token.

**Amharic normalization:** NFC, `፡፡` / `::` → `።`, punctuation split off, and the standard **homophone
normalization** used in Amharic NLP. Characters that sound identical in modern Amharic are mapped to
one form: ሐ/ኀ/ሃ → ሀ, ሠ → ሰ, ዐ/ኣ → አ, ፀ → ጸ (all seven vowel orders), and a few labialized
forms. This reduces spelling variation, e.g. *ኃጢአት / ሀጢአት / ኀጢአት* → *ሀጢአት*.
The model therefore outputs normalized spelling. It is readable, but it is not always the official
orthography.

Keeping one translation per English source also ensures that **no test source sentence occurs in training**.

### 1.4 Tokenization & vocabulary
**SentencePiece unigram** models (identity normalization, digits split), trained on the training split
only, one per language:

| | English | Amharic |
|---|---:|---:|
| Vocabulary (incl. `<pad>=0, <unk>=1, <s>=2, </s>=3`) | 8,000 | 8,000 |
| Mean subwords / sentence (train) | 17.2 | 16.8 |

Subwords let the decoder build rare inflected Amharic forms from pieces
(e.g. `▁ያ` + `ንጸባረቀ` + `ውን`). They also keep the output layer small enough for CPU training.

### 1.5 Split
Random shuffle (seed 42) after de-duplication:

| Split | Pairs | Mean EN tokens | Mean AM tokens |
|---|---:|---:|---:|
| Train | 427,665 | 15.5 | 11.1 |
| Validation | 3,000 | 15.5 | 11.1 |
| Test | 5,000 | 15.6 | 11.2 |

---

## 2. Model development & training

### 2.1 Architectures (`src/models.py`)
Both models share **the same encoder**, so the comparison isolates the effect of attention.

* **Encoder:** embedding (256) → 2-layer **bidirectional LSTM** (128 units per direction = 256).
  The final forward/backward states of each layer are concatenated to initialise the decoder.
* **Basic Seq2Seq + LSTM** (Sutskever et al., 2014): the decoder is a 2-layer LSTM (256) initialised
  with the encoder's final states. It sees nothing else from the source, so the whole sentence must fit
  in a fixed-size vector. Output: `Linear(256 → 8000)` + softmax.
* **Attention Seq2Seq + LSTM** (Luong et al., 2015, global "general" attention): the same decoder, plus
  at every step *t*:
  * `score(h_t, h̄_s) = h_tᵀ W_a h̄_s`, `a_t = softmax(score)` over all source positions (padding masked)
  * `c_t = Σ_s a_t,s h̄_s` (context vector), `h̃_t = tanh(W_c [h_t ; c_t])`, `p(y_t) = softmax(W_o h̃_t)`

  This adds only `W_a` (256×256) and `W_c` (512×256), i.e. +196k parameters (+2.5 %).
  Input feeding was left out so that the whole target sequence runs through the LSTM in one call.
  That made training 2× faster on CPU.

### 2.2 Training configuration (`src/train.py`)
| Hyper-parameter | Value (both models) |
|---|---|
| Embedding size | 256 (source and target) |
| Hidden units | 256 (encoder: 2 × 128 bidirectional; decoder: 256) |
| Layers | 2 encoder + 2 decoder |
| Dropout | 0.3 (embeddings, between LSTM layers, before output) |
| Batch size | 128 sentence pairs, length-bucketed |
| Optimizer | Adam, learning rate 1e-3, ReduceLROnPlateau (×0.5, patience 0) on validation loss |
| Loss | Token-level cross-entropy with label smoothing 0.1, padding ignored (test loss reported **without** smoothing) |
| Gradient clipping | global norm 1.0 |
| Teacher forcing | 100 % during training |
| Epochs | 5 planned; stopped by a **150-minute budget** per model → 3 full epochs + 65–73 % of the 4th |
| Hardware | 4-core CPU, no GPU; both models trained in parallel with 2 threads each |
| Checkpointing | best validation-loss model saved as `models/{seq2seq,attention}.pt`; resumable training state |
| Decoding | greedy, max length 2·|src| + 10 |

![training curves](../results/figures/training_curves.png)

Both models were still improving when the budget ran out, so more epochs (or a GPU) would raise both
scores. The gap between the models, however, opens in epoch 1 and keeps growing.

---

## 3. Evaluation & comparison (`src/evaluate.py`)

Test set: 5,000 sentences, greedy decoding. BLEU and chrF are corpus-level scores from sacrebleu 2.6 on
normalized, punctuation-tokenized Amharic.

| Metric | Seq2Seq + LSTM | Attention Seq2Seq + LSTM |
|---|---:|---:|
| **BLEU** ↑ | 2.97 | **8.57** |
| **chrF** ↑ | 12.16 | **19.46** |
| **Test loss (CE)** ↓ | 3.974 | **3.548** |
| Test perplexity ↓ | 53.2 | **34.7** |
| **Parameters** | 7,995,200 | 8,191,808 |
| Model size | 30.5 MB | 31.3 MB |
| **Training time** | 150.1 min (4 epochs*) | 150.2 min (4 epochs*) |
| **Inference time**, whole test set (batch 100) | **9.7 s** | 16.8 s |
| Inference per sentence, batched | **1.9 ms** | 3.4 ms |
| Inference per sentence, single request (API setting) | **19.5 ms** | 22.1 ms |

\* the 4th epoch was partial (see §2.2).

**The attention model is clearly better:** +5.6 BLEU (2.9× the baseline) and +7.3 chrF, with a lower
test loss for almost the same number of parameters. The price is roughly 1.7× slower batched decoding,
because attention over all source states runs at every step. For single-sentence requests the difference
is only about 3 ms (22 ms vs 19.5 ms on CPU), which does not matter for an interactive application.

BLEU is low in absolute terms. That is expected for small LSTMs, trained for 2.5 CPU-hours, on a
morphologically rich target language where one wrong affix makes the whole word count as a BLEU miss.
chrF, which gives credit for partly correct words, shows the same ranking.

### 3.1 Translation examples
Source → Reference → Seq2Seq output → Attention-LSTM output (test set; full table with 25 rows in
[`results/examples.md`](../results/examples.md); all 5,000 outputs in `results/test_predictions.tsv`).

| Source (EN) | Reference (AM) | Seq2Seq + LSTM | Attention-LSTM |
|---|---|---|---|
| paul and other first-century christians learned this kind of love from the teachings of jesus. | (ለ) ኢየሱስ ያንጸባረቀውን አይነት ፍቅርና ትህትና ማሳየት ምን ያህል አስፈላጊ ነው? *(misaligned reference)* | ጳውሎስና ጳውሎስ ክርስቲያኖችን በተመለከተ ኢየሱስ ክርስቶስ በፊልጵስዩስ ክርስቲያኖች ላይ እምነት ነበራቸው። | ጳውሎስና ሌሎች በመጀመሪያው መቶ ዘመን ክርስቲያኖች ከኢየሱስ ትምህርቶች ጋር ፍቅር እንዳላቸው አሳይተዋል። |
| indeed, "his loving-kindness is to time indefinite." - psalm 100:5. | በእርግጥም "ምህረቱ … ለዘላለም" ነው። - መዝሙር 100: 5 | በእርግጥም "ይሆዋ" ታላቅ ሰው ነው።" - 1 ቆሮንቶስ 00: 10 | በእርግጥም "ፍቅራዊ ደግነትን ለዘላለም ነው።" - መዝሙር 100: 5 |
| six years later, my father died. | ከስድስት አመት በኋላ አባቴ ሞተ። | አባቴ አባቴን ወለድኩ። | ከሁለት አመት በኋላ አባቴ ሞተ። |
| songs: 100, 87 | መዝሙሮች፦ 100, 87 | መዝሙሮች፦ 10, 70 | መዝሙሮች፦ 100, 87 |
| what are some factors that promote this unity? | ለዚህ አንድነት አስተዋጽኦ ያደረጉት አንዳንድ ነገሮች ምንድን ናቸው? | ይህ ሲባል ምን ማለት ነው? | ይህን አንድነት ለማጠናከር አንዳንድ ምክንያቶች ምንድን ናቸው? |
| therefore, they asked him: "lord, teach us how to pray." | በዚህም የተነሳ "መጸለይን አስተምረን" ብለው ጠይቀውት ነበር። | ስለዚህ "እግዚአብሄር ሆይ፣ … " የሚለውን ቃል ጸልዩ። | ስለዚህ "ጌታ ሆይ፣ መጸለይ እንዴት እንደሚጸልዩ አስተምረን" ብለው ይመለሱ ነበር። |
| but in order for your children to find happiness, you also need to teach them to love god and to learn from him. | ሆኖም ልጆቻችሁ ደስተኞች እንዲሆኑ አምላክን እንዲወዱና እሱ የሚላቸውን ነገር እንዲሰሙ ማስተማርም ያስፈልጋችኋል። | ይሁን እንጂ ልጆቻችሁን ለአምላክና ፍቅርን በማዳመጥ ረገድ ጥሩ ምሳሌ ማግኘት ትችላለህ። | ይሁን እንጂ ልጆቻችሁ ደስታ ለማግኘትና ከእሱ መማር እንዲችሉ አምላክን እንዲያውቁ ለመርዳት ጥረት ማድረግ ይኖርብሀል። |
| peru has put great effort into reducing its maternal mortality rate. | በፔሩ የእናቶችን ሞት ለመቀነስ ከፍተኛ ጥረት እየተደረገ ነው። | ምስጢን በጭንት ላይ የሚሰነዘርበት ጊዜ ምስጢን ውሸት። | ሪፖርቱ ሪፖርት ሪፖርት ፕሬድ ሪፖርት … *(degenerate repetition)* |

New sentences (not from the corpus):

| Input | Attention-LSTM | Seq2Seq + LSTM |
|---|---|---|
| Jesus taught his disciples to love one another. | ኢየሱስ ደቀ መዛሙርቱ እርስ በርሳቸው እንዲወዱ አስተምሯቸዋል። ✔ | ኢየሱስ ደቀ መዛሙርቱን ፍቅር አሳይቷል። |
| We must read the Bible every day. | መጽሀፍ ቅዱስን በየእለቱ ማንበብ ይኖርብናል። ✔ | መጽሀፍ ቅዱስን ማጥናት ይኖርብናል። |
| that one is none other than jehovah god. | ይህ ሰው ከይሆዋ አምላክ ሌላ ሌላ አይደለም። | ይሆዋ አምላክ ነው። |
| I am going to the university. | እኔ ደግሞ በስሜት ቆየሁ። ✘ | እኔ ግን እኔ ነኝ። ✘ |
| My mother is cooking dinner for the family. | እናቴ ቤተሰብን ለመንከባከብ ፈቃደኛ ነው። ✘ | እናቴን ቤተሰቦቼን ወስደዋል። ✘ |

---

## 4. Error & attention analysis

### 4.1 Automatic error indicators (test set)
| Error type | Indicator | Seq2Seq | Attention |
|---|---|---:|---:|
| Missing words | outputs < 70 % of reference length | 26.6 % | **17.2 %** |
| Additional words | outputs > 130 % of reference length | **10.1 %** | 13.7 % |
| (length) | mean length ratio hyp/ref (ideal 1.0) | 0.90 | **1.01** |
| Repeated words | outputs with a repeated word or bigram | **18.9 %** | 24.1 % |
| Incorrect word order | mean order agreement of shared words (ideal 1.0) | 0.940 | **0.942** |
| | sentences with order agreement < 0.6 | 5.0 % | **4.4 %** |
| Incorrect morphology | output words with the right stem but the wrong inflection | **3.7 %** | 5.7 % |
| Named entities | 25 frequent names (Jehovah, Jesus, Moses, Israel, Egypt, …) translated correctly | 85.1 % | **89.4 %** |
| Numbers | all source numbers copied correctly | 38.3 % | **93.6 %** |
| Unknown / rare words | BLEU on sentences with a word seen < 5 times in training | 0.7 | **3.2** |
| | BLEU on the other sentences | 3.2 | **9.3** |

**Long-sentence errors: BLEU by source length**

| Source length | n | Seq2Seq | Attention |
|---|---:|---:|---:|
| 1–10 words | 1,066 | 7.0 | **15.5** |
| 11–15 words | 1,291 | 3.3 | **9.7** |
| 16–20 words | 1,416 | 2.6 | **7.3** |
| 21–25 words | 1,227 | 1.9 | **7.0** |

![BLEU by length](../results/figures/bleu_by_length.png)

Examples for each category are in [`results/error_analysis.md`](../results/error_analysis.md).

### 4.2 Discussion of errors
* **Long sentences.** The baseline loses 73 % of its short-sentence BLEU on 21–25-word sentences
  (7.0 → 1.9); the attention model loses 55 % (15.5 → 7.0). This is the fixed-length bottleneck:
  a 256-dim vector cannot hold a 25-word sentence, while attention can look back at any source word.
* **Missing words / under-translation.** The baseline tends to produce short, generic sentences, e.g.
  *"what are some factors that promote this unity?"* → *"ይህ ሲባል ምን ማለት ነው?"* ("What does this mean?").
  It forgets the content and keeps only the sentence type. 27 % of its outputs are too short.
  The attention model's length ratio is 1.01.
* **Repeated words / over-translation.** This is the main weakness of the attention model (24 % of outputs).
  Without a coverage mechanism, attention can return to the same source word several times
  (*"ይሆዋ፣ ይሆዋ፣ ይሆዋ ትህትናን፣ ትህትናን …"*). For unfamiliar input it can fall into a loop
  (*"ሪፖርቱ ሪፖርት ሪፖርት ፕሬድ …"* for the Peru sentence). Remedies: coverage/input feeding,
  beam search with a repetition penalty, and longer training.
* **Morphology.** 4–6 % of output words have the right stem but the wrong affix, e.g. *ምድርንን*
  (doubled object marker) or the wrong person or gender on the verb. The attention model's higher
  rate here partly reflects the fact that it produces more *correct stems* in the first place.
* **Word order.** Both models learned the SOV structure well: 94 % pair-wise order agreement on the
  words they get right. Order errors are not the main problem; lexical and morphological errors are.
* **Named entities & numbers.** Attention copies names and numbers far better. The baseline "remembers"
  that a verse reference exists but invents the numbers (*psalm 100:5 → 1 ቆሮንቶስ 00:10*,
  *songs 100, 87 → 10, 70*). The attention model attends straight to the digit tokens (93.6 % correct).
  Rare names (Klaus, Tychicus) are still often wrong.
* **Unknown / rare words and domain shift.** Sentences with rare words score far lower for both models.
  Everyday sentences outside the religious domain (*university, cooking dinner*) are pulled towards
  frequent corpus phrases: *"I am going to the university"* → *"እኔ ደግሞ በስሜት ቆየሁ"*.
  This is the most important practical limitation of the demo, and it comes from the training data,
  not from the architecture.
* **Noisy references.** Some test references are misaligned (row 1 above, or the Qur'an verses with
  bracketed glosses), which caps achievable BLEU for any model.

### 4.3 Attention visualisations
The heatmaps are in `results/figures/attention_*.png` (rows = generated Amharic subwords, columns =
English subwords, brighter = more weight).

![attention 2](../results/figures/attention_2.png)

**"Jesus taught his disciples to love one another."** → *ኢየሱስ ደቀ መዛሙርቱ እርስ በርሳቸው እንዲወዱ አስተምሯቸዋል።*
The alignment is clean and shows the reordering needed for **SOV** Amharic. *ኢየሱስ* attends to
*jesus*, and *ደቀ መዛሙርቱ* to *disciples*. *እርስ በርሳቸው* ("one another") attends to *one/another*,
and *እንዲወዱ* ("that they love") to *to love*. The main verb *አስተምሯቸዋል* ("he taught them") is
generated **last**, yet attends back to *taught* (the 2nd source word): attention has learned to jump
backwards for the sentence-final Amharic verb. The final `።` and `</s>` attend to the English period.

![attention 5](../results/figures/attention_5.png)

**"We must read the Bible every day."** → *መጽሀፍ ቅዱስን በየእለቱ ማንበብ ይኖርብናል።*
The order is almost the reverse of the English: *Bible → every day → read → must*. The object
*መጽሀፍ ቅዱስን* attends to *the bible*, *በየእለቱ* to *every day*, and *ማንበብ* to *read*. The modal
verb *ይኖርብናል* ("we must"), which Amharic puts at the end, attends sharply to *must*. The subject
*we* has no separate Amharic word. It shows up as the suffix *-ናል* in *ይኖርብናል*, which is why no row
attends to *we* strongly.

![attention 1](../results/figures/attention_1.png)

**"I am going to the university."** (a failure case) → *እኔ ደግሞ በስሜት ቆየሁ።*
*university* is rare in the corpus. The attention for the middle of the output is **diffuse**, spread
across *to / the / university / .*, and the model fills that part with a frequent phrase (*በስሜት*).
Even so, the verb's first-person suffix *ሁ* attends clearly to *i*, so the model still marks
subject agreement on the verb correctly. This shows the attention mechanism works, and the error is
a gap in vocabulary and domain.

---

## 5. Deployment & application

| Component | What it does |
|---|---|
| `src/translate.py` — `Translator` | inference pipeline: normalize → SentencePiece → encoder → greedy decoder → SentencePiece decode → Amharic detokenization |
| `app.py` — **FastAPI** | `POST /translate` `{"text": "...", "model": "attention" \| "seq2seq"}` → `{"translation": "...", "model": "..."}`; `POST /translate/details` (tokens, attention matrix, latency); `GET /health`; OpenAPI docs at `/docs`; input validation (empty text / > 500 characters → HTTP 400) |
| `app.py` — **Gradio** UI at `/` | text box, examples, both models' translations side by side, live attention heatmap |
| `Dockerfile` | self-contained image with the trained models, tokenizers, dependencies and inference code; runs on Hugging Face Spaces, Render, Railway, Fly.io or locally |
| `web/` — **public browser demo** | the models are exported to ONNX (`src/export_onnx.py`; embedding and output matrices int8-quantized, 13 MB per model) and run in the visitor's browser with ONNX Runtime Web. The tokenizer is re-implemented in JavaScript and verified to match Python on 3,009 + 1,200 test cases (`tests/test_web_parity.py`). Published to GitHub Pages by `.github/workflows/pages.yml` |

**Live demo:** https://esraprojects.github.io/deep_learning-amharic_translation-using-seq2seq/

Quantization check (300 test sentences): the attention model scores 6.92 BLEU in ONNX-int8 vs 7.01 in
PyTorch-fp32. For seq2seq the scores are identical (2.66). Without quantization, the ONNX graphs
reproduce the PyTorch outputs exactly (attention weights match to 3·10⁻⁸).

Example:
```bash
$ curl -X POST localhost:7860/translate -H "Content-Type: application/json" \
       -d '{"text": "We must read the Bible every day."}'
{"translation":"መጽሀፍ ቅዱስን በየእለቱ ማንበብ ይኖርብናል።","model":"attention"}
```

---

## 6. Conclusion and future work

With the same encoder, data, training budget and almost the same number of parameters, **adding
attention nearly triples BLEU (2.97 → 8.57) and raises chrF by 7.3 points**. The gains are largest
exactly where theory predicts: long sentences, faithful copying of numbers and names, and avoiding
under-translation. The remaining errors are mostly repetition, wrong Amharic inflections and
out-of-domain vocabulary.

Future work, most promising first:
1. Train longer or on a GPU (both curves were still falling).
2. Beam search with length normalization and a repetition penalty.
3. Coverage / input feeding against repetition.
4. Add more general-domain data (e.g. the news portion of `habtew/english-amharic-translation`, once its
   license is clarified).
5. A Transformer baseline for comparison.

## References
* Sutskever, Vinyals, Le (2014). *Sequence to Sequence Learning with Neural Networks.*
* Bahdanau, Cho, Bengio (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.*
* Luong, Pham, Manning (2015). *Effective Approaches to Attention-based Neural Machine Translation.*
* Kudo (2018). *Subword Regularization* / SentencePiece.
* Tiedemann (2012). *Parallel Data, Tools and Interfaces in OPUS.* · Gowda et al. (2021) *Many-to-English MT (MT560).*
* Post (2018). *A Call for Clarity in Reporting BLEU Scores* (sacrebleu); Popović (2015) *chrF.*
