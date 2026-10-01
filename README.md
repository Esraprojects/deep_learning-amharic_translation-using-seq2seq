# English → Amharic Neural Machine Translation with Seq2Seq-LSTM and Attention

Deep Learning course project: develop, evaluate, compare and deploy an English→Amharic translation system
with a **basic Seq2Seq + LSTM** baseline and an **attention-based Seq2Seq + LSTM** model.

**▶ Live demo (runs in your browser, no install): https://esraprojects.github.io/deep_learning-amharic_translation-using-seq2seq/**

| | |
|---|---|
| Dataset | OPUS **MT560** English–Amharic ([`michsethowusu/english-amharic_sentence-pairs_mt560`](https://huggingface.co/datasets/michsethowusu/english-amharic_sentence-pairs_mt560), CC-BY-4.0) + [`habtew/english-amharic-translation`](https://huggingface.co/datasets/habtew/english-amharic-translation) (no license stated) · 517k training pairs |
| Models | Seq2Seq-LSTM (Sutskever et al. 2014) · Attention Seq2Seq-LSTM (Luong et al. 2015) · beam search decoding |
| Tokenization | SentencePiece unigram, 8k subwords per language |
| Deployment | FastAPI `POST /translate` + Gradio UI (`app.py`, Docker) · static web demo (ONNX Runtime Web, GitHub Pages) |
| Report | [`report/TECHNICAL_REPORT.md`](report/TECHNICAL_REPORT.md) · slides: [`report/presentation.pptx`](report/presentation.pptx) |

## Results (test set, 5,000 sentences, beam search)

| Metric | Seq2Seq + LSTM | **Attention Seq2Seq + LSTM** |
|---|---:|---:|
| BLEU ↑ | 5.26 | **11.35** |
| chrF ↑ | 13.68 | **23.13** |
| BLEU with greedy decoding | 4.91 | 10.36 |
| Test loss (CE) ↓ | 3.691 | **3.187** |
| Parameters | 7,995,200 | 8,191,808 |
| Training time (4-core CPU) | 420 min | 420 min |
| Inference per sentence (beam / greedy batched) | 33 / 2.0 ms | 36 / 2.22 ms |

**The attention model wins on every metric and in every sentence-length bucket.** Details:
[comparison & error analysis](results/comparison.md) · [translation examples](results/examples.md) ·
[error examples](results/error_analysis.md) · [full technical report](report/TECHNICAL_REPORT.md).

| Attention: "The children are playing in the garden." → ልጆቹ በአትክልት ቦታው ውስጥ እየተጫወቱ ነው። | BLEU by sentence length |
|---|---|
| ![attention](results/figures/attention_3.png) | ![length](results/figures/bleu_by_length.png) |

**Improvements over version 1** (attention model): adding the habtew corpus for everyday language,
training for 7 h instead of 2.5 h, and tuned beam search with repeat blocking raised BLEU from 8.57 to 11.27
and chrF from 19.46 to 22.84, and cut repeated-word outputs from 24 % to 8 %.

**Targeted fine-tuning (v3):** the error analysis found that many training pairs for everyday words are
wrong; for example, half of the "playing" pairs lack the Amharic verb, and "garden" is usually ገነት
("paradise"). `src/augment.py` writes 761 correct template pairs (`data/curated/play_garden.tsv`) for play,
cook, eat, read, write, work, study, run, *garden* and negation, and `src/finetune.py` fine-tunes the models on
them mixed with original data. Native speakers can edit or extend the TSV and re-run both scripts.

```bash
python src/augment.py && python src/finetune.py --model attention --upsample 10 --epochs 3
python src/finetune.py --model seq2seq --upsample 2 --epochs 1
```

> The training data is still mostly religious text, so that domain translates best; everyday sentences
> outside the curated patterns (e.g. "The weather is very cold today.") are still often wrong.

## Group members

| Name | ID |
|---|---|
| 1. Bethel Negusu | GSR/8221/18 |
| 2. Esrom Adugna | GSR/4064/18 |
| 3. Selamawit Siferh | GSR/6879/18 |
| 4. Aklilu Solomon | GSE/0756/18 |

## Repository layout

```
├── app.py                    FastAPI REST API + Gradio web UI (deployment)
├── Dockerfile                container for the API/UI (HF Spaces, Render, local …)
├── src/
│   ├── text.py               normalization & punctuation tokenization (EN + AM)
│   ├── preprocess.py         download, clean, de-duplicate, filter, split, train SentencePiece
│   ├── data.py               subword encoding, vocabulary, length-bucketed batching
│   ├── models.py             Seq2SeqLSTM and AttnSeq2SeqLSTM (+ greedy decoding)
│   ├── train.py              training loop (Adam, label-smoothed CE, LR schedule, checkpoints)
│   ├── evaluate.py           BLEU, chrF, loss, timing, #params, error & attention analysis
│   ├── translate.py          inference pipeline (used by the API/UI and CLI)
│   └── export_onnx.py        exports models/tokenizers for the browser demo
├── models/                   trained checkpoints (*.pt) and SentencePiece models
├── data/processed/           valid/test splits + dataset statistics (train split is regenerated)
├── results/                  metrics, comparison table, examples, error analysis, figures
├── web/                      static in-browser demo (GitHub Pages)
├── report/                   technical report + presentation
└── tests/                    JS ↔ Python tokenizer parity test
```

## Installation

Python 3.10+.

```bash
git clone https://github.com/Esraprojects/deep_learning-amharic_translation-using-seq2seq.git
cd deep_learning-amharic_translation-using-seq2seq
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # inference / app only
pip install -r requirements-train.txt    # + everything needed to retrain & evaluate
```

## Run the translation app (trained models are included)

```bash
python app.py                     # or: uvicorn app:app --host 0.0.0.0 --port 7860
```

* Web UI (Gradio): http://localhost:7860/
* Interactive API docs: http://localhost:7860/docs

```bash
curl -X POST http://localhost:7860/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "I am going to the university."}'
# {"translation": "እኔ ወደ ዩኒቨርሲቲ ሄድኩ።", "model": "attention"}
# optional fields: "model": "seq2seq", "decoding": "greedy" (default: attention + beam search)

# choose the baseline instead:
curl -X POST http://localhost:7860/translate -H "Content-Type: application/json" \
     -d '{"text": "God is love.", "model": "seq2seq"}'
```

`POST /translate/details` additionally returns the preprocessed input, subword tokens,
attention matrix and latency. Command line: `python src/translate.py "Where is my brother?"`.

With Docker: `docker build -t amharic-mt . && docker run -p 7860:7860 amharic-mt`.

## Reproduce the full pipeline

```bash
bash scripts/run_pipeline.sh              # everything below in one resumable command
python src/preprocess.py                  # ≈6 min: downloads both corpora, data/processed/*, models/spm_{en,am}.model
python src/train.py --model seq2seq   --emb 256 --hid 256 --layers 2 --batch_size 128 --epochs 12 --time_budget 420 --threads 2
python src/train.py --model attention --emb 256 --hid 256 --layers 2 --batch_size 128 --epochs 12 --time_budget 420 --threads 2
python src/evaluate.py                    # results/* and results/figures/*
python src/export_onnx.py                 # web/models/* for the browser demo
python tests/test_web_parity.py           # JS tokenizer == Python tokenizer
```

(The two training commands were run in parallel on a 4-core CPU without a GPU; with a GPU,
drop `--time_budget` and train longer for better scores.)

## Deployment

1. **Browser demo on GitHub Pages (live).** `web/` holds a static page that runs the exported
   ONNX models with ONNX Runtime Web, so translation happens on the visitor's device and no server
   is needed. The `Deploy web demo` workflow publishes it to the `gh-pages` branch on every push to `main`.
2. **FastAPI + Gradio server on Hugging Face Spaces (free).** Add a Hugging Face *write* token as
   the repository secret `HF_TOKEN` (Settings → Secrets and variables → Actions). The
   `Deploy to Hugging Face Space` workflow then builds the Docker Space
   `<hf-user>/amharic-seq2seq-translator`, which serves the same `POST /translate` API and Gradio UI publicly.
   The same `Dockerfile` also runs unchanged on Render, Railway or Fly.io.

## License

Code: MIT. OPUS MT560 is CC-BY-4.0 (please cite OPUS / MT560); habtew/english-amharic-translation states no license and is used here for coursework only.
Noto Sans Ethiopic font: SIL Open Font License.
