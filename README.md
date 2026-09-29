# English → Amharic Neural Machine Translation with Seq2Seq-LSTM and Attention

Deep Learning course project: develop, evaluate, compare and deploy an English→Amharic translation system
with a **basic Seq2Seq + LSTM** baseline and an **attention-based Seq2Seq + LSTM** model.

**▶ Live demo (runs in your browser, no install): https://esraprojects.github.io/deep_learning-amharic_translation-using-seq2seq/**

| | |
|---|---|
| Dataset | OPUS **MT560** English–Amharic ([HF: `michsethowusu/english-amharic_sentence-pairs_mt560`](https://huggingface.co/datasets/michsethowusu/english-amharic_sentence-pairs_mt560)), **CC-BY-4.0** |
| Models | Seq2Seq-LSTM (Sutskever et al. 2014) · Attention Seq2Seq-LSTM (Luong et al. 2015) |
| Tokenization | SentencePiece unigram, 8k subwords per language |
| Deployment | FastAPI `POST /translate` + Gradio UI (`app.py`, Docker) · static web demo (ONNX Runtime Web, GitHub Pages) |
| Report | [`report/TECHNICAL_REPORT.md`](report/TECHNICAL_REPORT.md) · slides: [`report/presentation.pptx`](report/presentation.pptx) |

<!-- RESULTS -->

## Group members

| Name | ID |
|---|---|
| _Member 1_ | |
| _Member 2_ | |
| _Member 3_ | |
| _Member 4_ | |

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
# {"translation": "…", "model": "attention"}

# choose the baseline instead:
curl -X POST http://localhost:7860/translate -H "Content-Type: application/json" \
     -d '{"text": "God is love.", "model": "seq2seq"}'
```

`POST /translate/details` additionally returns the preprocessed input, subword tokens,
attention matrix and latency. Command line: `python src/translate.py "Where is my brother?"`.

With Docker: `docker build -t amharic-mt . && docker run -p 7860:7860 amharic-mt`.

## Reproduce the full pipeline

```bash
python src/preprocess.py                  # ≈3 min: data/processed/*, models/spm_{en,am}.model
python src/train.py --model seq2seq   --emb 256 --hid 256 --layers 2 --batch_size 128 --epochs 5 --time_budget 170 --threads 2
python src/train.py --model attention --emb 256 --hid 256 --layers 2 --batch_size 128 --epochs 5 --time_budget 170 --threads 2
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

Code: MIT. The dataset (OPUS MT560) is CC-BY-4.0. Please cite OPUS / MT560 when you reuse the data.
Noto Sans Ethiopic font: SIL Open Font License.
