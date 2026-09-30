"""English -> Amharic translation service.

FastAPI REST API + Gradio web UI in one process.

    uvicorn app:app --host 0.0.0.0 --port 7860
    # UI:   http://localhost:7860/
    # API:  curl -X POST http://localhost:7860/translate \
    #            -H "Content-Type: application/json" \
    #            -d '{"text": "I am going to the university."}'
    # Docs: http://localhost:7860/docs
"""
import os
import sys
from typing import Literal

import gradio as gr
import matplotlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from translate import Translator  # noqa: E402

FONT = os.path.join(ROOT, "assets", "NotoSansEthiopic.ttf")
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = [font_manager.FontProperties(fname=FONT).get_name(), "DejaVu Sans"]

TRANSLATORS = {k: Translator(k) for k in ("attention", "seq2seq")}
MAX_CHARS = 500

# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------
api = FastAPI(title="English → Amharic Seq2Seq Translator",
              description="Basic Seq2Seq-LSTM and Attention Seq2Seq-LSTM models trained on OPUS MT560.",
              version="1.0")


class TranslateRequest(BaseModel):
    text: str = Field(..., examples=["I am going to the university."])
    model: Literal["attention", "seq2seq"] = "attention"
    decoding: Literal["beam", "greedy"] = "beam"


class TranslateResponse(BaseModel):
    translation: str
    model: str


@api.get("/health")
def health():
    return {"status": "ok", "models": list(TRANSLATORS)}


@api.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text must not be empty")
    if len(text) > MAX_CHARS:
        raise HTTPException(400, f"text must be at most {MAX_CHARS} characters")
    return {"translation": TRANSLATORS[req.model].translate(text, decoding=req.decoding), "model": req.model}


@api.post("/translate/details")
def translate_details(req: TranslateRequest):
    """Translation plus tokens, attention weights (attention model) and latency."""
    if not req.text.strip():
        raise HTTPException(400, "text must not be empty")
    return TRANSLATORS[req.model].translate(req.text[:MAX_CHARS], return_details=True, decoding=req.decoding)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def attention_figure(d):
    A = d["attention"]
    fig, ax = plt.subplots(figsize=(max(5, .5 * len(d["source_tokens"]) + 2),
                                    max(3.5, .4 * len(d["target_tokens"]) + 1.5)))
    ax.imshow(A, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(d["source_tokens"])))
    ax.set_xticklabels(d["source_tokens"], rotation=70, ha="right", fontsize=9)
    ax.set_yticks(range(len(d["target_tokens"])))
    ax.set_yticklabels(d["target_tokens"], fontsize=10)
    ax.set_xlabel("English (source subwords)")
    ax.set_ylabel("Amharic (output subwords)")
    fig.tight_layout()
    return fig


def ui_translate(text):
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        return "", "", None, ""
    att = TRANSLATORS["attention"].translate(text, return_details=True)
    base = TRANSLATORS["seq2seq"].translate(text, return_details=True)
    info = (f"Preprocessed input: `{att['normalized_input']}` · decoding: beam search (k=5)  \n"
            f"Latency — attention: {att['latency_ms']} ms · seq2seq: {base['latency_ms']} ms (CPU)")
    fig = attention_figure(att)
    return att["translation"], base["translation"], fig, info


EXAMPLES = ["I am going to the university.", "God is love.",
            "The children are playing in the garden.", "Jesus taught his disciples to pray.",
            "We must read the Bible every day.", "Where is my brother?"]

with gr.Blocks(title="English → Amharic Translator") as demo:
    gr.Markdown("# English → Amharic Neural Machine Translation\n"
                "Seq2Seq-LSTM vs. Attention Seq2Seq-LSTM, trained on the OPUS MT560 corpus "
                "(mostly religious/Watchtower texts, so that domain translates best).")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="English sentence", lines=3, placeholder="Type an English sentence…")
            btn = gr.Button("Translate", variant="primary")
            gr.Examples(EXAMPLES, inp)
        with gr.Column():
            out_att = gr.Textbox(label="Attention Seq2Seq + LSTM (best model)", lines=2)
            out_base = gr.Textbox(label="Basic Seq2Seq + LSTM", lines=2)
            info = gr.Markdown()
    plot = gr.Plot(label="Attention weights (what the model looks at for each output token)")
    btn.click(ui_translate, inp, [out_att, out_base, plot, info])
    inp.submit(ui_translate, inp, [out_att, out_base, plot, info])

app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
