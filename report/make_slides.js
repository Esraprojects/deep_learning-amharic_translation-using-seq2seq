// Builds report/presentation.pptx from results/metrics.json.
//   node report/make_slides.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.join(__dirname, "..");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "results/metrics.json"), "utf8"));
const S = JSON.parse(fs.readFileSync(path.join(ROOT, "data/processed/stats.json"), "utf8"));
const m = M.metrics, ea = M.error_analysis;
const fig = (f) => path.join(ROOT, "results/figures", f);

const C = { green: "1F5C4A", deep: "123A2F", mint: "DCEEE7", gold: "E0A526", ink: "1B1F24", muted: "5B6570", white: "FFFFFF", gray: "9AA3AE", light: "F4F7F6" };
const H = "Cambria", B = "Calibri", AM = "Nyala";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "English → Amharic NMT: Seq2Seq-LSTM vs Attention-LSTM";

function title(slide, text, sub) {
  slide.addText(text, { x: 0.6, y: 0.4, w: 12.1, h: 0.8, fontFace: H, fontSize: 34, bold: true, color: C.ink, margin: 0, isTextBox: true });
  if (sub) slide.addText(sub, { x: 0.6, y: 1.15, w: 12.1, h: 0.45, fontFace: B, fontSize: 16, color: C.muted, margin: 0, isTextBox: true });
}
function card(slide, x, y, w, h, fill = C.light) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
}
function bullets(slide, items, opts) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { fontFace: B, fontSize: 16, color: C.ink, paraSpaceAfter: 8, valign: "top", isTextBox: true, ...opts });
}
function stat(slide, x, y, w, value, label, color = C.green, labelColor = C.muted) {
  slide.addText(value, { x, y, w, h: 0.9, fontFace: H, fontSize: 44, bold: true, color, margin: 0, isTextBox: true });
  slide.addText(label, { x, y: y + 0.9, w, h: 0.6, fontFace: B, fontSize: 13, color: labelColor, margin: 0, valign: "top", isTextBox: true });
}
function num(n) { return n.toLocaleString("en-US"); }

// 1. Title -------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.deep };
  s.addText("English → Amharic", { x: 0.8, y: 1.6, w: 11.5, h: 1.0, fontFace: H, fontSize: 54, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText("Neural Machine Translation with Seq2Seq-LSTM and Attention", { x: 0.8, y: 2.6, w: 11.5, h: 0.7, fontFace: H, fontSize: 28, color: C.mint, margin: 0, isTextBox: true });
  s.addText("ወደ ዩኒቨርሲቲ እሄዳለሁ።", { x: 0.8, y: 3.5, w: 11.5, h: 0.7, fontFace: AM, fontSize: 30, color: C.gold, margin: 0, isTextBox: true });
  s.addText("Deep Learning Course Project  ·  Instructor: Fantahun Bogale Gereme", { x: 0.8, y: 4.45, w: 11.5, h: 0.4, fontFace: B, fontSize: 16, color: C.white, margin: 0, isTextBox: true });
  const members = [["Bethel Negusu", "GSR/8221/18"], ["Esrom Adugna", "GSR/4064/18"], ["Selamawit Siferh", "GSR/6879/18"], ["Aklilu Solomon", "GSE/0756/18"]];
  members.forEach(([n, id], i) => {
    const x = 0.8 + (i % 2) * 5.0, y = 4.95 + Math.floor(i / 2) * 0.45;
    s.addText([{ text: n, options: { bold: true, color: C.white } }, { text: "   " + id, options: { color: C.mint } }],
      { x, y, w: 4.8, h: 0.4, fontFace: B, fontSize: 15, margin: 0, isTextBox: true });
  });
  s.addText("Live demo: esraprojects.github.io/deep_learning-amharic_translation-using-seq2seq", { x: 0.8, y: 6.3, w: 11.5, h: 0.4, fontFace: B, fontSize: 14, color: C.gray, margin: 0, isTextBox: true });
  s.addNotes("Introduce the team and the goal: build, compare and deploy two LSTM translation models for English to Amharic.");
}

// 2. Objective & workflow -----------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "Objective and workflow", "Develop, evaluate, compare and deploy an English→Amharic translator");
  const steps = [["1", "Data", "OPUS MT560 + habtew, cleaning, normalization, split"], ["2", "Tokenize", "SentencePiece subwords, 8k per language"],
                 ["3", "Train", "Seq2Seq-LSTM and Attention-LSTM, same budget"], ["4", "Evaluate", "BLEU, chrF, loss, time, parameters"],
                 ["5", "Analyze", "error categories and attention maps"], ["6", "Deploy", "FastAPI + Gradio, live browser demo"]];
  steps.forEach(([n, h, d], i) => {
    const x = 0.6 + (i % 3) * 4.1, y = 2.0 + Math.floor(i / 3) * 2.45;
    card(s, x, y, 3.8, 2.1);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.6, h: 0.6, fill: { color: C.green }, line: { color: C.green } });
    s.addText(n, { x: x + 0.3, y: y + 0.3, w: 0.6, h: 0.6, fontFace: H, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(h, { x: x + 1.1, y: y + 0.3, w: 2.5, h: 0.6, fontFace: H, fontSize: 22, bold: true, color: C.ink, valign: "middle", margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.3, y: y + 1.1, w: 3.3, h: 0.8, fontFace: B, fontSize: 15, color: C.muted, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("The whole pipeline is reproducible with one script: scripts/run_pipeline.sh.");
}

// 3. Dataset -------------------------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "Dataset: MT560 + habtew", "HF: michsethowusu/english-amharic_sentence-pairs_mt560 (CC-BY-4.0)  ·  habtew/english-amharic-translation (no license stated)");
  stat(s, 0.6, 1.9, 3.0, num(S.raw.pairs), "raw sentence pairs");
  stat(s, 0.6, 3.5, 3.0, num(S.train.pairs), "training pairs after cleaning");
  stat(s, 0.6, 5.1, 3.0, "3k / 5k", "validation / test pairs");
  const log = S.cleaning_log;
  s.addChart(pres.charts.BAR, [{ name: "pairs", labels: ["Raw", "No empty / dup.", "Script filter", "Length / ratio", "De-dup. sources"],
    values: log.map((r) => r.pairs).filter((_, i) => i !== 1) }], {
    x: 4.0, y: 1.8, w: 5.0, h: 4.9, barDir: "bar", chartColors: [C.green], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "#,##0", dataLabelFontSize: 11, catAxisLabelFontSize: 12, valAxisHidden: true, valGridLine: { style: "none" },
    catAxisOrientation: "maxMin", showTitle: true, title: "Cleaning funnel", titleFontSize: 14, titleColor: C.ink, catAxisLabelColor: C.muted });
  bullets(s, ["MT560: religious (Bible, Qur'an, Watchtower); habtew adds news and everyday sentences", "Amharic has 3× more word types than English (rich morphology)",
              "SOV word order in Amharic vs SVO in English", "Removed misaligned, duplicate and wrong-script pairs; homophone normalization (ሐ/ኀ/ሃ→ሀ, ሠ→ሰ, ዐ→አ, ፀ→ጸ)"],
          { x: 9.3, y: 1.9, w: 3.5, h: 4.8, fontSize: 14 });
  s.addNotes("MT560 is the main corpus (documented license); habtew was added to cover everyday language. 88k habtew pairs remain after de-duplication.");
}

// 4. Models ----------------------------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "Two models, same encoder", "The only difference is the attention mechanism, so the comparison is fair");
  const cols = [["Basic Seq2Seq + LSTM", m.seq2seq.parameters, ["2-layer BiLSTM encoder (2×128)", "Final encoder state initializes a 2-layer LSTM decoder (256)",
      "Whole sentence squeezed into one fixed vector", "Sutskever et al., 2014"]],
    ["Attention Seq2Seq + LSTM", m.attention.parameters, ["Same encoder and decoder", "Luong global attention: score = hₜᵀ Wₐ h̄ₛ",
      "Context cₜ = Σ aₜₛ h̄ₛ;  h̃ₜ = tanh(W꜀[hₜ; cₜ])", "Looks back at every source word at each step"]]];
  cols.forEach(([h, p, items], i) => {
    const x = 0.6 + i * 6.2;
    card(s, x, 1.9, 5.9, 4.9, i ? C.mint : C.light);
    s.addText(h, { x: x + 0.35, y: 2.1, w: 5.2, h: 0.6, fontFace: H, fontSize: 24, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(num(p) + " parameters", { x: x + 0.35, y: 2.7, w: 5.2, h: 0.45, fontFace: B, fontSize: 15, color: C.green, bold: true, margin: 0, isTextBox: true });
    bullets(s, items, { x: x + 0.35, y: 3.35, w: 5.2, h: 3.2 });
  });
  s.addNotes("Attention adds only 196k parameters (+2.5%).");
}

// 5. Training config ----------------------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "Training configuration", "Identical for both models; 4-core CPU, no GPU");
  const rows = [["Embedding size", "256"], ["Hidden units", "256 (encoder 2×128 bidirectional)"], ["Layers", "2 encoder + 2 decoder"],
    ["Batch size", "128 (length-bucketed)"], ["Learning rate", "1e-3, halved on validation plateau"], ["Optimizer", "Adam, gradient clip 1.0"],
    ["Epochs", `${m.attention.epochs_trained} (7-hour budget per model)`], ["Loss", "Cross-entropy, label smoothing 0.1"], ["Dropout", "0.3"], ["Vocabulary", "SentencePiece unigram, 8k EN + 8k AM"]];
  s.addTable([[{ text: "Hyper-parameter", options: { bold: true, color: C.white, fill: { color: C.green } } }, { text: "Value", options: { bold: true, color: C.white, fill: { color: C.green } } }],
    ...rows.map(([a, b], i) => [{ text: a, options: { bold: true, fill: { color: i % 2 ? C.white : C.light } } }, { text: b, options: { fill: { color: i % 2 ? C.white : C.light } } }])],
    { x: 0.6, y: 1.85, w: 6.0, colW: [2.3, 3.7], fontFace: B, fontSize: 14, color: C.ink, rowH: 0.43, border: { type: "none" } });
  s.addImage({ path: fig("training_curves.png"), x: 6.9, y: 2.6, w: 5.9, h: 5.9 * 520 / 1950 });
  s.addText("The gap between the models opens in epoch 1 and keeps growing; both were still slowly improving.", { x: 6.9, y: 5.1, w: 5.9, h: 0.8, fontFace: B, fontSize: 14, italic: true, color: C.muted, margin: 0, isTextBox: true });
  s.addNotes("Training was resumable, with checkpoints every 250 steps.");
}

// 6. Results ------------------------------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "Results: attention wins clearly", "Test set, 5,000 sentences, beam search (k=5)");
  stat(s, 0.6, 1.9, 2.8, String(m.attention.bleu), `BLEU with attention (baseline ${m.seq2seq.bleu})`);
  stat(s, 0.6, 3.5, 2.8, String(m.attention.chrf), `chrF with attention (baseline ${m.seq2seq.chrf})`);
  stat(s, 0.6, 5.1, 2.8, String(m.attention.test_loss.toFixed(2)), `test loss with attention (baseline ${m.seq2seq.test_loss.toFixed(2)})`);
  s.addChart(pres.charts.BAR, [
    { name: "Seq2Seq + LSTM", labels: ["BLEU", "chrF"], values: [m.seq2seq.bleu, m.seq2seq.chrf] },
    { name: "Attention-LSTM", labels: ["BLEU", "chrF"], values: [m.attention.bleu, m.attention.chrf] }], {
    x: 3.7, y: 1.8, w: 4.4, h: 4.9, barDir: "col", chartColors: [C.gray, C.green], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFontSize: 12, dataLabelFormatCode: "0.0", showLegend: true, legendPos: "b", legendFontSize: 12, valAxisHidden: true, valGridLine: { style: "none" },
    catAxisLabelFontSize: 14, catAxisLabelColor: C.ink });
  const t = [["", "Seq2Seq", "Attention"], ["Parameters", num(m.seq2seq.parameters), num(m.attention.parameters)],
    ["Training time", `${m.seq2seq.training_time_min} min`, `${m.attention.training_time_min} min`],
    ["Test perplexity", String(m.seq2seq.test_perplexity), String(m.attention.test_perplexity)],
    ["Beam / greedy BLEU", `${m.seq2seq.bleu} / ${m.seq2seq.bleu_greedy}`, `${m.attention.bleu} / ${m.attention.bleu_greedy}`],
    ["Inference / sentence", `${m.seq2seq.inference_ms_per_sentence_single} ms`, `${m.attention.inference_ms_per_sentence_single} ms`],
    ["Test set, beam", `${m.seq2seq.inference_time_test_set_sec} s`, `${m.attention.inference_time_test_set_sec} s`]];
  s.addTable(t.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, align: j ? "right" : "left", fill: { color: i === 0 ? C.mint : C.white } } }))),
    { x: 8.4, y: 2.0, w: 4.6, colW: [1.7, 1.45, 1.45], fontFace: B, fontSize: 13, color: C.ink, rowH: 0.5, border: { type: "solid", pt: 0.5, color: "D9DEE3" } });
  s.addText(`+${(m.attention.bleu - m.seq2seq.bleu).toFixed(1)} BLEU for +2.5% parameters. Beam search adds +${(m.attention.bleu - m.attention.bleu_greedy).toFixed(1)} BLEU over greedy decoding.`, { x: 8.4, y: 5.95, w: 4.6, h: 0.9, fontFace: B, fontSize: 14, italic: true, color: C.green, margin: 0, isTextBox: true });
}

// 7. Examples --------------------------------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "Translation examples", "Source → Seq2Seq → Attention-LSTM (beam search); full table with references in results/examples.md");
  const AX = JSON.parse(fs.readFileSync(path.join(ROOT, "results/attention_examples.json"), "utf8"));
  const pick = [0, 1, 4, 3, 7].map((i) => AX[i]);
  const ex = pick.map((d) => [d.source, d.seq2seq, d.translation]);
  const hdr = ["Source (EN)", "Seq2Seq + LSTM", "Attention-LSTM"].map((t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.green }, fontFace: B } }));
  s.addTable([hdr, ...ex.map((r, i) => r.map((c, j) => ({ text: c, options: { fontFace: j ? AM : B, fontSize: j ? 15 : 13, fill: { color: i % 2 ? C.white : C.light } } })))],
    { x: 0.6, y: 1.8, w: 12.1, colW: [4.1, 4.0, 4.0], color: C.ink, rowH: 0.8, valign: "middle", border: { type: "none" } });
  s.addText("With the added habtew data, everyday sentences like “university” now work; long rare sentences remain hard.", { x: 0.6, y: 6.7, w: 12.1, h: 0.4, fontFace: B, fontSize: 13, italic: true, color: C.muted, margin: 0, isTextBox: true });
}

// 8. Error analysis ----------------------------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "Error analysis", "Automatic indicators on the test set");
  const L = ea.seq2seq.bleu_by_length.map((b) => b.bucket + " words");
  s.addChart(pres.charts.BAR, [
    { name: "Seq2Seq + LSTM", labels: L, values: ea.seq2seq.bleu_by_length.map((b) => b.bleu) },
    { name: "Attention-LSTM", labels: L, values: ea.attention.bleu_by_length.map((b) => b.bleu) }], {
    x: 0.6, y: 1.8, w: 5.6, h: 4.9, barDir: "col", chartColors: [C.gray, C.green], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFormatCode: "0.0",
    showLegend: true, legendPos: "b", legendFontSize: 12, valAxisHidden: true, valGridLine: { style: "none" }, catAxisLabelFontSize: 12,
    showTitle: true, title: "BLEU by source length (long-sentence errors)", titleFontSize: 14, titleColor: C.ink });
  const rows = [["Missing words (output too short)", ea.seq2seq.too_short_missing_words_pct + "%", ea.attention.too_short_missing_words_pct + "%"],
    ["Additional words (output too long)", ea.seq2seq.too_long_additional_words_pct + "%", ea.attention.too_long_additional_words_pct + "%"],
    ["Repeated words", ea.seq2seq.repeated_words_pct + "%", ea.attention.repeated_words_pct + "%"],
    ["Word-order agreement", String(ea.seq2seq.word_order_agreement), String(ea.attention.word_order_agreement)],
    ["Morphology (right stem, wrong affix)", ea.seq2seq.morphology_error_rate_pct + "%", ea.attention.morphology_error_rate_pct + "%"],
    ["Named entities correct", ea.seq2seq.named_entity_accuracy_pct + "%", ea.attention.named_entity_accuracy_pct + "%"],
    ["Numbers copied correctly", ea.seq2seq.number_copy_accuracy_pct + "%", ea.attention.number_copy_accuracy_pct + "%"],
    ["BLEU, sentences with rare words", String(ea.seq2seq.bleu_rare_word_sentences), String(ea.attention.bleu_rare_word_sentences)]];
  s.addTable([["Error type", "Seq2Seq", "Attention"].map((t) => ({ text: t, options: { bold: true, fill: { color: C.mint } } })),
    ...rows.map((r) => r.map((c, j) => ({ text: c, options: { align: j ? "right" : "left" } })))],
    { x: 6.6, y: 1.9, w: 6.2, colW: [3.8, 1.2, 1.2], fontFace: B, fontSize: 13, color: C.ink, rowH: 0.46, border: { type: "solid", pt: 0.5, color: "D9DEE3" } });
  s.addText("Attention fixes under-translation, long sentences, names and numbers. Its main weakness is repetition (no coverage mechanism).",
    { x: 6.6, y: 6.1, w: 6.2, h: 0.8, fontFace: B, fontSize: 13, italic: true, color: C.green, margin: 0, isTextBox: true });
}

// 9. Attention ----------------------------------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "What the attention model looks at", "Rows = generated Amharic subwords, columns = English source subwords");
  s.addImage({ path: fig("attention_2.png"), x: 0.6, y: 1.8, w: 5.0, h: 5.0 * 803 / 786 });
  s.addImage({ path: fig("attention_5.png"), x: 5.9, y: 1.9, w: 3.6, h: 3.6 * 605 / 728 });
  bullets(s, ["Learns the SOV reordering: the final verb አስተምሯቸዋል attends back to “taught”",
              "“We must read the Bible every day” → Amharic order: every day, Bible, read, must",
              "The subject “we” becomes the verb suffix -ናል, so it has no attention row of its own",
              "“I am going to the university” → ወደ ዩኒቨርሲቲው ሄድኩ: ዩኒቨርሲቲ attends to “university”, the suffix -ኩ to “I”"],
          { x: 9.7, y: 1.9, w: 3.2, h: 5.0, fontSize: 14 });
}

// 10. Deployment ----------------------------------------------------------------------------
{
  const s = pres.addSlide();
  title(s, "Deployment", "Input → preprocess → encoder → attention decoder → Amharic output");
  const items = [["REST API (FastAPI)", "POST /translate\n{\"text\": \"I am going to the university.\"}\n→ {\"translation\": \"…\"}"],
    ["Web UI (Gradio)", "Both models side by side, example sentences, live attention heatmap"],
    ["Docker", "Trained models, tokenizers, dependencies and inference pipeline in one image"],
    ["Live browser demo", "ONNX Runtime Web on GitHub Pages; the model runs on the visitor's device, free and always online"]];
  items.forEach(([h, d], i) => {
    const x = 0.6 + (i % 2) * 6.2, y = 1.9 + Math.floor(i / 2) * 2.5;
    card(s, x, y, 5.9, 2.2, i === 3 ? C.mint : C.light);
    s.addText(h, { x: x + 0.35, y: y + 0.25, w: 5.2, h: 0.55, fontFace: H, fontSize: 22, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(d, { x: x + 0.35, y: y + 0.85, w: 5.2, h: 1.2, fontFace: i === 0 ? "Courier New" : B, fontSize: i === 0 ? 13 : 15, color: C.muted, margin: 0, valign: "top", isTextBox: true });
  });
  s.addText("esraprojects.github.io/deep_learning-amharic_translation-using-seq2seq", { x: 0.6, y: 6.85, w: 12.1, h: 0.4, fontFace: B, fontSize: 15, bold: true, color: C.green, margin: 0, isTextBox: true });
}

// 11. Conclusion ----------------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.background = { color: C.deep };
  s.addText("Conclusion", { x: 0.8, y: 0.6, w: 11.5, h: 0.9, fontFace: H, fontSize: 40, bold: true, color: C.white, margin: 0, isTextBox: true });
  stat(s, 0.8, 1.8, 3.6, `${(m.attention.bleu / m.seq2seq.bleu).toFixed(1)}×`, "BLEU of the attention model vs the baseline", C.gold, C.mint);
  stat(s, 4.8, 1.8, 3.6, "+2.5%", "extra parameters for attention", C.gold, C.mint);
  stat(s, 8.8, 1.8, 3.6, `${ea.attention.number_copy_accuracy_pct}%`, "numbers copied correctly (baseline " + ea.seq2seq.number_copy_accuracy_pct + "%)", C.gold, C.mint);
  s.addText([
    { text: "Future work: ", options: { bold: true, color: C.gold } },
    { text: "GPU training for more epochs · coverage / input feeding against repetition · more general-domain data · Transformer baseline", options: { color: C.white } }],
    { x: 0.8, y: 4.4, w: 11.5, h: 1.0, fontFace: B, fontSize: 18, margin: 0, isTextBox: true });
  s.addText("Live demo  ·  esraprojects.github.io/deep_learning-amharic_translation-using-seq2seq", { x: 0.8, y: 6.2, w: 11.5, h: 0.5, fontFace: B, fontSize: 16, color: C.mint, margin: 0, isTextBox: true });
}

pres.writeFile({ fileName: path.join(__dirname, "presentation.pptx") }).then((f) => console.log("wrote", f));
