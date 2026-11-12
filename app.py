"""PredictiX: an N-gram language model lab built with Streamlit.

Course: Natural Language Processing (CS525PE), Unit V: Language Modeling.

Run locally:   streamlit run app.py
The language-model maths lives in ngram_engine.py; this file only builds the
user interface around it.
"""

import html
import json
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from corpora import CORPORA, DEFAULT_CORPUS
from ngram_engine import (EOS, ORDER_NAMES, NGramModel, context_words,
                          detokenize, finite, show, tokenize)

st.set_page_config(page_title="PredictiX", page_icon="\U0001F52E", layout="wide")

LABEL_TO_ORDER = {name: order for order, name in ORDER_NAMES.items()}
CUSTOM_CHOICE = "Your own text"
INK = "#2B2748"
VIOLET = "#7A68F0"
BLUSH_BAR = "#FFB3C7"
LILAC_BARS = ["#7A68F0", "#8F7FF3", "#A497F6", "#B6ABF8", "#C5BCFA", "#D3CCFB", "#DFDAFC", "#E9E5FD"]

esc = html.escape

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');
:root {
  --ink:#2B2748; --muted:#6F6A8F; --line:#E7E0FA; --accent:#7A68F0;
  --lilac:#E9E1FF; --blush:#FFE2EB; --sky:#DCEEFF; --mint:#DBF4E7; --butter:#FFF1C9; --peach:#FFE6D5;
}
.stApp { background:#FAF8FF; color:var(--ink); font-family:'Figtree',sans-serif; }
.block-container { max-width:1080px; padding-top:2.2rem; padding-bottom:3rem; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
h1, h2, h3 { font-family:'Bricolage Grotesque',sans-serif !important; color:var(--ink) !important; letter-spacing:-0.01em; }

.hero { display:grid; grid-template-columns:1.25fr 1fr; gap:28px; align-items:center;
  background:var(--lilac); border-radius:36px; padding:42px 44px; margin-bottom:22px; }
.hero h1 { font-size:2.7rem; line-height:1.08; font-weight:800; margin:0 0 14px 0; padding:0; }
.hero .lead { font-size:1.05rem; color:#4A4570; max-width:46ch; margin:0 0 18px 0; line-height:1.55; }
.hero .tags span { display:inline-block; background:rgba(255,255,255,.78); border-radius:999px;
  padding:5px 14px; margin:0 6px 6px 0; font-size:.85rem; font-weight:600; }
.kb { background:#fff; border-radius:28px; padding:18px 18px 16px; box-shadow:0 14px 34px rgba(80,60,160,.14); }
.kb .typed { font-size:1.2rem; font-weight:600; padding:6px 8px 14px; border-bottom:1.5px solid var(--line); }
.kb .caret { display:inline-block; width:2px; height:1.1em; background:var(--accent); margin-left:3px;
  vertical-align:-0.15em; animation:blink 1.1s steps(1) infinite; }
@keyframes blink { 50% { opacity:0; } }
.kb .keys { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:12px; }
.kb .key { border-radius:16px; padding:10px 6px; text-align:center; font-weight:700; }
.kb .key small { display:block; font-weight:500; font-size:.78rem; opacity:.7; }
.kb .k0 { background:var(--blush); } .kb .k1 { background:var(--butter); } .kb .k2 { background:var(--mint); }
.kb .kb-foot { font-size:.8rem; color:var(--muted); margin-top:10px; text-align:center; }
@media (max-width:800px) {
  .hero { grid-template-columns:1fr; padding:28px 22px; }
  .hero h1 { font-size:2rem; }
}
@media (prefers-reduced-motion:reduce) { .kb .caret { animation:none; } }

.status { background:#fff; border:1.5px solid var(--line); border-radius:999px; padding:9px 18px;
  display:inline-block; font-size:.92rem; margin-top:1.7rem; }
.sec h3 { margin:6px 0 2px 0; font-size:1.55rem; font-weight:700; }
.sec p { color:var(--muted); margin:0 0 14px 0; }
.tile { border-radius:22px; padding:16px 18px; margin-bottom:12px; }
.tile .lbl { font-size:.85rem; font-weight:600; color:#5A5580; }
.tile .val { font-family:'Bricolage Grotesque',sans-serif; font-size:1.7rem; font-weight:700;
  line-height:1.2; margin-top:2px; word-break:break-word; }
.tile .sub { font-size:.82rem; color:#5A5580; margin-top:2px; }
.t-lilac { background:var(--lilac); } .t-blush { background:var(--blush); } .t-sky { background:var(--sky); }
.t-mint { background:var(--mint); } .t-butter { background:var(--butter); } .t-peach { background:var(--peach); }
.paper { background:#fff; border:1.5px solid var(--line); border-radius:24px; padding:22px 26px;
  font-size:1.12rem; line-height:1.75; }
.note { background:var(--sky); border-radius:16px; padding:10px 14px; font-size:.92rem; margin:10px 0; }
.foot { text-align:center; color:var(--muted); font-size:.85rem; margin-top:36px; }

.stTabs [data-baseweb="tab-list"] { gap:6px; background:#fff; padding:6px; border-radius:999px;
  border:1.5px solid var(--line); flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { height:42px; border-radius:999px; padding:0 20px; background:transparent; }
.stTabs [data-baseweb="tab"] p { font-weight:600; margin:0; color:#5A5580; }
.stTabs [aria-selected="true"] { background:var(--lilac); }
.stTabs [aria-selected="true"] p { color:var(--ink); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

.stButton > button, .stDownloadButton > button { border-radius:999px; border:1.5px solid #D9CFFF;
  background:#fff; color:#4B3FBF; font-weight:600; padding:.35rem 1.2rem; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#F1ECFF; border-color:#A99CFF; color:#4B3FBF; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background:var(--accent); border-color:var(--accent); color:#fff; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background:#6755E0; border-color:#6755E0; color:#fff; }
div[data-baseweb="textarea"], div[data-baseweb="input"], div[data-baseweb="select"] > div {
  border-radius:16px !important; background-color:#fff !important; }
</style>
"""


# --------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------
def section(title, description):
    st.markdown(f'<div class="sec"><h3>{esc(title)}</h3><p>{esc(description)}</p></div>',
                unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{esc(text)}</div>', unsafe_allow_html=True)


def tile_html(label, value, tone="lilac", sub=""):
    sub_html = f'<div class="sub">{esc(str(sub))}</div>' if sub else ""
    return (f'<div class="tile t-{tone}"><div class="lbl">{esc(label)}</div>'
            f'<div class="val">{esc(str(value))}</div>{sub_html}</div>')


def tiles(items):
    """Show a row of pastel stat tiles. items = [(label, value, tone, sub), ...]"""
    for col, item in zip(st.columns(len(items)), items):
        col.markdown(tile_html(*item), unsafe_allow_html=True)


def style_fig(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, sans-serif", color=INK),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    fig.update_xaxes(gridcolor="#EEE9FF", zeroline=False)
    fig.update_yaxes(gridcolor="#EEE9FF", zeroline=False)
    return fig


def show_fig(fig):
    st.plotly_chart(fig, config={"displayModeBar": False})


def to_json(module, payload):
    return json.dumps({"module": module, **payload}, indent=2, ensure_ascii=False)


def fmt_ppl(value):
    if value is None:
        return "n/a"
    return "\u221e" if value == float("inf") else f"{value:,.2f}"


def fmt_prob(p):
    return "0" if p == 0 else f"{p:.4f}"


@st.cache_resource(show_spinner=False)
def get_model(text):
    """Build (and remember) a model for this exact training text."""
    return NGramModel(text)


# --------------------------------------------------------------------------
# Hero and corpus selection
# --------------------------------------------------------------------------
def render_hero():
    demo = get_model(CORPORA[DEFAULT_CORPUS]).predict(["the"], 3, 3)["items"]
    keys = "".join(
        f'<div class="key k{i}">{esc(show(w))}<small>{p:.0%}</small></div>'
        for i, (w, _count, p) in enumerate(demo)
    )
    st.markdown(
        '<div class="hero"><div>'
        '<h1>Type a few words. The model guesses what comes next.</h1>'
        '<p class="lead">PredictiX trains a small N-gram language model on a text you choose. '
        'Then you can predict words, generate sentences, score them and see how smoothing '
        'rescues unseen words.</p>'
        '<div class="tags"><span>Unigram</span><span>Bigram</span><span>Trigram</span>'
        '<span>Perplexity</span><span>Smoothing</span></div></div>'
        '<div class="kb"><div class="typed">The<span class="caret"></span></div>'
        f'<div class="keys">{keys}</div>'
        '<div class="kb-foot">Real trigram predictions from the campus text</div></div></div>',
        unsafe_allow_html=True,
    )


def choose_training_text():
    """Corpus picker. Returns (model, name of the text being used)."""
    left, right = st.columns([1, 1.7])
    with left:
        choice = st.selectbox("Training text", list(CORPORA) + [CUSTOM_CHOICE], key="corpus_choice")
    custom_box = st.container()

    text, used_name = CORPORA.get(choice, ""), choice
    if choice == CUSTOM_CHOICE:
        with custom_box:
            upload = st.file_uploader("Upload a .txt file", type=["txt"], key="corpus_upload")
            pasted = st.text_area("Or paste your own text here", height=140, key="corpus_paste",
                                  placeholder="Paste a few paragraphs. More text gives better predictions.")
            custom = upload.getvalue().decode("utf-8", errors="ignore") if upload is not None else pasted
            if len(tokenize(custom)) >= 30:
                text = custom
            else:
                note(f"Add at least 30 words to train on your own text. Using \u201c{DEFAULT_CORPUS}\u201d for now.")
                text, used_name = CORPORA[DEFAULT_CORPUS], DEFAULT_CORPUS

    model = get_model(text)
    s = model.stats()
    with right:
        st.markdown(
            f'<div class="status">Trained on <b>{s["sentences"]}</b> sentences, '
            f'<b>{s["tokens"]}</b> words and <b>{s["vocabulary"]}</b> different words</div>',
            unsafe_allow_html=True,
        )
    return model, used_name


# --------------------------------------------------------------------------
# Tab 1: next-word predictor
# --------------------------------------------------------------------------
def add_word(word):
    """Button callback: append the chosen suggestion to the text box."""
    text = st.session_state.get("pred_text", "")
    if word == EOS:
        stripped = text.rstrip()
        if stripped and stripped[-1] not in ".!?":
            text = stripped + ". "
    elif text.strip():
        text = text.rstrip() + " " + word + " "
    else:
        text = word + " "
    st.session_state["pred_text"] = text


def tab_predict(model):
    section("Next-word predictor",
            "Type a few words and the model suggests what comes next. Tap a suggestion to add it.")
    if "pred_text" not in st.session_state:
        st.session_state["pred_text"] = "the students "

    left, right = st.columns(2)
    with left:
        st.text_area("Your text", key="pred_text", height=120)
        order_label = st.radio("Model", list(ORDER_NAMES.values()), index=2, horizontal=True, key="pred_order")
        top_k = st.slider("Suggestions to show", 3, 8, 5, key="pred_k")

        order = LABEL_TO_ORDER[order_label]
        text = st.session_state["pred_text"]
        words = context_words(text)
        result = model.predict(words, order, top_k)
        items, used = result["items"], result["used_order"]

        st.caption("Tap a word to add it")
        for start in range(0, len(items), 4):
            row = st.columns(4)
            for col, (i, (w, _count, _p)) in zip(row, enumerate(items[start:start + 4], start=start)):
                col.button(show(w), key=f"suggest_{i}", on_click=add_word, args=(w,))

    with right:
        if used < order:
            note(f"This context never appeared in the training text, so the model backed off "
                 f"from {ORDER_NAMES[order].lower()} to {ORDER_NAMES[used].lower()}.")
        elif used == 1:
            note("A unigram model ignores the previous words and simply suggests the most common words.")
        else:
            ctx = model.context_label(words, used)
            note(f"The model looked at \u201c{ctx}\u201d, which appears {result['context_total']} times "
                 f"in the training text.")

        labels = [show(w) for w, _c, _p in items]
        probs = [p for _w, _c, p in items]
        fig = go.Figure(go.Bar(
            x=probs, y=labels, orientation="h",
            marker=dict(color=LILAC_BARS[:len(items)], line=dict(color="#CFC6FF", width=1)),
            text=[f"{p:.1%}" for p in probs], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickformat=".0%", range=[0, max(probs) * 1.3])
        show_fig(style_fig(fig, height=300))

    with st.expander("How this works"):
        st.markdown(
            "The model counts how often every word follows the words before it. "
            "For a trigram model it looks at the last two words and picks the words "
            "that followed them most often. If that pair was never seen, it **backs off** "
            "to the last word only (bigram), and finally to plain word frequency (unigram)."
        )
        st.latex(r"P(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}")

    st.download_button(
        "Download results (JSON)", key="download_predict", mime="application/json",
        file_name="predictix_next_word.json",
        data=to_json("next_word_predictor", {
            "input": text, "model": order_label, "model_used": ORDER_NAMES[used],
            "suggestions": [{"word": show(w), "count": c, "probability": finite(p)}
                            for w, c, p in items],
        }),
    )


# --------------------------------------------------------------------------
# Tab 2: text generator
# --------------------------------------------------------------------------
def reroll():
    st.session_state["gen_seed"] = random.randint(0, 999_999)


def tab_generate(model):
    section("Text generator",
            "The model writes new text one word at a time by sampling from what it has learned.")
    if "gen_seed" not in st.session_state:
        st.session_state["gen_seed"] = 11

    left, right = st.columns([1, 1.5])
    with left:
        start = st.text_input("Starting words", value="the students", key="gen_start")
        length = st.slider("Words to generate", 10, 80, 30, key="gen_len")
        order_label = st.radio("Model", list(ORDER_NAMES.values()), index=2, horizontal=True, key="gen_order")
        creativity = st.slider("Creativity", 0.3, 2.0, 1.0, 0.1, key="gen_temp",
                               help="Low values stick to the most likely words. High values pick rarer words more often.")
        st.button("Generate new text", type="primary", on_click=reroll, key="gen_button")

    order = LABEL_TO_ORDER[order_label]
    seed = st.session_state["gen_seed"]
    tokens = model.generate(tokenize(start), length, order, creativity, seed)
    output = detokenize(tokens)

    with right:
        st.markdown(f'<div class="paper">{esc(output)}</div>', unsafe_allow_html=True)
        st.write("")
        tiles([("Model", order_label, "sky", ""),
               ("Creativity", f"{creativity:.1f}", "butter", ""),
               ("Random seed", seed, "blush", "same seed, same text")])
    if order == 1:
        note("A unigram model has no memory of the previous word, so its sentences are random. "
             "Switch to bigram or trigram to see the text become more fluent.")

    with st.expander("How this works"):
        st.markdown(
            "At every step the model looks up the words that followed the current context, "
            "then picks one at random, weighted by how often it appeared. When it picks the "
            "end-of-sentence marker it writes a full stop and starts a new sentence. "
            "Creativity changes the weights: each count is raised to the power 1 / creativity."
        )
    st.download_button(
        "Download results (JSON)", key="download_generate", mime="application/json",
        file_name="predictix_generated_text.json",
        data=to_json("text_generator", {
            "starting_words": start, "model": order_label, "words_requested": length,
            "creativity": creativity, "seed": seed, "generated_text": output,
        }),
    )


# --------------------------------------------------------------------------
# Tab 3: perplexity
# --------------------------------------------------------------------------
def tab_perplexity(model):
    section("Perplexity check",
            "Score a sentence with each model. Lower perplexity means the model is less surprised by it.")
    left, right = st.columns([1.6, 1])
    with left:
        eval_text = st.text_area("Sentence to score", value="The students build small projects in the lab.",
                                 key="ppl_text", height=100)
    with right:
        k = st.slider("Smoothing strength (k)", 0.05, 1.0, 1.0, 0.05, key="ppl_k",
                      help="k = 1 is Laplace (add-one) smoothing. Smaller values trust the counts more.")

    results = {(n, s): model.evaluate(eval_text, n, k if s else 0.0) for n in (1, 2, 3) for s in (0, 1)}
    if results[(1, 0)] is None:
        note("Type a sentence above to see its perplexity.")
        return

    best = min((1, 2, 3), key=lambda n: results[(n, 1)]["perplexity"])
    unseen = results[(1, 0)]["unseen_words"]
    tiles([
        ("Best model with smoothing", ORDER_NAMES[best], "mint", f"perplexity {fmt_ppl(results[(best, 1)]['perplexity'])}"),
        ("Words not in the training text", len(unseen), "blush", ", ".join(unseen[:4]) if unseen else "none"),
        ("Words scored", results[(1, 0)]["tokens"], "sky", "including the sentence end"),
    ])

    table = pd.DataFrame([{
        "Model": ORDER_NAMES[n],
        "Without smoothing": fmt_ppl(results[(n, 0)]["perplexity"]),
        f"With smoothing (k = {k:g})": fmt_ppl(results[(n, 1)]["perplexity"]),
    } for n in (1, 2, 3)])
    st.dataframe(table, hide_index=True)

    def finite_or_none(n, s):
        v = results[(n, s)]["perplexity"]
        return None if v == float("inf") else v

    names = [ORDER_NAMES[n] for n in (1, 2, 3)]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Without smoothing", x=names, y=[finite_or_none(n, 0) for n in (1, 2, 3)],
                         marker_color=BLUSH_BAR))
    fig.add_trace(go.Bar(name="With smoothing", x=names, y=[finite_or_none(n, 1) for n in (1, 2, 3)],
                         marker_color=VIOLET))
    fig.update_layout(barmode="group")
    show_fig(style_fig(fig, height=300))
    st.caption("A missing bar means infinite perplexity: the model gave at least one word a probability of zero.")

    with st.expander("See the probability of each word"):
        c1, c2 = st.columns(2)
        detail_label = c1.radio("Model", list(ORDER_NAMES.values()), index=1, horizontal=True, key="ppl_detail_order")
        smoothed = c2.checkbox("Apply smoothing", value=True, key="ppl_detail_smooth")
        detail = results[(LABEL_TO_ORDER[detail_label], 1 if smoothed else 0)]
        st.dataframe(pd.DataFrame([{
            "Word": r["word"], "Context": r["context"], "Probability": fmt_prob(r["probability"]),
            "Seen in training": "yes" if r["count"] > 0 else "no",
        } for r in detail["rows"]]), hide_index=True)

    with st.expander("How this works"):
        st.markdown(
            "Perplexity is the model's average surprise per word. A model that predicts every "
            "word perfectly has perplexity 1. Without smoothing, one unseen word or n-gram gets "
            "probability 0 and the perplexity becomes infinite. Smoothing gives unseen events a "
            "small probability, so the score stays finite. On a small text a large k can "
            "over-smooth, so try lowering it."
        )
        st.latex(r"PP(W) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\ln P(w_i \mid \text{context})\right)")

    st.download_button(
        "Download results (JSON)", key="download_perplexity", mime="application/json",
        file_name="predictix_perplexity.json",
        data=to_json("perplexity_check", {
            "input": eval_text, "k": k, "words_scored": results[(1, 0)]["tokens"],
            "words_not_in_training_text": unseen,
            "perplexity": {ORDER_NAMES[n]: {"without_smoothing": finite(results[(n, 0)]["perplexity"]),
                                            "with_smoothing": finite(results[(n, 1)]["perplexity"])}
                           for n in (1, 2, 3)},
        }),
    )


# --------------------------------------------------------------------------
# Tab 4: smoothing lab
# --------------------------------------------------------------------------
def tab_smoothing(model):
    section("Smoothing lab",
            "See how maximum likelihood, Laplace and add-k smoothing estimate the chance that one word follows another.")
    c1, c2, c3 = st.columns(3)
    w1_raw = c1.text_input("Previous word", value="the", key="sm_w1")
    w2_raw = c2.text_input("Next word", value="students", key="sm_w2")
    k = c3.slider("Add-k value", 0.01, 1.0, 0.1, 0.01, key="sm_k")

    first, second = tokenize(w1_raw), tokenize(w2_raw)
    if not first or not second:
        note("Enter one word in each box to compare the estimates.")
        return
    w1, w2 = first[0], second[0]
    rep = model.bigram_report(w1, w2, k)

    tiles([
        (f"Times \u201c{w1}\u201d appears", rep["c_ctx"], "sky", ""),
        (f"Times \u201c{w1} {w2}\u201d appears", rep["c_pair"], "butter", "unseen" if rep["c_pair"] == 0 else ""),
        ("Vocabulary size V", rep["V"], "peach", "words + end marker + unknown"),
    ])
    tiles([
        ("Maximum likelihood", "undefined" if rep["mle"] is None else f"{rep['mle']:.4f}", "blush",
         "0 means the model thinks it is impossible" if rep["mle"] == 0 else ""),
        ("Laplace (k = 1)", f"{rep['laplace']:.4f}", "lilac", ""),
        (f"Add-k (k = {k:g})", f"{rep['add_k']:.4f}", "mint", ""),
    ])

    mle_line = (rf"\frac{{{rep['c_pair']}}}{{{rep['c_ctx']}}}" if rep["c_ctx"] else r"\text{undefined}")
    st.latex(r"P_{MLE}(w_2 \mid w_1) = \frac{C(w_1, w_2)}{C(w_1)} = " + mle_line)
    st.latex(rf"P_{{Laplace}}(w_2 \mid w_1) = \frac{{C(w_1, w_2) + 1}}{{C(w_1) + V}} = "
             rf"\frac{{{rep['c_pair']} + 1}}{{{rep['c_ctx']} + {rep['V']}}}")

    followers = sorted(model.next2.get(w1, {}).items(), key=lambda item: (-item[1], item[0]))[:6]
    shown = [w for w, _c in followers]
    if w2 not in shown:
        shown.append(w2)
    seen = model.next2.get(w1, {})
    mle_vals = [(seen.get(w, 0) / rep["c_ctx"]) if rep["c_ctx"] else 0.0 for w in shown]
    smooth_vals = [(seen.get(w, 0) + k) / (rep["c_ctx"] + k * rep["V"]) for w in shown]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Maximum likelihood", x=[show(w) for w in shown], y=mle_vals, marker_color=BLUSH_BAR))
    fig.add_trace(go.Bar(name=f"Add-k (k = {k:g})", x=[show(w) for w in shown], y=smooth_vals, marker_color=VIOLET))
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    show_fig(style_fig(fig, height=300))

    note(f"After smoothing, {rep['unseen_mass']:.1%} of the probability after \u201c{w1}\u201d is shared among "
         f"words that never followed it in training. Maximum likelihood gives them exactly 0%.")

    with st.expander("How this works"):
        st.markdown(
            "Maximum likelihood trusts the counts completely, so any word pair it has not seen "
            "gets probability 0. **Add-k smoothing** pretends every possible pair was seen k extra "
            "times. That takes a little probability from frequent pairs and shares it with unseen "
            "ones. Laplace smoothing is the special case k = 1."
        )
        st.latex(r"P_{add\text{-}k}(w_2 \mid w_1) = \frac{C(w_1, w_2) + k}{C(w_1) + kV}")

    st.download_button(
        "Download results (JSON)", key="download_smoothing", mime="application/json",
        file_name="predictix_smoothing.json",
        data=to_json("smoothing_lab", {
            "previous_word": w1, "next_word": w2, "k": k, "count_previous": rep["c_ctx"],
            "count_pair": rep["c_pair"], "vocabulary_size": rep["V"],
            "maximum_likelihood": finite(rep["mle"]), "laplace": finite(rep["laplace"], 6),
            "add_k": finite(rep["add_k"], 6), "mass_given_to_unseen_words": finite(rep["unseen_mass"]),
        }),
    )


# --------------------------------------------------------------------------
# Tab 5: corpus insights
# --------------------------------------------------------------------------
def tab_corpus(model, name):
    section("Corpus insights", f"What the model learned from \u201c{name}\u201d.")
    s = model.stats()
    tiles([("Sentences", s["sentences"], "lilac", f"about {s['avg_sentence_length']:.1f} words each"),
           ("Words in total", s["tokens"], "sky", ""),
           ("Different words", s["vocabulary"], "mint", "")])
    tiles([("Words seen only once", f"{s['seen_once_pct']:.0%}", "blush", "the reason smoothing is needed"),
           ("Different bigrams", s["unique_bigrams"], "butter", ""),
           ("Different trigrams", s["unique_trigrams"], "peach", "")])

    c1, c2 = st.columns([1.4, 1])
    order_label = c1.radio("N-gram type", list(ORDER_NAMES.values()), index=1, horizontal=True, key="corpus_order")
    top_n = c2.slider("How many to show", 5, 20, 10, key="corpus_top")
    top = model.top_ngrams(LABEL_TO_ORDER[order_label], top_n)

    fig = go.Figure(go.Bar(
        x=[c for _l, c in top], y=[label for label, _c in top], orientation="h",
        marker=dict(color="#A497F6"), text=[c for _l, c in top], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(c for _l, c in top) * 1.15])
    show_fig(style_fig(fig, height=max(260, 30 * len(top) + 60)))

    st.download_button(
        "Download results (JSON)", key="download_corpus", mime="application/json",
        file_name="predictix_corpus_insights.json",
        data=to_json("corpus_insights", {
            "corpus": name,
            "statistics": {key: finite(value) if isinstance(value, float) else value for key, value in s.items()},
            "top_ngrams": {"type": order_label, "items": [{"ngram": label, "count": c} for label, c in top]},
        }),
    )


# --------------------------------------------------------------------------
# Page layout
# --------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
render_hero()
model, corpus_name = choose_training_text()

tab_names = ["Predict", "Generate", "Perplexity", "Smoothing", "Corpus"]
tabs = st.tabs(tab_names)
with tabs[0]:
    tab_predict(model)
with tabs[1]:
    tab_generate(model)
with tabs[2]:
    tab_perplexity(model)
with tabs[3]:
    tab_smoothing(model)
with tabs[4]:
    tab_corpus(model, corpus_name)

st.markdown('<div class="foot">PredictiX is a learning project for Natural Language Processing (CS525PE), '
            'Unit V: Language Modeling. Built with Python, Streamlit and Plotly.</div>',
            unsafe_allow_html=True)
