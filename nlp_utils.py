"""
NLP utilities for Story Weaver.

keyword_extraction  — KeyBERT with a sentence-transformer (distilRoBERTa-based)
story_classification — GPT-4o-mini JSON mode (avoids loading a 500 MB+ NLI model
                       locally, which would exceed Streamlit Cloud's free-tier RAM)
"""

import re
import streamlit as st


# ── HTML helpers ──────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove all HTML tags — used to clean Quill editor output before AI calls."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


# ── Keyword extraction ────────────────────────────────────────────────────────
# Model: paraphrase-MiniLM-L6-v2  (~80 MB, distilRoBERTa-family embeddings)
# Swap to "paraphrase-distilroberta-base-v2" (~290 MB) for richer representations
# if you are running locally and want a purer RoBERTa model.

_KW_MODEL_NAME = "paraphrase-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading keyword model…")
def _kw_model():
    from keybert import KeyBERT
    return KeyBERT(model=_KW_MODEL_NAME)


def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Return [(keyword, relevance_score), …] sorted by score descending.
    Uses sentence-transformer embeddings to find the most semantically
    representative phrases in the text.
    """
    clean = strip_html(text)
    if len(clean.split()) < 5:
        return []
    return _kw_model().extract_keywords(
        clean,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_maxsum=True,
        nr_candidates=20,
        top_n=top_n,
    )


# ── Story classification ──────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
You are a literary analyst. Read the story excerpt below and return a JSON object
with exactly these fields — no extra keys, no markdown:

{{
  "predicted_genre":   "<Fantasy|Sci-Fi|Mystery|Romance|Horror|Comedy|Thriller|Drama>",
  "genre_confidence":  <float 0.0–1.0>,
  "tone":              "<dark|lighthearted|suspenseful|emotional|action-packed|mysterious|melancholic>",
  "tone_confidence":   <float 0.0–1.0>,
  "themes":            ["<theme1>", "<theme2>", "<theme3>"],
  "top_genres":        [["<genre>", <score>], ["<genre>", <score>], ["<genre>", <score>]]
}}

Story excerpt (last 2000 characters):
{text}"""


def classify_story(text: str) -> dict:
    """
    Classify genre, tone, and themes using GPT-4o-mini in JSON mode.
    Returns {} if the text is too short or if the call fails.
    """
    from llm_client import generate_json
    clean = strip_html(text)
    if len(clean.split()) < 20:
        return {}
    prompt = _CLASSIFY_PROMPT.format(text=clean[-2000:])
    result = generate_json(prompt)
    return result if isinstance(result, dict) else {}


# ── Keyword highlighting ───────────────────────────────────────────────────────

def highlight_keywords_html(text: str, keywords: list) -> str:
    """
    Return `text` as HTML with keyword occurrences wrapped in <mark> tags.
    Opacity of each highlight scales with the keyword's relevance score.
    Longer phrases are replaced first to avoid partial-match collisions.
    """
    if not keywords:
        return text

    sorted_kw = sorted(keywords, key=lambda x: len(x[0]), reverse=True)
    result = text
    for kw, score in sorted_kw:
        alpha = round(0.25 + min(score, 1.0) * 0.5, 2)
        style = (
            f"background:rgba(124,58,237,{alpha});"
            "border-radius:3px;padding:1px 4px;"
            "color:#1e1b4b;font-weight:600;"
        )
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        result = pattern.sub(
            lambda m, s=style: f'<mark style="{s}">{m.group(0)}</mark>',
            result,
        )
    return result
