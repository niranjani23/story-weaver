"""
NarrateAI — AI-powered collaborative storytelling app.
Run with:  python -m streamlit run app.py
"""

import html
import re

import streamlit as st
from streamlit_quill import st_quill

from llm_client import RateLimitError, generate_json, generate_text
from nlp_utils import (
    classify_story,
    extract_keywords,
    highlight_keywords_html,
    strip_html,
)
from prompts import (
    GENRE_RULES,
    build_branch_prompt,
    build_character_prompt,
    build_choices_prompt,
    build_continue_prompt,
    build_opening_prompt,
    build_remix_prompt,
    build_visualization_prompt,
)

GENRES = list(GENRE_RULES.keys())

GENRE_COLORS = {
    "Fantasy": "#7c3aed",
    "Sci-Fi":  "#0891b2",
    "Mystery": "#b45309",
    "Romance": "#db2777",
    "Horror":  "#dc2626",
    "Comedy":  "#16a34a",
}

_CHAR_REFRESH_EVERY = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def creativity_label(temp: float) -> str:
    if temp <= 0.30: return "Conservative"
    if temp <= 0.55: return "Balanced"
    if temp <= 0.80: return "Creative"
    return "Wildly Creative"

def creativity_desc(temp: float) -> str:
    if temp <= 0.30: return "Grounded, predictable prose"
    if temp <= 0.55: return "Natural story flow"
    if temp <= 0.80: return "Surprising turns ahead"
    return "Bold, experimental writing"

def word_count(segments: list) -> int:
    return sum(len(s["text"].split()) for s in segments)

def parse_choices(text: str) -> list:
    matches = re.findall(r"^\d+\.\s+(.+)$", text, re.MULTILINE)
    return [m.strip() for m in matches[:3]]

def maybe_refresh_characters(force: bool = False) -> None:
    if not st.session_state.story_segments:
        return
    if not force and st.session_state.ai_turn_count % _CHAR_REFRESH_EVERY != 0:
        return
    try:
        result = generate_json(build_character_prompt(st.session_state.story_segments))
        if isinstance(result, dict) and result:
            st.session_state.characters = result
    except Exception:
        pass

def full_story_text() -> str:
    """Plain text of the entire story for NLP analysis."""
    return " ".join(s["text"] for s in st.session_state.story_segments)

def init_state() -> None:
    defaults = {
        "started":          False,
        "title":            "",
        "genre":            GENRES[0],
        "story_segments":   [],
        "characters":       {},
        "ai_turn_count":    0,
        "pending_choices":  [],
        "last_remix_genre": "",
        "remix_result":     "",
        "viz_prompt":       "",
        "temperature":      0.70,
        # NLP
        "keywords":         [],
        "classification":   {},
        "show_highlights":  False,
        "quill_counter":    0,   # increment to reset the Quill editor
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Story display ─────────────────────────────────────────────────────────────

def render_story() -> None:
    segs = st.session_state.story_segments
    kws  = st.session_state.keywords if st.session_state.show_highlights else []

    if not segs:
        st.markdown(
            "<div style='text-align:center;padding:3em 1em;color:#9ca3af;"
            "border:2px dashed #e5e7eb;border-radius:10px;'>"
            "Your story will appear here.</div>",
            unsafe_allow_html=True,
        )
        return

    parts = []
    for seg in segs:
        # HTML-escape, then optionally apply keyword highlights
        safe = html.escape(seg["text"]).replace("\n", "<br>")
        if kws:
            safe = highlight_keywords_html(safe, kws)

        if seg["author"] == "ai":
            badge = ("<span style='font-size:0.7rem;font-weight:700;color:#7c3aed;"
                     "text-transform:uppercase;letter-spacing:0.07em;'>AI</span>")
            box   = ("margin-bottom:1.1em;padding:1em 1.2em;"
                     "background:#f5f3ff;border-left:4px solid #7c3aed;"
                     "border-radius:6px;line-height:1.85;font-size:0.98rem;color:#1e1b4b;"
                     "font-family:Georgia,'Times New Roman',serif;")
        else:
            badge = ("<span style='font-size:0.7rem;font-weight:700;color:#d97706;"
                     "text-transform:uppercase;letter-spacing:0.07em;'>You</span>")
            box   = ("margin-bottom:1.1em;padding:1em 1.2em;"
                     "background:#fffbeb;border-left:4px solid #f59e0b;"
                     "border-radius:6px;line-height:1.85;font-size:0.98rem;color:#1c1917;"
                     "font-style:italic;font-family:Georgia,'Times New Roman',serif;")
        parts.append(f'<div style="{box}">{badge}<br><br>{safe}</div>')

    st.markdown(
        "<div style='max-height:520px;overflow-y:auto;padding:1em;"
        "background:#fafafa;border-radius:10px;"
        "border:1px solid #e5e7eb;'>"
        + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        genre = st.session_state.genre
        color = GENRE_COLORS.get(genre, "#6c63ff")

        st.markdown(f"## {html.escape(st.session_state.title)}")
        st.markdown(
            f"<span style='display:inline-block;padding:3px 12px;border-radius:20px;"
            f"background:{color};color:#fff;font-size:0.8rem;font-weight:600;'>{genre}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Story Rules**")
        st.info(GENRE_RULES[genre])

        st.markdown("---")
        st.markdown("**Creativity Level**")
        st.session_state.temperature = st.slider(
            "temp", 0.0, 1.0,
            value=st.session_state.temperature,
            step=0.05, label_visibility="collapsed",
        )
        st.caption(f"{creativity_label(st.session_state.temperature)} — {creativity_desc(st.session_state.temperature)}")

        st.markdown("---")
        st.markdown("**Live Character Tracker**")
        chars = st.session_state.characters
        if chars:
            for name, desc in chars.items():
                st.markdown(f"**{html.escape(name)}** — {html.escape(desc)}")
        else:
            st.caption("Characters appear here once introduced.")
        if st.button("🔄 Refresh Characters", use_container_width=True):
            with st.spinner("Extracting…"):
                maybe_refresh_characters(force=True)
            st.rerun()

        st.markdown("---")
        segs   = st.session_state.story_segments
        ai_n   = sum(1 for s in segs if s["author"] == "ai")
        user_n = sum(1 for s in segs if s["author"] == "user")
        st.caption(f"📝 {word_count(segs)} words · AI ×{ai_n} · You ×{user_n}")

        st.markdown("---")
        if st.button("↩ Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ── Setup screen ──────────────────────────────────────────────────────────────

def setup_screen() -> None:
    # ── Hero ──────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='text-align:center;color:#7c3aed;margin-bottom:0.1em;'>📖 NarrateAI</h1>"
        "<p style='text-align:center;font-size:1.1rem;color:#6b7280;margin-top:0;'>"
        "Your AI-powered collaborative storytelling companion</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────────────────
    st.markdown("### How it works")
    step1, step2, step3, step4 = st.columns(4)
    for col, icon, title_t, body in [
        (step1, "1️⃣", "Set the scene",
         "Give your story a title, pick a genre, and describe the opening situation."),
        (step2, "2️⃣", "AI writes the opening",
         "Click **Start the Story** and the AI generates a vivid 150–250 word opening paragraph."),
        (step3, "3️⃣", "Co-write together",
         "Add your own lines in the rich-text editor, then let the AI continue — or pick from 3 branching paths."),
        (step4, "4️⃣", "Explore & analyse",
         "Remix into a new genre, generate a DALL·E image prompt, or analyse keywords and tone."),
    ]:
        with col:
            st.markdown(
                f"<div style='background:#f5f3ff;border-radius:10px;padding:1em;text-align:center;height:160px;'>"
                f"<div style='font-size:1.8rem;'>{icon}</div>"
                f"<strong style='color:#5b21b6;'>{title_t}</strong>"
                f"<p style='font-size:0.83rem;color:#4b5563;margin-top:0.4em;'>{body}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature overview ──────────────────────────────────────────────
    with st.expander("✨ See all features", expanded=False):
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("""
**Writing tools**
- 🖊️ **Rich text editor** — bold, italic, lists, and more
- ▶️ **Continue with AI** — appends 1–2 coherent paragraphs
- 🔀 **Give Me Choices** — 3 branching paths, you pick one
- 📏 **Creativity slider** — from conservative prose to wildly experimental
""")
        with f2:
            st.markdown("""
**Analysis & extras**
- 👥 **Live Character Tracker** — auto-extracts characters as they appear
- 🎭 **Genre Remix** — rewrite any section in a completely different genre
- 🎨 **Visualization Prompt** — generate a DALL·E / Flux / Midjourney prompt
- 🔍 **Story Analysis** — keyword highlighting + genre & tone classification
""")

    # ── Tips ──────────────────────────────────────────────────────────
    with st.expander("💡 Tips for a great story", expanded=False):
        st.markdown("""
- **Be specific in your hook** — "A detective finds a glowing letter on a rainy night" works better than "A detective investigates something strange."
- **Use the Creativity slider** in the sidebar to control how adventurous the AI gets. Start at 0.7 and adjust from there.
- **Mix your lines with the AI's** — click ✍️ *Add My Lines* to steer the plot, then let AI continue.
- **Stuck?** Hit *Give Me Choices* to get three fresh directions and pick your favourite.
- **Genre Remix** is great for seeing your story from a new angle — remix a Horror section as Comedy for a fun twist.
- **Analyse your story** after a few paragraphs to see which keywords and themes are emerging.
""")

    st.markdown("---")
    st.markdown("### Start your story")

    col1, col2 = st.columns([3, 2])
    with col1:
        title = st.text_input("Story Title", placeholder="The Last Ember of Aethon…")
    with col2:
        genre = st.selectbox("Genre", GENRES)

    st.info(f"**{genre} rules:** {GENRE_RULES[genre]}")

    hook = st.text_area(
        "Initial Hook / Setting",
        placeholder="Describe the opening scene or inciting event…",
        height=130,
    )

    can_start = bool(title and hook)
    if not can_start:
        st.caption("Fill in a title and hook to enable the button.")

    if st.button("🚀 Start the Story", type="primary", disabled=not can_start, use_container_width=True):
        with st.spinner("Weaving your opening…"):
            try:
                opening = generate_text(
                    build_opening_prompt(title, genre, hook),
                    st.session_state.temperature,
                )
                st.session_state.title          = title
                st.session_state.genre          = genre
                st.session_state.story_segments = [{"author": "ai", "text": opening}]
                st.session_state.ai_turn_count  = 1
                st.session_state.started        = True
                st.rerun()
            except RateLimitError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")


# ── Main storytelling screen ──────────────────────────────────────────────────

def main_screen() -> None:
    render_sidebar()

    genre = st.session_state.genre
    color = GENRE_COLORS.get(genre, "#6c63ff")
    segs  = st.session_state.story_segments

    # Header
    st.markdown(
        f"<h2 style='margin-bottom:0.2em;'>{html.escape(st.session_state.title)}"
        f" <span style='font-size:0.6em;padding:3px 12px;border-radius:20px;"
        f"background:{color};color:#fff;font-weight:600;vertical-align:middle;'>"
        f"{genre}</span></h2>",
        unsafe_allow_html=True,
    )

    # ── SECTION 1: Full Story So Far ──────────────────────────────────
    st.markdown("### 📖 Full Story So Far")

    # Keyword highlight toggle (only shown after analysis has been run)
    if st.session_state.keywords:
        st.session_state.show_highlights = st.toggle(
            "Highlight keywords in story",
            value=st.session_state.show_highlights,
        )

    render_story()

    st.markdown("---")

    # ── SECTION 2: Your Contribution (WYSIWYG) ───────────────────────
    st.markdown("### ✍️ Your Contribution")
    st.caption("Use the rich text editor below — bold names, italicise thoughts, add emphasis. Click **Add My Lines** when ready.")

    quill_html = st_quill(
        placeholder="She hesitated at the threshold, then stepped inside…",
        html=True,
        key=f"quill_{st.session_state.quill_counter}",
    )

    user_text_clean = strip_html(quill_html) if quill_html else ""

    if st.button("➕ Add My Lines", disabled=not bool(user_text_clean)):
        segs.append({"author": "user", "text": user_text_clean})
        st.session_state.pending_choices = []
        st.session_state.quill_counter  += 1   # resets the editor
        st.rerun()

    st.markdown("---")

    # ── SECTION 3: AI Controls ────────────────────────────────────────
    st.markdown("### 🤖 AI Controls")
    st.caption(
        "**Continue with AI** appends 1–2 paragraphs. "
        "**Give Me Choices** shows 3 branches — pick one to continue. "
        "Full story history is sent on every call."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ Continue with AI", type="primary", use_container_width=True):
            with st.spinner("Writing the next part…"):
                try:
                    text = generate_text(
                        build_continue_prompt(
                            segs, genre,
                            f"{creativity_label(st.session_state.temperature)} — "
                            f"{creativity_desc(st.session_state.temperature)}",
                        ),
                        st.session_state.temperature,
                    )
                    segs.append({"author": "ai", "text": text})
                    st.session_state.pending_choices = []
                    st.session_state.ai_turn_count  += 1
                    maybe_refresh_characters()
                    st.rerun()
                except RateLimitError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")

    with col2:
        if st.button("🔀 Give Me Choices", use_container_width=True):
            with st.spinner("Generating 3 branching paths…"):
                try:
                    raw = generate_text(build_choices_prompt(segs, genre), temperature=0.85)
                    choices = parse_choices(raw)
                    if choices:
                        st.session_state.pending_choices = choices
                    else:
                        st.warning("Couldn't parse choices — try again.")
                    st.rerun()
                except RateLimitError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")

    # Branch choices
    if st.session_state.pending_choices:
        st.markdown("**Choose your path:**")
        for i, choice in enumerate(st.session_state.pending_choices):
            if st.button(f"{i+1}. {choice}", key=f"branch_{i}", use_container_width=True):
                with st.spinner("Continuing your chosen path…"):
                    try:
                        text = generate_text(
                            build_branch_prompt(segs, genre, choice),
                            st.session_state.temperature,
                        )
                        segs.append({"author": "ai", "text": text})
                        st.session_state.pending_choices = []
                        st.session_state.ai_turn_count  += 1
                        maybe_refresh_characters()
                        st.rerun()
                    except RateLimitError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")

    st.markdown("---")

    # ── SECTION 4: Advanced Tools ─────────────────────────────────────
    st.markdown("### 🛠️ Advanced Tools")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown("#### 🎭 Genre Remix")
        st.caption("Rewrite the latest section in a new genre — plot preserved, tone transformed.")
        other_genres = [g for g in GENRES if g != genre]
        target_genre = st.selectbox("Remix to:", other_genres, key="remix_target")
        if st.button("🎭 Genre Remix", use_container_width=True):
            if not segs:
                st.warning("No story yet.")
            else:
                with st.spinner(f"Remixing as {target_genre}…"):
                    try:
                        remixed = generate_text(
                            build_remix_prompt(segs[-1]["text"], genre, target_genre),
                            temperature=0.8,
                        )
                        st.session_state.remix_result     = remixed
                        st.session_state.last_remix_genre = target_genre
                        st.rerun()
                    except RateLimitError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")

    with t2:
        st.markdown("#### 👥 Character Tracker")
        st.caption("Auto-updates every 3rd AI turn.")
        chars = st.session_state.characters
        if chars:
            for name, desc in chars.items():
                st.markdown(f"**{html.escape(name)}** — {html.escape(desc)}")
        else:
            st.caption("No characters detected yet.")
        if st.button("🔄 Refresh Characters", use_container_width=True, key="char_main"):
            with st.spinner("Extracting…"):
                maybe_refresh_characters(force=True)
            st.rerun()

    with t3:
        st.markdown("#### 🎨 Visualization Prompt")
        st.caption("DALL·E / Flux / Midjourney prompt from the latest paragraph.")
        st.markdown("")
        if st.button("🎨 Generate Viz Prompt", use_container_width=True):
            if not segs:
                st.warning("No story yet.")
            else:
                with st.spinner("Generating image prompt…"):
                    try:
                        viz = generate_text(
                            build_visualization_prompt(segs[-1]["text"]),
                            temperature=0.7,
                        )
                        st.session_state.viz_prompt = viz
                        st.rerun()
                    except RateLimitError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")

    # Result panels
    if st.session_state.remix_result:
        rg = st.session_state.last_remix_genre or "other genre"
        with st.expander(f"🎭 Remix result — rewritten as {rg}", expanded=True):
            st.markdown(st.session_state.remix_result)
            if st.button("➕ Add remix to story", key="add_remix"):
                segs.append({"author": "ai", "text": st.session_state.remix_result})
                st.session_state.remix_result   = ""
                st.session_state.ai_turn_count += 1
                maybe_refresh_characters()
                st.rerun()

    if st.session_state.viz_prompt:
        with st.expander("🎨 Visualization Prompt", expanded=True):
            st.caption("Paste into DALL·E / Flux / Midjourney.")
            st.text_area("Image Prompt", value=st.session_state.viz_prompt, height=120, key="viz_display")

    st.markdown("---")

    # ── SECTION 5: Story Analysis ─────────────────────────────────────
    st.markdown("### 🔍 Story Analysis")
    st.caption(
        "Extracts keywords using **sentence-transformer embeddings** (distilRoBERTa-family). "
        "Classifies genre and tone via GPT-4o-mini."
    )

    ana1, ana2 = st.columns(2)

    with ana1:
        if st.button("🔍 Analyze Story", use_container_width=True, type="primary"):
            if not segs:
                st.warning("No story yet.")
            else:
                full = full_story_text()
                with st.spinner("Extracting keywords…"):
                    st.session_state.keywords = extract_keywords(full, top_n=12)
                with st.spinner("Classifying genre and tone…"):
                    st.session_state.classification = classify_story(full)
                st.session_state.show_highlights = True
                st.rerun()

    with ana2:
        if st.session_state.keywords:
            if st.button("Clear Analysis", use_container_width=True):
                st.session_state.keywords        = []
                st.session_state.classification  = {}
                st.session_state.show_highlights = False
                st.rerun()

    # Classification results
    clf = st.session_state.classification
    if clf:
        st.markdown("#### Classification")
        c1, c2 = st.columns(2)
        with c1:
            genre_label = clf.get("predicted_genre", "—")
            genre_conf  = clf.get("genre_confidence", 0)
            st.markdown(f"**Predicted Genre:** {genre_label}")
            st.progress(float(genre_conf), text=f"{genre_conf:.0%} confidence")

            top_genres = clf.get("top_genres", [])
            if top_genres:
                st.markdown("**Top genre probabilities:**")
                for g, s in top_genres:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:0.5em;margin-bottom:3px;'>"
                        f"<span style='width:120px;font-size:0.85rem;'>{g}</span>"
                        f"<div style='flex:1;background:#e5e7eb;border-radius:4px;height:10px;'>"
                        f"<div style='width:{s*100:.0f}%;background:#7c3aed;border-radius:4px;height:10px;'></div>"
                        f"</div><span style='font-size:0.8rem;color:#6b7280;'>{s:.0%}</span></div>",
                        unsafe_allow_html=True,
                    )

        with c2:
            tone_label = clf.get("tone", "—")
            tone_conf  = clf.get("tone_confidence", 0)
            st.markdown(f"**Dominant Tone:** {tone_label}")
            st.progress(float(tone_conf), text=f"{tone_conf:.0%} confidence")

            themes = clf.get("themes", [])
            if themes:
                st.markdown("**Key Themes:**")
                chips = " ".join(
                    f"<span style='display:inline-block;background:#f0fdf4;"
                    f"border:1px solid #bbf7d0;color:#15803d;border-radius:20px;"
                    f"padding:2px 10px;font-size:0.82rem;margin:2px;'>{t}</span>"
                    for t in themes
                )
                st.markdown(chips, unsafe_allow_html=True)

    # Keyword results
    kws = st.session_state.keywords
    if kws:
        st.markdown("#### Keywords")
        st.caption("Sorted by semantic relevance score. Toggle **Highlight keywords** above the story to see them highlighted.")
        kw_html = ""
        for kw, score in sorted(kws, key=lambda x: x[1], reverse=True):
            alpha    = round(0.25 + score * 0.5, 2)
            bg_color = f"rgba(124,58,237,{alpha})"
            kw_html += (
                f"<span style='display:inline-block;background:{bg_color};"
                f"color:#1e1b4b;border-radius:20px;padding:3px 12px;"
                f"font-size:0.85rem;font-weight:500;margin:3px;'>"
                f"{html.escape(kw)} <span style='font-size:0.75rem;opacity:0.8;'>{score:.2f}</span></span>"
            )
        st.markdown(
            f"<div style='line-height:2.2;'>{kw_html}</div>",
            unsafe_allow_html=True,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NarrateAI",
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background-color: #ffffff; color: #1a1a2e; }
        section[data-testid="stSidebar"] {
            background-color: #f8f8ff;
            border-right: 1px solid #e5e7eb;
        }
        .stButton > button {
            border-radius: 8px; font-weight: 500;
            border: 1px solid #d1d5db;
        }
        .stButton > button[kind="primary"] {
            background: #7c3aed; color: #fff; border-color: #7c3aed;
        }
        .stButton > button[kind="primary"]:hover { background: #6d28d9; }
        .stTextArea textarea {
            font-family: Georgia, 'Times New Roman', serif;
            line-height: 1.7; color: #1a1a2e; border-radius: 8px;
        }
        /* Give the Quill editor a clean border */
        .ql-container { border-radius: 0 0 8px 8px !important; }
        .ql-toolbar { border-radius: 8px 8px 0 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_state()

    if not st.session_state.started:
        setup_screen()
    else:
        main_screen()


if __name__ == "__main__":
    main()
