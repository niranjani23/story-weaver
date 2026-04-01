# NarrateAI

Your AI-powered collaborative storytelling companion.

---

## Why NarrateAI instead of ChatGPT or Claude?

General-purpose chat tools like ChatGPT and Claude are powerful, but they are blank slates. Every time you start a new conversation you have to re-explain your story, your characters, your world rules, and your tone. NarrateAI is purpose-built for collaborative fiction:

| | ChatGPT / Claude | NarrateAI |
|---|---|---|
| **Story memory** | Lost between sessions; manually re-pasted each time | Full story history sent automatically on every call |
| **Genre consistency** | You must remind it of the genre and rules | Genre rules are baked into every prompt |
| **Branching choices** | Not available out of the box | "Give Me Choices" generates 3 meaningful branches |
| **Character tracking** | You track characters yourself | Live Character Tracker extracts and updates automatically |
| **Genre Remix** | Requires careful prompting | One-click rewrite in any genre, plot preserved |
| **Image prompts** | Manual prompt engineering | Auto-generates a structured DALL·E / Flux prompt |
| **Keyword analysis** | Not available | Highlights key phrases using transformer embeddings |
| **Tone & genre classification** | Not available | Classifies predicted genre, tone, and themes per session |
| **Rich text input** | Plain textarea | Full WYSIWYG editor (bold, italic, lists, headers) |

In short: NarrateAI removes all the scaffolding work so you can focus entirely on the story.

---

## What has been built

- **Setup screen** — title, genre selector (6 genres), hook input, live genre-rules preview
- **Rich text editor** — Quill.js WYSIWYG for user contributions (bold, italic, lists, headers)
- **Full story display** — scrollable, colour-coded by author (AI = purple, User = amber)
- **Continue with AI** — appends 1–2 coherent paragraphs; full story history sent on every call
- **Give Me Choices** — generates 3 meaningfully different branching options; user selects one
- **Creativity slider** — temperature control mapped to readable labels (Conservative → Wildly Creative)
- **Live Character Tracker** — auto-extracts named characters with descriptions after every 3rd AI turn
- **Genre Remix** — rewrites the latest section in a different genre, all plot events preserved
- **Visualization Prompt** — generates a structured image prompt (scene, characters, lighting, mood, art style) ready for DALL·E, Flux, or Midjourney
- **Story Analysis** — keyword extraction via sentence-transformer embeddings + genre/tone classification via GPT-4o-mini
- **Keyword highlighting** — toggle to highlight extracted keywords directly in the story display
- **Error handling** — specific messages for rate limits, quota exhaustion, auth failures, and timeouts
- **Streamlit Cloud ready** — secrets loaded from `st.secrets` (cloud) or `.env` (local) automatically

---

## Tech stack

| Layer | Technology |
|---|---|
| **UI framework** | [Streamlit](https://streamlit.io) |
| **Rich text editor** | [streamlit-quill](https://github.com/JackismyShephard/streamlit-quill) (Quill.js) |
| **LLM** | OpenAI GPT-4o-mini |
| **Keyword extraction** | [KeyBERT](https://github.com/MaartenGr/KeyBERT) + `paraphrase-MiniLM-L6-v2` sentence embeddings |
| **NLP runtime** | [sentence-transformers](https://www.sbert.net) + PyTorch (CPU) |
| **Story classification** | GPT-4o-mini in JSON mode |
| **Environment config** | python-dotenv |
| **Language** | Python 3.9+ |

---

## Next steps

- [ ] **User accounts & story persistence** — save and resume stories across sessions (SQLite or Supabase)
- [ ] **Export** — download the full story as a `.pdf` or `.docx` file
- [ ] **Image generation** — send the visualization prompt directly to DALL·E 3 or Stable Diffusion and display the image inline
- [ ] **Voice narration** — read the story aloud using the OpenAI TTS API
- [ ] **Multiplayer mode** — multiple users contributing to the same story in real time
- [ ] **Custom personas** — let the user define named characters with backstories that the AI references consistently
- [ ] **Story templates** — pre-built opening hooks for common scenarios (heist, romance, haunted house, etc.)
- [ ] **Mobile-friendly layout** — responsive design for smaller screens
- [ ] **Swap LLM backend** — provider toggle (OpenAI / Gemini / Mistral) without changing any other code

---

## Requirements

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) with billing enabled

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/YOUR_USERNAME/narrateai.git
cd narrateai
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your API key**

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-...your_key_here
```

## Run

```bash
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## Demo


