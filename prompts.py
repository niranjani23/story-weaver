"""
All 7 prompt builders + genre rules for Story Weaver.

Design principles:
- Every prompt that touches the narrative receives the full story history
  wrapped in --- FULL STORY SO FAR --- delimiters so the model has complete context.
- Prompts are instructional, not conversational — they tell the model exactly
  what structure and constraints to follow.
- JSON-mode prompts use a concrete example in the prompt body to anchor the schema.
"""

GENRE_RULES: dict[str, str] = {
    "Fantasy":  "Magic exists and has costs. World-building details matter. Heroes face epic stakes.",
    "Sci-Fi":   "Technology is plausible but advanced. Explore human nature through science. Logical cause and effect.",
    "Mystery":  "Clues are planted early. Red herrings are allowed. The detective/protagonist is observant.",
    "Romance":  "Emotional tension drives the story. Internal conflict matters as much as external. Chemistry is key.",
    "Horror":   "Build dread slowly. Use sensory details. The unknown is scarier than the revealed.",
    "Comedy":   "Timing is everything. Misunderstandings and irony fuel the plot. Keep the tone light and absurd.",
}

_SYSTEM = (
    "You are a masterful collaborative storyteller. "
    "Continue the story in the chosen genre while staying 100% consistent with all previous events, "
    "character personalities, and world rules. Never contradict earlier parts of the story. "
    "Write in vivid but concise third-person narrative. Keep the tone engaging and fun."
)


def _story_block(segments: list[dict]) -> str:
    """Format story segments into a readable block for prompt injection."""
    parts = []
    for seg in segments:
        label = "User" if seg["author"] == "user" else "AI"
        parts.append(f"[{label}]: {seg['text']}")
    return "\n\n".join(parts)


# ── 1. Opening ────────────────────────────────────────────────────────────────

def build_opening_prompt(title: str, genre: str, hook: str) -> str:
    """
    4-part structure: scene → atmosphere → character hint → cliffhanger.
    Enforces word count so the opening is substantial but not bloated.
    """
    rules = GENRE_RULES[genre]
    return f"""{_SYSTEM}

Genre: {genre}
Genre Rules: {rules}
Story Title: "{title}"
Initial Hook / Setting: {hook}

Write a compelling opening paragraph (150–250 words) that follows this four-part structure:
1. Establish the scene and physical setting with vivid sensory detail
2. Convey the mood and atmosphere of the world
3. Introduce at least one character with a single distinctive trait that feels true to {genre}
4. End on a hook or cliffhanger that makes the reader desperate to continue

Write only the opening paragraph. No title, no labels, no commentary — just the narrative."""


# ── 2. Continue ───────────────────────────────────────────────────────────────

def build_continue_prompt(
    story_so_far: list[dict],
    genre: str,
    creativity_label: str,
) -> str:
    """
    Injects the creativity_label so prose style adapts to the temperature slider:
    conservative → grounded; wildly creative → bold and surprising.
    """
    rules = GENRE_RULES[genre]
    story_text = _story_block(story_so_far)
    return f"""{_SYSTEM}

Genre: {genre}
Genre Rules: {rules}
Creativity Level: {creativity_label}

--- FULL STORY SO FAR ---
{story_text}
--- END OF STORY ---

Continue the story with 1–2 coherent paragraphs that:
- Follow naturally from the exact moment the story stopped
- Advance the plot meaningfully without rushing to a resolution
- Remain consistent with every established character, location, and event
- Match the creativity level: "{creativity_label}"
  • If Conservative: keep events grounded, predictable, and emotionally satisfying
  • If Wildly Creative: introduce a surprising but logically consistent twist

Write only the continuation. No labels, no commentary."""


# ── 3. Choices ────────────────────────────────────────────────────────────────

def build_choices_prompt(story_so_far: list[dict], genre: str) -> str:
    """
    Asks for 'meaningfully different' options — the key instruction that prevents
    the model from returning three vague variations of the same idea.
    """
    rules = GENRE_RULES[genre]
    story_text = _story_block(story_so_far)
    return f"""{_SYSTEM}

Genre: {genre}
Genre Rules: {rules}

--- FULL STORY SO FAR ---
{story_text}
--- END OF STORY ---

Generate exactly 3 branching story options for what happens next.
Each option MUST be meaningfully different — different in direction, stakes, or emotional register.
Do NOT offer three variations of the same idea.

Each option:
- Is a 1–2 sentence description of a specific event or decision
- Is consistent with established characters and world rules
- Is intriguing enough that any option feels worth choosing

Return ONLY a numbered list in this exact format (no other text):
1. [First option]
2. [Second option]
3. [Third option]"""


# ── 4. Branch ─────────────────────────────────────────────────────────────────

def build_branch_prompt(
    story_so_far: list[dict],
    genre: str,
    chosen_option: str,
) -> str:
    """
    Explicitly names the chosen path and says 'honour this direction precisely'
    — prevents the model from softening or ignoring the user's choice.
    """
    rules = GENRE_RULES[genre]
    story_text = _story_block(story_so_far)
    return f"""{_SYSTEM}

Genre: {genre}
Genre Rules: {rules}

--- FULL STORY SO FAR ---
{story_text}
--- END OF STORY ---

The user has chosen this story direction:
"{chosen_option}"

Honour this direction precisely — begin it immediately and concretely. Do not soften, delay, or reinterpret the choice.

Write 1–2 vivid paragraphs that:
- Open directly on the chosen event or decision
- Maintain all established character voices and world logic
- End at a natural pause that invites further continuation

Write only the continuation. No labels, no commentary."""


# ── 5. Genre Remix ────────────────────────────────────────────────────────────

def build_remix_prompt(
    latest_section: str,
    original_genre: str,
    target_genre: str,
) -> str:
    """
    Separates 'preserve ALL plot events' from 'transform EVERYTHING ELSE' —
    the critical distinction that produces a real remix rather than a paraphrase.
    """
    target_rules = GENRE_RULES[target_genre]
    return f"""{_SYSTEM}

Original Genre: {original_genre}
Target Genre: {target_genre}
Target Genre Rules: {target_rules}

--- SECTION TO REMIX ---
{latest_section}
--- END OF SECTION ---

Rewrite this section as if it were always written in the {target_genre} genre.

PRESERVE (these must stay identical):
- All plot events and their sequence
- All character names and their relationships
- The core action: who does what to whom

TRANSFORM (change everything else to fully match {target_genre}):
- Tone, mood, and emotional register
- Prose style, sentence rhythm, and vocabulary
- Sensory details and atmosphere
- Pacing (e.g., Horror slows down; Comedy speeds up)

Make it feel native to {target_genre} — not a translation, but an original.

Write only the remixed section. No labels, no commentary."""


# ── 6. Character Tracker ──────────────────────────────────────────────────────

def build_character_prompt(story_so_far: list[dict]) -> str:
    """
    'Based ONLY on what the text reveals, do not invent' — keeps the tracker
    factual rather than hallucinating traits the story hasn't established.
    Uses JSON mode, so no regex parsing needed.
    """
    story_text = _story_block(story_so_far)
    return f"""Read the following story and extract every named character.

--- FULL STORY ---
{story_text}
--- END OF STORY ---

Return a JSON object where:
- Each key is a character's full name as it appears in the text
- Each value is a single sentence describing that character based ONLY on what the text explicitly reveals

Do NOT invent details, traits, or backstory that the text does not establish.
If a name appears only once with no description, include it with "Mentioned briefly."

Example format:
{{"Aria Voss": "A sharp-tongued mage who hides a painful secret.", "Kron": "A gruff soldier with surprising loyalty to Aria."}}

Return only valid JSON. No other text."""


# ── 7. Visualization Prompt ───────────────────────────────────────────────────

def build_visualization_prompt(latest_paragraph: str) -> str:
    """
    Asks for 5 specific visual dimensions — scene, characters, lighting, mood,
    art style — which is what makes image-generator prompts actually effective
    rather than generic descriptions.
    """
    return f"""Based on the following story paragraph, generate a detailed image prompt for an AI image generator (DALL·E, Flux, Midjourney, Stable Diffusion).

--- PARAGRAPH ---
{latest_paragraph}
--- END OF PARAGRAPH ---

Your prompt must cover exactly these 5 visual dimensions:
1. Scene: What is happening and the specific location
2. Characters: Who is present, their physical appearance, expression, and pose
3. Lighting: Time of day, light sources, direction, and quality of light
4. Mood / Atmosphere: Emotional feel, dominant color palette, overall vibe
5. Art Style: One specific style (e.g., "cinematic photorealism", "dark fantasy oil painting", "Studio Ghibli watercolor", "noir comic book ink")

Write the final prompt as a single flowing paragraph of 60–100 words, optimized for image generation. Weave all 5 dimensions together naturally — no numbering or labels in the output."""
