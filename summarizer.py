"""ProtocolX summarizer: local Qwen3-0.6B via llama-cpp-python.

Everything runs on-device. No conversation text is logged or sent anywhere.
"""
import os
import re
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# ---------------------------------------------------------------- config
MODEL_PATH = os.environ.get("PROTOCOLX_MODEL", "Qwen3-0.6B-Q8_0.gguf")
N_CTX = int(os.environ.get("PROTOCOLX_CTX", "4096"))
N_THREADS = int(os.environ.get("PROTOCOLX_THREADS", str(max(2, (os.cpu_count() or 4) // 2))))
MAX_INPUT_CHARS = 200_000

# length option -> (max output tokens, bullet instruction)
LENGTHS = {
    "short": (350, "Keep each section to 1-2 bullets."),
    "medium": (600, "Use at most 4 bullets per section."),
    "detailed": (900, "Use at most 8 bullets per section and keep concrete details (names, numbers, dates)."),
}

SYSTEM_PROMPT = """You summarize chat conversations for a person who missed them.
Rules:
- Use ONLY facts stated in the conversation. Never invent names, dates or facts.
- A suggestion or question is NOT a decision. Only list a decision if people agreed on it.
- Do not repeat information. Skip small talk.
Write Markdown using only these sections, and leave out any section with nothing relevant:
## Overview
## Key Information
## Decisions Made
## Progress
## Problems and Unresolved Questions
## Action Items
In Action Items write each line as: - [High|Medium|Low] task - owner, deadline
(use 'unassigned' or 'no deadline' if unknown; High = urgent or near deadline)."""

MERGE_SYSTEM_PROMPT = SYSTEM_PROMPT + """
You are given partial summaries of consecutive parts of ONE conversation.
Merge them into a single summary. Remove duplicates. If a later part changes or
cancels an earlier decision, keep only the latest version."""


class SummarizerError(Exception):
    """Errors that are safe to show directly to the user."""


# ---------------------------------------------------------------- model
_llm = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # llama.cpp contexts are not thread-safe


def is_loaded() -> bool:
    return _llm is not None


def load_model(path: Optional[str] = None) -> str:
    """Load the model once. Returns a status message. Raises SummarizerError."""
    global _llm
    with _load_lock:
        if _llm is not None:
            return f"Model already loaded (context {N_CTX}, {N_THREADS} threads)."
        model_path = Path(path or MODEL_PATH)
        if not model_path.is_file():
            raise SummarizerError(
                f"Model file not found: {model_path.resolve()}\n"
                "Set the PROTOCOLX_MODEL environment variable to the full path of your .gguf file."
            )
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise SummarizerError("llama-cpp-python is not installed in this environment.") from e
        try:
            _llm = Llama(
                model_path=str(model_path),
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            raise SummarizerError(f"Could not load the model: {e}") from e
        return f"Model loaded: {model_path.name} (context {N_CTX}, {N_THREADS} threads)."


def _count_tokens(text: str) -> int:
    return len(_llm.tokenize(text.encode("utf-8"), add_bos=False))


# ---------------------------------------------------------------- input helpers
def clean_conversation(raw: str) -> str:
    """Trim blank lines and drop consecutive duplicate messages."""
    lines, prev = [], None
    for line in (raw or "").replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line or line == prev:
            continue
        lines.append(line)
        prev = line
    return "\n".join(lines)


def split_into_chunks(text: str, budget: int) -> List[str]:
    """Split on message (line) boundaries so each chunk fits in `budget` tokens."""
    chunks, current, current_tokens = [], [], 0
    for line in text.split("\n"):
        t = _count_tokens(line) + 1
        if t > budget:  # one gigantic line: hard-split by words
            if current:
                chunks.append("\n".join(current))
                current, current_tokens = [], 0
            words, piece = line.split(" "), []
            for w in words:
                piece.append(w)
                if _count_tokens(" ".join(piece)) > budget:
                    piece.pop()
                    chunks.append(" ".join(piece))
                    piece = [w]
            if piece:
                current, current_tokens = [" ".join(piece)], _count_tokens(" ".join(piece))
            continue
        if current_tokens + t > budget and current:
            chunks.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += t
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------------------------------------------------------------- generation
_THINK_DONE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    text = _THINK_DONE.sub("", text)
    if "<think>" in text:  # still thinking: hide it
        text = text.split("<think>")[0]
    return text.strip()


def _generate(system: str, user: str, max_tokens: int) -> Iterator[str]:
    """Stream the model output; yields the accumulated (cleaned) text."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user + "\n/no_think"},  # disables Qwen3 thinking mode
    ]
    acc = ""
    try:
        with _infer_lock:
            stream = _llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=True,
            )
            for part in stream:
                delta = part["choices"][0]["delta"].get("content")
                if delta:
                    acc += delta
                    yield _strip_think(acc)
    except ValueError as e:  # llama-cpp raises ValueError on context overflow
        raise SummarizerError(f"Context window exceeded: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise SummarizerError(f"Inference failed: {e}") from e


def _complete(system: str, user: str, max_tokens: int) -> str:
    last = ""
    for last in _generate(system, user, max_tokens):
        pass
    return last


# ---------------------------------------------------------------- highlights (regex, not AI)
_DEADLINE = re.compile(
    r"\b(today|tonight|tomorrow|eod|asap|urgent|deadline|due|by (?:mon|tue|wed|thu|fri|sat|sun)\w*|"
    r"next week|this week|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}(?::\d{2})?\s?(?:am|pm))\b",
    re.IGNORECASE,
)


def scan_highlights(text: str, my_name: str = "") -> str:
    """Cheap, reliable scan for mentions and deadlines the small model might miss."""
    mention_lines, deadline_lines = [], []
    name = my_name.strip().lstrip("@")
    name_re = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE) if name else None
    for line in text.split("\n"):
        short = line if len(line) <= 160 else line[:157] + "..."
        if name_re and (name_re.search(line) or f"@{name.lower()}" in line.lower()):
            mention_lines.append(short)
        if _DEADLINE.search(line):
            deadline_lines.append(short)
    out = []
    if mention_lines:
        out.append(f"### Mentions of you\n" + "\n".join(f"- {l}" for l in mention_lines[:10]))
    if deadline_lines:
        out.append("### Deadlines / urgent messages\n" + "\n".join(f"- {l}" for l in deadline_lines[:10]))
    if not out:
        return ""
    return "---\n## Flagged by keyword scan (not AI)\n" + "\n\n".join(out)


# ---------------------------------------------------------------- main entry
def summarize(conversation: str, length: str = "medium", my_name: str = "") -> Iterator[Tuple[str, str]]:
    """Yields (status, summary_markdown). Raises SummarizerError for user-facing problems."""
    if not is_loaded():
        raise SummarizerError("Model is not loaded yet. Click 'Load Model' first.")
    text = clean_conversation(conversation)
    if not text:
        raise SummarizerError("Paste a conversation first.")
    if len(text) < 30:
        raise SummarizerError("That is too short to summarize. Paste a longer conversation.")
    if len(text) > MAX_INPUT_CHARS:
        raise SummarizerError(f"Input is too long ({len(text):,} chars). Limit is {MAX_INPUT_CHARS:,}.")

    max_out, length_hint = LENGTHS.get(length, LENGTHS["medium"])
    system = SYSTEM_PROMPT + "\n" + length_hint
    merge_system = MERGE_SYSTEM_PROMPT + "\n" + length_hint

    # context = system prompt + conversation + output, so reserve room for all three
    overhead = _count_tokens(merge_system) + 80
    budget = N_CTX - max_out - overhead - 100
    if budget < 300:
        raise SummarizerError("Context window is too small for these settings. Increase PROTOCOLX_CTX.")

    highlights = scan_highlights(text, my_name)
    chunks = split_into_chunks(text, budget)
    user_tmpl = 'Conversation:\n"""\n{}\n"""\n\nWrite the summary now.'

    if len(chunks) == 1:
        yield "Summarizing...", ""
        final = ""
        for final in _generate(system, user_tmpl.format(chunks[0]), max_out):
            yield "Summarizing...", final
        yield "Done.", final + ("\n\n" + highlights if highlights else "")
        return

    # long input: summarize each chunk, then merge
    partials = []
    for i, chunk in enumerate(chunks, 1):
        yield f"Long conversation: reading part {i} of {len(chunks)}...", ""
        partials.append(_complete(system, user_tmpl.format(chunk), 400))

    while True:
        joined = "\n\n".join(f"--- Part {i} ---\n{p}" for i, p in enumerate(partials, 1))
        if _count_tokens(joined) <= budget:
            break
        groups, cur, cur_t = [], [], 0
        for p in partials:
            t = _count_tokens(p) + 10
            if cur and cur_t + t > budget:
                groups.append(cur)
                cur, cur_t = [], 0
            cur.append(p)
            cur_t += t
        groups.append(cur)
        if len(groups) >= len(partials):
            raise SummarizerError("Could not condense this conversation into the context window.")
        yield "Condensing partial summaries...", ""
        partials = [
            _complete(merge_system, "Partial summaries:\n" + "\n\n".join(g) + "\n\nWrite the merged summary now.", 400)
            if len(g) > 1 else g[0]
            for g in groups
        ]

    yield "Merging parts into the final summary...", ""
    final = ""
    for final in _generate(merge_system, "Partial summaries:\n" + joined + "\n\nWrite the merged summary now.", max_out):
        yield "Merging...", final
    yield "Done.", final + ("\n\n" + highlights if highlights else "")
