"""summarizer.py - the summarization pipeline: what we ask the model and how we cope with long chats.

Pipeline (see README for a diagram):

    raw text --clean--> tidy chat --validate--> (cache hit? return it)
         --priority scan--> "Needs your attention" list (instant, no AI)
         --split to fit the context window--> 1 chunk ............ summarize directly
                                               many chunks ....... summarize each, then merge
         --> final Markdown summary (cached for identical re-runs)

The model itself is reached only through model_process.py.
"""
import hashlib
import logging
import re
from collections import OrderedDict
from typing import Iterator, List, Optional, Tuple

import config
import highlights
import model_process as model

log = logging.getLogger("protocolx.summarizer")


class InputError(ValueError):
    """The pasted text cannot be summarized (empty, too short, too long). Message is user-safe."""


# ----------------------------------------------------------------------------- cleaning
# Characters WhatsApp exports hide inside lines: narrow no-break space, direction marks, BOM.
_INVISIBLE = str.maketrans({"\u202f": " ", "\u00a0": " ", "\u200e": None, "\u200f": None, "\ufeff": None})

# WhatsApp export prefixes, e.g. "12/05/2024, 10:15 pm - " (Android) or "[12/05/24, 10:15:32 PM] " (iPhone).
# Timestamps are noise for the summary, and the clock times would confuse the deadline scan.
_WA_PREFIX = re.compile(
    r"^\[?\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4},?\s+\d{1,2}[:.]\d{2}(?::\d{2})?(?:\s?[AaPp]\.?[Mm]\.?)?\]?\s*(?:-\s+)?"
)

# Lines that carry no information: media placeholders, deleted messages, the encryption notice.
# fullmatch() is used so a real message that merely *contains* these words is never dropped.
_NOISE = re.compile(
    r"(?:[^:]{1,40}:\s*)?(?:<media omitted>|<attached: [^>]*>|(?:image|video|audio|sticker|gif|document) omitted|"
    r"this message was deleted|you deleted this message|messages and calls are end-to-end encrypted\..*)",
    re.IGNORECASE,
)


def clean_conversation(raw: str) -> str:
    """Tidy pasted text: strip WhatsApp timestamps, drop noise lines, blank lines and repeated messages."""
    lines: List[str] = []
    previous: Optional[str] = None
    for line in (raw or "").translate(_INVISIBLE).replace("\r\n", "\n").split("\n"):
        line = _WA_PREFIX.sub("", line.strip()).strip()
        if not line or _NOISE.fullmatch(line):
            continue
        if line == previous:  # accidental double paste or a repeated line
            continue
        lines.append(line)
        previous = line
    return "\n".join(lines)


# ----------------------------------------------------------------------------- chunking
def split_into_chunks(text: str, budget: int, run: int) -> List[str]:
    """Split the chat at message boundaries so every chunk fits in `budget` tokens.

    Splitting on lines (messages) keeps each message whole, so the model never sees half a sentence.
    All lines are counted in a single call to the model process, which is far faster than one by one.
    """
    lines = text.split("\n")
    counts = model.count_tokens(lines, run)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for line, n in zip(lines, counts):
        tokens = n + 1  # +1 for the newline between messages
        if tokens > budget:
            # A single enormous message: flush what we have, then cut the message itself into pieces.
            if current:
                chunks.append("\n".join(current))
                current, current_tokens = [], 0
            sep = " " if " " in line else ""
            units = line.split(sep) if sep else list(line)
            per_piece = max(1, int(len(units) * budget * 0.8 / tokens))  # 0.8 = safety margin
            for i in range(0, len(units), per_piece):
                chunks.append(sep.join(units[i:i + per_piece]))
            continue
        if current and current_tokens + tokens > budget:  # this message would overflow: start a new chunk
            chunks.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += tokens
    if current:
        chunks.append("\n".join(current))
    return chunks


# ----------------------------------------------------------------------------- cache
# Re-summarizing the same chat with the same options returns instantly instead of re-running the model.
# Only the finished summaries are kept (in memory, capped, never written to disk).
_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_key(text: str, length: str, my_name: str) -> str:
    # We hash the input so the chat itself is not stored as a dictionary key.
    return hashlib.sha256(f"{length}|{my_name}|{text}".encode("utf-8")).hexdigest()


def clear_cache() -> None:
    """Forget all remembered summaries (called by the Clear button)."""
    _cache.clear()


# ----------------------------------------------------------------------------- main entry
def summarize(conversation: str, length: str = "medium", my_name: str = "") -> Iterator[Tuple[str, str]]:
    """Summarize a chat. A generator that yields (status_text, markdown_so_far) as work progresses.

    Raises InputError (bad input), model.ModelError (model problem) or model.RunStopped (Stop pressed).
    """
    # ---- 1. validate before touching the model ----
    text = clean_conversation(conversation)
    if not text:
        raise InputError("Paste a conversation first.")
    if len(text) < config.MIN_INPUT_CHARS:
        raise InputError("That is too short to summarize. Paste a longer conversation.")
    if len(text) > config.MAX_INPUT_CHARS:
        raise InputError(f"Input is too long ({len(text):,} characters). The limit is {config.MAX_INPUT_CHARS:,}.")
    if not model.is_loaded():
        raise model.ModelError("Model is not loaded (or is restarting). Press Load Model and wait for 'Model loaded'.")

    # ---- 2. identical request seen before? answer instantly ----
    key = _cache_key(text, length, my_name)
    if key in _cache:
        _cache.move_to_end(key)
        yield "Done (same input as before, no need to run the model again).", _cache[key]
        return

    max_out, length_hint = config.LENGTHS.get(length, config.LENGTHS["medium"])
    system = config.SYSTEM_PROMPT + "\n" + length_hint
    merge_system = config.MERGE_SYSTEM_PROMPT + "\n" + length_hint

    # ---- 3. instant, rule-based priority list (shown first, before the AI has produced anything) ----
    priorities = highlights.format_priorities(highlights.find_priorities(text, my_name))
    prefix = priorities + "\n\n---\n\n" if priorities else ""

    with model.run_session() as run:
        # The context window must hold: system prompt + chat + the answer. Reserve room for all three.
        overhead = model.count_tokens([merge_system], run)[0] + 80  # +80 for the chat-template tokens
        budget = config.N_CTX - max_out - overhead - 100  # -100 safety margin
        if budget < 300:
            raise InputError("The context window is too small for these settings. Increase PROTOCOLX_CTX.")
        chunks = split_into_chunks(text, budget, run)
        log.info("Summarizing %d characters in %d part(s)", len(text), len(chunks))

        # ---- 4a. short chat: one pass ----
        if len(chunks) == 1:
            yield "Summarizing...", prefix
            final = ""
            for final in model.stream_chat(system, config.CHAT_TEMPLATE.format(chunks[0]), max_out, run):
                yield "Summarizing...", prefix + final
        else:
            # ---- 4b. long chat (map-reduce): summarize every chunk, then merge the partial summaries ----
            partials: List[str] = []
            for i, chunk in enumerate(chunks, 1):
                yield f"Long conversation: reading part {i} of {len(chunks)}...", prefix
                partials.append(model.complete_chat(
                    system, config.CHAT_TEMPLATE.format(chunk), config.PART_SUMMARY_TOKENS, run))

            # If the partial summaries still do not fit in one prompt, merge them in groups first.
            while True:
                joined = "\n\n".join(f"--- Part {i} ---\n{p}" for i, p in enumerate(partials, 1))
                if model.count_tokens([joined], run)[0] <= budget:
                    break
                sizes = model.count_tokens(partials, run)
                groups: List[List[str]] = []
                group: List[str] = []
                group_tokens = 0
                for p, n in zip(partials, sizes):
                    if group and group_tokens + n + 10 > budget:
                        groups.append(group)
                        group, group_tokens = [], 0
                    group.append(p)
                    group_tokens += n + 10
                groups.append(group)
                if len(groups) >= len(partials):  # no progress possible: avoid an endless loop
                    raise model.ModelError("Could not condense this conversation into the context window.")
                yield "Condensing partial summaries...", prefix
                partials = [
                    model.complete_chat(merge_system, config.MERGE_TEMPLATE.format("\n\n".join(g)),
                                        config.PART_SUMMARY_TOKENS, run) if len(g) > 1 else g[0]
                    for g in groups
                ]

            yield "Merging parts into the final summary...", prefix
            final = ""
            for final in model.stream_chat(merge_system, config.MERGE_TEMPLATE.format(joined), max_out, run):
                yield "Merging...", prefix + final

    # ---- 5. finish: sanity check, remember the result, return it ----
    if not final.strip():
        raise model.ModelError("The model returned an empty summary. Please try again.")
    result = prefix + final
    _cache[key] = result
    while len(_cache) > config.CACHE_SIZE:  # oldest entries fall out first
        _cache.popitem(last=False)
    yield "Done.", result
