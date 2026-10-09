"""highlights.py - a fast, rule-based "what needs my attention?" scan. No AI involved.

Why not just ask the model?
    A 0.6B model sometimes misses a deadline or a mention. Simple rules never do, they
    run in milliseconds, and they appear on screen *before* the model has finished.
    The AI summary gives the big picture; this gives a reliable to-do list.

How it works:
    Every message line gets a score. Each "signal" that matches adds its weight:

        urgent word (urgent, asap, deadline...)   +3
        you were mentioned by name                +3
        time reference (tomorrow, Friday, 5pm)    +2
        request (please, can you, need to...)     +1
        question mark (something is open)         +1

    score >= 5 -> High, 3-4 -> Medium. Lower scores are ignored as noise.
"""
import re
from dataclasses import dataclass
from typing import List, Pattern

_FLAGS = re.IGNORECASE


@dataclass(frozen=True)
class Signal:
    """One thing we look for in a message and how much it matters."""
    label: str
    pattern: Pattern
    weight: int


SIGNALS = (
    Signal("urgent", re.compile(r"\b(urgent|asap|immediately|emergency|critical|important|deadline|overdue)\b", _FLAGS), 3),
    Signal("time", re.compile(
        r"\b(today|tonight|tomorrow|eod|noon|midnight|next week|this week|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tues?|thu(?:rs?)?|fri|"
        r"\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}(?::\d{2})?\s?(?:am|pm))\b", _FLAGS), 2),
    Signal("request", re.compile(r"\b(please|can you|could you|need to|need you|must|don't forget|remember to)\b", _FLAGS), 1),
    Signal("question", re.compile(r"\?"), 1),
)
MENTION_WEIGHT = 3
HIGH_SCORE = 5  # >= this is "High"
MEDIUM_SCORE = 3  # >= this is "Medium"; anything lower is not shown
MAX_ITEMS = 6  # keep the list short, it is a to-do list, not a transcript
MAX_LINE_CHARS = 160

# Level -> (emoji, label)
_LEVELS = {"high": ("🔴", "High"), "medium": ("🟠", "Medium")}


@dataclass(frozen=True)
class Priority:
    """A message that probably needs the reader's attention."""
    level: str  # "high" or "medium"
    score: int
    line: str
    reasons: List[str]


def find_priorities(text: str, my_name: str = "") -> List[Priority]:
    """Score every line of `text` and return the most urgent ones (best first)."""
    name = my_name.strip().lstrip("@")
    # Match "Sam" or "@Sam" as a whole word, case-insensitive. None if no name was given.
    name_re = re.compile(rf"(?<!\w)@?{re.escape(name)}\b", _FLAGS) if name else None

    found: List[Priority] = []
    for line in text.split("\n"):
        score, reasons = 0, []
        if name_re and name_re.search(line):
            score += MENTION_WEIGHT
            reasons.append("mentions you")
        for signal in SIGNALS:
            if signal.pattern.search(line):
                score += signal.weight
                reasons.append(signal.label)
        if score >= MEDIUM_SCORE:
            level = "high" if score >= HIGH_SCORE else "medium"
            short = line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS - 3] + "..."
            found.append(Priority(level, score, short, reasons))

    # sorted() is stable, so equal scores keep their original (chronological) order
    return sorted(found, key=lambda p: -p.score)[:MAX_ITEMS]


def format_priorities(items: List[Priority]) -> str:
    """Render the priority list as Markdown. Returns "" when there is nothing to flag."""
    if not items:
        return ""
    rows = []
    for p in items:
        emoji, label = _LEVELS[p.level]
        rows.append(f"- {emoji} **{label}**: {p.line}  \n  _({', '.join(p.reasons)})_")
    return "## 🔔 Needs your attention\n_Rule-based scan, not AI._\n\n" + "\n".join(rows)
