"""ProtocolX summarizer: local Qwen3-0.6B via llama-cpp-python.

The model runs in a separate process (llm_worker.py) so that Stop can kill it
instantly and restart it clean. Everything stays on-device; no conversation text
is logged or sent anywhere.
"""
import atexit
import itertools
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# ---------------------------------------------------------------- config
MODEL_PATH = os.environ.get("PROTOCOLX_MODEL", "Qwen3-0.6B-Q8_0.gguf")
N_CTX = int(os.environ.get("PROTOCOLX_CTX", "4096"))
N_THREADS = int(os.environ.get("PROTOCOLX_THREADS", str(max(2, (os.cpu_count() or 4) // 2))))
MAX_INPUT_CHARS = 200_000
WORKER_FILE = Path(__file__).with_name("llm_worker.py")

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


class SummarizerStopped(Exception):
    """Raised when the user pressed Stop (or Clear) during a run."""


# ---------------------------------------------------------------- model process
class _Worker:
    """One llm_worker.py process plus a thread that collects its JSON replies."""

    def __init__(self, model_path: str):
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW on Windows
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER_FILE), model_path, str(N_CTX), str(N_THREADS)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=flags,
        )
        self.q: "queue.Queue[dict]" = queue.Queue()
        self.ready = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        try:
            for line in self.proc.stdout:
                try:
                    self.q.put(json.loads(line))
                except ValueError:
                    continue  # stray non-JSON output
        except (OSError, ValueError):
            pass
        self.q.put({"type": "closed"})

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        self.ready = False
        try:
            self.proc.kill()
        except OSError:
            pass


_worker: Optional[_Worker] = None
_last_path: Optional[str] = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # one request at a time to the worker
_stop_lock = threading.Lock()
_active_lock = threading.Lock()
_active = 0  # number of summaries currently running
_run_id = 0  # every run gets an id; stop() invalidates the current one
_ids = itertools.count(1)


def _kill_at_exit() -> None:
    if _worker is not None:
        _worker.kill()


atexit.register(_kill_at_exit)


def is_loaded() -> bool:
    return _worker is not None and _worker.ready and _worker.alive()


def is_busy() -> bool:
    return _active > 0


def load_model(path: Optional[str] = None, force: bool = False) -> str:
    """Start the model process (once). Returns a status message. Raises SummarizerError."""
    global _worker, _last_path
    with _load_lock:
        if _worker is not None and not force and is_loaded():
            return f"Model already loaded (context {N_CTX}, {N_THREADS} threads)."
        if _worker is not None:
            _worker.kill()
            _worker = None
        model_path = Path(path or _last_path or MODEL_PATH)
        if not model_path.is_file():
            raise SummarizerError(
                f"Model file not found: {model_path.resolve()}\n"
                "Set the PROTOCOLX_MODEL environment variable to the full path of your .gguf file."
            )
        if not WORKER_FILE.is_file():
            raise SummarizerError(f"llm_worker.py is missing (expected next to summarizer.py): {WORKER_FILE}")
        _last_path = str(model_path.resolve())
        w = _Worker(_last_path)
        deadline = time.time() + 300
        while True:
            try:
                msg = w.q.get(timeout=0.5)
            except queue.Empty:
                if not w.alive():
                    raise SummarizerError("The model process exited while loading. "
                                          "Check that llama-cpp-python works in this environment.")
                if time.time() > deadline:
                    w.kill()
                    raise SummarizerError("Timed out while loading the model.")
                continue
            if msg.get("type") == "ready":
                w.ready = True
                break
            if msg.get("type") in ("error", "closed"):
                w.kill()
                raise SummarizerError(msg.get("error", "The model process ended while loading."))
        _worker = w
        return f"Model loaded: {model_path.name} (context {N_CTX}, {N_THREADS} threads)."


def stop() -> bool:
    """Stop the running summary by killing the model process and starting a fresh one.

    Returns True if something was running. Blocks until the new model is ready.
    """
    global _run_id
    if _active == 0:
        return False
    if not _stop_lock.acquire(blocking=False):
        return True  # a stop is already in progress
    try:
        _run_id += 1  # the running summary notices this and exits
        w = _worker
        if w is not None:
            w.kill()
        try:
            load_model(force=True)
        except SummarizerError:
            pass  # is_loaded() will report False; the UI tells the user
    finally:
        _stop_lock.release()
    return True


def _check(run: int) -> None:
    if run != _run_id:
        raise SummarizerStopped()


def _require_worker() -> _Worker:
    if not is_loaded():
        raise SummarizerError("Model is not loaded (or is restarting). Wait a few seconds or press Load Model.")
    return _worker  # type: ignore[return-value]


def _send(w: _Worker, obj: dict, run: int) -> None:
    try:
        w.proc.stdin.write(json.dumps(obj) + "\n")
        w.proc.stdin.flush()
    except (OSError, ValueError) as e:
        _check(run)  # killed by Stop? then it is not an error
        raise SummarizerError("Lost connection to the model process.") from e


def _next_msg(w: _Worker, run: int) -> dict:
    """Wait for the next reply, checking for Stop every 0.2 s."""
    while True:
        _check(run)
        try:
            msg = w.q.get(timeout=0.2)
        except queue.Empty:
            if not w.alive():
                _check(run)
                raise SummarizerError("The model process stopped unexpectedly. Press Load Model.")
            continue
        if msg.get("type") == "closed":
            _check(run)
            raise SummarizerError("The model process stopped unexpectedly. Press Load Model.")
        return msg


def _count_many(texts: List[str], run: int) -> List[int]:
    w = _require_worker()
    with _infer_lock:
        _check(run)
        rid = next(_ids)
        _send(w, {"op": "count", "id": rid, "texts": texts}, run)
        while True:
            msg = _next_msg(w, run)
            if msg.get("id") != rid:
                continue
            if msg["type"] == "counts":
                return msg["counts"]
            if msg["type"] == "error":
                raise SummarizerError(f"Token counting failed: {msg.get('error')}")


def _count_tokens(text: str, run: int) -> int:
    return _count_many([text], run)[0]


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


def split_into_chunks(text: str, budget: int, run: int) -> List[str]:
    """Split on message (line) boundaries so each chunk fits in `budget` tokens."""
    lines = text.split("\n")
    counts = _count_many(lines, run)
    chunks, current, current_tokens = [], [], 0
    for line, n in zip(lines, counts):
        t = n + 1
        if t > budget:  # one gigantic message: cut it into pieces
            if current:
                chunks.append("\n".join(current))
                current, current_tokens = [], 0
            sep = " " if " " in line else ""
            units = line.split(sep) if sep else list(line)
            per = max(1, int(len(units) * budget * 0.8 / t))
            for i in range(0, len(units), per):
                chunks.append(sep.join(units[i:i + per]))
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


def _generate(system: str, user: str, max_tokens: int, run: int) -> Iterator[str]:
    """Stream the model output; yields the accumulated (cleaned) text."""
    w = _require_worker()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user + "\n/no_think"},  # disables Qwen3 thinking mode
    ]
    with _infer_lock:
        _check(run)
        rid = next(_ids)
        _send(w, {"op": "chat", "id": rid, "messages": messages, "max_tokens": max_tokens}, run)
        acc = ""
        while True:
            msg = _next_msg(w, run)
            if msg.get("id") != rid:
                continue
            kind = msg["type"]
            if kind == "token":
                acc += msg["text"]
                yield _strip_think(acc)
            elif kind == "done":
                return
            elif kind == "error":
                raise SummarizerError(f"Inference failed: {msg.get('error')}")


def _complete(system: str, user: str, max_tokens: int, run: int) -> str:
    last = ""
    for last in _generate(system, user, max_tokens, run):
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
        out.append("### Mentions of you\n" + "\n".join(f"- {l}" for l in mention_lines[:10]))
    if deadline_lines:
        out.append("### Deadlines / urgent messages\n" + "\n".join(f"- {l}" for l in deadline_lines[:10]))
    if not out:
        return ""
    return "---\n## Flagged by keyword scan (not AI)\n" + "\n\n".join(out)


# ---------------------------------------------------------------- main entry
def summarize(conversation: str, length: str = "medium", my_name: str = "") -> Iterator[Tuple[str, str]]:
    """Yields (status, summary_markdown). Raises SummarizerError / SummarizerStopped."""
    global _active
    with _active_lock:
        _active += 1
    try:
        yield from _summarize(conversation, length, my_name)
    finally:
        with _active_lock:
            _active -= 1


def _summarize(conversation: str, length: str, my_name: str) -> Iterator[Tuple[str, str]]:
    global _run_id
    if not is_loaded():
        raise SummarizerError("Model is not loaded (or is restarting). Press Load Model and wait for 'Model loaded'.")
    text = clean_conversation(conversation)
    if not text:
        raise SummarizerError("Paste a conversation first.")
    if len(text) < 30:
        raise SummarizerError("That is too short to summarize. Paste a longer conversation.")
    if len(text) > MAX_INPUT_CHARS:
        raise SummarizerError(f"Input is too long ({len(text):,} chars). Limit is {MAX_INPUT_CHARS:,}.")

    _run_id += 1  # new run; also invalidates any older run still finishing
    run = _run_id

    max_out, length_hint = LENGTHS.get(length, LENGTHS["medium"])
    system = SYSTEM_PROMPT + "\n" + length_hint
    merge_system = MERGE_SYSTEM_PROMPT + "\n" + length_hint

    # context = system prompt + conversation + output, so reserve room for all three
    overhead = _count_tokens(merge_system, run) + 80
    budget = N_CTX - max_out - overhead - 100
    if budget < 300:
        raise SummarizerError("Context window is too small for these settings. Increase PROTOCOLX_CTX.")

    highlights = scan_highlights(text, my_name)
    chunks = split_into_chunks(text, budget, run)
    user_tmpl = 'Conversation:\n"""\n{}\n"""\n\nWrite the summary now.'
    merge_tail = "\n\nWrite the merged summary now."

    if len(chunks) == 1:
        yield "Summarizing...", ""
        final = ""
        for final in _generate(system, user_tmpl.format(chunks[0]), max_out, run):
            yield "Summarizing...", final
        yield "Done.", final + ("\n\n" + highlights if highlights else "")
        return

    # long input: summarize each chunk, then merge
    partials = []
    for i, chunk in enumerate(chunks, 1):
        yield f"Long conversation: reading part {i} of {len(chunks)}...", ""
        partials.append(_complete(system, user_tmpl.format(chunk), 400, run))

    while True:
        joined = "\n\n".join(f"--- Part {i} ---\n{p}" for i, p in enumerate(partials, 1))
        if _count_tokens(joined, run) <= budget:
            break
        sizes = _count_many(partials, run)
        groups, cur, cur_t = [], [], 0
        for p, n in zip(partials, sizes):
            t = n + 10
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
            _complete(merge_system, "Partial summaries:\n" + "\n\n".join(g) + merge_tail, 400, run)
            if len(g) > 1 else g[0]
            for g in groups
        ]

    yield "Merging parts into the final summary...", ""
    final = ""
    for final in _generate(merge_system, "Partial summaries:\n" + joined + merge_tail, max_out, run):
        yield "Merging...", final
    yield "Done.", final + ("\n\n" + highlights if highlights else "")
