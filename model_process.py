"""model_process.py - starts, talks to, and stops the model process (llm_worker.py).

This module is the "backend" of ProtocolX. The rest of the app never touches llama.cpp
directly; it only calls the small API below:

    load_model()        start the model process and wait until it is ready
    is_loaded()         is the model ready to use?
    run_session()       context manager that marks "a summary is running" and gives it an id
    count_tokens()      how many tokens are these texts?
    stream_chat()       generate a reply, yielding the text as it grows
    complete_chat()     same, but returns only the final text
    stop()              kill the model process and start a fresh one (the Stop button)

Stop mechanism, in short: every running summary has a "run id". stop() bumps the global
run id and kills the process. The running summary notices the mismatch within 0.2 s and
raises RunStopped. Because the whole process is killed, even a long prompt-reading phase
ends immediately.
"""
import atexit
import itertools
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import config

# Log events only (loading, stopping...). Conversation text is NEVER logged.
log = logging.getLogger("protocolx.model")


class ModelError(Exception):
    """A problem the user should see (missing file, crashed process...). Messages are user-safe."""


class RunStopped(Exception):
    """Raised inside a running summary when the user pressed Stop or Clear."""


# ----------------------------------------------------------------------------- worker handle
class _Worker:
    """One running llm_worker.py process plus a thread that collects its replies."""

    def __init__(self, model_path: str):
        # On Windows, CREATE_NO_WINDOW stops a black console window from popping up.
        flags = 0x08000000 if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [sys.executable, str(config.WORKER_FILE), model_path, str(config.N_CTX), str(config.N_THREADS)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # DEBUG=1 lets the worker's own errors show up in your console.
            stderr=None if config.DEBUG else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,  # line buffered: each JSON line is delivered immediately
            creationflags=flags,
        )
        self.q: "queue.Queue[dict]" = queue.Queue()  # replies from the worker
        self.ready = False
        # A background thread turns the worker's stdout lines into messages on the queue,
        # so the rest of the app can wait with a timeout instead of blocking forever.
        threading.Thread(target=self._reader, daemon=True).start()
        log.info("Model process started (pid %s)", self.proc.pid)

    def _reader(self) -> None:
        try:
            for line in self.proc.stdout:
                try:
                    self.q.put(json.loads(line))
                except ValueError:
                    continue  # stray non-JSON output from native libraries
        except (OSError, ValueError):
            pass
        self.q.put({"type": "closed"})  # the pipe ended: process exited or was killed

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        self.ready = False
        try:
            self.proc.kill()
        except OSError:
            pass


# ----------------------------------------------------------------------------- module state
_worker: Optional[_Worker] = None
_last_path: Optional[str] = None  # remembered so stop() can restart the same model
_load_lock = threading.Lock()  # only one load/restart at a time
_infer_lock = threading.Lock()  # the worker handles one request at a time
_stop_lock = threading.Lock()  # ignore a second Stop click while one is in progress
_state_lock = threading.Lock()  # protects the two counters below
_active = 0  # number of summaries currently running
_run_id = 0  # id of the newest run; older ids are "stopped"
_ids = itertools.count(1)  # request ids, so replies can be matched to requests


def _kill_at_exit() -> None:
    """Never leave an orphan model process behind when the app closes."""
    if _worker is not None:
        _worker.kill()


atexit.register(_kill_at_exit)


# ----------------------------------------------------------------------------- lifecycle
def is_loaded() -> bool:
    """True when the model process is running and has finished loading."""
    return _worker is not None and _worker.ready and _worker.alive()


def is_busy() -> bool:
    """True while at least one summary is running."""
    return _active > 0


def load_model(path: Optional[str] = None, force: bool = False) -> str:
    """Start the model process and wait until it reports "ready".

    Does nothing (and says so) if the model is already loaded, unless force=True.
    Raises ModelError with a readable message if anything goes wrong.
    """
    global _worker, _last_path
    with _load_lock:
        if _worker is not None and not force and is_loaded():
            return f"Model already loaded (context {config.N_CTX}, {config.N_THREADS} threads)."
        if _worker is not None:  # a dead or unwanted process: clean it up first
            _worker.kill()
            _worker = None

        model_path = Path(path or _last_path or config.MODEL_PATH)
        if not model_path.is_file():
            raise ModelError(
                f"Model file not found: {model_path}\n"
                "Download Qwen3-0.6B-Q8_0.gguf (see the README) and put it next to ui.py, "
                "or set PROTOCOLX_MODEL to its full path."
            )
        if not config.WORKER_FILE.is_file():
            raise ModelError(f"llm_worker.py is missing (expected at {config.WORKER_FILE}).")

        _last_path = str(model_path.resolve())
        started = time.time()
        worker = _Worker(_last_path)

        # Wait for the worker's first message: "ready" or "error".
        while True:
            try:
                msg = worker.q.get(timeout=0.5)
            except queue.Empty:
                if not worker.alive():
                    raise ModelError("The model process exited while loading. Check that "
                                     "llama-cpp-python works in this environment (try PROTOCOLX_DEBUG=1).")
                if time.time() - started > config.LOAD_TIMEOUT_S:
                    worker.kill()
                    raise ModelError("Timed out while loading the model.")
                continue
            if msg.get("type") == "ready":
                worker.ready = True
                break
            if msg.get("type") in ("error", "closed"):
                worker.kill()
                raise ModelError(msg.get("error", "The model process ended while loading."))

        _worker = worker
        log.info("Model ready in %.1fs", time.time() - started)
        return f"Model loaded: {model_path.name} (context {config.N_CTX}, {config.N_THREADS} threads)."


def stop() -> bool:
    """Stop the running summary: kill the model process and start a fresh one.

    Returns True if a summary was running, False if there was nothing to stop.
    Blocks for a few seconds until the new model process is ready.
    """
    global _run_id
    with _state_lock:
        if _active == 0:
            return False
    if not _stop_lock.acquire(blocking=False):
        return True  # another stop is already in progress
    try:
        log.info("Stop requested: restarting the model process")
        with _state_lock:
            _run_id += 1  # step 1: running summaries now see "my id is outdated"
        if _worker is not None:
            _worker.kill()  # step 2: end the process, even if it is mid-prompt
        try:
            load_model(force=True)  # step 3: bring up a fresh one
        except ModelError as e:
            log.warning("Restart after stop failed: %s", e)  # is_loaded() will be False
    finally:
        _stop_lock.release()
    return True


@contextmanager
def run_session() -> Iterator[int]:
    """Mark the start and end of one summary and hand out its run id.

    Usage:  with run_session() as run: ... pass `run` to count_tokens / stream_chat.
    Starting a session also invalidates any older run that is still finishing.
    """
    global _active, _run_id
    with _state_lock:
        _active += 1
        _run_id += 1
        run = _run_id
    try:
        yield run
    finally:
        with _state_lock:
            _active -= 1


# ----------------------------------------------------------------------------- talking to the worker
def _check(run: int) -> None:
    """Raise RunStopped if this run has been replaced or stopped."""
    if run != _run_id:
        raise RunStopped()


def _require_worker() -> _Worker:
    if not is_loaded():
        raise ModelError("Model is not loaded (or is restarting). Wait a few seconds or press Load Model.")
    return _worker  # type: ignore[return-value]


def _send(worker: _Worker, obj: dict, run: int) -> None:
    """Write one JSON request to the worker."""
    try:
        worker.proc.stdin.write(json.dumps(obj) + "\n")
        worker.proc.stdin.flush()
    except (OSError, ValueError) as e:
        _check(run)  # if Stop killed it, that is not an error
        raise ModelError("Lost connection to the model process.") from e


def _next_msg(worker: _Worker, run: int) -> dict:
    """Wait for the next reply. Wakes up every 0.2 s to notice Stop or a crashed process."""
    while True:
        _check(run)
        try:
            msg = worker.q.get(timeout=0.2)
        except queue.Empty:
            if not worker.alive():
                _check(run)
                raise ModelError("The model process stopped unexpectedly. Press Load Model.")
            continue
        if msg.get("type") == "closed":
            _check(run)
            raise ModelError("The model process stopped unexpectedly. Press Load Model.")
        return msg


def count_tokens(texts: List[str], run: int) -> List[int]:
    """Token count for each text. One round trip for the whole list (fast)."""
    worker = _require_worker()
    with _infer_lock:
        _check(run)
        rid = next(_ids)
        _send(worker, {"op": "count", "id": rid, "texts": texts}, run)
        while True:
            msg = _next_msg(worker, run)
            if msg.get("id") != rid:
                continue  # leftover message from an older request
            if msg["type"] == "counts":
                return msg["counts"]
            if msg["type"] == "error":
                raise ModelError(f"Token counting failed: {msg.get('error')}")


# Qwen3 can emit a hidden "<think>...</think>" block before answering. We switch it off with
# "/no_think" (below) and also strip any block that still appears.
_THINK_DONE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove Qwen3 reasoning blocks (finished ones, and an unfinished one still streaming)."""
    text = _THINK_DONE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def stream_chat(system: str, user: str, max_tokens: int, run: int) -> Iterator[str]:
    """Ask the model and yield the reply so far, growing token by token."""
    worker = _require_worker()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user + "\n/no_think"},  # Qwen3 switch: answer directly
    ]
    with _infer_lock:
        _check(run)
        rid = next(_ids)
        _send(worker, {"op": "chat", "id": rid, "messages": messages, "max_tokens": max_tokens}, run)
        reply = ""
        while True:
            msg = _next_msg(worker, run)
            if msg.get("id") != rid:
                continue
            kind = msg["type"]
            if kind == "token":
                reply += msg["text"]
                yield strip_think(reply)
            elif kind == "done":
                return
            elif kind == "error":
                raise ModelError(f"Inference failed: {msg.get('error')}")


def complete_chat(system: str, user: str, max_tokens: int, run: int) -> str:
    """Like stream_chat but returns only the final text (used for the partial summaries)."""
    last = ""
    for last in stream_chat(system, user, max_tokens, run):
        pass
    return last
