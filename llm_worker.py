"""llm_worker.py - hosts the Qwen model in its own process.

Why a separate process?
    A running llama.cpp call cannot be interrupted from Python, especially while it is
    still reading a long prompt. If the model lives in its own process, the Stop button
    can simply kill that process and start a fresh one. It also keeps the (heavy) model
    memory out of the web server process.

How it talks to the app (one JSON object per line):
    app -> worker (stdin)
        {"op": "count", "id": 1, "texts": ["...", "..."]}          count tokens
        {"op": "chat",  "id": 2, "messages": [...], "max_tokens": 600}   generate a reply
    worker -> app (stdout)
        {"type": "ready"}                                         model loaded
        {"type": "counts", "id": 1, "counts": [12, 40]}           answer to "count"
        {"type": "token",  "id": 2, "text": "..."}                one streamed piece of text
        {"type": "done",   "id": 2}                               reply finished
        {"type": "error",  "id": 2, "error": "..."}               something failed

This is the ONLY file that imports llama_cpp. Do not run it by hand; model_process.py
starts it as: python llm_worker.py <model_path> <n_ctx> <n_threads>
"""
import json
import sys


def emit(obj: dict) -> None:
    """Send one JSON message to the app. flush() makes sure it is not held in a buffer."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    model_path, n_ctx, n_threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    # ---- 1. load the model (the slow part, a few seconds) ----
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
    except Exception as e:  # noqa: BLE001 - any failure must be reported to the app
        emit({"type": "error", "error": f"Could not load the model: {e}"})
        return
    emit({"type": "ready"})

    # ---- 2. serve requests until the app closes the pipe or kills us ----
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue  # ignore garbage instead of crashing
        rid = req.get("id")
        try:
            if req["op"] == "count":
                # add_bos=False: count only the text itself, matching how we budget the context
                counts = [len(llm.tokenize(t.encode("utf-8"), add_bos=False)) for t in req["texts"]]
                emit({"type": "counts", "id": rid, "counts": counts})

            elif req["op"] == "chat":
                # Low temperature = consistent, factual summaries (no creative drifting).
                # repeat_penalty discourages the small model from looping on the same bullet.
                stream = llm.create_chat_completion(
                    messages=req["messages"],
                    max_tokens=req["max_tokens"],
                    temperature=0.1,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    stream=True,
                )
                for part in stream:
                    delta = part["choices"][0]["delta"].get("content")
                    if delta:
                        emit({"type": "token", "id": rid, "text": delta})
                emit({"type": "done", "id": rid})
        except Exception as e:  # noqa: BLE001 - report, then keep serving
            emit({"type": "error", "id": rid, "error": str(e)})


if __name__ == "__main__":
    main()
