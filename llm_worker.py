"""llm_worker.py - runs the Qwen model in its own process.

summarizer.py starts this script and talks to it with JSON lines over stdin/stdout.
Because the model lives in a separate process, the Stop button can kill it instantly,
even while it is still reading a long input, and then start a fresh one.
Do not run this file by hand.
"""
import json
import sys


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    model_path, n_ctx, n_threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "error": f"Could not load the model: {e}"})
        return
    emit({"type": "ready"})

    for line in sys.stdin:  # ends when the parent closes the pipe or exits
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid = req.get("id")
        try:
            if req["op"] == "count":
                counts = [len(llm.tokenize(t.encode("utf-8"), add_bos=False)) for t in req["texts"]]
                emit({"type": "counts", "id": rid, "counts": counts})
            elif req["op"] == "chat":
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
        except Exception as e:  # noqa: BLE001
            emit({"type": "error", "id": rid, "error": str(e)})


if __name__ == "__main__":
    main()
