# ProtocolX - local "What Did I Miss?" chat summarizer

Qwen3-0.6B (GGUF) + llama-cpp-python + Gradio. Everything runs on your machine.

Files: `ui.py`, `summarizer.py`, `llm_worker.py` (keep all three in the same folder).

## Run (PyCharm terminal, venv active)
    pip install -r requirements.txt        # llama-cpp-python is already installed, so this only adds gradio
    set PROTOCOLX_MODEL=C:\full\path\Qwen3-0.6B-Q8_0.gguf   # skip if the .gguf sits next to ui.py
    python ui.py

Open http://127.0.0.1:7860 . The model auto-loads on startup (or press Load Model).

Optional env vars: PROTOCOLX_CTX (default 4096), PROTOCOLX_THREADS (default cpu/2).

## Expected output structure
    ## Overview
    ## Key Information
    ## Decisions Made
    ## Progress
    ## Problems and Unresolved Questions
    ## Action Items
    - [High] Book lab room - Riya, today
    ---
    ## Flagged by keyword scan (not AI)   <- mentions of you + deadlines (regex)

Use "Load sample chat" in the UI for a ready test conversation.

## Get the model (it is not in this repo, GitHub blocks files over 100 MB)
1. Open https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/main
2. Download `Qwen3-0.6B-Q8_0.gguf`.
3. Paste it into the same folder as `ui.py` (or set `PROTOCOLX_MODEL` to its full path).

## Using it
- Paste a chat into the box, or drop a `.txt` file on the upload area (it fills the box).
- **Stop** kills the model process, even mid-way through a long input, and starts a fresh one (takes a few seconds). When the status says the model restarted, you can summarize again.
- **Clear** empties the boxes and also stops a running summary.
