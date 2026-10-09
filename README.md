# ProtocolX - local "What Did I Miss?" chat summarizer

Qwen3-0.6B (GGUF) + llama-cpp-python + Gradio. Everything runs on your machine.

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
