"""ui.py - the Gradio web interface (presentation layer only).

This file contains no summarization logic. It collects input, calls summarizer.py /
model_process.py, and shows the results. Run it with:  python ui.py

Privacy by design (the environment variables below must be set BEFORE gradio is imported):
    * analytics are switched off,
    * every temporary file (uploads, the downloadable summary) lives in ONE private temp
      folder that is deleted when the app exits,
    * the server listens on 127.0.0.1 only and never creates a public share link.
"""
import atexit
import inspect
import logging
import os
import shutil
import tempfile
from pathlib import Path

# ---- private working folder + privacy settings (must come before `import gradio`) ----
WORK_DIR = Path(tempfile.mkdtemp(prefix="protocolx_"))
os.environ["GRADIO_TEMP_DIR"] = str(WORK_DIR)  # uploads and downloads are stored here
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage telemetry
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)  # wipe everything on exit

import gradio as gr  # noqa: E402  (deliberately imported after the environment is prepared)

import config  # noqa: E402
import model_process as model  # noqa: E402
import summarizer  # noqa: E402

log = logging.getLogger("protocolx.ui")

SAMPLE_CHAT = """Riya: Team, demo is on Friday 5pm. We need the drone sim ready by Thursday.
Arjun: I trained the PPO agent, it reaches the target 8/10 times now.
Riya: Nice. Should we switch to SAC? I read it's more stable.
Arjun: Let's not switch now, too risky before the demo. Stay with PPO.
Riya: Agreed, PPO it is.
Meera: @Sam can you record the demo video by Thursday noon?
Sam: Sure, I'll do it. Still need the final reward curve plot though.
Arjun: Plot is blocked, the logging callback crashes on Windows. Not fixed yet.
Meera: Also nobody has booked the lab room for Friday. Urgent!
Riya: I'll book it today. Arjun please fix the logger tomorrow."""

# Small stylesheet: ProtocolX red accent, a card look for the summary, a readable page width.
CSS = """
.gradio-container { max-width: 1250px !important; margin: auto !important; }
#go_btn { background: #e11d2e !important; border-color: #e11d2e !important; color: #fff !important; }
#go_btn:hover { filter: brightness(1.12); }
#stop_btn { border-color: #e11d2e !important; }
#summary_card { border: 1px solid rgba(225, 29, 46, 0.45); border-radius: 12px; padding: 14px 18px; min-height: 320px; }
#counter { opacity: 0.7; font-size: 0.9em; }
"""

PLACEHOLDER = "*Your summary will appear here.*"  # shown in the output card before the first run

# Runs in the browser: copies the text to the clipboard (localhost counts as a secure context).
COPY_JS = "(text) => { if (text) { navigator.clipboard.writeText(text); } return text; }"


# ----------------------------------------------------------------------------- event handlers
def count_text(text: str) -> str:
    """Live word/character counter under the input box, with a hint for long chats."""
    text = text or ""
    info = f"{len(text.split()):,} words | {len(text):,} characters"
    if len(text) > config.LONG_CHAT_HINT_CHARS:
        info += " | long chat: it will be summarized in parts"
    return info


def do_load() -> str:
    """Start the model (also runs automatically when the page opens)."""
    try:
        return model.load_model()
    except model.ModelError as e:
        return f"Error: {e}"


def load_file(f):
    """A .txt file was dropped: read it into the conversation box (limited size, tolerant encoding)."""
    if f is None:
        return gr.update(), "No file."
    path = Path(getattr(f, "name", f))  # Gradio gives a path string or a file-like object
    try:
        if path.stat().st_size > config.MAX_FILE_BYTES:
            return gr.update(), f"File too large (limit {config.MAX_FILE_BYTES // 1_000_000} MB)."
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")  # utf-8, with or without a BOM (WhatsApp exports)
        except UnicodeDecodeError:
            text = raw.decode("latin-1")  # last resort so odd encodings still load
    except OSError as e:
        return gr.update(), f"Could not read the file: {e}"
    return text, f"Loaded {path.name} ({len(text):,} characters). Press Summarize."


def do_summarize(conversation: str, length: str, my_name: str):
    """Run the summary and stream it to the page.

    Yields 4-tuples matching the outputs: (status, rendered summary, raw text for copy, download file).
    """
    last, last_status = "", "Done."
    try:
        for last_status, markdown in summarizer.summarize(conversation, length, my_name):
            last = markdown
            yield last_status, markdown, markdown, None
        # Finished: save the summary as a .md file the user can download.
        out_file = WORK_DIR / "protocolx_summary.md"
        out_file.write_text(last, encoding="utf-8")
        yield last_status, last, last, str(out_file)  # same status text, now with the download file
    except model.RunStopped:
        # Stop/Clear was pressed; that button sets the status text, so leave it untouched.
        yield gr.update(), last, last, None
    except (summarizer.InputError, model.ModelError) as e:
        yield f"Error: {e}", "", "", None
    except Exception as e:  # noqa: BLE001 - never show a stack trace to the user
        log.exception("Unexpected error while summarizing")
        yield f"Unexpected error: {e}", "", "", None


def do_stop():
    """Stop button: kill the running summary and restart the model so the next run works."""
    if not model.is_busy():
        yield "Nothing is running."
        return
    yield "Stopping and restarting the model (a few seconds)..."
    model.stop()
    if model.is_loaded():
        yield "Stopped. Model restarted and ready for the next run."
    else:
        yield "Stopped, but the model did not restart. Press Load Model."


def do_clear():
    """Clear button: empty everything, stop a running summary and forget cached summaries."""
    stopped = model.stop()
    summarizer.clear_cache()
    message = "Cleared. Model restarted." if stopped and model.is_loaded() else "Cleared."
    # Order must match the `outputs` list of clear_btn.click below.
    return "", PLACEHOLDER, "", None, message, None


def copy_status(text: str) -> str:
    """Status message after the Copy button (the actual copying happens in COPY_JS)."""
    return "Summary copied to clipboard." if text else "Nothing to copy yet."


# ----------------------------------------------------------------------------- layout
def _css_target_is_launch() -> bool:
    """Newer Gradio versions take `css` in launch(), older ones in gr.Blocks(). Detect which."""
    try:
        return "css" in inspect.signature(gr.Blocks.launch).parameters
    except (TypeError, ValueError, AttributeError):
        return False


def build_ui() -> gr.Blocks:
    """Create all components and wire them to the handlers above."""
    blocks_kwargs = {} if _css_target_is_launch() else {"css": CSS}
    with gr.Blocks(title="ProtocolX - What Did I Miss?", **blocks_kwargs) as demo:
        gr.Markdown(
            "# 📨 ProtocolX: What Did I Miss?\n"
            "Turn a long, unread chat into a short summary with decisions, action items and what "
            "needs *your* attention. **Runs 100% on your computer. Nothing is sent anywhere.**"
        )

        with gr.Row():
            load_btn = gr.Button("Load Model", scale=1)
            status = gr.Textbox(label="Status", interactive=False, scale=4)

        with gr.Row():
            # ---------------- left column: input ----------------
            with gr.Column(scale=1):
                file_in = gr.File(label="1. Drop a .txt chat (WhatsApp export works) or paste below", file_types=[".txt"])
                chat = gr.Textbox(label="Conversation", lines=14, max_lines=24,
                                  placeholder="One message per line, e.g.\nRiya: Demo is on Friday 5pm")
                counter = gr.Markdown(count_text(""), elem_id="counter")
                with gr.Row():
                    length = gr.Radio(["short", "medium", "detailed"], value="medium", label="Summary length")
                    my_name = gr.Textbox(label="Your name (flags messages about you)", placeholder="e.g. Sam")
                with gr.Row():
                    go_btn = gr.Button("2. Summarize Conversation", variant="primary", elem_id="go_btn", scale=3)
                    stop_btn = gr.Button("Stop", variant="stop", elem_id="stop_btn", scale=1)
                with gr.Row():
                    sample_btn = gr.Button("Load sample chat")
                    clear_btn = gr.Button("Clear")

            # ---------------- right column: output ----------------
            with gr.Column(scale=1):
                gr.Markdown("### 3. Your summary")
                output = gr.Markdown(PLACEHOLDER, elem_id="summary_card")
                raw_summary = gr.Textbox(visible=False)  # plain-text copy of the summary for the Copy button
                with gr.Row():
                    copy_btn = gr.Button("Copy Summary")
                download = gr.File(label="Download summary (.md)", interactive=False)

        with gr.Accordion("How it works and privacy", open=False):
            gr.Markdown(
                "- **Local only:** the Qwen3-0.6B model runs in a separate process on your machine; "
                "no internet connection is used for summarizing.\n"
                "- **Needs your attention** is a rule-based scan (urgent words, deadlines, your name), "
                "so it is instant and never misses a keyword. The AI writes the rest.\n"
                "- **Long chats** are split at message boundaries, summarized in parts and merged.\n"
                "- **Stop** kills and restarts the model process, so it works even on a very long input.\n"
                "- Temporary files are deleted when you close the app. Conversations are never logged."
            )

        # ---------------- wiring: which event calls which handler ----------------
        demo.load(do_load, outputs=status)  # load the model when the page opens
        load_btn.click(do_load, outputs=status)
        file_in.upload(load_file, file_in, [chat, status])
        chat.change(count_text, chat, counter)
        sample_btn.click(lambda: SAMPLE_CHAT, outputs=chat)
        go_btn.click(do_summarize, [chat, length, my_name], [status, output, raw_summary, download])
        stop_btn.click(do_stop, outputs=status)
        clear_btn.click(do_clear, outputs=[chat, output, raw_summary, download, status, file_in])
        # js runs first (copies the text), then copy_status shows a confirmation message.
        copy_btn.click(copy_status, inputs=raw_summary, outputs=status, js=COPY_JS)
    return demo


if __name__ == "__main__":
    # Log events (model loaded, stop pressed...) to the console; conversation text is never logged.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = build_ui().queue()  # queue() lets long summaries stream without blocking other clicks
    launch_kwargs = {"server_name": "127.0.0.1", "share": False, "inbrowser": True}
    if _css_target_is_launch():
        launch_kwargs["css"] = CSS
    app.launch(**launch_kwargs)
