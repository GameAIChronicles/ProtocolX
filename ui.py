"""ProtocolX UI: paste or drop a chat, get a local AI summary of what you missed."""
import tempfile
from pathlib import Path

import gradio as gr

import summarizer

MAX_FILE_BYTES = 5_000_000

SAMPLE = """Riya: Team, demo is on Friday 5pm. We need the drone sim ready by Thursday.
Arjun: I trained the PPO agent, it reaches the target 8/10 times now.
Riya: Nice. Should we switch to SAC? I read it's more stable.
Arjun: Let's not switch now, too risky before the demo. Stay with PPO.
Riya: Agreed, PPO it is.
Meera: @Sam can you record the demo video by Thursday noon?
Sam: Sure, I'll do it. Still need the final reward curve plot though.
Arjun: Plot is blocked, the logging callback crashes on Windows. Not fixed yet.
Meera: Also nobody has booked the lab room for Friday. Urgent!
Riya: I'll book it today. Arjun please fix the logger tomorrow."""


def count_text(text):
    text = text or ""
    return f"{len(text.split()):,} words | {len(text):,} characters"


def do_load():
    try:
        return summarizer.load_model()
    except summarizer.SummarizerError as e:
        return f"Error: {e}"


def load_file(f):
    """Dropped .txt file -> conversation textbox."""
    if f is None:
        return gr.update(), "No file."
    path = Path(getattr(f, "name", f))
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return gr.update(), f"File too large (limit {MAX_FILE_BYTES // 1_000_000} MB)."
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    except OSError as e:
        return gr.update(), f"Could not read the file: {e}"
    return text, f"Loaded {path.name} ({len(text):,} characters). Press Summarize."


def do_summarize(conversation, length, my_name):
    if not summarizer.is_loaded():
        yield "Model not loaded. Click 'Load Model' first.", "", None
        return
    last = ""
    try:
        for status, md in summarizer.summarize(conversation, length, my_name):
            last = md
            yield status, md, None
        f = tempfile.NamedTemporaryFile("w", suffix=".md", prefix="protocolx_summary_",
                                        delete=False, encoding="utf-8")
        f.write(last)
        f.close()
        yield "Done.", last, f.name
    except summarizer.SummarizerStopped:
        yield gr.update(), last, None  # status text is set by the Stop button
    except summarizer.SummarizerError as e:
        yield f"Error: {e}", "", None
    except Exception as e:  # noqa: BLE001
        yield f"Unexpected error: {e}", "", None


def do_stop():
    if not summarizer.is_busy():
        yield "Nothing is running."
        return
    yield "Stopping and restarting the model (a few seconds)..."
    summarizer.stop()  # kills the model process and starts a fresh one
    if summarizer.is_loaded():
        yield "Stopped. Model restarted and ready for the next run."
    else:
        yield "Stopped, but the model did not restart. Press Load Model."


def do_clear():
    stopped = summarizer.stop()  # Clear also stops a running summary
    msg = "Cleared. Model restarted." if stopped and summarizer.is_loaded() else "Cleared."
    return "", "", None, msg, None


with gr.Blocks(title="ProtocolX") as demo:
    gr.Markdown("# ProtocolX - What Did I Miss?\n"
                "Paste a chat or drop a .txt file and get a structured summary. "
                "Runs 100% locally; nothing leaves your device.")

    with gr.Row():
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False, scale=3)

    file_in = gr.File(label="Drop a .txt chat file here (optional)", file_types=[".txt"])
    chat = gr.Textbox(label="Conversation", lines=14, max_lines=30,
                      placeholder="Paste the conversation here, one message per line...")
    counter = gr.Markdown(count_text(""))

    with gr.Row():
        length = gr.Radio(["short", "medium", "detailed"], value="medium", label="Summary length")
        my_name = gr.Textbox(label="Your name (to flag mentions of you)", placeholder="e.g. Sam")

    with gr.Row():
        go_btn = gr.Button("Summarize Conversation", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop")
        sample_btn = gr.Button("Load sample chat")
        clear_btn = gr.Button("Clear")

    output = gr.Markdown(label="Summary")
    download = gr.File(label="Download summary (.md)")

    demo.load(do_load, outputs=status)  # auto-load on startup
    load_btn.click(do_load, outputs=status)
    file_in.upload(load_file, file_in, [chat, status])
    chat.change(count_text, chat, counter)
    sample_btn.click(lambda: SAMPLE, outputs=chat)
    go_btn.click(do_summarize, [chat, length, my_name], [status, output, download])
    stop_btn.click(do_stop, outputs=status)
    clear_btn.click(do_clear, outputs=[chat, output, download, status, file_in])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1")
