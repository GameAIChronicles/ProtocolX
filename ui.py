"""ProtocolX UI: paste a chat, get a local AI summary of what you missed."""
import tempfile

import gradio as gr

import summarizer

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


def do_summarize(conversation, length, my_name):
    if not summarizer.is_loaded():
        yield "Model not loaded. Click 'Load Model' first.", "", None
        return
    try:
        last = ""
        for status, md in summarizer.summarize(conversation, length, my_name):
            last = md
            yield status, md, None
        f = tempfile.NamedTemporaryFile("w", suffix=".md", prefix="protocolx_summary_",
                                        delete=False, encoding="utf-8")
        f.write(last)
        f.close()
        yield "Done.", last, f.name
    except summarizer.SummarizerError as e:
        yield f"Error: {e}", "", None
    except Exception as e:  # noqa: BLE001
        yield f"Unexpected error: {e}", "", None


with gr.Blocks(title="ProtocolX") as demo:
    gr.Markdown("# ProtocolX - What Did I Miss?\n"
                "Paste a chat and get a structured summary. Runs 100% locally; nothing leaves your device.")

    with gr.Row():
        load_btn = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False, scale=3)

    chat = gr.Textbox(label="Conversation", lines=14, max_lines=30,
                      placeholder="Paste the conversation here, one message per line...")
    counter = gr.Markdown(count_text(""))

    with gr.Row():
        length = gr.Radio(["short", "medium", "detailed"], value="medium", label="Summary length")
        my_name = gr.Textbox(label="Your name (to flag mentions of you)", placeholder="e.g. Sam")

    with gr.Row():
        go_btn = gr.Button("Summarize Conversation", variant="primary")
        sample_btn = gr.Button("Load sample chat")
        clear_btn = gr.Button("Clear")

    output = gr.Markdown(label="Summary")
    download = gr.File(label="Download summary (.md)")

    demo.load(do_load, outputs=status)  # auto-load on startup
    load_btn.click(do_load, outputs=status)
    chat.change(count_text, chat, counter)
    sample_btn.click(lambda: SAMPLE, outputs=chat)
    go_btn.click(do_summarize, [chat, length, my_name], [status, output, download])
    clear_btn.click(lambda: ("", "", None, ""), outputs=[chat, output, download, status])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1")
