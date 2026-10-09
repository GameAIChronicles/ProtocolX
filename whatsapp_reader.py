"""whatsapp_reader.py - read a WhatsApp exported chat and print it as plain text.

WhatsApp has no official way to read your personal chats from a program, so the
reliable local-first route is WhatsApp's own "Export chat" feature:

  Android: open the chat -> ⋮ menu -> More -> Export chat -> Without media
  iPhone : open the chat -> tap the contact/group name -> Export Chat -> Without Media

Send the exported .txt (or .zip) to your PC (email, Drive, "message yourself" on
WhatsApp Web), then run:

    python whatsapp_reader.py path\\to\\chat.txt
    python whatsapp_reader.py path\\to\\chat.zip --last 50
    python whatsapp_reader.py                      (opens a file picker)

Output is "Sender: message" lines, ready to paste into ProtocolX.
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

# invisible characters WhatsApp inserts (narrow no-break space before am/pm, direction marks)
_CLEAN = str.maketrans({"\u202f": " ", "\u00a0": " ", "\u200e": None, "\u200f": None, "\ufeff": None})

_DATE = r"(\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4})"
_TIME = r"(\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)"
# Android: 12/05/2024, 10:15 pm - Riya: hello
_ANDROID = re.compile(rf"^{_DATE},?\s+{_TIME}\s+-\s+(.*)$")
# iPhone:  [12/05/24, 10:15:32 PM] Riya: hello
_IOS = re.compile(rf"^\[{_DATE},?\s+{_TIME}\]\s+(.*)$")

_SKIP = re.compile(
    r"(?i)^(<media omitted>|<attached: .*>|.*\b(image|video|audio|sticker|gif|document|contact card) omitted"
    r"|.*\(file attached\)|this message was deleted|you deleted this message|null)$"
)


def load_text(path: str) -> str:
    """Read an exported chat from a .txt or a .zip (first .txt inside)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not names:
                raise ValueError("No .txt chat file found inside the zip.")
            raw = z.read(names[0])
    else:
        raw = p.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_chat(text: str) -> List[Dict[str, str]]:
    """Return messages as dicts: date, time, sender, text. System lines are dropped."""
    messages: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    for line in text.translate(_CLEAN).replace("\r\n", "\n").split("\n"):
        m = _ANDROID.match(line) or _IOS.match(line)
        if m:
            date, time, rest = m.groups()
            if ": " in rest:
                sender, body = rest.split(": ", 1)
                current = {"date": date, "time": time, "sender": sender.strip(), "text": body.strip()}
                messages.append(current)
            else:
                current = None  # system message (encryption notice, "X joined", ...)
        elif current is not None and line.strip():
            current["text"] += " " + line.strip()  # multi-line message continues
    return [m for m in messages if m["text"] and not _SKIP.match(m["text"])]


def format_messages(messages: List[Dict[str, str]], with_time: bool = False) -> str:
    lines = []
    for m in messages:
        prefix = f"[{m['date']} {m['time']}] " if with_time else ""
        lines.append(f"{prefix}{m['sender']}: {m['text']}")
    return "\n".join(lines)


def load_chat(path: str, last: Optional[int] = None, with_time: bool = False) -> str:
    """One-call helper (also usable from ui.py): file path -> 'Sender: message' text."""
    messages = parse_chat(load_text(path))
    if last:
        messages = messages[-last:]
    return format_messages(messages, with_time)


def _pick_file() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select exported WhatsApp chat",
        filetypes=[("WhatsApp export", "*.txt *.zip"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Print a WhatsApp exported chat as plain text.")
    ap.add_argument("path", nargs="?", help="exported chat .txt or .zip (omit to pick a file)")
    ap.add_argument("--last", type=int, help="only the last N messages")
    ap.add_argument("--timestamps", action="store_true", help="include date and time")
    args = ap.parse_args()

    path = args.path or _pick_file()
    if not path:
        sys.exit("No file selected.")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emojis on Windows consoles

    try:
        messages = parse_chat(load_text(path))
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as e:
        sys.exit(f"Error: {e}")
    if not messages:
        sys.exit("No messages found. Is this a WhatsApp 'Export chat' file?")
    if args.last:
        messages = messages[-args.last:]

    senders = {m["sender"] for m in messages}
    print(f"# {len(messages)} messages from {len(senders)} people", file=sys.stderr)
    print(format_messages(messages, args.timestamps))


if __name__ == "__main__":
    main()
