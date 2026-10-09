"""test_core.py - quick tests for the parts that do not need the model.

Run from the project folder:   python -m unittest -v
No GPU, no model file and no Gradio are required, so these also run on any laptop or CI.
"""
import unittest
from unittest import mock

import highlights
import model_process
import summarizer


class CleanConversationTests(unittest.TestCase):
    def test_removes_blank_lines_and_consecutive_duplicates(self):
        raw = "Riya: hi\n\nRiya: hi\nSam: hello\n"
        self.assertEqual(summarizer.clean_conversation(raw), "Riya: hi\nSam: hello")

    def test_strips_android_whatsapp_prefix(self):
        raw = "12/05/2024, 10:15 pm - Riya: Demo is Friday"
        self.assertEqual(summarizer.clean_conversation(raw), "Riya: Demo is Friday")

    def test_strips_iphone_whatsapp_prefix_with_invisible_characters(self):
        raw = "\u200e[12/05/24, 10:15:32\u202fPM] Riya: hello there"
        self.assertEqual(summarizer.clean_conversation(raw), "Riya: hello there")

    def test_drops_media_and_system_lines_but_keeps_real_messages(self):
        raw = ("Arjun: <Media omitted>\n"
               "Messages and calls are end-to-end encrypted. No one outside of this chat can read them.\n"
               "Sam: we should discuss end-to-end encrypted storage")
        self.assertEqual(summarizer.clean_conversation(raw), "Sam: we should discuss end-to-end encrypted storage")

    def test_empty_input(self):
        self.assertEqual(summarizer.clean_conversation(""), "")
        self.assertEqual(summarizer.clean_conversation(None), "")


class PriorityTests(unittest.TestCase):
    CHAT = ("Meera: @Sam can you record the demo video by Thursday noon?\n"
            "Meera: Also nobody has booked the lab room for Friday. Urgent!\n"
            "Riya: I'll book it today. Arjun please fix the logger tomorrow.\n"
            "Arjun: Thus we sat and talked.")

    def test_mention_plus_deadline_is_high_and_ranked_first(self):
        items = highlights.find_priorities(self.CHAT, "Sam")
        self.assertEqual(items[0].level, "high")
        self.assertIn("mentions you", items[0].reasons)

    def test_urgent_word_with_day_is_high(self):
        items = highlights.find_priorities(self.CHAT, "")
        self.assertTrue(any(p.level == "high" and "urgent" in p.reasons for p in items))

    def test_small_talk_and_false_friends_are_ignored(self):
        lines = [p.line for p in highlights.find_priorities(self.CHAT, "Sam")]
        self.assertFalse(any("Thus we sat" in line for line in lines))  # "thus"/"sat" are not days

    def test_nothing_to_flag_gives_empty_text(self):
        self.assertEqual(highlights.format_priorities(highlights.find_priorities("A: hi\nB: hello", "")), "")


class ThinkBlockTests(unittest.TestCase):
    def test_finished_block_removed(self):
        self.assertEqual(model_process.strip_think("<think>hmm</think>\n## Overview"), "## Overview")

    def test_unfinished_block_hidden_while_streaming(self):
        self.assertEqual(model_process.strip_think("<think>still thinking"), "")


class ChunkingTests(unittest.TestCase):
    """split_into_chunks needs a token counter; we replace the model with 'one word = one token'."""

    @staticmethod
    def _fake_count(texts, run):
        return [len(t.split()) for t in texts]

    def test_chunks_respect_budget_and_keep_every_message(self):
        text = "\n".join(f"U{i}: " + "word " * 9 for i in range(20))  # 20 messages of ~10 tokens
        with mock.patch.object(model_process, "count_tokens", self._fake_count):
            chunks = summarizer.split_into_chunks(text, budget=50, run=1)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("\n".join(chunks), text)  # nothing lost, nothing duplicated
        for chunk in chunks:
            self.assertLessEqual(sum(self._fake_count([line], 1)[0] + 1 for line in chunk.split("\n")), 50)

    def test_one_huge_message_is_cut_into_pieces(self):
        with mock.patch.object(model_process, "count_tokens", self._fake_count):
            chunks = summarizer.split_into_chunks("word " * 500, budget=100, run=1)
        self.assertGreater(len(chunks), 1)


class InputValidationTests(unittest.TestCase):
    def test_empty_and_short_input_rejected_before_the_model_is_needed(self):
        with self.assertRaises(summarizer.InputError):
            next(summarizer.summarize("   "))
        with self.assertRaises(summarizer.InputError):
            next(summarizer.summarize("hi"))


if __name__ == "__main__":
    unittest.main()
