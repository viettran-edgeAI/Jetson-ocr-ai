from __future__ import annotations

import json
import unittest

import llm_service.main as llm_main
from llm_service.main import AnswerRequest, ConversationMessage


class LlmServiceConfigTests(unittest.TestCase):
    def test_llama_command_disables_gemma_thinking_and_uses_short_context(self) -> None:
        command = llm_main.build_llama_command()

        self.assertEqual(command[command.index("--ctx-size") + 1], "4096")
        self.assertIn("--reasoning", command)
        self.assertEqual(command[command.index("--reasoning") + 1], "off")
        self.assertIn("--reasoning-budget", command)
        self.assertEqual(command[command.index("--reasoning-budget") + 1], "0")

        template_kwargs = command[command.index("--chat-template-kwargs") + 1]
        self.assertEqual(json.loads(template_kwargs), {"enable_thinking": False})

    def test_payload_defaults_for_one_shot_answers(self) -> None:
        payload = llm_main.build_chat_payload(
            AnswerRequest(ocr_markdown="Which option? Product Owner", user_request="answer question")
        )

        self.assertFalse(payload["stream"])
        self.assertEqual(payload["max_tokens"], 160)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("Do not include hidden reasoning", payload["messages"][0]["content"])
        self.assertIn("OCR Markdown for this file session", payload["messages"][0]["content"])

    def test_payload_includes_session_conversation_before_latest_request(self) -> None:
        payload = llm_main.build_chat_payload(
            AnswerRequest(
                ocr_markdown="Invoice total is $42.",
                user_request="Why?",
                conversation_history=[
                    ConversationMessage(role="user", content="What is the total?"),
                    ConversationMessage(role="assistant", content="The total is $42."),
                ],
            )
        )

        self.assertEqual(payload["messages"][1], {"role": "user", "content": "What is the total?"})
        self.assertEqual(payload["messages"][2], {"role": "assistant", "content": "The total is $42."})
        self.assertEqual(payload["messages"][3], {"role": "user", "content": "Why?"})

    def test_conversation_history_is_bounded(self) -> None:
        original_budget = llm_main.DEFAULT_MAX_HISTORY_CHARS
        try:
            llm_main.DEFAULT_MAX_HISTORY_CHARS = 40
            history = llm_main.prepare_conversation_history(
                [
                    ConversationMessage(role="user", content="old question"),
                    ConversationMessage(role="assistant", content="x" * 100),
                ]
            )
        finally:
            llm_main.DEFAULT_MAX_HISTORY_CHARS = original_budget

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "assistant")
        self.assertLessEqual(len(history[0]["content"]), 40)
        self.assertIn("Conversation history truncated", history[0]["content"])

    def test_ocr_markdown_is_truncated_to_context_budget(self) -> None:
        original_budget = llm_main.DEFAULT_MAX_OCR_CHARS
        try:
            llm_main.DEFAULT_MAX_OCR_CHARS = 40
            prepared = llm_main.prepare_ocr_markdown("x" * 100)
        finally:
            llm_main.DEFAULT_MAX_OCR_CHARS = original_budget

        self.assertTrue(prepared["truncated"])
        self.assertEqual(prepared["original_chars"], 100)
        self.assertIn("OCR Markdown truncated", prepared["text"])


if __name__ == "__main__":
    unittest.main()
