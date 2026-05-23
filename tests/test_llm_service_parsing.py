from __future__ import annotations

import unittest

from llm_service.main import (
    AnswerRequest,
    DEFAULT_MAX_TOKENS,
    DEFAULT_THINKING_MAX_TOKENS,
    DISABLE_THINKING,
    build_chat_payload,
    extract_delta_content,
    extract_delta_parts,
    extract_message_parts,
    extract_message_text,
    resolve_max_tokens,
    split_reasoning_output,
)


class LlmServiceParsingTests(unittest.TestCase):
    def test_extract_message_prefers_content(self) -> None:
        message = {
            "content": "final answer",
            "reasoning": "internal reasoning",
            "reasoning_content": "internal reasoning fallback",
        }
        self.assertEqual(extract_message_text(message), "final answer")

    def test_extract_message_falls_back_to_reasoning(self) -> None:
        message = {
            "content": "",
            "reasoning": "answer emitted in reasoning",
            "reasoning_content": "",
        }
        self.assertEqual(extract_message_text(message), "answer emitted in reasoning")

    def test_extract_message_supports_openai_content_parts(self) -> None:
        message = {
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "output_text", "text": "world"},
            ]
        }
        self.assertEqual(extract_message_text(message), "hello world")

    def test_extract_delta_content_falls_back_to_reasoning(self) -> None:
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "",
                        "reasoning_content": "token from reasoning",
                    },
                }
            ]
        }
        self.assertEqual(extract_delta_content(chunk), "token from reasoning")

    def test_extract_delta_parts_separates_answer_and_reasoning(self) -> None:
        chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "final token",
                        "reasoning_content": "thinking token",
                    },
                }
            ]
        }
        self.assertEqual(
            extract_delta_parts(chunk),
            {"answer": "final token", "reasoning": "thinking token"},
        )

    def test_extract_message_parts_prefers_content_and_preserves_reasoning(self) -> None:
        message = {
            "content": "final answer",
            "reasoning_content": "private reasoning",
        }
        self.assertEqual(extract_message_parts(message), ("final answer", "private reasoning"))

    def test_split_reasoning_output_separates_step_trace_from_final_answer(self) -> None:
        combined = (
            "1. Analyze: inspect the question.\n"
            "2. Decide: prepare the explanation.\n\n"
            "The four main factors are latency, price, service availability, and compliance."
        )
        self.assertEqual(
            split_reasoning_output(combined),
            (
                "The four main factors are latency, price, service availability, and compliance.",
                "1. Analyze: inspect the question.\n2. Decide: prepare the explanation.",
            ),
        )

    def test_resolve_max_tokens_fast_mode_uses_default(self) -> None:
        request = AnswerRequest(user_request="hello", thinking_mode="fast")
        self.assertEqual(resolve_max_tokens(request), DEFAULT_MAX_TOKENS)

    def test_resolve_max_tokens_thinking_mode_uses_thinking_default(self) -> None:
        request = AnswerRequest(user_request="hello", thinking_mode="thinking")
        expected = DEFAULT_MAX_TOKENS if DISABLE_THINKING else DEFAULT_THINKING_MAX_TOKENS
        self.assertEqual(resolve_max_tokens(request), expected)

    def test_resolve_max_tokens_explicit_override_wins(self) -> None:
        request = AnswerRequest(user_request="hello", thinking_mode="thinking", max_tokens=321)
        self.assertEqual(resolve_max_tokens(request), 321)

    def test_build_chat_payload_fast_mode_disables_template_thinking(self) -> None:
        request = AnswerRequest(user_request="hello", thinking_mode="fast")
        payload = build_chat_payload(request)

        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("Do not include hidden reasoning", payload["messages"][0]["content"])

    def test_build_chat_payload_thinking_mode_allows_concise_reasoning(self) -> None:
        if DISABLE_THINKING:
            self.skipTest("thinking mode is globally disabled")

        request = AnswerRequest(user_request="hello", thinking_mode="thinking")
        payload = build_chat_payload(request)

        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual(payload["max_tokens"], DEFAULT_THINKING_MAX_TOKENS)
        self.assertIn("Thinking mode is active", payload["messages"][0]["content"])
        self.assertIn("concise and user-facing", payload["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
