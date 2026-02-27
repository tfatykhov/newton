"""Tests for 007.2 spike: topic persistence via _resolve_focus_text."""

import pytest


class FakeCognitiveLayer:
    """Minimal stub to test _resolve_focus_text without full init."""

    _FOLLOWUP_PRONOUNS = {"it", "that", "this", "them", "they", "those", "these", "he", "she"}
    _FOLLOWUP_STARTERS = {
        "what about", "how about", "tell me more", "more about",
        "and what", "and how", "what else", "anything else",
        "go on", "continue", "keep going", "elaborate",
    }
    _FOLLOWUP_QUESTION_WORDS = {"why", "how", "when", "where", "who"}

    def _resolve_focus_text(self, user_input: str, session_id: str) -> str | None:
        text = user_input.strip()
        if len(text) < 5:
            return None

        words = text.lower().split()
        if len(words) <= 3 and words[0] in self._FOLLOWUP_PRONOUNS:
            return None

        text_lower = text.lower()

        stripped = text_lower.rstrip("?!. ")
        if stripped in self._FOLLOWUP_QUESTION_WORDS:
            return None

        for starter in self._FOLLOWUP_STARTERS:
            if text_lower.startswith(starter):
                remainder = text_lower[len(starter):].strip()
                if starter in ("tell me more", "more about") and len(remainder) > 10:
                    return text[:200]
                return None

        if len(words) <= 5:
            _stop = {"the", "and", "for", "are", "was", "were", "has", "have",
                     "does", "did", "can", "could", "would", "should", "will",
                     "not", "but", "with", "from", "about", "what", "how",
                     "is", "a", "an", "do", "its", "it's", "what's", "right",
                     "really", "sure", "just", "so", "then", "well", "ok"}
            non_stop = [w.rstrip("?!.,") for w in words
                        if w.rstrip("?!.,") not in _stop]
            if non_stop and all(w in self._FOLLOWUP_PRONOUNS for w in non_stop):
                return None

        return text[:200]


@pytest.fixture
def layer():
    return FakeCognitiveLayer()


class TestTopicPersistence:
    """Verify that ambiguous/follow-up inputs preserve existing topic."""

    # --- Should PRESERVE topic (return None) ---

    @pytest.mark.parametrize("input_text", [
        "yes",
        "ok",
        "no",
        "hm",
        "?",
        "",
        "   ",
    ])
    def test_very_short_inputs_preserve_topic(self, layer, input_text):
        assert layer._resolve_focus_text(input_text, "s1") is None

    @pytest.mark.parametrize("input_text", [
        "it works",
        "that one",
        "this please",
        "them too",
    ])
    def test_pronoun_inputs_preserve_topic(self, layer, input_text):
        assert layer._resolve_focus_text(input_text, "s1") is None

    @pytest.mark.parametrize("input_text", [
        "what about it?",
        "how about that?",
        "tell me more",
        "what else?",
        "anything else?",
        "go on",
        "continue",
        "keep going",
        "elaborate",
        "and what happened?",
    ])
    def test_followup_phrases_preserve_topic(self, layer, input_text):
        assert layer._resolve_focus_text(input_text, "s1") is None

    @pytest.mark.parametrize("input_text", [
        "why?",
        "how?",
        "when?",
        "where?",
        "who?",
    ])
    def test_bare_question_words_preserve_topic(self, layer, input_text):
        assert layer._resolve_focus_text(input_text, "s1") is None

    def test_pronoun_question_preserves(self, layer):
        assert layer._resolve_focus_text("what's that about?", "s1") is None

    def test_is_that_right_preserves(self, layer):
        assert layer._resolve_focus_text("is that right?", "s1") is None

    # --- Should UPDATE topic (return text) ---

    @pytest.mark.parametrize("input_text", [
        "tell me about cognition-engines",
        "let's talk about membrain",
        "what do you know about the attractor dynamics?",
        "explain the compaction algorithm",
        "how does the CSTP server work?",
        "I want to discuss the trading strategy",
    ])
    def test_clear_topic_updates(self, layer, input_text):
        result = layer._resolve_focus_text(input_text, "s1")
        assert result is not None
        assert result == input_text[:200]

    def test_tell_me_more_with_clear_object_updates(self, layer):
        result = layer._resolve_focus_text(
            "tell me more about the attractor dynamics in membrain", "s1"
        )
        assert result is not None

    def test_truncation_at_200(self, layer):
        long_input = "explain " + "x" * 300
        result = layer._resolve_focus_text(long_input, "s1")
        assert result is not None
        assert len(result) == 200

    def test_multiword_topic_updates(self, layer):
        result = layer._resolve_focus_text("the compaction algorithm needs work", "s1")
        assert result is not None

    def test_how_does_with_object_updates(self, layer):
        """'how does X work' has a clear topic — should update."""
        result = layer._resolve_focus_text("how does the CSTP server work?", "s1")
        assert result is not None
