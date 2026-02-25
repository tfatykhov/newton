"""Tests for Telegram-specific formatting (#29, #39).

Tests cover:
- sanitize_telegram() converts tables, headers, HRs (plain text, intermediate streaming)
- format_telegram_html() converts markdown to Telegram HTML (final sends)
- _strip_html_tags() strips HTML for plain text fallback
- Platform param flows through to system prompt
"""

import pytest

from nous.telegram_bot import format_telegram_html, sanitize_telegram, _strip_html_tags


class TestSanitizeTelegram:
    """Tests for the sanitize_telegram() fallback sanitizer."""

    def test_table_to_bullets(self):
        """Markdown table rows converted to bullet points."""
        text = "| Feature | Status |\n|---|---|\n| Brain | Shipped |\n| Heart | Shipped |"
        result = sanitize_telegram(text)
        assert "|" not in result
        assert "• Brain — Shipped" in result
        assert "• Heart — Shipped" in result

    def test_table_separator_removed(self):
        """|---|---| separator rows are stripped."""
        text = "|---|---|\n|:-:|:-:|"
        result = sanitize_telegram(text)
        assert "---" not in result

    def test_single_column_table(self):
        """Single column table row becomes simple bullet."""
        text = "| Just one column |"
        result = sanitize_telegram(text)
        assert "• Just one column" in result

    def test_three_column_table(self):
        """Three+ column rows join with em dash."""
        text = "| A | B | C |"
        result = sanitize_telegram(text)
        assert "•" in result
        assert "|" not in result
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_header_to_bold(self):
        """## headers converted to **bold**."""
        text = "## My Header\nSome text\n### Sub Header"
        result = sanitize_telegram(text)
        assert "##" not in result
        assert "**My Header**" in result
        assert "**Sub Header**" in result

    def test_horizontal_rule_removed(self):
        """--- horizontal rules stripped."""
        text = "Above\n---\nBelow"
        result = sanitize_telegram(text)
        assert "---" not in result
        assert "Above" in result
        assert "Below" in result

    def test_excessive_blank_lines_collapsed(self):
        """Multiple blank lines from removals collapsed to max 2."""
        text = "Above\n\n\n\n\nBelow"
        result = sanitize_telegram(text)
        assert "\n\n\n" not in result

    def test_plain_text_unchanged(self):
        """Normal text passes through unchanged."""
        text = "Hello, this is a normal message with **bold** and _italic_."
        result = sanitize_telegram(text)
        assert result == text

    def test_bullet_lists_unchanged(self):
        """Bullet lists are not affected."""
        text = "• Item one\n• Item two\n- Item three"
        result = sanitize_telegram(text)
        assert result == text

    def test_code_blocks_unchanged(self):
        """Code blocks pass through."""
        text = "```python\nprint('hello')\n```"
        result = sanitize_telegram(text)
        assert "```python" in result

    def test_table_inside_code_block_preserved(self):
        """Tables inside fenced code blocks are NOT sanitized."""
        text = "```\n| col1 | col2 |\n|------|------|\n| a    | b    |\n```"
        result = sanitize_telegram(text)
        assert "| col1 | col2 |" in result
        assert "|------|------|" in result

    def test_header_inside_code_block_preserved(self):
        """Headers inside fenced code blocks are NOT sanitized."""
        text = "```markdown\n## My Header\n```"
        result = sanitize_telegram(text)
        assert "## My Header" in result
        assert "**My Header**" not in result

    def test_mixed_code_and_tables(self):
        """Code blocks preserved while tables outside are sanitized."""
        text = (
            "## Status\n"
            "| Feature | Done |\n"
            "|---------|------|\n"
            "| Brain | Yes |\n\n"
            "```sql\n"
            "SELECT * FROM brain.decisions\n"
            "WHERE status = 'pending'\n"
            "```\n\n"
            "All good!"
        )
        result = sanitize_telegram(text)
        assert "**Status**" in result
        assert "• Brain — Yes" in result
        assert "SELECT * FROM brain.decisions" in result
        assert "```sql" in result

    def test_mixed_content(self):
        """Real-world message with tables + headers + normal text."""
        text = (
            "Here's the status:\n\n"
            "## Features\n"
            "| Feature | Status |\n"
            "|---------|--------|\n"
            "| Brain | Shipped |\n"
            "| Heart | Shipped |\n\n"
            "---\n\n"
            "That's everything!"
        )
        result = sanitize_telegram(text)
        assert "|" not in result
        assert "##" not in result
        assert "---" not in result
        assert "**Features**" in result
        assert "• Brain — Shipped" in result
        assert "That's everything!" in result

    def test_empty_string(self):
        """Empty string returns empty."""
        assert sanitize_telegram("") == ""

    def test_pipe_in_code_not_affected(self):
        """Pipes inside inline code should ideally not be affected.

        Note: The regex-based approach may catch some edge cases.
        This test documents current behavior.
        """
        text = "Use `a | b` for OR operations"
        result = sanitize_telegram(text)
        # Inline code pipes are not wrapped in |...|$ pattern, so safe
        assert "`a | b`" in result


class TestFormatTelegramHtml:
    """Tests for format_telegram_html() — markdown to Telegram HTML (#39)."""

    def test_html_entity_escaping(self):
        """HTML special chars are escaped to prevent injection."""
        text = "a < b & c > d"
        result = format_telegram_html(text)
        assert "a &lt; b &amp; c &gt; d" == result

    def test_header_to_html_bold(self):
        """## headers converted to <b>bold</b>."""
        text = "## My Header\nSome text\n### Sub Header"
        result = format_telegram_html(text)
        assert "##" not in result
        assert "<b>My Header</b>" in result
        assert "<b>Sub Header</b>" in result

    def test_bold_to_html(self):
        """**bold** converted to <b>bold</b>."""
        text = "This is **important** stuff"
        result = format_telegram_html(text)
        assert "**" not in result
        assert "This is <b>important</b> stuff" in result

    def test_inline_code_to_html(self):
        """`code` converted to <code>code</code>."""
        text = "Use `print()` to output"
        result = format_telegram_html(text)
        assert "`" not in result
        assert "<code>print()</code>" in result

    def test_inline_code_html_escaped(self):
        """HTML entities inside inline code are escaped."""
        text = "Use `a < b` in code"
        result = format_telegram_html(text)
        assert "<code>a &lt; b</code>" in result

    def test_fenced_code_block_with_language(self):
        """Fenced code blocks with language get <pre><code class=language-X>."""
        text = "```python\nprint('hello')\n```"
        result = format_telegram_html(text)
        assert "```" not in result
        assert '<pre><code class="language-python">' in result
        assert "print('hello')" in result
        assert "</code></pre>" in result

    def test_fenced_code_block_without_language(self):
        """Fenced code blocks without language get <pre>."""
        text = "```\nsome code\n```"
        result = format_telegram_html(text)
        assert "```" not in result
        assert "<pre>some code</pre>" in result

    def test_code_block_html_escaped(self):
        """HTML entities inside code blocks are escaped."""
        text = "```\na < b && c > d\n```"
        result = format_telegram_html(text)
        assert "a &lt; b &amp;&amp; c &gt; d" in result

    def test_link_to_html(self):
        """[text](url) converted to <a href=url>text</a>."""
        text = "Visit [Nous](https://example.com) for more"
        result = format_telegram_html(text)
        assert '<a href="https://example.com">Nous</a>' in result
        assert "[Nous]" not in result

    def test_table_to_bullets(self):
        """Tables converted to bullet lists (same as sanitize_telegram)."""
        text = "| Feature | Status |\n|---|---|\n| Brain | Shipped |"
        result = format_telegram_html(text)
        assert "• Brain — Shipped" in result
        assert "|" not in result

    def test_hr_removed(self):
        """--- horizontal rules removed."""
        text = "Above\n---\nBelow"
        result = format_telegram_html(text)
        assert "---" not in result
        assert "Above" in result
        assert "Below" in result

    def test_excessive_blank_lines_collapsed(self):
        """Multiple blank lines collapsed to max 2."""
        text = "Above\n\n\n\n\nBelow"
        result = format_telegram_html(text)
        assert "\n\n\n" not in result

    def test_plain_text_escaped(self):
        """Plain text without markdown passes through with HTML escaping."""
        text = "Hello, world!"
        result = format_telegram_html(text)
        assert result == "Hello, world!"

    def test_empty_string(self):
        """Empty string returns empty."""
        assert format_telegram_html("") == ""

    def test_table_inside_code_block_preserved(self):
        """Tables inside fenced code blocks are NOT converted to bullets."""
        text = "```\n| col1 | col2 |\n|------|\n| a    | b    |\n```"
        result = format_telegram_html(text)
        assert "| col1 | col2 |" in result
        assert "•" not in result

    def test_header_inside_code_block_preserved(self):
        """Headers inside code blocks are NOT converted to <b>."""
        text = "```markdown\n## My Header\n```"
        result = format_telegram_html(text)
        assert "## My Header" in result
        assert "<b>My Header</b>" not in result

    def test_mixed_content(self):
        """Real-world message with tables, headers, code, and bold."""
        text = (
            "## Status\n\n"
            "This is **important**.\n\n"
            "| Feature | Done |\n"
            "|---------|------|\n"
            "| Brain | Yes |\n\n"
            "```python\nx = 1\n```\n\n"
            "Use `code` here."
        )
        result = format_telegram_html(text)
        assert "<b>Status</b>" in result
        assert "<b>important</b>" in result
        assert "• Brain — Yes" in result
        assert '<pre><code class="language-python">x = 1</code></pre>' in result
        assert "<code>code</code>" in result

    def test_bold_with_html_entities_inside(self):
        """Bold text containing HTML special chars is properly escaped."""
        text = "**a < b**"
        result = format_telegram_html(text)
        assert "<b>a &lt; b</b>" in result

    def test_link_with_ampersand_in_url(self):
        """URLs with & are properly escaped in HTML."""
        text = "[search](https://example.com?a=1&b=2)"
        result = format_telegram_html(text)
        assert 'href="https://example.com?a=1&amp;b=2"' in result
        assert ">search</a>" in result


class TestStripHtmlTags:
    """Tests for _strip_html_tags() plain text fallback."""

    def test_strips_bold_tags(self):
        text = "This is <b>bold</b> text"
        assert _strip_html_tags(text) == "This is bold text"

    def test_strips_code_tags(self):
        text = "Use <code>print()</code>"
        assert _strip_html_tags(text) == "Use print()"

    def test_strips_pre_tags(self):
        text = "<pre>code block</pre>"
        assert _strip_html_tags(text) == "code block"

    def test_unescapes_html_entities(self):
        text = "a &lt; b &amp; c &gt; d"
        assert _strip_html_tags(text) == "a < b & c > d"

    def test_combined_strip_and_unescape(self):
        text = "<b>a &lt; b</b>"
        assert _strip_html_tags(text) == "a < b"

    def test_plain_text_unchanged(self):
        text = "Hello, world!"
        assert _strip_html_tags(text) == "Hello, world!"
