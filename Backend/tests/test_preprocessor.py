"""
Tests for rag/helper_models/Preprocessor/Proprocessor.py  →  Preprocessor
No external deps — pure unit tests on the regex logic.
"""

import pytest
from rag.helper_models.Preprocessor.Proprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    return Preprocessor()


# ═════════════════════════════════════════════════════════════════════════════
# process() — happy paths
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocessorProcess:

    def test_returns_string(self, preprocessor):
        result = preprocessor.process("Hello world")
        assert isinstance(result, str)

    def test_removes_complete_html_tag(self, preprocessor):
        result = preprocessor.process("Hello <b>world</b>")
        assert "<b>" not in result
        assert "</b>" not in result

    def test_removes_self_closing_tag(self, preprocessor):
        result = preprocessor.process("Line break <br/> here")
        assert "<br/>" not in result

    def test_removes_img_tag_with_attributes(self, preprocessor):
        html = 'Image: <img src="photo.jpg" alt="photo"> here'
        result = preprocessor.process(html)
        assert "<img" not in result

    def test_removes_truncated_base64_img_tag(self, preprocessor):
        """Unclosed tags (e.g. truncated base64 images) should be stripped."""
        html = "data: <img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"
        result = preprocessor.process(html)
        assert "<img" not in result

    def test_preserves_text_content(self, preprocessor):
        result = preprocessor.process("<p>Keep this text</p>")
        assert "Keep this text" in result

    def test_strips_leading_trailing_whitespace(self, preprocessor):
        result = preprocessor.process("  <b>text</b>  ")
        assert result == result.strip()

    def test_multiline_tag_removed(self, preprocessor):
        html = "before <div\n  class='x'\n>content</div> after"
        result = preprocessor.process(html)
        assert "<div" not in result

    def test_removes_script_tag(self, preprocessor):
        # The regex removes the <script> and </script> tags themselves,
        # but NOT the text content between them (it is not a DOTALL block tag stripper).
        html = "text <script>alert('xss')</script> more"
        result = preprocessor.process(html)
        assert "<script>" not in result
        assert "</script>" not in result

    def test_multiple_tags_all_removed(self, preprocessor):
        html = "<p>First</p> <span>Second</span> <em>Third</em>"
        result = preprocessor.process(html)
        assert "<" not in result
        assert "First" in result
        assert "Second" in result
        assert "Third" in result


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestPreprocessorEdgeCases:

    def test_empty_string_returns_empty(self, preprocessor):
        result = preprocessor.process("")
        assert result == ""

    def test_plain_text_unchanged(self, preprocessor):
        plain = "I feel very anxious and stressed lately."
        result = preprocessor.process(plain)
        assert result == plain

    def test_only_tags_returns_empty_or_whitespace(self, preprocessor):
        result = preprocessor.process("<p></p>")
        assert result.strip() == ""

    def test_arabic_text_preserved(self, preprocessor):
        text = "أنا أشعر بالقلق <b>الشديد</b>"
        result = preprocessor.process(text)
        assert "أنا أشعر بالقلق" in result
        assert "<b>" not in result

    def test_very_long_plain_text(self, preprocessor):
        text = "word " * 1000
        result = preprocessor.process(text)
        assert len(result) > 0

    def test_very_long_html(self, preprocessor):
        html = "<p>paragraph</p>" * 500
        result = preprocessor.process(html)
        assert "<p>" not in result

    def test_nested_tags(self, preprocessor):
        html = "<div><p><span>deep text</span></p></div>"
        result = preprocessor.process(html)
        assert "deep text" in result
        assert "<" not in result

    def test_no_tag_brackets_in_output(self, preprocessor):
        html = "<b>bold</b> and <i>italic</i> content"
        result = preprocessor.process(html)
        assert "<" not in result

    def test_text_with_angle_bracket_like_content(self, preprocessor):
        """Mathematical expressions that look like tags should be handled gracefully."""
        text = "x < y and y > z"
        result = preprocessor.process(text)
        # regex only matches <...> pairs; "x < y" alone won't form a complete tag
        assert isinstance(result, str)

    def test_unclosed_tag_at_end_stripped(self, preprocessor):
        text = "Some content <unclosed"
        result = preprocessor.process(text)
        assert "<unclosed" not in result
