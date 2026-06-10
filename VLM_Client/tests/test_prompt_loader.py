import os
import tempfile

import pytest

from prompt_loader import load_prompt, parse_prompt_arg


@pytest.fixture
def plain_prompt_file():
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    f.write("Navigate to the landing pad.\n")
    f.close()
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def frontmatter_prompt_file():
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    f.write("---\ntitle: Mission 1\nauthor: test\n---\n\nFind the red barrel.\n")
    f.close()
    yield f.name
    os.unlink(f.name)


class TestLoadPrompt:
    def test_reads_plain_file(self, plain_prompt_file):
        content = load_prompt(plain_prompt_file)
        assert content == "Navigate to the landing pad."

    def test_strips_yaml_frontmatter(self, frontmatter_prompt_file):
        content = load_prompt(frontmatter_prompt_file)
        assert "Find the red barrel." in content
        assert "title:" not in content
        assert "---" not in content

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("/nonexistent/path/prompt.md")

    def test_strips_surrounding_whitespace(self, plain_prompt_file):
        content = load_prompt(plain_prompt_file)
        assert content == content.strip()

    def test_frontmatter_stripped_leaving_body(self, frontmatter_prompt_file):
        content = load_prompt(frontmatter_prompt_file)
        assert content.startswith("Find")

    def test_file_without_frontmatter_unchanged(self, plain_prompt_file):
        content = load_prompt(plain_prompt_file)
        assert "Navigate" in content


class TestParsePromptArg:
    def test_returns_default_when_no_args(self, plain_prompt_file):
        from mission_config import DEFAULT_DIRECT_PROMPT
        result = parse_prompt_arg([])
        assert result == DEFAULT_DIRECT_PROMPT

    def test_parses_space_separated_arg(self, plain_prompt_file):
        result = parse_prompt_arg(["--prompt", plain_prompt_file])
        assert result == plain_prompt_file

    def test_parses_equals_separated_arg(self, plain_prompt_file):
        result = parse_prompt_arg([f"--prompt={plain_prompt_file}"])
        assert result == plain_prompt_file

    def test_first_prompt_arg_wins(self, plain_prompt_file):
        f2 = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
        f2.close()
        try:
            result = parse_prompt_arg(["--prompt", plain_prompt_file, "--prompt", f2.name])
            assert result == plain_prompt_file
        finally:
            os.unlink(f2.name)

    def test_ignores_unrelated_args(self, plain_prompt_file):
        result = parse_prompt_arg(["--verbose", "--prompt", plain_prompt_file, "--debug"])
        assert result == plain_prompt_file

    def test_incomplete_prompt_arg_returns_default(self):
        from mission_config import DEFAULT_DIRECT_PROMPT
        result = parse_prompt_arg(["--prompt"])
        assert result == DEFAULT_DIRECT_PROMPT
