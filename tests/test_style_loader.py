import logging
import os
import time

import pytest

from core.style_loader import FALLBACK, StyleLoader, strip_frontmatter


def write(path, content: str, encoding: str = "utf-8") -> None:
    path.write_bytes(content.encode(encoding))
    stamp = time.time() + 1
    os.utime(path, (stamp, stamp))


def touch_future(path) -> None:
    stamp = time.time() + 2
    os.utime(path, (stamp, stamp))


# --- frontmatter -----------------------------------------------------------


def test_frontmatter_is_stripped() -> None:
    text = "---\nname: x\ndescription: y\n---\n\n# Стиль\nКоротко.\n"

    assert strip_frontmatter(text).strip() == "# Стиль\nКоротко."


def test_unterminated_frontmatter_is_kept() -> None:
    text = "---\nтекст без закрытия"

    assert strip_frontmatter(text) == text


def test_horizontal_rule_without_closing_is_kept() -> None:
    text = "---\n\n- пункт один\n- пункт два"

    assert strip_frontmatter(text) == text


def test_text_starting_with_dashes_but_not_delimiter() -> None:
    text = "-- это не frontmatter\nтекст"

    assert strip_frontmatter(text) == text


def test_empty_closing_line_is_not_a_delimiter() -> None:
    text = "---\nname: x\n---\nстиль"

    assert strip_frontmatter(text) == "стиль"


def test_frontmatter_with_crlf() -> None:
    text = "---\r\nname: x\r\n---\r\nстиль"

    assert strip_frontmatter(text) == "стиль"


def test_no_frontmatter() -> None:
    text = "# Стиль\nКоротко."

    assert strip_frontmatter(text) == text


def test_second_dash_block_is_kept_as_content() -> None:
    text = "---\nname: x\n---\n---\nстиль"

    assert strip_frontmatter(text) == "---\nстиль"


def test_loader_strips_frontmatter(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "---\nname: x\ndescription: y\n---\n\n# Стиль\nКоротко.\n")

    assert StyleLoader(skill).load() == "# Стиль\nКоротко."


def test_loader_keeps_unterminated_frontmatter(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "---\nтекст")

    assert StyleLoader(skill).load() == "---\nтекст"


# --- read failures ---------------------------------------------------------


def test_missing_file_falls_back(tmp_path) -> None:
    assert StyleLoader(tmp_path / "nope.md").load() == FALLBACK


def test_empty_file_falls_back(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "   \n")

    assert StyleLoader(skill).load() == FALLBACK


def test_wrong_encoding_falls_back(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "стиль в windows-1251: привет", encoding="cp1251")

    loader = StyleLoader(skill)

    with pytest.raises(UnicodeDecodeError):
        skill.read_text(encoding="utf-8")
    assert loader.load() == FALLBACK


def test_last_good_style_survives_encoding_break(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "рабочий стиль")
    loader = StyleLoader(skill)
    assert loader.load() == "рабочий стиль"

    write(skill, "сломанный стиль в cp1251", encoding="cp1251")
    touch_future(skill)

    assert loader.load() == "рабочий стиль"


def test_deleted_file_survives_with_last_style(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "рабочий стиль")
    loader = StyleLoader(skill)
    assert loader.load() == "рабочий стиль"

    skill.unlink()

    assert loader.load() == "рабочий стиль"


def test_deleted_file_without_prior_load_returns_fallback(tmp_path) -> None:
    assert StyleLoader(tmp_path / "gone.md").load() == FALLBACK


def test_recovers_after_file_returns(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    loader = StyleLoader(skill)
    assert loader.load() == FALLBACK

    write(skill, "новый стиль")

    assert loader.load() == "новый стиль"


def test_warning_is_logged_once_without_file_contents(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "секретный стиль", encoding="cp1251")
    loader = StyleLoader(skill)

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            loader.load()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "секретный стиль" not in caplog.text


# --- caching ---------------------------------------------------------------


def test_change_on_disk_is_picked_up_without_restart(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "первый стиль")

    loader = StyleLoader(skill)
    assert loader.load() == "первый стиль"

    time.sleep(0.01)
    write(skill, "второй стиль")
    assert loader.load() == "второй стиль"


def test_unchanged_file_uses_cache(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "стиль")

    loader = StyleLoader(skill)
    assert loader.load() == "стиль"
    mtime = skill.stat().st_mtime

    assert loader.load() == "стиль"
    assert loader._mtime == mtime


def test_real_skill_file_loads() -> None:
    text = StyleLoader().load()

    assert text
    assert not text.startswith("---")
    assert "Инструкции стиля" in text
