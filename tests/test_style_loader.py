import logging
import os
import time

import pytest

from core.style_loader import (
    FALLBACK,
    MAX_REFERENCE_BYTES,
    SKILL_PATH,
    StyleLoader,
    discover_references,
    strip_frontmatter,
)


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


# --- references -----------------------------------------------------------


def make_tree(tmp_path, skill: str = "# Стиль\nсм. references/genres.md\n", **references: str):
    """SKILL.md plus a references/ folder; values are file contents (str or (text, enc))."""
    root = tmp_path / "skill"
    root.mkdir(parents=True, exist_ok=True)
    refs = root / "references"
    refs.mkdir(exist_ok=True)
    write(root / "SKILL.md", skill)
    for name, value in references.items():
        content, encoding = value if isinstance(value, tuple) else (value, "utf-8")
        (refs / name).write_bytes(content.encode(encoding))
    return root


def test_reference_in_plain_text_is_included(tmp_path) -> None:
    root = make_tree(tmp_path, "Стиль, см. references/genres.md\n", **{"genres.md": "Жанры"})
    style = StyleLoader(root / "SKILL.md").load()

    assert "Жанры" in style
    assert "## Справочник: genres.md" in style


def test_reference_in_code_span_is_included(tmp_path) -> None:
    root = make_tree(tmp_path, "Смотри `references/genres.md` для деталей\n", **{"genres.md": "Жанры"})
    style = StyleLoader(root / "SKILL.md").load()

    assert "## Справочник: genres.md" in style
    assert "Жанры" in style


def test_reference_in_markdown_link_is_included(tmp_path) -> None:
    root = make_tree(
        tmp_path,
        "Подробности: [жанры](references/genres.md)\n",
        **{"genres.md": "Жанры"},
    )
    style = StyleLoader(root / "SKILL.md").load()

    assert "## Справочник: genres.md" in style
    assert "Жанры" in style


def test_reference_order_follows_first_appearance(tmp_path) -> None:
    root = make_tree(
        tmp_path,
        "сначала references/b.md, потом references/a.md\n",
        **{"genres.md": "unused", "a.md": "AAA", "b.md": "BBB"},
    )
    style = StyleLoader(root / "SKILL.md").load()

    assert style.index("## Справочник: b.md") < style.index("## Справочник: a.md")


def test_duplicate_links_are_loaded_once(tmp_path) -> None:
    root = make_tree(
        tmp_path,
        "references/genres.md и снова references/genres.md и опять references/genres.md\n",
        **{"genres.md": "Жанры"},
    )
    style = StyleLoader(root / "SKILL.md").load()

    assert style.count("## Справочник: genres.md") == 1


def test_file_without_link_is_not_included(tmp_path) -> None:
    root = make_tree(tmp_path, "Стиль без ссылок\n", **{"genres.md": "Жанры"})
    style = StyleLoader(root / "SKILL.md").load()

    assert "Жанры" not in style
    assert "## Справочник" not in style


def test_several_references_are_all_included(tmp_path) -> None:
    root = make_tree(
        tmp_path,
        "a: references/a.md, b: references/b.md\n",
        **{"a.md": "AAA", "b.md": "BBB"},
    )
    style = StyleLoader(root / "SKILL.md").load()

    assert "## Справочник: a.md" in style
    assert "## Справочник: b.md" in style
    assert "AAA" in style and "BBB" in style


def test_assembled_layout_has_fixed_separators(tmp_path) -> None:
    root = make_tree(tmp_path, "ТЕЛО\nсм. references/genres.md\n", **{"genres.md": "СПРАВОЧНИК"})
    style = StyleLoader(root / "SKILL.md").load()

    assert style == "ТЕЛО\nсм. references/genres.md\n\n---\n## Справочник: genres.md\nСПРАВОЧНИК"


def test_heading_comes_from_filename_not_content(tmp_path) -> None:
    root = make_tree(tmp_path, "x references/genres.md", **{"genres.md": "# Свой заголовок"})
    style = StyleLoader(root / "SKILL.md").load()

    assert "## Справочник: genres.md" in style
    assert "# Свой заголовок" in style


def test_reference_frontmatter_is_stripped(tmp_path) -> None:
    root = make_tree(
        tmp_path,
        "x references/genres.md",
        **{"genres.md": "---\nname: g\ndescription: d\n---\n\nтело справочника\n"},
    )
    style = StyleLoader(root / "SKILL.md").load()

    assert "тело справочника" in style
    assert "name: g" not in style


def test_empty_reference_is_skipped(tmp_path) -> None:
    root = make_tree(tmp_path, "x references/genres.md", **{"genres.md": "   \n"})
    style = StyleLoader(root / "SKILL.md").load()

    assert "## Справочник" not in style
    assert style.strip() == "x references/genres.md"


def test_missing_reference_yields_skill_only(tmp_path, caplog) -> None:
    root = make_tree(tmp_path, "x references/genres.md")

    with caplog.at_level(logging.WARNING):
        style = StyleLoader(root / "SKILL.md").load()

    assert "## Справочник" not in style
    assert style.strip() == "x references/genres.md"
    assert any("genres.md" in r.message for r in caplog.records)


# --- path safety -----------------------------------------------------------


def test_parent_traversal_is_rejected(tmp_path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    write(root / "SKILL.md", "x references/../secret.md")
    write(root / "secret.md", "СЕКРЕТ")

    style = StyleLoader(root / "SKILL.md").load()

    assert "СЕКРЕТ" not in style
    assert "## Справочник" not in style


def test_nested_folder_is_rejected(tmp_path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    write(root / "SKILL.md", "x references/sub/genres.md")
    (root / "references" / "sub").mkdir(parents=True)
    write(root / "references" / "sub" / "genres.md", "ВЛОЖЕННЫЙ")

    style = StyleLoader(root / "SKILL.md").load()

    assert "ВЛОЖЕННЫЙ" not in style


def test_absolute_path_link_is_ignored(tmp_path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    write(root / "SKILL.md", f"x {tmp_path.as_posix()}/outside.md")

    style = StyleLoader(root / "SKILL.md").load()

    assert "## Справочник" not in style


def test_non_md_extension_is_ignored(tmp_path) -> None:
    root = make_tree(tmp_path, "x references/notes.txt")
    write(root / "references" / "notes.txt", "ТЕКСТОВЫЙ")

    style = StyleLoader(root / "SKILL.md").load()

    assert "ТЕКСТОВЫЙ" not in style
    assert "## Справочник" not in style


def test_symlink_escaping_references_is_rejected(tmp_path) -> None:
    root = tmp_path / "skill"
    root.mkdir()
    write(root / "SKILL.md", "x references/leak.md")
    refs = root / "references"
    refs.mkdir()
    outside = tmp_path / "outside.md"
    write(outside, "УТЕКЛО")
    try:
        (refs / "leak.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")

    style = StyleLoader(root / "SKILL.md").load()

    assert "УТЕКЛО" not in style
    assert "## Справочник" not in style


def test_cp1251_reference_is_skipped_others_still_load(tmp_path, caplog) -> None:
    root = make_tree(
        tmp_path,
        "a references/broken.md и b references/good.md",
        **{"broken.md": ("сломан", "cp1251"), "good.md": "рабочий"},
    )

    with caplog.at_level(logging.WARNING):
        style = StyleLoader(root / "SKILL.md").load()

    assert "сломан" not in style
    assert "## Справочник: broken.md" not in style
    assert "рабочий" in style
    assert "## Справочник: good.md" in style


# --- size limits -----------------------------------------------------------


def test_oversized_reference_is_skipped(tmp_path, caplog) -> None:
    big = "A" * (MAX_REFERENCE_BYTES + 10)
    root = make_tree(tmp_path, "x references/big.md", **{"big.md": big})

    with caplog.at_level(logging.WARNING):
        style = StyleLoader(root / "SKILL.md").load()

    assert "## Справочник" not in style
    assert any("big.md" in r.message and "too large" in r.message for r in caplog.records)


def test_total_limit_is_enforced(tmp_path, caplog) -> None:
    chunk = "B" * 20_000
    root = make_tree(
        tmp_path,
        "a references/a.md b references/b.md c references/c.md d references/d.md",
        **{"a.md": chunk, "b.md": chunk, "c.md": chunk, "d.md": chunk},
    )

    with caplog.at_level(logging.WARNING):
        style = StyleLoader(root / "SKILL.md").load()

    included = [n for n in "abcd" if f"## Справочник: {n}.md" in style]
    assert len(included) == 3, "3 x 20KB fits into the 64KB total budget"
    assert any("total limit" in r.message for r in caplog.records)


def test_size_warning_has_no_file_content(tmp_path, caplog) -> None:
    big = "СЕКРЕТНЫЙ" * 5000
    root = make_tree(tmp_path, "x references/big.md", **{"big.md": big})

    with caplog.at_level(logging.WARNING):
        StyleLoader(root / "SKILL.md").load()

    assert "СЕКРЕТНЫЙ" not in caplog.text


# --- caching of references -------------------------------------------------


def test_changing_only_reference_updates_style(tmp_path) -> None:
    root = make_tree(tmp_path, "x references/genres.md", **{"genres.md": "версия один"})
    loader = StyleLoader(root / "SKILL.md")
    assert "версия один" in loader.load()

    time.sleep(0.01)
    write(root / "references" / "genres.md", "версия два")

    style = loader.load()
    assert "версия два" in style
    assert "версия один" not in style


def test_deleted_reference_keeps_last_good_version(tmp_path) -> None:
    root = make_tree(tmp_path, "x references/genres.md", **{"genres.md": "версия один"})
    loader = StyleLoader(root / "SKILL.md")
    assert "версия один" in loader.load()

    (root / "references" / "genres.md").unlink()

    style = loader.load()
    assert "версия один" in style, "a missing reference must not empty the appendix"
    assert "## Справочник: genres.md" in style


def test_broken_reference_keeps_last_good_version(tmp_path) -> None:
    root = make_tree(tmp_path, "x references/genres.md", **{"genres.md": "хорошая версия"})
    loader = StyleLoader(root / "SKILL.md")
    assert "хорошая версия" in loader.load()

    time.sleep(0.01)
    write(root / "references" / "genres.md", "сломанная версия", encoding="cp1251")

    style = loader.load()
    assert "хорошая версия" in style
    assert "сломанная версия" not in style


def test_adding_a_newly_linked_reference_is_picked_up(tmp_path) -> None:
    root = make_tree(tmp_path, "x references/genres.md")
    loader = StyleLoader(root / "SKILL.md")
    assert "## Справочник" not in loader.load()

    time.sleep(0.01)
    write(root / "references" / "genres.md", "появился")
    assert "появился" in loader.load()


def test_unlinked_reference_does_not_invalidate_cache(tmp_path) -> None:
    root = make_tree(tmp_path, "x references/genres.md", **{"genres.md": "жанр"})
    loader = StyleLoader(root / "SKILL.md")
    assert "жанр" in loader.load()

    # touching an unlinked file must not trigger a rebuild
    time.sleep(0.01)
    write(root / "references" / "other.md", "не подключён")

    assert "не подключён" not in loader.load()


# --- privacy ---------------------------------------------------------------


def test_reference_content_is_never_logged(tmp_path, caplog) -> None:
    root = make_tree(
        tmp_path,
        "x references/genres.md",
        **{"genres.md": ("---\nname: g\n---\n", "cp1251")},
    )

    with caplog.at_level(logging.DEBUG):
        StyleLoader(root / "SKILL.md").load()

    assert "name: g" not in caplog.text


def test_reference_content_is_never_structlogged(tmp_path) -> None:
    from structlog.testing import capture_logs

    root = make_tree(tmp_path, "x references/missing.md")

    with capture_logs() as captured:
        StyleLoader(root / "SKILL.md").load()

    assert "missing.md" not in str(captured)


def test_warning_is_not_repeated_per_request(tmp_path, caplog) -> None:
    root = make_tree(tmp_path, "x references/missing.md")
    loader = StyleLoader(root / "SKILL.md")

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            loader.load()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_warned_flag_resets_after_successful_load(tmp_path, caplog) -> None:
    root = make_tree(tmp_path, "x references/genres.md", **{"genres.md": "ок"})
    loader = StyleLoader(root / "SKILL.md")
    loader.load()

    time.sleep(0.01)
    (root / "references" / "genres.md").unlink()
    with caplog.at_level(logging.WARNING):
        loader.load()
    assert any("genres.md" in r.message for r in caplog.records)

    # after a healthy load the warning is allowed once more
    caplog.clear()
    time.sleep(0.01)
    (root / "references" / "genres.md").write_text("снова ок", encoding="utf-8")
    loader.load()
    (root / "references" / "genres.md").unlink()
    with caplog.at_level(logging.WARNING):
        loader.load()

    assert any("genres.md" in r.message for r in caplog.records)


# --- real tree -------------------------------------------------------------


def test_real_tree_includes_genres_reference() -> None:
    style = StyleLoader().load()

    assert "Инструкции стиля" in style
    assert "## Справочник: genres.md" in style
    assert "Жанры сообщений" in style


def test_real_skill_links_resolve_to_existing_files() -> None:
    content = (SKILL_PATH).read_text(encoding="utf-8")
    links = discover_references(content)

    assert links
    for link in links:
        assert (SKILL_PATH.parent / link).is_file(), f"{link} is linked but missing"


# --- caching ---------------------------------------------------------------


def test_change_on_disk_is_picked_up_without_restart(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "первый стиль")

    loader = StyleLoader(skill)
    assert loader.load() == "первый стиль"

    time.sleep(0.01)
    write(skill, "второй стиль")
    assert loader.load() == "второй стиль"


class CountingLoader(StyleLoader):
    """Counts how many times SKILL.md is actually read from disk."""

    def __init__(self, path) -> None:
        super().__init__(path)
        self.reads = 0

    def _read_skill(self, mtime: float | None) -> str | None:
        self.reads += 1
        return super()._read_skill(mtime)


def test_unchanged_file_uses_cache(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "стиль")

    loader = CountingLoader(skill)
    assert loader.load() == "стиль"
    assert loader.reads == 1

    assert loader.load() == "стиль"
    assert loader.reads == 1, "unchanged SKILL.md must not be re-read"
    assert loader.load() == "стиль"
    assert loader.reads == 1


def test_failed_skill_is_not_reread_on_every_request(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "рабочий стиль", encoding="cp1251")

    loader = CountingLoader(skill)
    for _ in range(5):
        assert loader.load() == FALLBACK

    assert loader.reads == 1, "a known-broken file must be read once, not per request"


def test_real_skill_file_loads() -> None:
    text = StyleLoader().load()

    assert text
    assert not text.startswith("---")
    assert "Инструкции стиля" in text
