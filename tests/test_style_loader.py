import os
import time

from core.style_loader import FALLBACK, StyleLoader


def write(path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    # make sure mtime actually differs on coarse-grained filesystems
    stamp = time.time() + 1
    os.utime(path, (stamp, stamp))


def test_reads_file_and_strips_frontmatter(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "---\nname: x\ndescription: y\n---\n\n# Стиль\nКоротко.\n")

    assert StyleLoader(skill).load() == "# Стиль\nКоротко."


def test_missing_file_falls_back(tmp_path) -> None:
    assert StyleLoader(tmp_path / "nope.md").load() == FALLBACK


def test_empty_file_falls_back(tmp_path) -> None:
    skill = tmp_path / "SKILL.md"
    write(skill, "   \n")

    assert StyleLoader(skill).load() == FALLBACK


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
