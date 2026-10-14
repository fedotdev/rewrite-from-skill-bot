from pathlib import Path

SKILL_PATH = Path(__file__).resolve().parent.parent.parent / "runtime_skills" / "rewrite" / "SKILL.md"
FALLBACK = "Стиль не задан. Переписывай естественно, сохраняя смысл."


def load_style_instructions() -> str:
    if not SKILL_PATH.exists():
        return FALLBACK
    content = SKILL_PATH.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) == 3:
            content = parts[2]
    return content.strip() or FALLBACK
