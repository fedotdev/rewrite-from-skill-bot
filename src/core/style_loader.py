from pathlib import Path

SKILL_PATH = Path(__file__).resolve().parent.parent.parent / "runtime_skills" / "rewrite" / "SKILL.md"
FALLBACK = "Стиль не задан. Переписывай естественно, сохраняя смысл."


class StyleLoader:
    """Reads the style file on demand and caches it until the file changes on disk."""

    def __init__(self, path: Path = SKILL_PATH) -> None:
        self._path = path
        self._cached: str = ""
        self._mtime: float | None = None

    def load(self) -> str:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            self._cached, self._mtime = FALLBACK, None
            return FALLBACK

        if mtime == self._mtime:
            return self._cached

        content = self._path.read_text(encoding="utf-8")
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) == 3:
                content = parts[2]
        self._cached = content.strip() or FALLBACK
        self._mtime = mtime
        return self._cached
