import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_PATH = Path(__file__).resolve().parent.parent.parent / "runtime_skills" / "rewrite" / "SKILL.md"
FALLBACK = "Стиль не задан. Переписывай естественно, сохраняя смысл."


def strip_frontmatter(content: str) -> str:
    """Drop a leading YAML frontmatter block, but only if it is properly closed."""
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return content
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1 :])
    # a lone horizontal rule is not frontmatter; keep the content as-is
    return content


class StyleLoader:
    """Reads the style file on demand and caches it until the file changes on disk."""

    def __init__(self, path: Path = SKILL_PATH) -> None:
        self._path = path
        self._cached: str = ""
        self._mtime: float | None = None
        self._warned = False

    def _warn_once(self, reason: str) -> None:
        if self._warned:
            return
        self._warned = True
        # never include the file contents in the log
        logger.warning("Style file unavailable (%s), keeping last known style", reason)

    def load(self) -> str:
        try:
            mtime = self._path.stat().st_mtime
        except OSError as e:
            self._warn_once(f"stat failed: {e.__class__.__name__}")
            return self._cached or FALLBACK

        if self._cached and mtime == self._mtime:
            return self._cached

        try:
            content = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            self._warn_once(f"read failed: {e.__class__.__name__}")
            return self._cached or FALLBACK

        style = strip_frontmatter(content).strip() or FALLBACK
        self._cached = style
        self._mtime = mtime
        return style
