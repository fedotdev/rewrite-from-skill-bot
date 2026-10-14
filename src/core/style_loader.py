import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_PATH = Path(__file__).resolve().parent.parent.parent / "runtime_skills" / "rewrite" / "SKILL.md"
FALLBACK = "Стиль не задан. Переписывай естественно, сохраняя смысл."

# only flat files directly under references/, never nested folders
REFERENCE_RE = re.compile(r"references/[A-Za-z0-9._-]+\.md")
REFERENCES_DIRNAME = "references"
MAX_REFERENCE_BYTES = 32 * 1024
MAX_REFERENCES_TOTAL_BYTES = 64 * 1024


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


def discover_references(content: str) -> list[str]:
    """Linked reference files, in order of first appearance, without duplicates."""
    ordered: dict[str, None] = {}
    for match in REFERENCE_RE.finditer(content):
        ordered.setdefault(match.group(0), None)
    return list(ordered)


class StyleLoader:
    """Builds the runtime style from SKILL.md plus the reference files it links to.

    The cache is replaced only after SKILL.md has been read successfully, so a
    broken SKILL.md never costs the last working style.
    """

    def __init__(self, path: Path = SKILL_PATH) -> None:
        self._path = path
        self._cached: str = ""
        self._skill_body: str = ""
        self._links: list[str] = []
        self._skill_mtime: float | None = None
        self._skill_failed_mtime: float | None = None
        self._ref_mtimes: dict[str, float | None] = {}
        self._ref_failed_mtime: dict[str, float | None] = {}
        self._ref_cache: dict[str, str] = {}
        self._warned: set[tuple[str, str]] = set()

    # --- helpers ----------------------------------------------------------

    @property
    def _skill_dir(self) -> Path:
        return self._path.parent.resolve()

    def _warn_once(self, name: str, problem: str) -> None:
        key = (name, problem)
        if key in self._warned:
            return
        self._warned.add(key)
        # file contents are never logged
        logger.warning("Style reference %s skipped: %s", name, problem)

    @staticmethod
    def _mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _resolve(self, link: str) -> Path | None:
        """Resolve a link, refusing anything that escapes references/."""
        references_root = self._skill_dir / REFERENCES_DIRNAME
        candidate = (self._skill_dir / link).resolve()
        if not candidate.is_relative_to(references_root.resolve()):
            return None
        if candidate.suffix != ".md":
            return None
        return candidate

    # --- loading ----------------------------------------------------------

    def _references_changed(self) -> bool:
        for link in self._links:
            path = self._resolve(link)
            if path is None:
                continue
            if self._mtime(path) != self._ref_mtimes.get(link):
                return True
        return False

    def _read_text(self, path: Path, name: str) -> str | None:
        mtime = self._mtime(path)
        if mtime is None:
            self._warn_once(name, "missing")
            return None
        if self._ref_failed_mtime.get(name) == mtime:
            # already failed for this exact version, do not re-read on every request
            return None
        if path.stat().st_size > MAX_REFERENCE_BYTES:
            self._warn_once(name, f"too large ({path.stat().st_size} bytes)")
            self._ref_failed_mtime[name] = mtime
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            self._warn_once(name, e.__class__.__name__)
            self._ref_failed_mtime[name] = mtime
            return None
        self._ref_failed_mtime.pop(name, None)
        return content

    def _load_reference(self, link: str, budget: list[int]) -> str | None:
        name = link.split("/", 1)[1]
        path = self._resolve(link)
        if path is None:
            self._warn_once(name, "outside references")
            return None

        size = path.stat().st_size if path.exists() else 0
        if budget[0] + size > MAX_REFERENCES_TOTAL_BYTES:
            self._warn_once(name, f"total limit exceeded ({size} bytes)")
            return None

        content = self._read_text(path, name)
        if content is None:
            # keep the last good version of this file, drop it only if there is none
            return self._ref_cache.get(link)

        body = strip_frontmatter(content).strip()
        if not body:
            self._warn_once(name, "empty")
            return None

        self._ref_cache[link] = body
        self._ref_mtimes[link] = self._mtime(path)
        budget[0] += size
        return body

    def _assemble(self) -> str:
        blocks = [self._skill_body]
        budget = [0]
        for link in self._links:
            body = self._load_reference(link, budget)
            if body:
                name = link.split("/", 1)[1]
                blocks.append(f"---\n## Справочник: {name}\n{body}")
        return "\n\n".join(blocks).strip()

    # --- public -----------------------------------------------------------

    def load(self) -> str:
        skill_mtime = self._mtime(self._path)
        if skill_mtime is None:
            self._warn_once(self._path.name, "missing")
            return self._cached or FALLBACK

        if skill_mtime == self._skill_failed_mtime:
            # this exact version already failed; do not touch the disk on every request
            return self._cached or FALLBACK

        skill_changed = skill_mtime != self._skill_mtime
        if self._cached and not skill_changed and not self._references_changed():
            return self._cached

        if skill_changed or not self._skill_body:
            content = self._read_skill(skill_mtime)
            if content is None:
                return self._cached or FALLBACK
            self._skill_body = strip_frontmatter(content).strip() or FALLBACK
            self._links = discover_references(content)

        style = self._assemble()
        # single atomic swap; partial state never reaches the cache
        self._cached = style
        self._skill_mtime = skill_mtime
        self._skill_failed_mtime = None
        self._warned.clear()
        return style

    def _read_skill(self, mtime: float | None) -> str | None:
        if mtime is not None and self._skill_failed_mtime == mtime:
            # already failed for this exact version, do not re-read on every request
            return None
        try:
            return self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            self._warn_once(self._path.name, e.__class__.__name__)
            self._skill_failed_mtime = mtime
            return None
