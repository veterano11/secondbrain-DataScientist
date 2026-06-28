from pathlib import Path
from typing import Optional
from datetime import datetime
import re

VAULT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(path: str) -> Path:
    resolved = (VAULT_ROOT / path).resolve()
    if not str(resolved).startswith(str(VAULT_ROOT)):
        raise ValueError(f"Access denied: path escapes vault root: {path}")
    return resolved


def read_note(path: str) -> dict:
    full = _resolve_path(path)
    if not full.exists():
        raise FileNotFoundError(f"Note not found: {path}")
    content = full.read_text(encoding="utf-8")
    frontmatter = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            for line in fm_text.split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    frontmatter[key.strip()] = val.strip()
    return {
        "path": path,
        "frontmatter": frontmatter,
        "content": body,
        "size": len(content),
    }


def write_note(path: str, content: str, frontmatter: Optional[dict] = None) -> dict:
    full = _resolve_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        fm_lines = ["---"]
        now = datetime.now().strftime("%Y-%m-%d")
        if "created" not in frontmatter and not full.exists():
            frontmatter["created"] = now
        frontmatter["modified"] = now
        for k, v in frontmatter.items():
            if isinstance(v, list):
                fm_lines.append(f"{k}: [{', '.join(v)}]")
            else:
                fm_lines.append(f"{k}: {v}")
        fm_lines.append("---")
        full_content = "\n".join(fm_lines) + "\n\n" + content
    else:
        full_content = content
    full.write_text(full_content, encoding="utf-8")
    return {"path": path, "size": len(full_content), "created": not full.exists()}


def search_notes(query: str, limit: int = 10) -> list[dict]:
    results = []
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for f in VAULT_ROOT.rglob("*.md"):
        rel = str(f.relative_to(VAULT_ROOT))
        if ".obsidian" in rel or "mcp_server" in rel or ".opencode" in rel:
            continue
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if pattern.search(content):
            results.append({"path": rel, "title": f.stem})
    results.sort(key=lambda x: x["path"])
    return results[:limit]


def list_directory(path: str = "") -> list[dict]:
    full = _resolve_path(path)
    if not full.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    entries = []
    for entry in sorted(full.iterdir()):
        if entry.name.startswith("."):
            continue
        entries.append(
            {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "path": str(entry.relative_to(VAULT_ROOT)),
            }
        )
    return entries


def vault_stats() -> dict:
    stats = {"total_notes": 0, "total_directories": 0, "by_folder": {}}
    for entry in VAULT_ROOT.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            if ".obsidian" in str(entry):
                continue
            md_files = list(entry.rglob("*.md"))
            count = len(md_files)
            stats["by_folder"][entry.name] = {"notes": count, "path": entry.name}
            stats["total_directories"] += 1
            stats["total_notes"] += count
    return stats
