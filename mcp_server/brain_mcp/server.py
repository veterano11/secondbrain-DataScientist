from mcp.server.fastmcp import FastMCP
from brain_mcp import vault as v

mcp = FastMCP("second-brain")


@mcp.tool()
def read_note(path: str) -> str:
    """Read a note from the vault by its relative path.
    Example: path='3 - Resources/my-note.md'
    Returns the full note content with frontmatter.
    """
    try:
        note = v.read_note(path)
        lines = [f"# {note['path']}"]
        if note["frontmatter"]:
            lines.append("")
            lines.append("**Frontmatter:**")
            for k, val in note["frontmatter"].items():
                lines.append(f"  {k}: {val}")
        lines.append("")
        lines.append(note["content"])
        return "\n".join(lines)
    except FileNotFoundError as e:
        return f"Error: {e}"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def write_note(path: str, content: str, frontmatter: str = "") -> str:
    """Create or update a note in the vault.
    Args:
        path: Relative path (e.g. '3 - Resources/my-topic.md')
        content: Markdown body content
        frontmatter: Optional YAML frontmatter as a JSON string (e.g. '{"tags": ["ai", "ml"]}')
    """
    try:
        import json
        fm = json.loads(frontmatter) if frontmatter else None
    except json.JSONDecodeError as e:
        return f"Error: invalid frontmatter JSON: {e}"
    try:
        result = v.write_note(path, content, frontmatter=fm)
        action = "created" if result["created"] else "updated"
        return f"Note {action}: {path} ({result['size']} bytes)"
    except ValueError as e:
        return f"Error: {e}"


@mcp.tool()
def search_notes(query: str, limit: int = 10) -> str:
    """Search for notes by content across the vault.
    Returns matching note paths and titles.
    """
    results = v.search_notes(query, limit=limit)
    if not results:
        return "No notes found."
    lines = [f"Found {len(results)} note(s):", ""]
    for r in results:
        lines.append(f"  - {r['path']}")
    return "\n".join(lines)


@mcp.tool()
def list_directory(path: str = "") -> str:
    """List files and directories in the vault.
    Args:
        path: Relative path (e.g. '3 - Resources'). Empty string lists root.
    """
    try:
        entries = v.list_directory(path)
    except (FileNotFoundError, ValueError) as e:
        return f"Error: {e}"
    if not entries:
        return "Empty directory."
    lines = [f"Contents of /{path}:", ""]
    for e in entries:
        icon = "📁" if e["type"] == "directory" else "📄"
        lines.append(f"  {icon} {e['name']}")
    return "\n".join(lines)


@mcp.tool()
def vault_stats() -> str:
    """Get overview statistics of the vault."""
    stats = v.vault_stats()
    lines = [
        f"**Vault: {v.VAULT_ROOT.name}**",
        "",
        f"Total notes: {stats['total_notes']}",
        f"Total folders: {stats['total_directories']}",
        "",
        "By folder:",
    ]
    for folder, info in sorted(stats["by_folder"].items()):
        lines.append(f"  - {folder}: {info['notes']} notes")
    return "\n".join(lines)


@mcp.resource("vault://root")
def vault_root() -> str:
    """List the vault root directory."""
    entries = v.list_directory()
    lines = ["Vault Root:", ""]
    for e in entries:
        icon = "📁" if e["type"] == "directory" else "📄"
        lines.append(f"  {icon} {e['name']}")
    return "\n".join(lines)


@mcp.resource("vault://stats")
def vault_stats_resource() -> str:
    return vault_stats()
