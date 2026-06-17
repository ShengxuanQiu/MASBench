"""Thin local software-engineering tool wrappers.

The workflow emits trace events around these functions. The wrappers keep the
tool surface deliberately small and bounded so later real runs can reuse them
without changing motif trace semantics.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .tools import safe_list_files, safe_read_files


def search_code(query: str, repo_dir: str | os.PathLike[str] | None, *, max_files: int = 40) -> dict[str, Any]:
    if not repo_dir:
        return {"status": "skipped", "reason": "repo_dir_unavailable", "query": query, "files": [], "snippets": {}}
    root = Path(repo_dir).expanduser().resolve()
    files = safe_list_files(str(root), max_files=500)
    terms = [term.lower() for term in query.replace("/", " ").replace("_", " ").split() if len(term) >= 3]
    scored = []
    for rel in files:
        lower = rel.lower()
        score = sum(1 for term in terms if term in lower)
        if score or rel.endswith((".py", ".js", ".ts", ".java", ".go", ".rs", ".md", ".rst")):
            scored.append((score, rel))
    selected = [rel for _, rel in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_files]]
    return {"status": "success", "query": query, "files": selected, "snippets": safe_read_files(str(root), selected[:8], max_chars_per_file=2000)}


def read_file(path: str, repo_dir: str | os.PathLike[str] | None, *, max_chars: int = 8000) -> dict[str, Any]:
    if not repo_dir:
        return {"status": "skipped", "reason": "repo_dir_unavailable", "path": path, "content": ""}
    root = Path(repo_dir).expanduser().resolve()
    data = safe_read_files(str(root), [path], max_chars_per_file=max_chars)
    return {"status": "success" if path in data else "missing", "path": path, "content": data.get(path, "")}


def apply_patch(patch: str, repo_dir: str | os.PathLike[str] | None, *, dry_run: bool = True, timeout: int = 60) -> dict[str, Any]:
    if not repo_dir:
        return {"status": "skipped", "reason": "repo_dir_unavailable", "dry_run": dry_run}
    if not patch.strip():
        return {"status": "skipped", "reason": "empty_patch", "dry_run": dry_run}
    if dry_run:
        return {"status": "dry_run", "patch_chars": len(patch), "dry_run": True}
    start = time.perf_counter()
    proc = subprocess.run(
        ["git", "apply", "-"],
        input=patch,
        cwd=Path(repo_dir).expanduser().resolve(),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "status": "success" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-8000:],
        "duration_sec": round(time.perf_counter() - start, 6),
        "dry_run": False,
    }


def run_tests(command: str, repo_dir: str | os.PathLike[str] | None, *, dry_run: bool = True, timeout: int = 120) -> dict[str, Any]:
    if not repo_dir:
        return {"status": "skipped", "reason": "repo_dir_unavailable", "command": command, "dry_run": dry_run}
    if dry_run:
        return {"status": "dry_run", "command": command, "dry_run": True, "returncode": None, "stdout": "", "stderr": ""}
    parts = command.split()
    if not parts or not (parts[0] == "pytest" or parts[:3] == ["python", "-m", "pytest"]):
        return {"status": "refused", "reason": "only pytest commands are allowed", "command": command}
    start = time.perf_counter()
    try:
        proc = subprocess.run(parts, cwd=Path(repo_dir).expanduser().resolve(), text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "status": "success" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "duration_sec": round(time.perf_counter() - start, 6),
            "dry_run": False,
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": 124,
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
            "duration_sec": round(time.perf_counter() - start, 6),
            "dry_run": False,
            "command": command,
        }


def revert_patch(repo_dir: str | os.PathLike[str] | None, *, dry_run: bool = True, timeout: int = 60) -> dict[str, Any]:
    if not repo_dir:
        return {"status": "skipped", "reason": "repo_dir_unavailable", "dry_run": dry_run}
    if dry_run:
        return {"status": "dry_run", "dry_run": True}
    start = time.perf_counter()
    proc = subprocess.run(["git", "diff", "--quiet"], cwd=Path(repo_dir).expanduser().resolve(), text=True, capture_output=True, timeout=timeout, check=False)
    if proc.returncode == 0:
        return {"status": "clean", "duration_sec": round(time.perf_counter() - start, 6), "dry_run": False}
    reset = subprocess.run(["git", "checkout", "--", "."], cwd=Path(repo_dir).expanduser().resolve(), text=True, capture_output=True, timeout=timeout, check=False)
    return {
        "status": "success" if reset.returncode == 0 else "failed",
        "returncode": reset.returncode,
        "stdout": reset.stdout[-8000:],
        "stderr": reset.stderr[-8000:],
        "duration_sec": round(time.perf_counter() - start, 6),
        "dry_run": False,
    }
