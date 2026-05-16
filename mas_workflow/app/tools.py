"""受限本地工具。"""

from __future__ import annotations

import os
import random
import shlex
import subprocess
import time
from pathlib import Path


DENY_TOKENS = {
    "rm",
    "sudo",
    "curl",
    "wget",
    "chmod",
    "chown",
    "ssh",
    "scp",
    "nc",
    "netcat",
    "dd",
    "mkfs",
}


def simulated_tool_delay(profile: str = "none", *, burst: bool = False) -> float:
    """模拟工具等待时间，用于构造 tool stall 和 tool return burst。"""
    profile = (profile or "none").lower()
    if profile == "none":
        delay = 0.0
    elif burst:
        if profile == "short":
            delay = random.uniform(0.8, 1.2)
        elif profile == "medium":
            delay = random.uniform(5.0, 8.0)
        elif profile == "long":
            delay = random.uniform(25.0, 32.0)
        else:
            delay = random.uniform(4.0, 9.0)
    elif profile == "short":
        delay = random.uniform(0.5, 1.5)
    elif profile == "medium":
        delay = random.uniform(5.0, 10.0)
    elif profile == "long":
        delay = random.uniform(20.0, 40.0)
    elif profile == "random":
        delay = min(40.0, random.lognormvariate(1.2, 0.8))
    else:
        delay = 0.0
    if delay > 0:
        time.sleep(delay)
    return delay


def _repo_root(repo_path: str | os.PathLike[str]) -> Path:
    root = Path(repo_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"repo_path 不存在或不是目录: {root}")
    return root


def _ensure_inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"路径越界，拒绝访问: {resolved}")
    return resolved


def safe_list_files(repo_path: str, max_files: int = 100) -> list[str]:
    """列出 repo 内有限数量的源码和测试文件。"""
    root = _repo_root(repo_path)
    ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignored]
        for filename in sorted(filenames):
            path = _ensure_inside(root, Path(dirpath) / filename)
            rel = path.relative_to(root).as_posix()
            results.append(rel)
            if len(results) >= max_files:
                return results
    return results


def safe_read_files(
    repo_path: str,
    file_paths: list[str],
    max_chars_per_file: int = 6000,
) -> dict[str, str]:
    """读取 repo 内有限文件内容。"""
    root = _repo_root(repo_path)
    contents: dict[str, str] = {}
    for file_path in file_paths:
        path = _ensure_inside(root, root / file_path)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            text = f"[读取失败: {exc}]"
        contents[file_path] = text[:max_chars_per_file]
    return contents


def _validate_pytest_command(test_command: str) -> list[str]:
    parts = shlex.split(test_command)
    if not parts:
        raise ValueError("test_command 不能为空")
    lowered = [p.lower() for p in parts]
    if any(token in lowered for token in DENY_TOKENS):
        raise ValueError(f"命令包含危险 token，拒绝执行: {test_command}")
    allowed = parts[0] == "pytest" or parts[:3] == ["python", "-m", "pytest"]
    if not allowed:
        raise ValueError("第一版只允许 pytest 或 python -m pytest")
    return parts


def run_pytest(
    repo_path: str,
    test_command: str = "pytest -q",
    timeout: int = 120,
) -> dict[str, object]:
    """在 repo_path 内运行受限 pytest 命令。"""
    root = _repo_root(repo_path)
    parts = _validate_pytest_command(test_command)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            parts,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": test_command,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "duration_sec": round(time.perf_counter() - start, 6),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": test_command,
            "returncode": 124,
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
            "duration_sec": round(time.perf_counter() - start, 6),
            "error": "pytest 超时",
        }
