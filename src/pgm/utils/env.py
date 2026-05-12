"""Capture reproducibility metadata for a run."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pgm.utils.paths import ensure_parents, results_reports_dir


def _git_sha(cfg) -> str | None:
    root = cfg.resolved_root
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def _pip_freeze() -> list[str]:
    pip = shutil.which("pip") or shutil.which("pip.exe")
    if pip is None:
        return []
    try:
        out = subprocess.run(
            [pip, "freeze"], check=False, capture_output=True, text=True
        )
        return [ln for ln in out.stdout.splitlines() if ln.strip()]
    except OSError:
        return []


def snapshot_run_environment(cfg, tag: str = "run") -> Path:
    """
    Save python version, optional pip freeze lines, git SHA under ``reports/results``.
    """
    out_dir = results_reports_dir(cfg)
    ensure_parents(out_dir / ".touch")
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "tag": tag,
        "utc_time": ts,
        "python": sys.version,
        "executable": sys.executable,
        "git_sha": _git_sha(cfg),
        "pip_freeze": _pip_freeze(),
    }
    path = out_dir / f"{tag}_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
