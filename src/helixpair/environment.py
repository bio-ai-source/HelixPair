from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from helixpair.io_utils import ensure_dir, write_json


def bootstrap_environment(project_root: str | Path, python_executable: str | None = None) -> dict[str, str]:
    project_root = Path(project_root)
    repro_root = ensure_dir(project_root / "repro")
    environment = {
        "python": python_executable or sys.executable,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    try:
        import torch

        environment["torch_version"] = torch.__version__
        environment["cuda_available"] = str(bool(torch.cuda.is_available()))
        if torch.cuda.is_available():
            environment["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - best effort runtime capture
        environment["torch_error"] = repr(exc)

    try:
        completed = subprocess.run(
            [environment["python"], "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            (repro_root / "requirements.lock.txt").write_text(completed.stdout, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort runtime capture
        environment["pip_freeze_error"] = repr(exc)

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            (repro_root / "commit_hash.txt").write_text(completed.stdout.strip() + "\n", encoding="utf-8")
            environment["git_commit"] = completed.stdout.strip()
    except Exception as exc:  # pragma: no cover - best effort runtime capture
        environment["git_error"] = repr(exc)

    write_json(repro_root / "environment_bootstrap.json", environment)
    return environment
