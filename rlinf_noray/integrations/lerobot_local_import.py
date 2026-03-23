from __future__ import annotations

import importlib
import sys
from pathlib import Path


class LocalLerobotImportError(RuntimeError):
    pass


def ensure_local_lerobot() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if sys.path[0] != repo_root_str:
        if repo_root_str in sys.path:
            sys.path.remove(repo_root_str)
        sys.path.insert(0, repo_root_str)

    lerobot = importlib.import_module("lerobot")
    lerobot_file = Path(getattr(lerobot, "__file__", "")).resolve()
    expected_root = (repo_root / "lerobot").resolve()

    if expected_root not in lerobot_file.parents:
        raise LocalLerobotImportError(
            "LeRobot import is not resolved from local workspace. "
            f"Expected under {expected_root}, got {lerobot_file}."
        )
