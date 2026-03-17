#!/usr/bin/env python3
"""Reproduce, diagnose, and optionally fix environment warnings (4/5/6).

Warnings covered:
4) Gym unmaintained notice spam.
5) robosuite macros_private warning.
6) OpenGL_accelerate missing info log.

Usage:
  uv run --no-sync python toolkits/debug_env_warnings_456.py
  uv run --no-sync python toolkits/debug_env_warnings_456.py --repeat 3
  uv run --no-sync python toolkits/debug_env_warnings_456.py --fix
  uv run --no-sync python toolkits/debug_env_warnings_456.py --fix --silence-gym-notice
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CmdResult:
    code: int
    stdout: str
    stderr: str


def run_py(code: str, env: dict[str, str] | None = None) -> CmdResult:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    return CmdResult(proc.returncode, proc.stdout, proc.stderr)


def pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def print_header(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def reproduce_once() -> None:
    print_header("Reproduce Warnings")

    gym_res = run_py("import gym")
    print("[4] import gym")
    print(f"exit={gym_res.code}")
    if gym_res.stderr.strip():
        print("stderr:\n" + gym_res.stderr.strip())
    else:
        print("stderr: <empty>")

    robo_res = run_py("import robosuite")
    print("\n[5] import robosuite")
    print(f"exit={robo_res.code}")
    if robo_res.stderr.strip():
        print("stderr:\n" + robo_res.stderr.strip())
    else:
        print("stderr: <empty>")

    ogl_code = (
        "import logging; "
        "logging.basicConfig(level=logging.INFO); "
        "import OpenGL.acceleratesupport"
    )
    ogl_res = run_py(ogl_code)
    print("\n[6] import OpenGL.acceleratesupport")
    print(f"exit={ogl_res.code}")
    if ogl_res.stderr.strip():
        print("stderr:\n" + ogl_res.stderr.strip())
    else:
        print("stderr: <empty>")


def diagnose() -> tuple[bool, bool, bool]:
    print_header("Diagnosis")

    gym_ver = pkg_version("gym")
    gym_notice_ver = pkg_version("gym-notices")
    gymnasium_ver = pkg_version("gymnasium")
    print(f"gym={gym_ver}")
    print(f"gym-notices={gym_notice_ver}")
    print(f"gymnasium={gymnasium_ver}")
    gym_notice_expected = gym_ver != "NOT_INSTALLED" and gym_notice_ver != "NOT_INSTALLED"
    print(
        "cause[4]: gym imports gym_notices and prints maintenance notice to stderr on every process import"
    )

    robosuite_ver = pkg_version("robosuite")
    robo_has_private = False
    robo_private_path = "<unknown>"
    if robosuite_ver != "NOT_INSTALLED":
        robosuite = importlib.import_module("robosuite")
        robo_root = Path(robosuite.__path__[0])
        robo_private = robo_root / "macros_private.py"
        robo_private_path = str(robo_private)
        robo_has_private = robo_private.exists()
    print(f"robosuite={robosuite_ver}")
    print(f"robosuite_macros_private={robo_private_path}")
    print(f"robosuite_macros_private_exists={robo_has_private}")
    print(
        "cause[5]: robosuite/macros.py warns when robosuite.macros_private cannot be imported"
    )

    pyopengl_ver = pkg_version("PyOpenGL")
    accelerate_spec = importlib.util.find_spec("OpenGL_accelerate")
    accelerate_ver = pkg_version("PyOpenGL-accelerate")
    accelerate_ok = accelerate_spec is not None
    print(f"PyOpenGL={pyopengl_ver}")
    print(f"PyOpenGL-accelerate={accelerate_ver}")
    print(f"OpenGL_accelerate_importable={accelerate_ok}")
    print(
        "cause[6]: OpenGL.acceleratesupport logs info when OpenGL_accelerate is unavailable"
    )

    return gym_notice_expected, robo_has_private, accelerate_ok


def fix_robosuite_macros() -> bool:
    try:
        robosuite = importlib.import_module("robosuite")
    except Exception as exc:  # pragma: no cover
        print(f"[fix][5] skip: robosuite import failed: {exc}")
        return False

    setup_script = Path(robosuite.__path__[0]) / "scripts" / "setup_macros.py"
    if not setup_script.exists():
        print(f"[fix][5] skip: setup script not found: {setup_script}")
        return False

    print(f"[fix][5] running: {sys.executable} {setup_script}")
    proc = subprocess.run([sys.executable, str(setup_script)], text=True, check=False)
    ok = proc.returncode == 0
    print(f"[fix][5] {'ok' if ok else 'failed'} (exit={proc.returncode})")
    return ok


def fix_opengl_accelerate() -> bool:
    if importlib.util.find_spec("OpenGL_accelerate") is not None:
        print("[fix][6] OpenGL_accelerate already installed")
        return True

    print("[fix][6] installing PyOpenGL-accelerate via pip")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "PyOpenGL-accelerate"],
        text=True,
        check=False,
    )
    ok = proc.returncode == 0 and importlib.util.find_spec("OpenGL_accelerate") is not None
    print(f"[fix][6] {'ok' if ok else 'failed'} (exit={proc.returncode})")
    return ok


def silence_gym_notice() -> bool:
    if pkg_version("gym-notices") == "NOT_INSTALLED":
        print("[fix][4] gym-notices already absent")
        return True

    print("[fix][4] uninstalling gym-notices to silence import-time stderr notice")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "gym-notices"],
        text=True,
        check=False,
    )
    ok = proc.returncode == 0 and pkg_version("gym-notices") == "NOT_INSTALLED"
    print(f"[fix][4] {'ok' if ok else 'failed'} (exit={proc.returncode})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to reproduce (simulates multi-process repeated output).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply automatic fixes for [5] and [6].",
    )
    parser.add_argument(
        "--silence-gym-notice",
        action="store_true",
        help="Also remove gym-notices package to silence [4] import notice.",
    )
    args = parser.parse_args()

    print_header("Environment")
    print(f"python={sys.executable}")
    print(f"cwd={os.getcwd()}")

    for i in range(args.repeat):
        print(f"\n--- reproduce round {i + 1}/{args.repeat} ---")
        reproduce_once()

    gym_notice_expected, robo_has_private, accelerate_ok = diagnose()

    if args.fix:
        print_header("Apply Fixes")

        if not robo_has_private:
            fix_robosuite_macros()
        else:
            print("[fix][5] robosuite macros_private already present")

        if not accelerate_ok:
            fix_opengl_accelerate()
        else:
            print("[fix][6] OpenGL_accelerate already importable")

        if args.silence_gym_notice and gym_notice_expected:
            silence_gym_notice()
        elif gym_notice_expected:
            print(
                "[fix][4] gym notice is expected as long as gym + gym-notices are installed; "
                "use --silence-gym-notice for pragmatic suppression, or migrate deps to gymnasium."
            )

        print_header("Re-check After Fix")
        reproduce_once()
        diagnose()
    else:
        print_header("Planned Fixes")
        print("[4] gym notice: migrate dependency chain to gymnasium (long-term) or use --silence-gym-notice.")
        print("[5] robosuite warning: run with --fix to generate macros_private.py.")
        print("[6] OpenGL warning: run with --fix to install PyOpenGL-accelerate.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
