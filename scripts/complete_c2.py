"""Reproduce C2 completion with raw subprocess logs, avoiding shell stderr wrappers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    meta = ROOT / "outputs/metadata"
    meta.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(), "steps": []}
    destination = meta / "C2_completion_run.json"
    env = os.environ.copy()
    env.update(
        PYTHONIOENCODING="utf-8",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    commands = [
        ("core", ["run_pipeline.py", "--steps", "models"], "C2_completion_pipeline.log"),
        ("diagnostics", ["-m", "src.models.diagnostics"], "C2_window_diagnostics.log"),
        ("verification", ["scripts/verify_c2.py"], "C2_completion_verification.log"),
    ]
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        for name, arguments, log in commands:
            command = [sys.executable, *arguments]
            print(f"C2 {name}: {' '.join(arguments)}", flush=True)
            with (meta / log).open("w", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            report["steps"].append(
                {"step": name, "command": command, "exit_code": result.returncode, "log": log}
            )
            print((meta / log).read_text(encoding="utf-8"), end="", flush=True)
            if result.returncode:
                raise RuntimeError(f"C2 step {name} failed; inspect {log}")
        report["status"] = "passed"
    except Exception as exc:
        report["status"], report["error"] = "failed", str(exc)
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
