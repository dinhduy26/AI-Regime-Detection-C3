"""Repeat C1/C2 tests in a fresh process and independently audit persisted artifacts.

Run from any directory: python scripts/verify_c2.py
This is a reproducible local verification, not a claim of third-party review.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "outputs" / "metadata"
TEST_INVENTORY = ROOT / "tests" / "fixtures" / "c2_test_inventory.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_inventory():
    paths = [ROOT / name for name in ("config/config.yaml", "config/schema.yaml", "requirements-lock.txt")]
    paths.append(Path(__file__).resolve())
    paths.append(TEST_INVENTORY)
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("data/processed", "*.parquet")):
        paths += sorted((ROOT / folder).rglob(pattern))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def read_junit(path):
    tree = ET.parse(path)
    cases = tree.findall(".//testcase")
    identities = {(c.get("classname"), c.get("name")) for c in cases}
    failures = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
    skipped = sum(c.find("skipped") is not None for c in cases)
    suites = tree.findall(".//testsuite")
    return {
        "tests": len(cases),
        "failures_or_errors": failures,
        "skipped": skipped,
        "suite_errors": sum(int(s.get("errors", 0)) for s in suites),
    }, identities


def verify_test_inventory(identities, evidence_dir=META):
    """Verify historical test coverage without requiring machine-generated logs in Git."""
    inventory = json.loads(TEST_INVENTORY.read_text(encoding="utf-8"))
    if inventory["schema_version"] != 1:
        raise ValueError("Unsupported test inventory schema")
    result = {}
    for size in (47, 55):
        records = [case for case in inventory["cases"] if case["introduced_in"] <= size]
        expected = {(case["classname"], case["name"]) for case in records}
        covered = len(expected & identities)
        if len(records) != size or len(expected) != size or covered != size:
            raise ValueError(f"Full suite no longer covers all {size} historical test identities")
        baseline = Path(evidence_dir) / f"pytest_baseline_{size}.xml"
        entry = {
            "baseline": {"tests": size},
            "covered": covered,
            "inventory_sha256": sha256(TEST_INVENTORY),
            "historical_junit_sha256": inventory["baselines"][str(size)]["historical_junit_sha256"],
            "historical_log_available": baseline.exists(),
        }
        if baseline.exists():
            counts, original = read_junit(baseline)
            if (
                original != expected
                or counts["failures_or_errors"]
                or counts["suite_errors"]
                or counts["skipped"]
            ):
                raise ValueError(f"Historical {size}-test log disagrees with the portable inventory")
            entry.update(baseline=counts, baseline_sha256=sha256(baseline))
        result[f"original_{size}_tests"] = entry
    return result


def audit_artifacts():
    """Use library likelihoods directly, without src.models filtering/report helpers."""
    import joblib
    import numpy as np
    import pandas as pd

    manifest = json.loads((META / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Persisted model run is not complete")
    if sha256(ROOT / "config/config.yaml")[: len(manifest["config_hash"])] != manifest["config_hash"]:
        raise ValueError("Current configuration differs from model manifest")
    for relative, expected in manifest["output_hashes"].items():
        if sha256(ROOT / relative)[: len(expected)] != expected:
            raise ValueError(f"Artifact hash mismatch: {relative}")
    compared, metadata_checks = [], []
    b07 = pd.read_csv(ROOT / "outputs/tables/B07_hmm_vs_gmm.csv")
    features = ["z_log_return", "z_rv21", "z_log_dollar_volume"]
    for asset, info in manifest["assets"].items():
        bundle = joblib.load(ROOT / f"outputs/models/hmm_{asset.lower()}.joblib")
        hmm = bundle["model"]
        gmm_bundle = joblib.load(ROOT / f"outputs/models/gmm_{asset.lower()}.joblib")
        gmm = gmm_bundle["model"]
        standalone = json.loads((META / f"model_{asset.lower()}.json").read_text(encoding="utf-8"))
        if info != bundle["metadata"] or info != gmm_bundle["metadata"] or info != standalone:
            raise ValueError(f"Final metadata differs across joblib/JSON/manifest: {asset}")
        if info["test_evaluated"] is not True or not info.get("finalized_at_utc"):
            raise ValueError("Saved artifact metadata was not finalized")
        for kind in ("filtered", "smoothed", "gmm"):
            output = pd.read_csv(ROOT / f"outputs/predictions/regime_{kind}_{asset.lower()}.csv")
            expected_mapping = info["gmm_label_map"] if kind == "gmm" else info["label_map"]
            expected_risk = info["gmm_high_risk_cluster"] if kind == "gmm" else info["high_risk_state"]
            if (
                output["split"].value_counts().to_dict() != info["n_by_period"]
                or not output["model_run_id"].eq(info["model_run_id"]).all()
                or not output["K"].eq(info["K"]).all()
                or not output["high_risk_state"].eq(expected_risk).all()
                or not output["label_map"].map(json.loads).map(lambda value: value == expected_mapping).all()
            ):
                raise ValueError(f"CSV metadata disagrees with final model: {asset} {kind}")
        metadata_checks.append(
            {"asset": asset, "hmm_gmm_json_manifest_equal": True, "csv_metadata_equal": True}
        )
        # A changed implementation must not pass using stale artifacts.
        for name, digest in info["source_sha256"].items():
            if sha256(ROOT / "src/models" / name)[: len(digest)] != digest:
                raise ValueError(f"Stale model source fingerprint: {asset} {name}")
        frame = pd.read_parquet(ROOT / f"data/processed/scaled_features_{asset.lower()}.parquet")
        previous_ll = 0.0
        for index, period in enumerate(("train", "validation", "test")):
            prefix = frame.loc[frame["split"].isin(("train", "validation", "test")[: index + 1])]
            dates = pd.to_datetime(prefix["date_utc"], utc=True)
            boundary = dates.diff().ne(pd.Timedelta(days=1)) | prefix["segment_id"].ne(
                prefix["segment_id"].shift()
            )
            offsets = np.flatnonzero(boundary.to_numpy())
            lengths = np.diff(np.r_[offsets, len(prefix)]).tolist()
            ll = float(hmm.score(prefix[features].to_numpy(), lengths=lengths))
            part = frame.loc[frame["split"] == period]
            # Difference of prefix likelihoods equals the conditional log predictive score.
            scores = {
                "HMM": (ll - previous_ll) / len(part),
                "GMM": float(gmm.score_samples(part[features].to_numpy()).mean()),
            }
            previous_ll = ll
            for method, actual in scores.items():
                row = b07.loc[(b07["asset"] == asset) & (b07["period"] == period) & (b07["method"] == method)]
                if len(row) != 1 or int(row.iloc[0]["n"]) != len(part):
                    raise ValueError("B07 sample size mismatch")
                expected = float(row.iloc[0]["avg_log_score"])
                if not np.isclose(actual, expected, rtol=1e-8, atol=1e-8):
                    raise ValueError(f"B07 library score mismatch: {asset} {method} {period}")
                compared.append(
                    {
                        "asset": asset,
                        "period": period,
                        "method": method,
                        "n": len(part),
                        "library_score": actual,
                        "stored_score": expected,
                    }
                )
    return {
        "output_hashes_checked": len(manifest["output_hashes"]),
        "library_score_checks": compared,
        "metadata_checks": metadata_checks,
        "model_quality_status": manifest["quality_status"],
    }


def main():
    META.mkdir(parents=True, exist_ok=True)
    report_path, junit = META / "C2_verification.json", META / "pytest_verification.xml"
    log_path = META / "pytest_verification.log"
    # Remove only previous verification outputs, so a failed launch cannot reuse old success.
    for path in (junit, log_path):
        if path.exists():
            path.unlink()
    started = time.perf_counter()
    before = input_inventory()
    report = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_kind": "fresh Python process plus direct library artifact audit",
        "review_scope": "local reproducible verification; not a third-party external review",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "input_sha256": before,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        env = os.environ.copy()
        env.update(
            PYTHONHASHSEED="0",
            PYTHONIOENCODING="utf-8",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"]
        report["command"] = command
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        output = result.stdout + result.stderr
        log_path.write_text(output, encoding="utf-8")
        print(output, end="")
        report["pytest_exit_code"] = result.returncode
        counts, identities = read_junit(junit)
        report["pytest"] = counts
        if (
            result.returncode
            or not counts["tests"]
            or counts["failures_or_errors"]
            or counts["suite_errors"]
            or counts["skipped"]
        ):
            raise ValueError("Full suite did not finish with all tests passing and none skipped")
        report.update(verify_test_inventory(identities))
        report["artifact_audit"] = audit_artifacts()
        # Exercise the actual public consumer interface, separately from likelihood checks.
        sys.path.insert(0, str(ROOT))
        from src.models.integration import check_handoff
        from src.utils.config import load_config

        report["c2_c3_integration"] = check_handoff(load_config())
        (META / "C2_C3_integration_check.json").write_text(
            json.dumps(report["c2_c3_integration"], indent=2), encoding="utf-8"
        )
        from packaging.requirements import Requirement

        pinned = {}
        for line in (ROOT / "requirements-lock.txt").read_text(encoding="ascii").splitlines():
            if not line or line.startswith("#"):
                continue
            req = Requirement(line)
            version = importlib.metadata.version(req.name)
            if not req.specifier.contains(version):
                raise ValueError(f"Dependency differs from lock: {req.name} {version}")
            pinned[req.name] = version
        report["locked_dependencies_checked"] = pinned
        check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        report["pip_check"] = {"exit_code": check.returncode, "output": check.stdout + check.stderr}
        if check.returncode:
            raise ValueError("Dependency compatibility check failed")
        report["inputs_unchanged_during_verification"] = before == input_inventory()
        if not report["inputs_unchanged_during_verification"]:
            raise ValueError(
                "Sources, tests, configuration or processed input data changed during verification"
            )
        report["evidence_sha256"] = {p.name: sha256(p) for p in (junit, log_path)}
        report["status"] = "passed"
    except Exception as exc:
        report["status"], report["error"] = "failed", f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = round(time.perf_counter() - started, 3)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
