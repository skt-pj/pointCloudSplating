#!/usr/bin/env python3
"""Build-time contract checks for the Phase 1 observation hard gate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/Phase1DatasetEvaluator.java"
FINALIZER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DatasetFinalizer.java"
JOB = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/GaussianSplatJob.java"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"phase1 evaluation check failed: {why}: missing {needle!r}")


def forbid(text: str, needle: str, why: str) -> None:
    if needle in text:
        raise SystemExit(f"phase1 evaluation check failed: {why}: found {needle!r}")


def main() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    job = JOB.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    # Phase 1 evaluates captured observations only. Training/geometry build belongs to later phases.
    forbid(evaluator, "NativeGaussianTrainer", "Phase 1 evaluator must not start training")
    forbid(evaluator, "GaussianSplatJob", "Phase 1 evaluator must not run 3DGS jobs")
    forbid(evaluator, "ColmapDatasetExporter", "Phase 1 evaluator must not build trainer input")

    # Every requested hard-gate dimension must be present in code and in the persisted report.
    require(evaluator, "MIN_HIGH_RES_PIXELS = 8_000_000L", "high-resolution RGB gate missing")
    require(evaluator, "MAX_POSE_DELTA_NS = 75_000_000L", "75 ms pose gate missing")
    require(evaluator, "BitmapFactory.decodeFile", "actual JPEG decode-bounds validation missing")
    require(evaluator, "bracketed_pose_interpolation", "pose interpolation validation missing")
    require(evaluator, "pose_before_timestamp_ns", "pose bracket start validation missing")
    require(evaluator, "pose_after_timestamp_ns", "pose bracket end validation missing")
    require(evaluator, "identityError", "world/camera inverse validation missing")
    require(evaluator, "requireRigid", "rigid camera transform validation missing")
    require(evaluator, "inspectPly", "Depth PLY validation missing")
    require(evaluator, "Float.isFinite", "finite Depth point/confidence validation missing")
    require(evaluator, "confidence < 0f || confidence > 1f", "Depth confidence range gate missing")
    require(evaluator, "DiagnosticLog.snapshot()", "capture-session health validation missing")
    require(evaluator, "OpenGL/ARCore frame failed; frame loop latched", "ARCore fatal gate missing")
    require(evaluator, "Shared camera error=", "Camera2 fatal gate missing")
    require(evaluator, '"PHASE1_EVAL %s', "single PASS/FAIL diagnostic line missing")
    require(evaluator, '"phase1_evaluation.json"', "persistent evaluation report missing")
    require(evaluator, '"next_phase_allowed"', "next-phase gate result missing")

    require(finalizer, "Phase1DatasetEvaluator.evaluate(workingDirectory)",
            "finalization does not execute Phase 1 evaluation")
    require(finalizer, '"phase1_evaluation_status"', "manifest evaluation status missing")
    require(finalizer, '"phase1_evaluation_required_for_phase2", true',
            "Phase 2 hard-gate contract missing")

    # No current downstream path may bypass a failed/missing Phase 1 evaluation.
    require(job, "Phase1DatasetEvaluator.hasStoredPass(datasetDirectory)",
            "3D processing does not enforce the Phase 1 PASS gate")
    require(job, "Downstream 3D processing blocked: PHASE1_EVAL is not PASS",
            "blocked downstream processing is not diagnosable")

    require(version, "VERSION_NAME=1.0.3", "versionName must identify Phase 1 evaluator build")
    require(version, "VERSION_CODE=40", "versionCode must identify Phase 1 evaluator build")

    print("Phase 1 evaluation architecture checks passed")


if __name__ == "__main__":
    main()
