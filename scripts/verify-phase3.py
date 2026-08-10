#!/usr/bin/env python3
"""Build-time contract checks for the v1.0.9 Phase 3 pipeline and quality gate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting"
PHASE2 = JAVA / "Phase2DatasetEvaluator.java"
PHASE3 = JAVA / "Phase3DatasetEvaluator.java"
JOB = JAVA / "GaussianSplatJob.java"
TRAINER = JAVA / "NativeGaussianTrainer.java"
EXPORTER = JAVA / "ColmapDatasetExporter.java"
PROCESSING = JAVA / "ModelProcessingCoordinator.java"
NATIVE = ROOT / "app/src/main/cpp/native_3dgs.cpp"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"Phase 3 check failed: {why}: missing {needle!r}")


def forbid(text: str, needle: str, why: str) -> None:
    if needle in text:
        raise SystemExit(f"Phase 3 check failed: {why}: found {needle!r}")


def main() -> None:
    phase2 = PHASE2.read_text(encoding="utf-8")
    phase3 = PHASE3.read_text(encoding="utf-8")
    job = JOB.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    exporter = EXPORTER.read_text(encoding="utf-8")
    processing = PROCESSING.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    require(phase2, "DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX = 8.0",
            "reviewed Pixel 10a edge gate missing")
    require(phase2, "MAX_SYSTEMATIC_OFFSET_PX = 2.0",
            "reviewed systematic offset gate missing")
    require(job, "Phase2DatasetEvaluator.hasStoredPass(datasetDirectory)",
            "Phase 3 does not enforce stored Phase 2 PASS")
    require(processing, "PHASE3_PROCESSING_ENABLED = true", "Phase 3 switch is disabled")

    require(exporter, 'new File(dataset, "phase2_geometry_prior.ply")',
            "Phase 2 geometry prior is not consumed")
    require(exporter, "Phase2DatasetEvaluator.hasStoredPass(dataset)",
            "exporter can bypass Phase 2 PASS")
    require(exporter, 'File[] depthFiles = {phase2Geometry};',
            "Phase 3 geometry source is not exclusive")
    require(exporter, 'meta.put("geometry_source", "phase2_geometry_prior.ply")',
            "geometry provenance is not persisted")

    require(exporter, "PHASE3_LOW_LONG_EDGE = 720", "low-resolution stage missing")
    require(exporter, "MAX_TRAIN_LONG_EDGE = 1000", "medium-resolution stage missing")
    require(exporter, "PHASE3_HIGH_PATCH_WIDTH = 1280", "high-resolution patch width missing")
    require(exporter, "PHASE3_HIGH_PATCH_HEIGHT = 960", "high-resolution patch height missing")
    require(exporter, "BitmapRegionDecoder", "native-resolution patch decode missing")
    require(exporter, "frame.getDouble(\"cx\") - left", "patch principal point is not corrected")
    require(trainer, 'PHASE3_PROFILE = "progressive_low_mid_high_patch"',
            "progressive Phase 3 profile missing")
    require(trainer, "trainProgressiveInitial", "three-stage Phase 3 entry point missing")
    require(trainer, 'state.put("current_stage", stage)', "stage state is not persisted")
    require(trainer, 'state.put("completed_stage", stage)', "completed stage is not persisted")
    require(trainer, "phase3_stage_result.json", "intermediate stages can be confused with final output")
    require(job, "NativeGaussianTrainer.trainProgressiveInitial", "initial job bypasses progressive training")

    require(native, "executeProjectionForward", "projection forward missing")
    require(native, "executeRasterizeForward", "raster forward missing")
    require(native, "executeComputeSSIMGradient", "SSIM/L1 gradient missing")
    require(native, "executeRasterizeBackward", "raster backward missing")
    require(native, "executeFusedProjectionBackwardOptimizerStep", "parameter optimizer missing")
    require(native, "executeMCMCPostBackward", "density control missing")
    require(native, "restoreTrainingCheckpoint", "stage-to-stage optimizer checkpoint restore missing")

    require(native, "frameCount >= 8 ? 5", "hold-out split missing")
    require(native, "ValidationMetrics", "hold-out metric aggregation missing")
    require(native, "blockSsim", "hold-out SSIM missing")
    require(native, "phase3_holdout_render_", "hold-out render artifact missing")
    # C++ JSON keys are embedded in escaped string literals, so check the key tokens rather than
    # a Java/Python-style quoted literal.
    require(native, "validation_ssim", "native result does not report SSIM")
    require(native, "validation_view_count", "native result does not report hold-out view count")
    require(trainer, 'result.put("holdout_view_count", validationViews)',
            "hold-out view count is not persisted")
    require(trainer, 'result.put("validation_ssim", ssim)',
            "hold-out SSIM is not persisted")

    require(phase3, '"phase3_evaluation.json"', "Phase 3 report missing")
    require(phase3, 'String status = machinePass ? "REVIEW_REQUIRED" : "FAIL"',
            "machine metrics can incorrectly auto-PASS final quality")
    require(phase3, 'report.put("pass", false)', "Phase 3 evaluator can auto-PASS")
    require(phase3, 'report.put("final_visual_pass_requires_user", true)',
            "final visual review contract missing")
    require(phase3, '"visual_review_required"', "visual review warning missing")
    require(phase3, '"ab_fixed_view_pending"',
            "unimplemented A/B artifact must be explicit rather than faked")
    forbid(phase3, 'report.put("pass", true)', "numeric-only final Phase 3 PASS is forbidden")

    require(version, "VERSION_NAME=1.0.9", "versionName mismatch")
    require(version, "VERSION_CODE=46", "versionCode mismatch")
    print("Phase 3 architecture and quality-gate checks passed")


if __name__ == "__main__":
    main()
