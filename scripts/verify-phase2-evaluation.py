#!/usr/bin/env python3
"""Build-time contract checks for Phase 2 geometry/camera-view evaluation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/Phase2DatasetEvaluator.java"
COORDINATOR = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/Phase2EvaluationCoordinator.java"
MODEL_PROCESSING = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/ModelProcessingCoordinator.java"
APP = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/PointCloudApp.java"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"phase2 evaluation check failed: {why}: missing {needle!r}")


def forbid(text: str, needle: str, why: str) -> None:
    if needle in text:
        raise SystemExit(f"phase2 evaluation check failed: {why}: found {needle!r}")


def main() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    coordinator = COORDINATOR.read_text(encoding="utf-8")
    model_processing = MODEL_PROCESSING.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    # Phase 2 consumes observations and evaluates geometry/camera consistency only.
    forbid(evaluator, "NativeGaussianTrainer", "Phase 2 must not start the Phase 3 trainer")
    forbid(evaluator, "GaussianSplatJob", "Phase 2 must not invoke the 3DGS job")
    forbid(evaluator, "ColmapDatasetExporter", "Phase 2 geometry must use independent depth_obs truth")

    require(evaluator, 'name.startsWith("depth_obs_")',
            "independent Depth Observation source-of-truth missing")
    require(evaluator, "MIN_CONFIDENCE = 0.30f", "confidence filtering missing")
    require(evaluator, "FUSION_VOXEL_METERS = 0.008f", "voxel fusion missing")
    require(evaluator, "removeIsolatedFusionVoxels", "outlier/isolated voxel filtering missing")
    require(evaluator, '"phase2_geometry_prior.ply"', "fused geometry artifact missing")

    require(evaluator, "compareMutualNearest", "overlapping Depth surface consistency missing")
    require(evaluator, "MAX_SURFACE_MEDIAN_METERS = 0.03", "3 cm surface median gate missing")
    require(evaluator, "MAX_SURFACE_P90_METERS = 0.08", "8 cm surface p90 gate missing")
    require(evaluator, "camera_space_leave_one_out_nearest_surface_depth",
            "camera-space depth consistency metric missing")
    require(evaluator, "MAX_DEPTH_MEDIAN_METERS = 0.03", "3 cm depth median gate missing")
    require(evaluator, "MAX_DEPTH_P90_METERS = 0.08", "8 cm depth p90 gate missing")
    require(evaluator, "raw_depth_pixel_grid_not_persisted",
            "saved-Depth limitation must be explicit rather than faked")

    require(evaluator, "projectCamera", "RGB geometry projection missing")
    require(evaluator, "rgb_zero_projection", "zero-projection hard gate missing")
    require(evaluator, "countConnectedComponents", "view overlap graph check missing")
    require(evaluator, "rgb_overlap_graph_disconnected", "connected graph hard gate missing")
    require(evaluator, "REQUIRED_OVERLAYS = 5", "five-view overlay requirement missing")
    require(evaluator, '"phase2_overlay_%02d_view_%03d.jpg"', "overlay artifact output missing")
    require(evaluator, '"depth_edge_alignment_error_px"', "edge alignment metric missing")

    # First Pixel 10a measurement must not invent an edge threshold just to produce PASS.
    require(evaluator, "DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX = null",
            "first-run edge threshold must remain intentionally unset")
    require(evaluator, '"REVIEW_REQUIRED"', "baseline review state missing")
    require(evaluator, '"next_phase_allowed"', "Phase 3 gate result missing")
    require(evaluator, '"phase2_evaluation.json"', "persistent Phase 2 report missing")
    require(evaluator, '"PHASE2_EVAL %s', "single Phase 2 diagnostic summary missing")
    require(evaluator, "hasStoredPass", "stored Phase 2 PASS gate missing")

    require(coordinator, "Phase1DatasetEvaluator.hasStoredPass(dataset)",
            "coordinator may not evaluate a non-PASS Phase 1 dataset")
    require(coordinator, 'new File(file, ".saved").isFile()',
            "coordinator must use finalized datasets")
    require(coordinator, "hasActivelyChangingCapture",
            "Phase 2 must not compete with active capture")
    require(app, "Phase2EvaluationCoordinator.initialize(appContext);",
            "Phase 2 coordinator not initialized")

    # The legacy Phase 3 trainer remains compiled for regression checks, but the Phase 2 APK must
    # not permit UI paths to start it before Phase 2 has passed on a real device.
    require(model_processing, "PHASE3_PROCESSING_ENABLED = false",
            "Phase 3 processing is not disabled in the Phase 2 measurement build")
    require(model_processing,
            "Phase 3 model processing blocked: current build is Phase 2 evaluation only",
            "blocked Phase 3 attempts are not diagnosable")

    require(version, "VERSION_NAME=1.0.6", "Phase 2 build versionName mismatch")
    require(version, "VERSION_CODE=43", "Phase 2 build versionCode mismatch")

    print("Phase 2 evaluation architecture checks passed")


if __name__ == "__main__":
    main()
