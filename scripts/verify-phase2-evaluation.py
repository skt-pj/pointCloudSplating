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


def parse_version_properties(text: str) -> tuple[tuple[int, ...], int]:
    values = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    try:
        version_name = tuple(int(part) for part in values["VERSION_NAME"].split("."))
        version_code = int(values["VERSION_CODE"])
    except (KeyError, ValueError) as error:
        raise SystemExit(f"phase2 evaluation check failed: invalid version.properties: {error}")
    return version_name, version_code


def main() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    coordinator = COORDINATOR.read_text(encoding="utf-8")
    model_processing = MODEL_PROCESSING.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    forbid(evaluator, "NativeGaussianTrainer", "Phase 2 must not start the Phase 3 trainer")
    forbid(evaluator, "GaussianSplatJob", "Phase 2 must not invoke the 3DGS job")
    forbid(evaluator, "ColmapDatasetExporter", "Phase 2 geometry must use independent depth_obs truth")

    require(evaluator, 'name.startsWith("depth_obs_")',
            "independent Depth Observation source-of-truth missing")
    require(evaluator, "MIN_CONFIDENCE = 0.30f", "confidence filtering missing")
    require(evaluator, "depthExcludedQualityCount", "quality-only Depth exclusion tracking missing")
    require(evaluator, 'warn("depth_observation_no_usable_points"',
            "quality-only Depth observations must be excluded without hard-failing the dataset")
    require(evaluator, '"depth_observation_excluded_quality_count"',
            "quality-only exclusion count missing from report")
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

    # Pixel 10a v1.0.8 baseline was reviewed before Phase 3. PASS now depends on both the
    # original geometry gates and the frozen evidence-backed RGB/depth alignment gates.
    require(evaluator, "DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX = 8.0",
            "reviewed edge threshold missing")
    require(evaluator, "MAX_SYSTEMATIC_OFFSET_PX = 2.0",
            "systematic alignment gate missing")
    require(evaluator, "depth_edge_systematic_offset", "systematic alignment failure missing")
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

    require(model_processing, "PHASE3_PROCESSING_ENABLED = true",
            "Phase 3 build switch was not enabled after Phase 2 PASS review")
    version_name, version_code = parse_version_properties(version)
    if version_name < (1, 0, 9):
        raise SystemExit(
            f"phase2 evaluation check failed: Phase 3 transition versionName too old: {version_name}")
    if version_code < 46:
        raise SystemExit(
            f"phase2 evaluation check failed: Phase 3 transition versionCode too old: {version_code}")

    print("Phase 2 evaluation architecture checks passed")


if __name__ == "__main__":
    main()
