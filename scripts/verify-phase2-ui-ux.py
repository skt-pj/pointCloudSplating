#!/usr/bin/env python3
"""Build-time checks for saved-data UI semantics across the Phase 2 -> Phase 3 transition."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting"
EVALUATOR = JAVA / "Phase2DatasetEvaluator.java"
SCANNER = JAVA / "ScannerActivity.java"
JOB = JAVA / "GaussianSplatJob.java"
PROCESSING = JAVA / "ModelProcessingCoordinator.java"
LIBRARY = JAVA / "LibraryActivity.java"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"Phase 2/UI check failed: {why}: missing {needle!r}")


def forbid(text: str, needle: str, why: str) -> None:
    if needle in text:
        raise SystemExit(f"Phase 2/UI check failed: {why}: found {needle!r}")


def main() -> None:
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    scanner = SCANNER.read_text(encoding="utf-8")
    job = JOB.read_text(encoding="utf-8")
    processing = PROCESSING.read_text(encoding="utf-8")
    library = LIBRARY.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    require(evaluator, "int depthExcludedQualityCount;", "quality exclusion counter missing")
    require(evaluator, '"depth_observation_no_usable_points"', "quality exclusion reason missing")
    require(evaluator, 'warn("depth_observation_no_usable_points"',
            "low-confidence-only Depth must be a warning, not a hard failure")
    require(evaluator, '"depth_observation_excluded_quality_count"',
            "quality exclusion count is not persisted")
    require(evaluator, "MIN_CONFIDENCE = 0.30f",
            "confidence threshold must not be weakened just to obtain PASS")

    require(processing, "PHASE3_PROCESSING_ENABLED = true", "Phase 3 is not enabled")
    require(processing, "static boolean isPhase3ProcessingEnabled()",
            "UI cannot query Phase 3 availability")
    require(scanner, '"3Dモデルを作成"', "scanner Phase 3 action missing")
    require(scanner, '"撮影を保存しました"', "save success state missing")
    forbid(scanner, "3Dプレビューを作成できます。",
            "save success must not promise an old preview path")
    forbid(job, "安全のため", "meaningless internal safety wording leaked to users")
    require(job, "Phase2DatasetEvaluator.hasStoredPass(datasetDirectory)",
            "Phase 3 can bypass the Phase 2 PASS gate")

    require(library, "ModelProcessingCoordinator.isPhase3ProcessingEnabled()",
            "library does not use Phase 3 availability")
    require(library, "品質確認待ち",
            "library does not distinguish machine-complete model from final visual quality PASS")

    require(version, "VERSION_NAME=1.0.9", "versionName mismatch")
    require(version, "VERSION_CODE=46", "versionCode mismatch")

    print("Phase 2/3 UI transition checks passed")


if __name__ == "__main__":
    main()
