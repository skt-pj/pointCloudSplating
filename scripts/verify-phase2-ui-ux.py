#!/usr/bin/env python3
"""Build-time checks for the v1.0.7 Phase 2 classification and user-facing UI correction."""
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

    # Structurally valid Depth observations can legitimately contribute zero points after the
    # Phase 2 confidence filter. They are a quality exclusion, not a corrupt-PLY hard failure.
    require(evaluator, "int depthExcludedQualityCount;", "quality exclusion counter missing")
    require(evaluator, '"depth_observation_no_usable_points"', "quality exclusion reason missing")
    require(evaluator, 'warn("depth_observation_no_usable_points"',
            "low-confidence-only Depth must be a warning, not a hard failure")
    require(evaluator, '"depth_observation_excluded_quality_count"',
            "quality exclusion count is not persisted")
    require(evaluator, "MIN_CONFIDENCE = 0.30f",
            "confidence threshold must not be weakened just to obtain PASS")

    # Phase 3 remains blocked until Phase 2 is actually accepted, but the UI must say that plainly
    # instead of presenting an enabled action that inevitably fails.
    require(processing, "PHASE3_PROCESSING_ENABLED = false", "Phase 3 was enabled prematurely")
    require(processing, "static boolean isPhase3ProcessingEnabled()",
            "UI cannot query Phase 3 availability")
    require(scanner, '"3Dモデル作成 準備中"', "scanner does not expose unavailable state")
    require(scanner, "gaussianButton.setEnabled(phase3Enabled);",
            "scanner Phase 3 action is still enabled")
    require(scanner, '"撮影データは保存済みです。3Dモデル作成機能は現在準備中です。"',
            "scanner does not separate saved data from unavailable model creation")
    forbid(scanner, "3Dプレビューを作成できます。",
            "save success must not promise an unavailable downstream action")
    forbid(job, "安全のため", "meaningless internal safety wording leaked to users")

    require(library, "!ModelProcessingCoordinator.isPhase3ProcessingEnabled()",
            "library can still start unavailable Phase 3 processing")
    require(library, "撮影データ保存済み",
            "library must represent saved data without promising model creation")

    require(version, "VERSION_NAME=1.0.7", "versionName mismatch")
    require(version, "VERSION_CODE=44", "versionCode mismatch")

    print("Phase 2/UI correction checks passed")


if __name__ == "__main__":
    main()
