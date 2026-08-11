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
        raise SystemExit(f"Phase 2/UI check failed: invalid version.properties: {error}")
    return version_name, version_code


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

    version_name, version_code = parse_version_properties(version)
    if version_name < (1, 0, 9):
        raise SystemExit(f"Phase 2/UI check failed: versionName too old: {version_name}")
    if version_code < 46:
        raise SystemExit(f"Phase 2/UI check failed: versionCode too old: {version_code}")

    print("Phase 2/3 UI transition checks passed")


if __name__ == "__main__":
    main()
