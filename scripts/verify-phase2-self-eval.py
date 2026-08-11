#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/Phase2DatasetEvaluator.java"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"Phase 2 self-eval check failed: {why}: missing {needle!r}")


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
        raise SystemExit(f"Phase 2 self-eval check failed: invalid version.properties: {error}")
    return version_name, version_code


def main() -> None:
    java = JAVA.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    require(java, "nearestGradientEdgeOffset", "signed nearest edge measurement missing")
    require(java, '"signed_dx_median_px"', "signed X metric missing")
    require(java, '"signed_dy_median_px"', "signed Y metric missing")
    require(java, '"signed_dx_direction_consensus"', "X direction consensus missing")
    require(java, '"signed_dy_direction_consensus"', "Y direction consensus missing")
    require(java, '"regions"', "quadrant metrics missing")
    require(java, '"systematic_offset_summary"', "cross-view systematic summary missing")
    require(java, '"machine_metrics_in_drive_log_no_manual_file_export"',
            "remote review contract missing")

    # v1.0.8 Pixel 10a evidence was reviewed before Phase 3 activation. The old
    # REVIEW_REQUIRED-only state must now be replaced by evidence-backed fixed gates.
    require(java, "DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX = 8.0",
            "reviewed edge threshold is not frozen")
    require(java, "MAX_SYSTEMATIC_OFFSET_PX = 2.0",
            "reviewed systematic-offset threshold is missing")
    require(java, "pixel10a_baseline_20260810_reviewed_edge_p90_8px_systematic_2px",
            "review provenance is missing")
    require(java, '"next_phase_allowed"', "Phase 3 gate result missing")

    version_name, version_code = parse_version_properties(version)
    if version_name < (1, 0, 9):
        raise SystemExit(f"Phase 2 self-eval check failed: versionName too old: {version_name}")
    if version_code < 46:
        raise SystemExit(f"Phase 2 self-eval check failed: versionCode too old: {version_code}")

    print("Phase 2 self-evaluation checks passed")


if __name__ == "__main__":
    main()
