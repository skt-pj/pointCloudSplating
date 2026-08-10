#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/Phase2DatasetEvaluator.java"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"Phase 2 self-eval check failed: {why}: missing {needle!r}")


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
    require(java, "DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX = null",
            "absolute edge threshold must not be invented before evidence review")
    require(version, "VERSION_NAME=1.0.8", "versionName mismatch")
    require(version, "VERSION_CODE=45", "versionCode mismatch")

    print("Phase 2 self-evaluation checks passed")


if __name__ == "__main__":
    main()
