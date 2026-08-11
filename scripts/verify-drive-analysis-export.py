#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting"
STORE = JAVA / "DriveAnalysisExportStore.java"
LIBRARY = JAVA / "LibraryActivity.java"
APP = JAVA / "PointCloudApp.java"
FINALIZER = JAVA / "DatasetFinalizer.java"
PHASE2 = JAVA / "Phase2EvaluationCoordinator.java"
JOB = JAVA / "GaussianSplatJob.java"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"Drive analysis export check failed: {why}: missing {needle!r}")


def main() -> None:
    for path in (STORE, LIBRARY, APP, FINALIZER, PHASE2, JOB, VERSION):
        if not path.is_file():
            raise SystemExit(f"Drive analysis export check failed: missing {path}")

    store = STORE.read_text(encoding="utf-8")
    library = LIBRARY.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    phase2 = PHASE2.read_text(encoding="utf-8")
    job = JOB.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    require(store, 'FILE_NAME = "pointCloudSplating-analysis-current.zip"', "fixed analysis ZIP missing")
    require(store, 'Intent.ACTION_CREATE_DOCUMENT', "SAF destination setup missing")
    require(store, 'setType("application/zip")', "analysis file is not a ZIP")
    require(store, 'openOutputStream(uri, "wt")', "analysis bundle is not truncate-replaced")
    require(store, 'original_jpeg_bytes_preserved', "original JPEG preservation not declared")
    require(store, 'independent_depth_observations_preserved', "independent Depth preservation not declared")
    require(store, 'phase2_geometry_prior.ply', "Phase 2 fused geometry is not exported")
    require(store, 'splat.ply', "final splat is not exported")
    require(store, 'CHECKPOINT_NAME = "3dgs_checkpoint.bin"', "checkpoint omission is not explicit")
    require(store, 'sha256', "per-file hash manifest missing")
    require(store, 'analysis_manifest.json', "bundle manifest missing")
    require(store, 'cloud_upload_confirmation', "SAF cloud-sync limitation is not explicit")
    require(store, 'source changed during export:', "export does not reject a mutating source snapshot")

    require(library, '解析データをDriveへ更新', "manual analysis export menu missing")
    require(library, 'REQUEST_SAVE_ANALYSIS', "analysis destination request missing")
    require(library, 'DriveAnalysisExportStore.exportLatestNow', "manual export action missing")
    require(app, 'DriveAnalysisExportStore.initialize(appContext)', "analysis exporter not initialized")
    require(finalizer, 'DriveAnalysisExportStore.requestExport(finalDirectory)', "save-time source export hook missing")
    require(phase2, 'DriveAnalysisExportStore.requestExport(dataset)', "Phase 2 derived export hook missing")
    if job.count('DriveAnalysisExportStore.requestExport(datasetDirectory)') < 4:
        raise SystemExit("Drive analysis export check failed: Phase 3 success/failure export hooks incomplete")

    require(version, 'VERSION_NAME=1.0.12', "versionName mismatch")
    require(version, 'VERSION_CODE=49', "versionCode mismatch")
    print("Drive analysis export checks passed: originals + raw Depth + Phase2 + Phase3 in stable SAF ZIP")


if __name__ == "__main__":
    main()
