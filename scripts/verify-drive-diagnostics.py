#!/usr/bin/env python3
"""Build-time contract checks for persisted Google Drive diagnostic logging."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DriveDiagnosticLogStore.java"
LOG = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DiagnosticLog.java"
APP = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/PointCloudApp.java"
LIBRARY = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/LibraryActivity.java"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"Drive diagnostics check failed: {why}: missing {needle!r}")


def forbid(text: str, needle: str, why: str) -> None:
    if needle in text:
        raise SystemExit(f"Drive diagnostics check failed: {why}: found {needle!r}")


def main() -> None:
    store = STORE.read_text(encoding="utf-8")
    log = LOG.read_text(encoding="utf-8")
    app = APP.read_text(encoding="utf-8")
    library = LIBRARY.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    require(store, 'FILE_NAME = "pointCloudSplating-current-diagnostics.txt"',
            "remote file must have one stable name")
    require(store, "Intent.ACTION_CREATE_DOCUMENT", "SAF destination setup missing")
    require(store, "takePersistableUriPermission", "destination permission is not persisted")
    require(store, 'openOutputStream(uri, "wt")',
            "Drive write must explicitly truncate instead of append")
    require(store, "DiagnosticLog.currentProcessSnapshot()",
            "current-process diagnostic log missing")
    require(store, '"phase1_evaluation.json"', "Phase 1 evaluation is not exported")
    require(store, '"phase2_evaluation.json"', "Phase 2 evaluation is not exported")
    require(store, '"phase3_evaluation.json"', "Phase 3 evaluation slot is not exported")
    require(store, "AUTO_SYNC_DELAY_SECONDS", "automatic remote refresh is missing")
    forbid(store, 'openOutputStream(uri, "wa")', "append mode is forbidden")

    # v1.0.4 incorrectly let a newer capture_tmp directory hide the last finalized dataset.
    require(store, 'PREF_PRIMARY_DATASET_PATH = "primary_dataset_path"',
            "finalized dataset identity is not persisted")
    require(store, "public static void setPrimaryDataset(File dataset)",
            "evaluators cannot select the finalized diagnostic dataset")
    require(store, "datasetSelection=last_finalized_preferred",
            "report does not expose finalized-first selection policy")
    require(store, 'file.getName().startsWith("dataset_")',
            "finalized dataset selection missing")
    require(store, 'new File(file, ".saved").isFile()',
            "finalized dataset marker is not required")

    require(log, "public static synchronized String currentProcessSnapshot()",
            "current-process log view missing")
    require(log, "DriveDiagnosticLogStore.requestOverwrite();",
            "new diagnostic lines do not schedule Drive refresh")
    require(app, "DriveDiagnosticLogStore.initialize(appContext);",
            "Drive diagnostics are not initialized at process start")

    require(library, '"Driveログ保存先を設定"', "first-time Drive setup menu missing")
    require(library, '"Driveログを更新"', "manual Drive refresh menu missing")
    require(library, "DriveDiagnosticLogStore.registerDestination(this,data)",
            "library does not persist the selected Drive document")
    forbid(library, "pointCloudSplating-diagnostics-",
            "timestamped diagnostic copies must not be created")

    require(version, "VERSION_NAME=1.0.5", "versionName must identify Phase 2 build")
    require(version, "VERSION_CODE=42", "versionCode must identify Phase 2 build")

    print("Drive diagnostics overwrite checks passed")


if __name__ == "__main__":
    main()
