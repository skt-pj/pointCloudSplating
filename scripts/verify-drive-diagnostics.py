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


def parse_version_properties(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def semver_tuple(value: str) -> tuple[int, int, int]:
    try:
        major, minor, patch = value.split(".", 2)
        return int(major), int(minor), int(patch)
    except Exception as error:
        raise SystemExit(f"Drive diagnostics check failed: invalid VERSION_NAME={value!r}") from error


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

    require(store, 'PREF_PRIMARY_DATASET_PATH = "primary_dataset_path"',
            "finalized dataset identity is not persisted")
    require(store, "public static void setPrimaryDataset(File dataset)",
            "evaluators cannot identify the finalized diagnostic dataset")
    require(store, "datasetSelection=last_finalized_preferred",
            "report does not expose finalized-first selection policy")
    require(store, 'file.getName().startsWith("dataset_")',
            "finalized dataset selection missing")
    require(store, 'new File(file, ".saved").isFile()',
            "finalized dataset marker is not required")
    require(store, "Comparator.comparing(File::getName).reversed()",
            "newest finalized dataset is not selected by capture timestamp name")
    require(store, "preferred.getName().compareTo(newest.getName()) >= 0",
            "an older persisted primary can still hide a newer finalized dataset")
    require(store, 'file.getName().startsWith("capture_tmp_")',
            "temporary diagnostic fallback missing")

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

    # Drive overwrite support first shipped in v1.0.4. Keep this verifier tied to the
    # capability floor, not one exact app version, so later Phase builds do not fail CI.
    properties = parse_version_properties(version)
    version_name = properties.get("VERSION_NAME", "")
    try:
        version_code = int(properties.get("VERSION_CODE", "0"))
    except ValueError as error:
        raise SystemExit("Drive diagnostics check failed: VERSION_CODE is not an integer") from error
    if semver_tuple(version_name) < (1, 0, 4) or version_code < 41:
        raise SystemExit(
            "Drive diagnostics check failed: build predates Drive overwrite support "
            f"VERSION_NAME={version_name} VERSION_CODE={version_code}"
        )

    print("Drive diagnostics overwrite checks passed")


if __name__ == "__main__":
    main()
