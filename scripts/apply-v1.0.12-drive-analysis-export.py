#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"v1.0.12 patch failed: missing needle in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    point_app = JAVA / "PointCloudApp.java"
    replace_once(
        point_app,
        "        DriveDiagnosticLogStore.initialize(appContext);\n        DiagnosticLog.initialize(appContext);",
        "        DriveDiagnosticLogStore.initialize(appContext);\n"
        "        DriveAnalysisExportStore.initialize(appContext);\n"
        "        DiagnosticLog.initialize(appContext);",
    )

    finalizer = JAVA / "DatasetFinalizer.java"
    replace_once(
        finalizer,
        "            File finalDirectory = renameAsSavedDataset(workingDirectory);\n"
        "            return Result.ok(finalDirectory, sourceFrames.size());",
        "            File finalDirectory = renameAsSavedDataset(workingDirectory);\n"
        "            DriveAnalysisExportStore.requestExport(finalDirectory);\n"
        "            return Result.ok(finalDirectory, sourceFrames.size());",
    )

    phase2 = JAVA / "Phase2EvaluationCoordinator.java"
    replace_once(
        phase2,
        "            DriveDiagnosticLogStore.requestOverwrite();\n"
        "        } catch (Throwable error) {",
        "            DriveDiagnosticLogStore.requestOverwrite();\n"
        "            DriveAnalysisExportStore.requestExport(dataset);\n"
        "        } catch (Throwable error) {",
    )

    job = JAVA / "GaussianSplatJob.java"
    replace_once(
        job,
        "        if (!trained.success) {\n"
        "            DiagnosticLog.w(TAG, \"3DGS continuation failed: \" + trained.message",
        "        if (!trained.success) {\n"
        "            DriveAnalysisExportStore.requestExport(datasetDirectory);\n"
        "            DiagnosticLog.w(TAG, \"3DGS continuation failed: \" + trained.message",
    )
    replace_once(
        job,
        "        Phase3DatasetEvaluator.Result phase3 = Phase3DatasetEvaluator.evaluate(datasetDirectory);\n"
        "        if (!phase3.machineGatePassed) {",
        "        Phase3DatasetEvaluator.Result phase3 = Phase3DatasetEvaluator.evaluate(datasetDirectory);\n"
        "        DriveAnalysisExportStore.requestExport(datasetDirectory);\n"
        "        if (!phase3.machineGatePassed) {",
    )
    replace_once(
        job,
        "            if (!trained.success) {\n"
        "                DiagnosticLog.w(TAG, \"Full 3DGS failed: \" + trained.message);",
        "            if (!trained.success) {\n"
        "                DriveAnalysisExportStore.requestExport(datasetDirectory);\n"
        "                DiagnosticLog.w(TAG, \"Full 3DGS failed: \" + trained.message);",
    )
    replace_once(
        job,
        "            Phase3DatasetEvaluator.Result phase3 = Phase3DatasetEvaluator.evaluate(datasetDirectory);\n"
        "            if (!phase3.machineGatePassed) {",
        "            Phase3DatasetEvaluator.Result phase3 = Phase3DatasetEvaluator.evaluate(datasetDirectory);\n"
        "            DriveAnalysisExportStore.requestExport(datasetDirectory);\n"
        "            if (!phase3.machineGatePassed) {",
    )
    replace_once(
        job,
        "        } catch (Exception e) {\n"
        "            ModelProcessingCoordinator.exit();\n"
        "            DiagnosticLog.e(TAG, \"Full 3DGS job failed\", e);",
        "        } catch (Exception e) {\n"
        "            ModelProcessingCoordinator.exit();\n"
        "            DriveAnalysisExportStore.requestExport(datasetDirectory);\n"
        "            DiagnosticLog.e(TAG, \"Full 3DGS job failed\", e);",
    )

    library = JAVA / "LibraryActivity.java"
    replace_once(
        library,
        "    private static final int REQUEST_SAVE_LOG = 1201;\n"
        "    private static final int MENU_SAVE_LOG = 1;\n"
        "    private static final int MENU_CHANGE_LOG_DESTINATION = 2;",
        "    private static final int REQUEST_SAVE_LOG = 1201;\n"
        "    private static final int REQUEST_SAVE_ANALYSIS = 1202;\n"
        "    private static final int MENU_SAVE_LOG = 1;\n"
        "    private static final int MENU_CHANGE_LOG_DESTINATION = 2;\n"
        "    private static final int MENU_EXPORT_ANALYSIS = 3;\n"
        "    private static final int MENU_CHANGE_ANALYSIS_DESTINATION = 4;",
    )
    replace_once(
        library,
        "        boolean configured=DriveDiagnosticLogStore.hasDestination(this);\n"
        "        popup.getMenu().add(0,MENU_SAVE_LOG,0,configured?\"Driveログを更新\":\"Driveログ保存先を設定\");\n"
        "        if(configured)popup.getMenu().add(0,MENU_CHANGE_LOG_DESTINATION,1,\"Driveログ保存先を変更\");",
        "        boolean configured=DriveDiagnosticLogStore.hasDestination(this);\n"
        "        popup.getMenu().add(0,MENU_SAVE_LOG,0,configured?\"Driveログを更新\":\"Driveログ保存先を設定\");\n"
        "        if(configured)popup.getMenu().add(0,MENU_CHANGE_LOG_DESTINATION,1,\"Driveログ保存先を変更\");\n"
        "        boolean analysisConfigured=DriveAnalysisExportStore.hasDestination(this);\n"
        "        popup.getMenu().add(0,MENU_EXPORT_ANALYSIS,2,analysisConfigured?\"解析データをDriveへ更新\":\"解析データのDrive保存先を設定\");\n"
        "        if(analysisConfigured)popup.getMenu().add(0,MENU_CHANGE_ANALYSIS_DESTINATION,3,\"解析データのDrive保存先を変更\");",
    )
    replace_once(
        library,
        "            if(item.getItemId()==MENU_SAVE_LOG){saveDiagnosticLog();return true;}\n"
        "            if(item.getItemId()==MENU_CHANGE_LOG_DESTINATION){chooseDriveDiagnosticDestination();return true;}\n"
        "            return false;",
        "            if(item.getItemId()==MENU_SAVE_LOG){saveDiagnosticLog();return true;}\n"
        "            if(item.getItemId()==MENU_CHANGE_LOG_DESTINATION){chooseDriveDiagnosticDestination();return true;}\n"
        "            if(item.getItemId()==MENU_EXPORT_ANALYSIS){saveAnalysisBundle();return true;}\n"
        "            if(item.getItemId()==MENU_CHANGE_ANALYSIS_DESTINATION){chooseDriveAnalysisDestination();return true;}\n"
        "            return false;",
    )
    replace_once(
        library,
        "    private void chooseDriveDiagnosticDestination(){\n"
        "        Toast.makeText(this,\"Google Driveで保存先を選択してください\",Toast.LENGTH_SHORT).show();\n"
        "        startActivityForResult(DriveDiagnosticLogStore.createDestinationIntent(),REQUEST_SAVE_LOG);\n"
        "    }\n\n"
        "    @Override protected void onActivityResult(int requestCode,int resultCode,Intent data){\n"
        "        super.onActivityResult(requestCode,resultCode,data);\n"
        "        if(requestCode!=REQUEST_SAVE_LOG)return;\n"
        "        if(resultCode!=RESULT_OK||data==null||data.getData()==null)return;\n"
        "        if(!DriveDiagnosticLogStore.registerDestination(this,data)){\n"
        "            Toast.makeText(this,\"Driveの保存先を設定できませんでした\",Toast.LENGTH_LONG).show();\n"
        "            return;\n"
        "        }\n"
        "        Toast.makeText(this,\"Driveログを設定しました。以後は同じファイルを上書きします\",Toast.LENGTH_SHORT).show();\n"
        "        DriveDiagnosticLogStore.overwriteNow((success,message)->runOnUiThread(()->\n"
        "                Toast.makeText(this,message,success?Toast.LENGTH_SHORT:Toast.LENGTH_LONG).show()));\n"
        "    }",
        "    private void chooseDriveDiagnosticDestination(){\n"
        "        Toast.makeText(this,\"Google Driveで保存先を選択してください\",Toast.LENGTH_SHORT).show();\n"
        "        startActivityForResult(DriveDiagnosticLogStore.createDestinationIntent(),REQUEST_SAVE_LOG);\n"
        "    }\n\n"
        "    private void saveAnalysisBundle(){\n"
        "        if(!DriveAnalysisExportStore.hasDestination(this)){chooseDriveAnalysisDestination();return;}\n"
        "        Toast.makeText(this,\"原画像・点群・処理結果をDrive保存先へ書き込んでいます…\",Toast.LENGTH_SHORT).show();\n"
        "        DriveAnalysisExportStore.exportLatestNow((success,message)->runOnUiThread(()->\n"
        "                Toast.makeText(this,message,success?Toast.LENGTH_SHORT:Toast.LENGTH_LONG).show()));\n"
        "    }\n\n"
        "    private void chooseDriveAnalysisDestination(){\n"
        "        Toast.makeText(this,\"Google Driveで解析ZIPの保存先を選択してください\",Toast.LENGTH_SHORT).show();\n"
        "        startActivityForResult(DriveAnalysisExportStore.createDestinationIntent(),REQUEST_SAVE_ANALYSIS);\n"
        "    }\n\n"
        "    @Override protected void onActivityResult(int requestCode,int resultCode,Intent data){\n"
        "        super.onActivityResult(requestCode,resultCode,data);\n"
        "        if(resultCode!=RESULT_OK||data==null||data.getData()==null)return;\n"
        "        if(requestCode==REQUEST_SAVE_LOG){\n"
        "            if(!DriveDiagnosticLogStore.registerDestination(this,data)){\n"
        "                Toast.makeText(this,\"Driveの保存先を設定できませんでした\",Toast.LENGTH_LONG).show();\n"
        "                return;\n"
        "            }\n"
        "            Toast.makeText(this,\"Driveログを設定しました。以後は同じファイルを上書きします\",Toast.LENGTH_SHORT).show();\n"
        "            DriveDiagnosticLogStore.overwriteNow((success,message)->runOnUiThread(()->\n"
        "                    Toast.makeText(this,message,success?Toast.LENGTH_SHORT:Toast.LENGTH_LONG).show()));\n"
        "            return;\n"
        "        }\n"
        "        if(requestCode==REQUEST_SAVE_ANALYSIS){\n"
        "            if(!DriveAnalysisExportStore.registerDestination(this,data)){\n"
        "                Toast.makeText(this,\"解析データのDrive保存先を設定できませんでした\",Toast.LENGTH_LONG).show();\n"
        "                return;\n"
        "            }\n"
        "            Toast.makeText(this,\"解析データ保存先を設定しました。現在のdatasetを書き出します\",Toast.LENGTH_SHORT).show();\n"
        "            DriveAnalysisExportStore.exportLatestNow((success,message)->runOnUiThread(()->\n"
        "                    Toast.makeText(this,message,success?Toast.LENGTH_SHORT:Toast.LENGTH_LONG).show()));\n"
        "        }\n"
        "    }",
    )

    print("v1.0.12 Drive analysis export wiring applied")


if __name__ == "__main__":
    main()
