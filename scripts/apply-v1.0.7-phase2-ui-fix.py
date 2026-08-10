#!/usr/bin/env python3
"""Apply the v1.0.7 Phase 2 evaluator classification and user-facing UI correction.

This patch is intentionally idempotent. CI applies it before verification/build, then persists the
patched Java sources back to the agent branch so the checked-in source matches the APK.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"v1.0.7 patch failed: {label}: source pattern not found in {path}")
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def replace_all(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise SystemExit(f"v1.0.7 patch failed: {label}: source pattern not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_phase2() -> None:
    path = JAVA / "Phase2DatasetEvaluator.java"
    replace_once(
        path,
        "        int depthUsedCount;\n        int depthExcludedMetadataCount;\n",
        "        int depthUsedCount;\n        int depthExcludedMetadataCount;\n        int depthExcludedQualityCount;\n",
        "Depth quality exclusion counter",
    )
    replace_once(
        path,
        "                } catch (Exception error) {\n"
        "                    depthExcludedMetadataCount++;\n"
        "                    fail(\"depth_ply_read_failed\", ply.getName() + \": \" + error.getMessage());\n"
        "                }\n",
        "                } catch (Exception error) {\n"
        "                    String message = String.valueOf(error.getMessage());\n"
        "                    if (\"no confidence-filtered sample points\".equals(message)) {\n"
        "                        // The observation is structurally valid (Phase 1 already proved that),\n"
        "                        // but contributes no geometry after the Phase 2 quality filter. Exclude\n"
        "                        // it with an explicit reason; do not misclassify normal low-confidence\n"
        "                        // sensor output as a corrupt PLY or a dataset hard failure.\n"
        "                        depthExcludedQualityCount++;\n"
        "                        warn(\"depth_observation_no_usable_points\",\n"
        "                                ply.getName() + \": all points rejected by Phase 2 confidence filter\");\n"
        "                    } else {\n"
        "                        depthExcludedMetadataCount++;\n"
        "                        fail(\"depth_ply_read_failed\", ply.getName() + \": \" + message);\n"
        "                    }\n"
        "                }\n",
        "Depth low-confidence classification",
    )
    replace_once(
        path,
        "                fusionJson.put(\"depth_observation_excluded_metadata_count\", depthExcludedMetadataCount);\n",
        "                fusionJson.put(\"depth_observation_excluded_metadata_count\", depthExcludedMetadataCount);\n"
        "                fusionJson.put(\"depth_observation_excluded_quality_count\", depthExcludedQualityCount);\n",
        "Depth quality exclusion JSON",
    )


def patch_processing_gate() -> None:
    path = JAVA / "ModelProcessingCoordinator.java"
    replace_once(
        path,
        "    static boolean isActive() {\n        return active;\n    }\n",
        "    static boolean isPhase3ProcessingEnabled() {\n"
        "        return PHASE3_PROCESSING_ENABLED;\n"
        "    }\n\n"
        "    static boolean isActive() {\n        return active;\n    }\n",
        "Expose Phase 3 UI availability",
    )


def patch_scanner_ui() -> None:
    path = JAVA / "ScannerActivity.java"
    replace_once(
        path,
        "        gaussianButton = new Button(this);\n"
        "        gaussianButton.setText(\"3Dモデルを作成\");\n"
        "        styleButton(gaussianButton, 0xFF0B57D0, 15f);\n"
        "        gaussianButton.setContentDescription(\"撮影した写真から3Dモデルを作成する\");\n"
        "        gaussianButton.setOnClickListener(v -> startGaussianSplatting());\n",
        "        gaussianButton = new Button(this);\n"
        "        boolean phase3Enabled = ModelProcessingCoordinator.isPhase3ProcessingEnabled();\n"
        "        gaussianButton.setText(phase3Enabled ? \"3Dモデルを作成\" : \"3Dモデル作成 準備中\");\n"
        "        styleButton(gaussianButton, phase3Enabled ? 0xFF0B57D0 : 0xFF3C4043, 15f);\n"
        "        gaussianButton.setContentDescription(phase3Enabled\n"
        "                ? \"撮影した写真から3Dモデルを作成する\"\n"
        "                : \"3Dモデル作成は現在準備中\");\n"
        "        gaussianButton.setEnabled(phase3Enabled);\n"
        "        gaussianButton.setOnClickListener(v -> startGaussianSplatting());\n",
        "Scanner Phase 2 button state",
    )
    replace_all(
        path,
        "gaussianButton.setEnabled(true);",
        "gaussianButton.setEnabled(ModelProcessingCoordinator.isPhase3ProcessingEnabled());",
        "Do not re-enable Phase 3 button",
    )
    replace_once(
        path,
        "                    showState(\"撮影を保存しました\",\n"
        "                            result.frameCount + \"枚の写真を保存しました。3Dプレビューを作成できます。\");\n",
        "                    showState(\"撮影を保存しました\",\n"
        "                            result.frameCount + \"枚の写真を保存しました。\");\n",
        "Save-success wording",
    )
    replace_once(
        path,
        "    private void startGaussianSplatting() {\n        if (saveInProgress) {\n",
        "    private void startGaussianSplatting() {\n"
        "        if (!ModelProcessingCoordinator.isPhase3ProcessingEnabled()) {\n"
        "            runGaussianAfterSave = false;\n"
        "            showState(\"3Dモデル作成は準備中です\",\n"
        "                    captureFinalized\n"
        "                            ? \"撮影データは保存済みです。3Dモデル作成機能は現在準備中です。\"\n"
        "                            : \"撮影は続けられます。3Dモデル作成機能は現在準備中です。\");\n"
        "            return;\n"
        "        }\n"
        "        if (saveInProgress) {\n",
        "Scanner Phase 3 early UI gate",
    )


def patch_job_wording() -> None:
    path = JAVA / "GaussianSplatJob.java"
    replace_all(
        path,
        "変換用画面を開始できなかったため、安全のため3D処理を開始しませんでした。",
        "3Dモデル作成機能は現在準備中です。撮影データは保存されています。",
        "Remove meaningless safety wording from initial training",
    )
    replace_all(
        path,
        "変換用画面を開始できなかったため、安全のため追加学習を開始しませんでした。",
        "3Dモデルの追加学習は現在利用できません。保存済みモデルは変更されていません。",
        "Remove meaningless safety wording from continuation",
    )


def patch_library_ui() -> None:
    path = JAVA / "LibraryActivity.java"
    replace_once(
        path,
        "        if(isPhotometricComplete(dataset)){\n"
        "            Button more=new Button(this);",
        "        if(isPhotometricComplete(dataset)&&ModelProcessingCoordinator.isPhase3ProcessingEnabled()){\n"
        "            Button more=new Button(this);",
        "Hide unavailable additional training action",
    )
    replace_once(
        path,
        "    private void openOrTrain(File dataset,TextView status){migrateLegacyArtifacts(dataset);if(isPhotometricComplete(dataset)){openViewer(dataset);return;}showTrainingOptions(dataset,status,false);}\n",
        "    private void openOrTrain(File dataset,TextView status){migrateLegacyArtifacts(dataset);if(isPhotometricComplete(dataset)){openViewer(dataset);return;}if(!ModelProcessingCoordinator.isPhase3ProcessingEnabled()){Toast.makeText(this,\"撮影データは保存済みです。3Dモデル作成機能は現在準備中です。\",Toast.LENGTH_LONG).show();return;}showTrainingOptions(dataset,status,false);}\n",
        "Library Phase 3 gate",
    )
    replace_once(
        path,
        "if(isHqPreview(dataset))return photos+\"\\n旧プレビューあり・タップして学習\";if(new File(dataset,DEPTH_PRIOR).isFile())return photos+\"\\nタップして3Dモデルを作成\";return photos+\"\\nタップして3Dモデルを作成\";}",
        "if(!ModelProcessingCoordinator.isPhase3ProcessingEnabled())return photos+\"\\n撮影データ保存済み\";if(isHqPreview(dataset))return photos+\"\\n旧プレビューあり・タップして学習\";if(new File(dataset,DEPTH_PRIOR).isFile())return photos+\"\\nタップして3Dモデルを作成\";return photos+\"\\nタップして3Dモデルを作成\";}",
        "Library status wording",
    )


def main() -> None:
    patch_phase2()
    patch_processing_gate()
    patch_scanner_ui()
    patch_job_wording()
    patch_library_ui()
    print("v1.0.7 Phase 2/UI correction applied")


if __name__ == "__main__":
    main()
