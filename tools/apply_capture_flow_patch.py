from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing patch target: {label}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, repl: str, label: str) -> str:
    out, count = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"patch target {label}: expected 1 match, got {count}")
    return out


# ScannerActivity: visible Save/3DGS controls, explicit finalization, stable root anchor.
path = Path("app/src/main/java/com/sktpj/pointcloudsplatting/ScannerActivity.java")
s = path.read_text()
s = replace_once(s, "import com.google.ar.core.ArCoreApk;\n",
                 "import com.google.ar.core.Anchor;\nimport com.google.ar.core.ArCoreApk;\n",
                 "Anchor import")
s = replace_once(s, "import java.io.BufferedReader;\n",
                 "import java.io.BufferedReader;\nimport java.io.File;\n",
                 "File import")
s = replace_once(s,
                 "    private TextView statusView;\n    private Button menuButton;\n    private DisplayManager displayManager;\n",
                 "    private TextView statusView;\n    private Button menuButton;\n    private Button saveButton;\n    private Button gaussianButton;\n    private DisplayManager displayManager;\n",
                 "control fields")
s = replace_once(s,
                 "    private String lastStatus = \"\";\n    private String cameraConfigSummary = \"camera=?\";\n",
                 "    private String lastStatus = \"\";\n    private String cameraConfigSummary = \"camera=?\";\n    private volatile boolean captureFinalized;\n    private volatile boolean saveInProgress;\n    private volatile String finalizedDatasetPath;\n    private Anchor datasetRootAnchor;\n",
                 "capture state fields")
s = replace_once(s,
                 "        menuButton.setPadding(0, 0, 0, 0);\n        menuButton.setOnClickListener(v -> showMenu());\n\n        FrameLayout root = new FrameLayout(this);\n",
                 "        menuButton.setPadding(0, 0, 0, 0);\n        menuButton.setOnClickListener(v -> showMenu());\n\n        saveButton = new Button(this);\n        saveButton.setText(\"保存\");\n        saveButton.setTextColor(0xFFFFFFFF);\n        saveButton.setBackgroundColor(0xAA202020);\n        saveButton.setOnClickListener(v -> saveCurrentDataset());\n\n        gaussianButton = new Button(this);\n        gaussianButton.setText(\"3DGS化\");\n        gaussianButton.setTextColor(0xFFFFFFFF);\n        gaussianButton.setBackgroundColor(0xAA202020);\n        gaussianButton.setEnabled(false);\n        gaussianButton.setOnClickListener(v -> startGaussianSplatting());\n\n        FrameLayout root = new FrameLayout(this);\n",
                 "create controls")
s = replace_once(s,
                 "        root.addView(menuButton, menuParams);\n        setContentView(root);\n",
                 "        root.addView(menuButton, menuParams);\n\n        FrameLayout.LayoutParams saveParams = new FrameLayout.LayoutParams(dp(132), dp(52));\n        saveParams.gravity = Gravity.BOTTOM | Gravity.START;\n        saveParams.leftMargin = dp(12);\n        saveParams.bottomMargin = dp(16);\n        root.addView(saveButton, saveParams);\n\n        FrameLayout.LayoutParams gsParams = new FrameLayout.LayoutParams(dp(132), dp(52));\n        gsParams.gravity = Gravity.BOTTOM | Gravity.END;\n        gsParams.rightMargin = dp(12);\n        gsParams.bottomMargin = dp(16);\n        root.addView(gaussianButton, gsParams);\n        setContentView(root);\n",
                 "attach controls")
s = replace_once(s,
                 "                String path = datasetCaptureManager == null\n                        ? \"dataset unavailable\"\n                        : datasetCaptureManager.getCaptureDirectoryPath();\n",
                 "                String path = getCurrentDatasetPath();\n",
                 "menu dataset path")
insert_methods = r'''
    private String getCurrentDatasetPath() {
        if (finalizedDatasetPath != null) {
            return finalizedDatasetPath;
        }
        return datasetCaptureManager == null
                ? "dataset unavailable"
                : datasetCaptureManager.getCaptureDirectoryPath();
    }

    private void saveCurrentDataset() {
        DatasetCaptureManager manager = datasetCaptureManager;
        if (manager == null) {
            Toast.makeText(this, "保存対象がありません", Toast.LENGTH_SHORT).show();
            return;
        }
        if (captureFinalized) {
            Toast.makeText(this, "すでに保存済みです", Toast.LENGTH_SHORT).show();
            return;
        }
        if (saveInProgress) {
            return;
        }

        saveInProgress = true;
        saveButton.setEnabled(false);
        gaussianButton.setEnabled(false);
        setStatus("保存処理中: keyframe書き込みを確定しています...");

        new Thread(() -> {
            boolean flushed = manager.stopCaptureAndFlush(5_000L);
            if (!flushed) {
                manager.resumeCapture();
                runOnUiThread(() -> {
                    saveInProgress = false;
                    saveButton.setEnabled(true);
                    setStatus("保存できませんでした。撮影中のフレーム完了後にもう一度押してください。");
                });
                return;
            }

            DatasetFinalizer.Result result = DatasetFinalizer.finalizeDataset(
                    new File(manager.getCaptureDirectoryPath()));
            runOnUiThread(() -> {
                saveInProgress = false;
                if (result.success) {
                    captureFinalized = true;
                    finalizedDatasetPath = result.directory.getAbsolutePath();
                    saveButton.setText("保存済み");
                    saveButton.setEnabled(false);
                    gaussianButton.setEnabled(true);
                    setStatus("保存完了: " + result.frameCount + " keyframes\n"
                            + finalizedDatasetPath);
                    Toast.makeText(this, "データセットを保存しました", Toast.LENGTH_SHORT).show();
                } else {
                    manager.resumeCapture();
                    saveButton.setEnabled(true);
                    setStatus("保存失敗: " + result.message);
                }
            });
        }, "FinalizeDataset").start();
    }

    private void startGaussianSplatting() {
        if (!captureFinalized || finalizedDatasetPath == null) {
            Toast.makeText(this, "先に保存してください", Toast.LENGTH_SHORT).show();
            return;
        }

        gaussianButton.setEnabled(false);
        setStatus("3DGS入力を検証しています...");
        File datasetDirectory = new File(finalizedDatasetPath);
        new Thread(() -> {
            GaussianSplatJob.Result result = GaussianSplatJob.prepare(datasetDirectory);
            runOnUiThread(() -> {
                gaussianButton.setEnabled(true);
                if (result.success) {
                    setStatus("3DGS開始要求を作成: " + result.frameCount + " keyframes\n"
                            + "Native Vulkan trainerは次の実装段階です。");
                    Toast.makeText(this, "3DGS入力準備完了", Toast.LENGTH_SHORT).show();
                } else {
                    setStatus("3DGS開始不可: " + result.message);
                    Toast.makeText(this, result.message, Toast.LENGTH_LONG).show();
                }
            });
        }, "Prepare3DGS").start();
    }

'''
s = replace_once(s, "    private void copyLogsToClipboard() {\n",
                 insert_methods + "    private void copyLogsToClipboard() {\n",
                 "save and 3dgs methods")
s = replace_once(s,
                 "        String dataset = datasetCaptureManager == null\n                ? \"unavailable\" : datasetCaptureManager.getCaptureDirectoryPath();\n",
                 "        String dataset = getCurrentDatasetPath();\n",
                 "diagnostic dataset path")
s = replace_once(s,
                 "                .append(\"savedDatasetFrames=\").append(photos).append('\\n')\n",
                 "                .append(\"savedDatasetFrames=\").append(photos).append('\\n')\n                .append(\"captureFinalized=\").append(captureFinalized).append('\\n')\n",
                 "diagnostic finalized state")
s = replace_once(s,
                 "                still.set(CaptureRequest.JPEG_ORIENTATION, computeJpegOrientation());\n",
                 "                // Keep the encoded pixels in sensor/readout orientation so camera pose and\n"
                 "                // intrinsics have one deterministic convention for 3DGS input.\n"
                 "                still.set(CaptureRequest.JPEG_ORIENTATION, 0);\n",
                 "sensor-native jpeg")
s = replace_once(s,
                 "                if (datasetCaptureManager != null\n                        && datasetCaptureManager.onArFrame(frame, camera)) {\n                    requestHighResolutionStill();\n                }\n",
                 "                if (datasetRootAnchor == null && !captureFinalized && !saveInProgress) {\n"
                 "                    datasetRootAnchor = session.createAnchor(camera.getPose());\n"
                 "                    DiagnosticLog.i(TAG, \"Dataset root anchor created\");\n"
                 "                }\n"
                 "                com.google.ar.core.Pose datasetRootPose = datasetRootAnchor == null\n"
                 "                        ? camera.getPose() : datasetRootAnchor.getPose();\n\n"
                 "                if (datasetCaptureManager != null\n"
                 "                        && datasetCaptureManager.onArFrame(frame, camera, datasetRootPose)) {\n"
                 "                    requestHighResolutionStill();\n"
                 "                }\n",
                 "root anchored pose capture")
s = replace_once(s,
                 "                            datasetCaptureManager.onDepthFrame(depth);\n",
                 "                            datasetCaptureManager.onDepthFrame(depth, datasetRootPose);\n",
                 "root anchored depth")
s = replace_once(s,
                 "                setStatus(\n                        \"Raw depth: \" + pointCloudRenderer.getStoredFrameCount()\n                                + \" / saved sets: \" + photos\n                                + \"\\nphoto: \" + decision\n                                + \"\\n1/500 target / <=1/250 / ISO<=3200\"\n                                + \"\\n\" + cameraConfigSummary);\n",
                 "                String captureState = captureFinalized\n"
                 "                        ? \"saved\" : (saveInProgress ? \"saving\" : \"capturing\");\n"
                 "                setStatus(\n"
                 "                        \"Raw depth: \" + pointCloudRenderer.getStoredFrameCount()\n"
                 "                                + \" / keyframes: \" + photos + \" / \" + captureState\n"
                 "                                + \"\\nphoto: \" + decision\n"
                 "                                + \"\\n1/500 target / <=1/250 / ISO<=3200\"\n"
                 "                                + \"\\n\" + cameraConfigSummary);\n",
                 "status capture state")
s = replace_once(s,
                 "        if (session != null) {\n            session.close();\n            session = null;\n        }\n",
                 "        if (datasetRootAnchor != null) {\n"
                 "            datasetRootAnchor.detach();\n"
                 "            datasetRootAnchor = null;\n"
                 "        }\n"
                 "        if (session != null) {\n"
                 "            session.close();\n"
                 "            session = null;\n"
                 "        }\n",
                 "detach root anchor")
path.write_text(s)


# DatasetCaptureManager: keyframe capture into a temporary spool, stable root-anchor coordinates,
# and an explicit stop+flush used by the Save button.
path = Path("app/src/main/java/com/sktpj/pointcloudsplatting/DatasetCaptureManager.java")
s = path.read_text()
s = replace_once(s, "import java.util.concurrent.Executors;\n",
                 "import java.util.concurrent.Executors;\nimport java.util.concurrent.TimeUnit;\n",
                 "TimeUnit import")
s = replace_once(s,
                 "    private boolean captureInFlight;\n    private volatile int savedCount;\n",
                 "    private boolean captureInFlight;\n    private volatile boolean captureEnabled = true;\n    private volatile int savedCount;\n",
                 "capture enabled field")
s = replace_once(s,
                 "        captureRoot = new File(externalPicturesDir, \"dataset_\" + sessionName);\n",
                 "        captureRoot = new File(externalPicturesDir, \"capture_tmp_\" + sessionName);\n",
                 "temporary capture directory")
s = replace_once(s,
                 "    public synchronized void onDepthFrame(DepthData depth) {\n        try {\n            WorldPointCloudSnapshot snapshot = WorldPointCloudSnapshot.from(depth);\n",
                 "    public synchronized void onDepthFrame(DepthData depth, Pose rootPose) {\n        try {\n            WorldPointCloudSnapshot snapshot = WorldPointCloudSnapshot.from(depth, rootPose);\n",
                 "root depth signature")
s = replace_once(s,
                 "    public synchronized boolean onArFrame(Frame frame, Camera camera) {\n        long timestampNs = frame.getTimestamp();\n        Pose pose = camera.getPose();\n        Motion motion = measureMotion(pose, timestampNs);\n\n        poseSamples.addLast(PoseSample.from(timestampNs, camera));\n",
                 "    public synchronized boolean onArFrame(Frame frame, Camera camera, Pose rootPose) {\n"
                 "        long timestampNs = frame.getTimestamp();\n"
                 "        Pose pose = rootPose.inverse().compose(camera.getPose());\n"
                 "        Motion motion = measureMotion(pose, timestampNs);\n\n"
                 "        poseSamples.addLast(PoseSample.from(timestampNs, camera, rootPose));\n",
                 "root pose signature")
s = replace_once(s,
                 "        previousPose = pose;\n        previousFrameTimestampNs = timestampNs;\n        tryFinalizePendingLocked();\n\n        if (captureInFlight) {\n",
                 "        previousPose = pose;\n        previousFrameTimestampNs = timestampNs;\n        tryFinalizePendingLocked();\n\n"
                 "        if (!captureEnabled) {\n"
                 "            lastDecision = \"capture stopped; press 3DGS after Save\";\n"
                 "            return false;\n"
                 "        }\n"
                 "        if (captureInFlight) {\n",
                 "capture stop gate")
stop_flush = r'''
    /** Stops requesting new keyframes and waits for the current Camera2/JPEG + disk writes. */
    public boolean stopCaptureAndFlush(long timeoutMs) {
        long deadline = System.currentTimeMillis() + Math.max(1L, timeoutMs);
        synchronized (this) {
            captureEnabled = false;
            lastDecision = "capture stopped; finalizing";
        }

        while (true) {
            boolean settled;
            synchronized (this) {
                settled = !captureInFlight && pendingJpegs.isEmpty();
            }
            if (settled) {
                break;
            }
            if (System.currentTimeMillis() >= deadline) {
                synchronized (this) {
                    lastDecision = "save timed out waiting for Camera2 frame";
                }
                return false;
            }
            try {
                Thread.sleep(20L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }

        long remaining = Math.max(1L, deadline - System.currentTimeMillis());
        try {
            writer.submit(() -> {}).get(remaining, TimeUnit.MILLISECONDS);
            synchronized (this) {
                lastDecision = "capture stopped; ready to save";
            }
            return true;
        } catch (Exception e) {
            DiagnosticLog.w(TAG, "Timed out flushing dataset writer: " + e.getMessage());
            synchronized (this) {
                lastDecision = "save timed out flushing files";
            }
            return false;
        }
    }

    public synchronized void resumeCapture() {
        captureEnabled = true;
        lastDecision = "capture resumed";
    }

'''
s = replace_once(s, "    public void shutdown() {\n",
                 stop_flush + "    public void shutdown() {\n",
                 "stop and flush methods")
s = replace_once(s,
                 "        json.put(\"point_cloud_coordinate_system\", \"ARCore world coordinates\");\n",
                 "        json.put(\"point_cloud_coordinate_system\", \"ARCore root-anchor local coordinates\");\n",
                 "point cloud coordinate label")
s = replace_once(s,
                 "        static PoseSample from(long timestampNs, Camera camera) {\n            Pose pose = camera.getPose();\n",
                 "        static PoseSample from(long timestampNs, Camera camera, Pose rootPose) {\n            Pose pose = rootPose.inverse().compose(camera.getPose());\n",
                 "pose sample root")
s = replace_once(s,
                 "                    \"Each frame has .jpg + synchronized ARCore camera pose JSON + world-space Raw Depth .ply\");\n",
                 "                    \"Each frame has .jpg + synchronized root-anchor camera pose JSON + Raw Depth .ply\");\n",
                 "session dataset description")
path.write_text(s)


# WorldPointCloudSnapshot: point clouds use the same root-anchor local frame as camera poses.
path = Path("app/src/main/java/com/sktpj/pointcloudsplatting/WorldPointCloudSnapshot.java")
s = path.read_text()
s = replace_once(s, "package com.sktpj.pointcloudsplatting;\n\n",
                 "package com.sktpj.pointcloudsplatting;\n\nimport com.google.ar.core.Pose;\n\n",
                 "Pose import world cloud")
s = replace_once(s,
                 "/** Immutable copy of one Raw Depth frame transformed into the ARCore world coordinate system. */\n",
                 "/** Immutable copy of one Raw Depth frame transformed into the session root-anchor frame. */\n",
                 "world cloud comment")
s = replace_once(s,
                 "    public static WorldPointCloudSnapshot from(DepthData depth) {\n",
                 "    public static WorldPointCloudSnapshot from(DepthData depth, Pose rootPose) {\n",
                 "world cloud signature")
s = replace_once(s,
                 "        float[] model = new float[16];\n        depth.getModelMatrix(model);\n",
                 "        float[] model = new float[16];\n"
                 "        Pose rootFromDepthCamera = rootPose.inverse().compose(depth.getAnchor().getPose());\n"
                 "        rootFromDepthCamera.toMatrix(model, 0);\n",
                 "root relative depth matrix")
s = replace_once(s,
                 "                        + \"comment ARCore world coordinates; camera looks along local -Z\\n\"\n",
                 "                        + \"comment ARCore root-anchor local coordinates; camera looks along local -Z\\n\"\n",
                 "ply root comment")
path.write_text(s)


# Bump app version.
path = Path("version.properties")
s = path.read_text()
s = re.sub(r"VERSION_NAME=.*", "VERSION_NAME=0.4.3", s)
s = re.sub(r"VERSION_CODE=\\d+", "VERSION_CODE=7", s)
path.write_text(s)

print("capture flow patch applied")
