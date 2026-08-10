#!/usr/bin/env python3
"""Activate Phase 3 after the reviewed Pixel 10a Phase 2 baseline.

The patch is intentionally idempotent because CI runs the historical v1.0.7/v1.0.8 patch steps
first. Phase 3 uses the Phase 2 fused geometry as its only geometry source, trains progressively
(low full-frame -> medium full-frame -> native-resolution principal-point patch), preserves the
resumable Vulkan optimizer checkpoint between stages, and emits hold-out metrics/renders.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting"
CPP = ROOT / "app/src/main/cpp/native_3dgs.cpp"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"v1.0.9 patch failed: {label}: source pattern not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_all(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            return
        raise SystemExit(f"v1.0.9 patch failed: {label}: source pattern not found in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_phase2_pass_policy() -> None:
    path = JAVA / "Phase2DatasetEvaluator.java"
    replace_once(
        path,
        " * The RGB/depth-edge absolute pixel threshold remains intentionally unfrozen until Pixel 10a evidence\n"
        " * is reviewed. v1.0.8 adds signed and quadrant edge offsets to the Drive JSON so that review can be\n"
        " * performed remotely without asking the user to export overlay image files manually.\n",
        " * v1.0.8 measured the first Pixel 10a baseline and exposed signed/quadrant offsets in Drive.\n"
        " * That reviewed run had five-view median edge errors 4.47-6.00 px and zero cross-view systematic\n"
        " * offset, so v1.0.9 freezes a conservative 8 px aggregate edge-p90 gate plus a 2 px systematic\n"
        " * offset gate before Phase 3 may start.\n",
        "Phase 2 reviewed policy comment",
    )
    replace_once(
        path,
        "    // Intentionally unset for the first Pixel 10a Phase 2 run.\n"
        "    private static final Double DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX = null;\n",
        "    // Frozen after reviewing the 2026-08-10 Pixel 10a baseline in v1.0.8.\n"
        "    private static final Double DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX = 8.0;\n"
        "    private static final double MAX_SYSTEMATIC_OFFSET_PX = 2.0;\n",
        "Phase 2 reviewed thresholds",
    )
    replace_once(
        path,
        "            if (DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX != null\n"
        "                    && Double.isFinite(edgeP90Px)\n"
        "                    && edgeP90Px > DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX) {\n",
        "            if (!Double.isFinite(systematicOffsetMagnitudePx)) {\n"
        "                fail(\"depth_edge_systematic_offset_missing\",\n"
        "                        \"signed cross-view alignment metric is unavailable\");\n"
        "            } else if (systematicOffsetMagnitudePx > MAX_SYSTEMATIC_OFFSET_PX) {\n"
        "                fail(\"depth_edge_systematic_offset\",\n"
        "                        \"systematic offset=\" + systematicOffsetMagnitudePx\n"
        "                                + \"px > \" + MAX_SYSTEMATIC_OFFSET_PX + \"px\");\n"
        "            }\n\n"
        "            if (DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX != null\n"
        "                    && Double.isFinite(edgeP90Px)\n"
        "                    && edgeP90Px > DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX) {\n",
        "Phase 2 systematic alignment gate",
    )
    replace_once(
        path,
        "                systematic.put(\"review_source\", \"machine_metrics_in_drive_log_no_manual_file_export\");\n",
        "                systematic.put(\"review_source\", \"machine_metrics_in_drive_log_no_manual_file_export\");\n"
        "                systematic.put(\"hard_threshold_px\", MAX_SYSTEMATIC_OFFSET_PX);\n",
        "Phase 2 systematic threshold report",
    )
    replace_once(
        path,
        "                                ? \"FIRST_PIXEL10A_BASELINE_REQUIRED_DO_NOT_PASS_PHASE2_YET\"\n"
        "                                : \"fixed_from_prior_pixel10a_measurement\");\n",
        "                                ? \"FIRST_PIXEL10A_BASELINE_REQUIRED_DO_NOT_PASS_PHASE2_YET\"\n"
        "                                : \"pixel10a_baseline_20260810_reviewed_edge_p90_8px_systematic_2px\");\n",
        "Phase 2 threshold policy report",
    )


def patch_exporter() -> None:
    path = JAVA / "ColmapDatasetExporter.java"
    replace_once(
        path,
        "import android.graphics.BitmapFactory;\n",
        "import android.graphics.BitmapFactory;\nimport android.graphics.BitmapRegionDecoder;\nimport android.graphics.Rect;\n",
        "Bitmap region imports",
    )
    replace_once(
        path,
        "    private static final int MAX_TRAIN_LONG_EDGE = 1000;\n",
        "    private static final int MAX_TRAIN_LONG_EDGE = 1000;\n"
        "    private static final int PHASE3_LOW_LONG_EDGE = 720;\n"
        "    private static final int PHASE3_HIGH_PATCH_WIDTH = 1280;\n"
        "    private static final int PHASE3_HIGH_PATCH_HEIGHT = 960;\n"
        "    private static final String PHASE3_STATE_FILE = \"phase3_training_state.json\";\n",
        "progressive export constants",
    )
    replace_once(
        path,
        "            File root = new File(dataset, \"vksplat_data\");\n"
        "            File imageDir = new File(root, \"images_4\");\n",
        "            int phase3Stage = readPhase3Stage(dataset);\n"
        "            File root = new File(dataset, \"vksplat_data\");\n"
        "            File imageDir = new File(root, \"images_4\");\n",
        "read progressive stage",
    )
    replace_once(
        path,
        "                    int sourceW = frame.getInt(\"w\");\n"
        "                    int sourceH = frame.getInt(\"h\");\n"
        "                    double trainingScale = Math.min(\n"
        "                            1.0,\n"
        "                            (double) MAX_TRAIN_LONG_EDGE / Math.max(sourceW, sourceH));\n"
        "                    int targetW = Math.max(1, (int) Math.round(sourceW * trainingScale));\n"
        "                    int targetH = Math.max(1, (int) Math.round(sourceH * trainingScale));\n"
        "                    File target = new File(imageDir, source.getName());\n"
        "                    resizeJpeg(source, target, targetW, targetH);\n\n"
        "                    double sx = (double) targetW / sourceW;\n"
        "                    double sy = (double) targetH / sourceH;\n"
        "                    double fx = frame.getDouble(\"fl_x\") * sx;\n"
        "                    double fy = frame.getDouble(\"fl_y\") * sy;\n"
        "                    double cx = frame.getDouble(\"cx\") * sx;\n"
        "                    double cy = frame.getDouble(\"cy\") * sy;\n",
        "                    int sourceW = frame.getInt(\"w\");\n"
        "                    int sourceH = frame.getInt(\"h\");\n"
        "                    File target = new File(imageDir, source.getName());\n"
        "                    int targetW;\n"
        "                    int targetH;\n"
        "                    double fx;\n"
        "                    double fy;\n"
        "                    double cx;\n"
        "                    double cy;\n"
        "                    if (phase3Stage >= 3) {\n"
        "                        targetW = Math.min(sourceW, PHASE3_HIGH_PATCH_WIDTH);\n"
        "                        targetH = Math.min(sourceH, PHASE3_HIGH_PATCH_HEIGHT);\n"
        "                        int left = clampInt(\n"
        "                                (int) Math.round(frame.getDouble(\"cx\") - targetW * 0.5),\n"
        "                                0, Math.max(0, sourceW - targetW));\n"
        "                        int top = clampInt(\n"
        "                                (int) Math.round(frame.getDouble(\"cy\") - targetH * 0.5),\n"
        "                                0, Math.max(0, sourceH - targetH));\n"
        "                        cropJpeg(source, target, new Rect(left, top, left + targetW, top + targetH));\n"
        "                        fx = frame.getDouble(\"fl_x\");\n"
        "                        fy = frame.getDouble(\"fl_y\");\n"
        "                        cx = frame.getDouble(\"cx\") - left;\n"
        "                        cy = frame.getDouble(\"cy\") - top;\n"
        "                    } else {\n"
        "                        int longEdge = phase3Stage == 1 ? PHASE3_LOW_LONG_EDGE : MAX_TRAIN_LONG_EDGE;\n"
        "                        double trainingScale = Math.min(\n"
        "                                1.0, (double) longEdge / Math.max(sourceW, sourceH));\n"
        "                        targetW = Math.max(1, (int) Math.round(sourceW * trainingScale));\n"
        "                        targetH = Math.max(1, (int) Math.round(sourceH * trainingScale));\n"
        "                        resizeJpeg(source, target, targetW, targetH);\n"
        "                        double sx = (double) targetW / sourceW;\n"
        "                        double sy = (double) targetH / sourceH;\n"
        "                        fx = frame.getDouble(\"fl_x\") * sx;\n"
        "                        fy = frame.getDouble(\"fl_y\") * sy;\n"
        "                        cx = frame.getDouble(\"cx\") * sx;\n"
        "                        cy = frame.getDouble(\"cy\") * sy;\n"
        "                    }\n",
        "progressive image preparation",
    )
    replace_once(
        path,
        "            File[] depthFiles = dataset.listFiles((dir, name) ->\n"
        "                    name.endsWith(\".ply\")\n"
        "                            && (name.startsWith(\"frame_\") || name.startsWith(\"depth_prior_\")));\n"
        "            if (depthFiles == null) depthFiles = new File[0];\n"
        "            Arrays.sort(depthFiles, Comparator.comparing(File::getName));\n"
        "            DiagnosticLog.i(TAG, \"Depth prior source files=\" + depthFiles.length);\n",
        "            File phase2Geometry = new File(dataset, \"phase2_geometry_prior.ply\");\n"
        "            if (!Phase2DatasetEvaluator.hasStoredPass(dataset)\n"
        "                    || !phase2Geometry.isFile() || phase2Geometry.length() <= 0L) {\n"
        "                return Result.fail(\"Phase 2の3D形状検証が完了していません\");\n"
        "            }\n"
        "            File[] depthFiles = {phase2Geometry};\n"
        "            DiagnosticLog.i(TAG, \"Phase 3 geometry source=phase2_geometry_prior.ply\");\n",
        "Phase 2 geometry source of truth",
    )
    replace_once(
        path,
        "            meta.put(\"training_image_policy\",\n"
        "                    \"preserve aspect ratio; max long edge \" + MAX_TRAIN_LONG_EDGE + \" px; never upscale\");\n",
        "            meta.put(\"phase3_stage\", phase3Stage);\n"
        "            meta.put(\"geometry_source\", \"phase2_geometry_prior.ply\");\n"
        "            meta.put(\"training_image_policy\", phase3Stage >= 3\n"
        "                    ? \"native-resolution principal-point crop up to 1280x960; never upscale\"\n"
        "                    : \"preserve aspect ratio; max long edge \"\n"
        "                            + (phase3Stage == 1 ? PHASE3_LOW_LONG_EDGE : MAX_TRAIN_LONG_EDGE)\n"
        "                            + \" px; never upscale\");\n",
        "progressive export metadata",
    )
    replace_once(
        path,
        "    private static void resizeJpeg(File source, File target, int width, int height) throws IOException {\n",
        "    private static int readPhase3Stage(File dataset) {\n"
        "        File state = new File(dataset, PHASE3_STATE_FILE);\n"
        "        if (!state.isFile()) return 0;\n"
        "        try {\n"
        "            return new JSONObject(readText(state)).optInt(\"current_stage\", 0);\n"
        "        } catch (Exception ignored) {\n"
        "            return 0;\n"
        "        }\n"
        "    }\n\n"
        "    private static int clampInt(int value, int min, int max) {\n"
        "        return Math.max(min, Math.min(max, value));\n"
        "    }\n\n"
        "    private static void cropJpeg(File source, File target, Rect region) throws IOException {\n"
        "        BitmapRegionDecoder decoder = null;\n"
        "        Bitmap cropped = null;\n"
        "        try {\n"
        "            decoder = BitmapRegionDecoder.newInstance(source.getAbsolutePath(), false);\n"
        "            BitmapFactory.Options options = new BitmapFactory.Options();\n"
        "            options.inPreferredConfig = Bitmap.Config.ARGB_8888;\n"
        "            options.inScaled = false;\n"
        "            cropped = decoder.decodeRegion(region, options);\n"
        "            if (cropped == null) throw new IOException(\"region decode failed \" + source.getName());\n"
        "            try (FileOutputStream output = new FileOutputStream(target)) {\n"
        "                if (!cropped.compress(Bitmap.CompressFormat.JPEG, 96, output)) {\n"
        "                    throw new IOException(\"crop JPEG compression failed \" + source.getName());\n"
        "                }\n"
        "            }\n"
        "        } finally {\n"
        "            if (cropped != null) cropped.recycle();\n"
        "            if (decoder != null) decoder.recycle();\n"
        "        }\n"
        "    }\n\n"
        "    private static void resizeJpeg(File source, File target, int width, int height) throws IOException {\n",
        "progressive image helper methods",
    )


def patch_native_java() -> None:
    path = JAVA / "NativeGaussianTrainer.java"
    replace_once(
        path,
        "    private static final String CHECKPOINT_FILE = \"3dgs_checkpoint.bin\";\n",
        "    private static final String CHECKPOINT_FILE = \"3dgs_checkpoint.bin\";\n"
        "    private static final String PHASE3_STATE_FILE = \"phase3_training_state.json\";\n"
        "    private static final String PHASE3_PROFILE = \"progressive_low_mid_high_patch\";\n",
        "Phase 3 training constants",
    )
    replace_once(
        path,
        "        public final double validationPsnr;\n        public final double peakVramMb;\n",
        "        public final double validationPsnr;\n"
        "        public final double validationSsim;\n"
        "        public final int validationViewCount;\n"
        "        public final double peakVramMb;\n",
        "validation result fields",
    )
    replace_once(
        path,
        "                int steps, int addedSteps, boolean resumed, double validationPsnr,\n"
        "                double peakVramMb, String device) {\n",
        "                int steps, int addedSteps, boolean resumed, double validationPsnr,\n"
        "                double validationSsim, int validationViewCount,\n"
        "                double peakVramMb, String device) {\n",
        "validation result constructor args",
    )
    replace_once(
        path,
        "            this.validationPsnr = validationPsnr;\n            this.peakVramMb = peakVramMb;\n",
        "            this.validationPsnr = validationPsnr;\n"
        "            this.validationSsim = validationSsim;\n"
        "            this.validationViewCount = validationViewCount;\n"
        "            this.peakVramMb = peakVramMb;\n",
        "validation result assignment",
    )
    replace_once(
        path,
        "            return new Result(false, message, null, 0, 0, 0, false,\n"
        "                    Double.NaN, Double.NaN, \"\");\n",
        "            return new Result(false, message, null, 0, 0, 0, false,\n"
        "                    Double.NaN, Double.NaN, 0, Double.NaN, \"\");\n",
        "validation fail defaults",
    )
    replace_once(
        path,
        "    /** Runs exactly requestedSteps more optimizer iterations, resuming a checkpoint when present. */\n",
        "    /** Initial Phase 3 run: low full-frame -> medium full-frame -> high-resolution patch. */\n"
        "    public static Result trainProgressiveInitial(Context context, File dataset,\n"
        "            ProgressListener listener) {\n"
        "        int frames = readFrameCountForDefault(dataset);\n"
        "        return trainProgressiveInitial(context, dataset, defaultTrainingSteps(frames), listener);\n"
        "    }\n\n"
        "    public static Result trainProgressiveInitial(Context context, File dataset, int requestedSteps,\n"
        "            ProgressListener listener) {\n"
        "        if (requestedSteps < 3) return Result.fail(\"初回3D学習は3 step以上を指定してください\");\n"
        "        if (context == null || dataset == null || !dataset.isDirectory()) {\n"
        "            return Result.fail(\"撮影データが見つかりませんでした\");\n"
        "        }\n"
        "        int low = Math.max(1, requestedSteps * 30 / 100);\n"
        "        int mid = Math.max(1, requestedSteps * 40 / 100);\n"
        "        int high = requestedSteps - low - mid;\n"
        "        if (high < 1) { high = 1; if (mid > low) mid--; else low--; }\n"
        "        int[] stageSteps = {low, mid, high};\n"
        "        JSONObject state = readPhase3State(dataset);\n"
        "        boolean compatible = state != null\n"
        "                && PHASE3_PROFILE.equals(state.optString(\"profile\", \"\"))\n"
        "                && state.optInt(\"requested_steps\", -1) == requestedSteps;\n"
        "        if (!compatible) {\n"
        "            resetInitialPhase3State(dataset);\n"
        "            state = new JSONObject();\n"
        "            try {\n"
        "                state.put(\"profile\", PHASE3_PROFILE);\n"
        "                state.put(\"requested_steps\", requestedSteps);\n"
        "                state.put(\"completed_stage\", 0);\n"
        "            } catch (Exception ignored) {}\n"
        "            writePhase3State(dataset, state);\n"
        "        }\n"
        "        int completedStage = Math.max(0, Math.min(3, state.optInt(\"completed_stage\", 0)));\n"
        "        Result last = null;\n"
        "        for (int stage = completedStage + 1; stage <= 3; stage++) {\n"
        "            try {\n"
        "                state.put(\"current_stage\", stage);\n"
        "                state.put(\"stage_steps\", stageSteps[stage - 1]);\n"
        "                writePhase3State(dataset, state);\n"
        "            } catch (Exception error) {\n"
        "                return Result.fail(\"3D学習状態を保存できませんでした\");\n"
        "            }\n"
        "            final int currentStage = stage;\n"
        "            final int progressStart = stage == 1 ? 0 : (stage == 2 ? 30 : 70);\n"
        "            final int progressSpan = stage == 1 ? 30 : (stage == 2 ? 40 : 30);\n"
        "            last = train(context, dataset, stageSteps[stage - 1], (percent, message) -> {\n"
        "                int mapped = progressStart + Math.round(progressSpan * percent / 100f);\n"
        "                notifyProgress(listener, mapped,\n"
        "                        \"Phase 3 \" + currentStage + \"/3: \" + message);\n"
        "            });\n"
        "            if (!last.success) return last;\n"
        "            try {\n"
        "                state.put(\"completed_stage\", stage);\n"
        "                writePhase3State(dataset, state);\n"
        "            } catch (Exception error) {\n"
        "                return Result.fail(\"3D学習状態を更新できませんでした\");\n"
        "            }\n"
        "        }\n"
        "        return last == null ? Result.fail(\"3D学習を開始できませんでした\") : last;\n"
        "    }\n\n"
        "    /** Runs exactly requestedSteps more optimizer iterations, resuming a checkpoint when present. */\n",
        "progressive initial method",
    )
    replace_once(
        path,
        "            double psnr = json.optDouble(\"validation_psnr\", Double.NaN);\n"
        "            double peakVramMb = json.optDouble(\"peak_vram_mb\", Double.NaN);\n",
        "            double psnr = json.optDouble(\"validation_psnr\", Double.NaN);\n"
        "            double ssim = json.optDouble(\"validation_ssim\", Double.NaN);\n"
        "            int validationViews = json.optInt(\"validation_view_count\", 0);\n"
        "            double peakVramMb = json.optDouble(\"peak_vram_mb\", Double.NaN);\n",
        "validation JSON parsing",
    )
    replace_once(
        path,
        "            writeFinalResult(dataset, gaussians, completedSteps, addedSteps, resumed, psnr,\n"
        "                    peakVramMb, device, gaussianBudget, strategy, initialization, prepared);\n",
        "            writeFinalResult(dataset, gaussians, completedSteps, addedSteps, resumed, psnr,\n"
        "                    ssim, validationViews, peakVramMb, device, gaussianBudget, strategy,\n"
        "                    initialization, prepared);\n",
        "final result validation metrics",
    )
    replace_once(
        path,
        "                    + \" validationPsnr=\" + psnr + \" peakVramMb=\" + peakVramMb\n",
        "                    + \" validationPsnr=\" + psnr + \" validationSsim=\" + ssim\n"
        "                    + \" holdoutViews=\" + validationViews + \" peakVramMb=\" + peakVramMb\n",
        "validation diagnostics",
    )
    replace_once(
        path,
        "            return new Result(true, resumed ? \"3DGS追加学習完了\" : \"3DGS学習完了\", output,\n"
        "                    gaussians, completedSteps, addedSteps, resumed, psnr, peakVramMb, device);\n",
        "            return new Result(true, resumed ? \"3DGS追加学習完了\" : \"3DGS学習完了\", output,\n"
        "                    gaussians, completedSteps, addedSteps, resumed, psnr, ssim,\n"
        "                    validationViews, peakVramMb, device);\n",
        "validation result return",
    )
    replace_once(
        path,
        "    public static boolean hasTrainingCheckpoint(File dataset) {\n",
        "    private static JSONObject readPhase3State(File dataset) {\n"
        "        if (dataset == null) return null;\n"
        "        File file = new File(dataset, PHASE3_STATE_FILE);\n"
        "        if (!file.isFile()) return null;\n"
        "        try (InputStream input = new FileInputStream(file)) {\n"
        "            byte[] bytes = new byte[(int) Math.min(file.length(), 256 * 1024)];\n"
        "            int offset = 0; int n;\n"
        "            while (offset < bytes.length && (n = input.read(bytes, offset, bytes.length - offset)) > 0) offset += n;\n"
        "            return new JSONObject(new String(bytes, 0, offset, StandardCharsets.UTF_8));\n"
        "        } catch (Exception ignored) { return null; }\n"
        "    }\n\n"
        "    private static int readPhase3Stage(File dataset) {\n"
        "        JSONObject state = readPhase3State(dataset);\n"
        "        return state == null ? 0 : state.optInt(\"current_stage\", 0);\n"
        "    }\n\n"
        "    private static void writePhase3State(File dataset, JSONObject state) {\n"
        "        if (dataset == null || state == null) return;\n"
        "        try (FileOutputStream out = new FileOutputStream(new File(dataset, PHASE3_STATE_FILE), false)) {\n"
        "            out.write(state.toString(2).getBytes(StandardCharsets.UTF_8));\n"
        "            out.getFD().sync();\n"
        "        } catch (Exception error) {\n"
        "            throw new IllegalStateException(error);\n"
        "        }\n"
        "    }\n\n"
        "    private static void resetInitialPhase3State(File dataset) {\n"
        "        if (dataset == null) return;\n"
        "        File[] stale = {\n"
        "                new File(dataset, \"splat.ply\"),\n"
        "                new File(dataset, \"3dgs_result.json\"),\n"
        "                new File(dataset, \"3dgs_job.json\"),\n"
        "                new File(dataset, \"phase3_stage_result.json\"),\n"
        "                new File(new File(dataset, \"vksplat_data\"), CHECKPOINT_FILE)\n"
        "        };\n"
        "        for (File file : stale) if (file.isFile() && !file.delete()) {\n"
        "            DiagnosticLog.w(TAG, \"Could not remove stale Phase 3 artifact \" + file.getAbsolutePath());\n"
        "        }\n"
        "    }\n\n"
        "    public static boolean hasTrainingCheckpoint(File dataset) {\n",
        "Phase 3 state helpers",
    )
    replace_once(
        path,
        "    private static void writeFinalResult(File dataset, int gaussians, int steps, int addedSteps,\n"
        "            boolean resumed, double psnr, double peakVramMb, String device, int gaussianBudget,\n"
        "            String strategy, String initialization, ColmapDatasetExporter.Result prepared)\n",
        "    private static void writeFinalResult(File dataset, int gaussians, int steps, int addedSteps,\n"
        "            boolean resumed, double psnr, double ssim, int validationViews, double peakVramMb,\n"
        "            String device, int gaussianBudget, String strategy, String initialization,\n"
        "            ColmapDatasetExporter.Result prepared)\n",
        "final result signature",
    )
    replace_once(
        path,
        "        JSONObject result = new JSONObject();\n"
        "        result.put(\"format_version\", 8);\n"
        "        result.put(\"status\", \"COMPLETE\");\n",
        "        int phase3Stage = readPhase3Stage(dataset);\n"
        "        boolean progressive = phase3Stage > 0;\n"
        "        boolean finalStage = !progressive || phase3Stage >= 3;\n"
        "        JSONObject result = new JSONObject();\n"
        "        result.put(\"format_version\", 9);\n"
        "        result.put(\"status\", finalStage ? \"COMPLETE\" : \"STAGE_COMPLETE\");\n"
        "        result.put(\"phase3_pipeline_version\", progressive ? 1 : 0);\n"
        "        if (progressive) {\n"
        "            result.put(\"training_profile\", PHASE3_PROFILE);\n"
        "            result.put(\"phase3_stage\", phase3Stage);\n"
        "        }\n"
        "        result.put(\"geometry_source\", \"phase2_geometry_prior.ply\");\n",
        "Phase 3 final result header",
    )
    replace_once(
        path,
        "        result.put(\"training_resolution_scale\", 0.25);\n",
        "        result.put(\"training_image_policy\", progressive\n"
        "                ? \"720px full -> 1000px full -> native-resolution principal-point patch up to 1280x960\"\n"
        "                : \"1000px full-frame continuation\");\n",
        "progressive training result metadata",
    )
    replace_once(
        path,
        "        result.put(\"final_3dgs\", true);\n",
        "        result.put(\"final_3dgs\", finalStage);\n",
        "final stage marker",
    )
    replace_once(
        path,
        "        if (Double.isFinite(psnr)) result.put(\"validation_psnr\", psnr);\n"
        "        if (Double.isFinite(peakVramMb)) result.put(\"peak_vram_mb\", peakVramMb);\n"
        "        result.put(\"completed_at_unix_ms\", System.currentTimeMillis());\n"
        "        try (FileOutputStream out = new FileOutputStream(new File(dataset, \"3dgs_result.json\"))) {\n"
        "            out.write(result.toString(2).getBytes(StandardCharsets.UTF_8));\n"
        "        }\n\n"
        "        File jobFile = new File(dataset, \"3dgs_job.json\");\n",
        "        result.put(\"holdout_view_count\", validationViews);\n"
        "        if (Double.isFinite(psnr)) result.put(\"validation_psnr\", psnr);\n"
        "        if (Double.isFinite(ssim)) result.put(\"validation_ssim\", ssim);\n"
        "        if (Double.isFinite(peakVramMb)) result.put(\"peak_vram_mb\", peakVramMb);\n"
        "        result.put(\"completed_at_unix_ms\", System.currentTimeMillis());\n"
        "        File resultFile = new File(dataset, finalStage ? \"3dgs_result.json\" : \"phase3_stage_result.json\");\n"
        "        try (FileOutputStream out = new FileOutputStream(resultFile, false)) {\n"
        "            out.write(result.toString(2).getBytes(StandardCharsets.UTF_8));\n"
        "            out.getFD().sync();\n"
        "        }\n"
        "        if (!finalStage) return;\n\n"
        "        File jobFile = new File(dataset, \"3dgs_job.json\");\n",
        "stage-safe result writing",
    )
    replace_once(
        path,
        "        job.put(\"format_version\", 9);\n",
        "        job.put(\"format_version\", 10);\n"
        "        job.put(\"phase3_pipeline_version\", progressive ? 1 : 0);\n"
        "        if (progressive) job.put(\"training_profile\", PHASE3_PROFILE);\n"
        "        job.put(\"geometry_source\", \"phase2_geometry_prior.ply\");\n",
        "Phase 3 job metadata",
    )


def patch_job() -> None:
    path = JAVA / "GaussianSplatJob.java"
    replace_once(
        path,
        "        if (!Phase1DatasetEvaluator.hasStoredPass(datasetDirectory)) {\n"
        "            DiagnosticLog.w(TAG,\n"
        "                    \"Downstream 3D processing blocked: PHASE1_EVAL is not PASS dataset=\"\n"
        "                            + datasetDirectory.getAbsolutePath());\n"
        "            return Result.fail(\n"
        "                    \"撮影データの検証がPASSしていないため、3D処理を開始しません。診断ログを確認してください。\",\n"
        "                    0);\n"
        "        }\n",
        "        if (!Phase1DatasetEvaluator.hasStoredPass(datasetDirectory)) {\n"
        "            DiagnosticLog.w(TAG,\n"
        "                    \"Downstream 3D processing blocked: PHASE1_EVAL is not PASS dataset=\"\n"
        "                            + datasetDirectory.getAbsolutePath());\n"
        "            return Result.fail(\"撮影データの検証が完了していません。\", 0);\n"
        "        }\n"
        "        if (!Phase2DatasetEvaluator.hasStoredPass(datasetDirectory)) {\n"
        "            DiagnosticLog.w(TAG,\n"
        "                    \"Phase 3 blocked: PHASE2_EVAL is not PASS dataset=\"\n"
        "                            + datasetDirectory.getAbsolutePath());\n"
        "            return Result.fail(\"3D形状の検証が完了していません。少し待ってからもう一度お試しください。\", 0);\n"
        "        }\n",
        "Phase 2 hard gate before Phase 3",
    )
    replace_once(
        path,
        "                trained = requestedSteps > 0\n"
        "                        ? NativeGaussianTrainer.train(appContext, datasetDirectory, requestedSteps,\n"
        "                                (percent, message) -> notifyProgress(listener, percent, message))\n"
        "                        : NativeGaussianTrainer.train(appContext, datasetDirectory,\n"
        "                                (percent, message) -> notifyProgress(listener, percent, message));\n",
        "                trained = requestedSteps > 0\n"
        "                        ? NativeGaussianTrainer.trainProgressiveInitial(\n"
        "                                appContext, datasetDirectory, requestedSteps,\n"
        "                                (percent, message) -> notifyProgress(listener, percent, message))\n"
        "                        : NativeGaussianTrainer.trainProgressiveInitial(\n"
        "                                appContext, datasetDirectory,\n"
        "                                (percent, message) -> notifyProgress(listener, percent, message));\n",
        "progressive initial trainer",
    )
    replace_once(
        path,
        "            DiagnosticLog.i(TAG,\n"
        "                    \"Full 3DGS COMPLETE frames=\" + count\n",
        "            Phase3DatasetEvaluator.Result phase3 = Phase3DatasetEvaluator.evaluate(datasetDirectory);\n"
        "            if (!phase3.machineGatePassed) {\n"
        "                DiagnosticLog.e(TAG, \"Phase 3 machine evaluation failed after trainer completion\");\n"
        "                return Result.fail(\"3Dモデルは学習しましたが、結果検証に失敗しました。\", count);\n"
        "            }\n\n"
        "            DiagnosticLog.i(TAG,\n"
        "                    \"Full 3DGS COMPLETE frames=\" + count\n",
        "Phase 3 machine evaluation",
    )
    replace_once(
        path,
        "            notifyProgress(listener, 100, \"3Dモデルを作成しました\");\n"
        "            return Result.complete(\"3DGS学習完了\", count, trained.gaussianCount, trained.outputFile);\n",
        "            notifyProgress(listener, 100, \"3Dモデルを作成しました。表示して仕上がりを確認します\");\n"
        "            return Result.complete(\"3DGS学習完了・品質確認待ち\", count, trained.gaussianCount, trained.outputFile);\n",
        "Phase 3 visual review wording",
    )
    replace_once(
        path,
        "                && result.optBoolean(\"density_control\", false)\n"
        "                && result.optBoolean(\"final_3dgs\", false);\n",
        "                && result.optBoolean(\"density_control\", false)\n"
        "                && result.optBoolean(\"final_3dgs\", false)\n"
        "                && result.optInt(\"phase3_pipeline_version\", 0) >= 1\n"
        "                && \"progressive_low_mid_high_patch\".equals(\n"
        "                        result.optString(\"training_profile\", \"\"));\n",
        "Phase 3 verified final contract",
    )
    replace_once(
        path,
        "        deleteLegacyResumeHint(datasetDirectory);\n"
        "        DiagnosticLog.i(TAG, \"3DGS CONTINUED previousSteps=\" + previousSteps\n",
        "        deleteLegacyResumeHint(datasetDirectory);\n"
        "        Phase3DatasetEvaluator.Result phase3 = Phase3DatasetEvaluator.evaluate(datasetDirectory);\n"
        "        if (!phase3.machineGatePassed) {\n"
        "            return Result.fail(\"追加学習は完了しましたが、結果検証に失敗しました。\", frameCount);\n"
        "        }\n"
        "        DiagnosticLog.i(TAG, \"3DGS CONTINUED previousSteps=\" + previousSteps\n",
        "continuation Phase 3 evaluation",
    )


def patch_ui() -> None:
    scanner = JAVA / "ScannerActivity.java"
    replace_all(scanner, "3Dプレビューを準備中", "3Dモデルを学習中", "scanner training title")
    replace_once(
        scanner,
        "                    showOperation(\"高品質3Dモデルを作成しました\",\n"
        "                            \"完成した3Dモデルを表示します。\", 100);\n",
        "                    showOperation(\"3Dモデルを作成しました\",\n"
        "                            \"表示して仕上がりを確認します。\", 100);\n",
        "scanner final review wording",
    )

    library = JAVA / "LibraryActivity.java"
    replace_once(
        library,
        "String trained=steps>0?\"3Dモデル完成・\"+steps+\" step\":\"3Dモデル完成\";return photos+\"\\n\"+trained+(GaussianSplatJob.canContinueTraining(dataset)?\"・追加学習可\":\"\");}",
        "String trained=steps>0?\"3Dモデル作成済み・\"+steps+\" step\":\"3Dモデル作成済み\";if(Phase3DatasetEvaluator.hasMachineGatePass(dataset))trained+=\"・品質確認待ち\";return photos+\"\\n\"+trained+(GaussianSplatJob.canContinueTraining(dataset)?\"・追加学習可\":\"\");}",
        "library Phase 3 review status",
    )
    replace_once(
        library,
        "&&result.optBoolean(\"final_3dgs\",false)&&\"COMPLETE\".equals(result.optString(\"status\",\"\"));}",
        "&&result.optBoolean(\"final_3dgs\",false)&&result.optInt(\"phase3_pipeline_version\",0)>=1&&\"progressive_low_mid_high_patch\".equals(result.optString(\"training_profile\",\"\"))&&\"COMPLETE\".equals(result.optString(\"status\",\"\"));}",
        "library final Phase 3 contract",
    )


def patch_native_cpp() -> None:
    path = CPP
    replace_once(path, "#include <cstdint>\n", "#include <cstdint>\n#include <fstream>\n", "fstream include")
    replace_once(
        path,
        "    c.eval_interval = std::max(2, frameCount + 1);\n",
        "    // Keep roughly 20% of sufficiently large datasets out of optimization for real hold-out evaluation.\n"
        "    c.eval_interval = frameCount >= 8 ? 5 : std::max(2, frameCount + 1);\n",
        "hold-out split",
    )
    old = '''double validationPsnr(VulkanGSTrainer& trainer,\n                      VulkanGSRendererUniforms& uniforms,\n                      VulkanGSPipelineBuffers& buffers) {\n    if (trainer.num_val() == 0) return NAN;\n    trainer.get_val_camera(0, uniforms);\n    uniforms.active_sh = 3;\n    uniforms.step = trainer.getCompletedTrainingSteps();\n    {\n        auto guard = DeviceGuard(&trainer);\n        forward(trainer, uniforms, buffers, true);\n    }\n    trainer.copyFromDevice(buffers.pixel_state);\n    auto& reference = trainer.get_val_image(0).buffer;\n    size_t pixelCount = static_cast<size_t>(uniforms.image_width) * uniforms.image_height;\n    if (buffers.pixel_state.size() < pixelCount * 4 || reference.size() < pixelCount * 4) return NAN;\n    double mse = 0.0;\n    size_t samples = 0;\n    for (size_t i = 0; i < pixelCount; ++i) {\n        for (int c = 0; c < 3; ++c) {\n            double predicted = std::clamp(static_cast<double>(buffers.pixel_state[4*i+c]), 0.0, 1.0);\n            double expected = static_cast<double>(reference[4*i+c]) / 255.0;\n            double d = predicted - expected;\n            mse += d * d;\n            ++samples;\n        }\n    }\n    if (samples == 0) return NAN;\n    mse /= static_cast<double>(samples);\n    if (mse <= 1e-12) return 120.0;\n    return 10.0 * std::log10(1.0 / mse);\n}\n'''
    new = '''struct ValidationMetrics {\n    double psnr = NAN;\n    double ssim = NAN;\n    int views = 0;\n};\n\nvoid writeValidationPpm(const std::string& path, uint32_t width, uint32_t height,\n                        const Buffer<float>& pixels) {\n    std::ofstream out(path, std::ios::binary | std::ios::trunc);\n    if (!out) throw std::runtime_error("could not write hold-out render");\n    out << "P6\\n" << width << " " << height << "\\n255\\n";\n    for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i) {\n        unsigned char rgb[3];\n        for (int c = 0; c < 3; ++c) {\n            double value = std::clamp(static_cast<double>(pixels[4*i+c]), 0.0, 1.0);\n            rgb[c] = static_cast<unsigned char>(std::lround(value * 255.0));\n        }\n        out.write(reinterpret_cast<const char*>(rgb), 3);\n    }\n}\n\ndouble blockSsim(const Buffer<float>& predicted, const Buffer<uint8_t>& reference,\n                 uint32_t width, uint32_t height) {\n    constexpr int block = 8;\n    constexpr double c1 = 0.01 * 0.01;\n    constexpr double c2 = 0.03 * 0.03;\n    double total = 0.0;\n    size_t blocks = 0;\n    for (uint32_t by = 0; by < height; by += block) {\n        for (uint32_t bx = 0; bx < width; bx += block) {\n            double meanP = 0.0, meanR = 0.0;\n            size_t count = 0;\n            for (uint32_t y = by; y < std::min<uint32_t>(height, by + block); ++y) {\n                for (uint32_t x = bx; x < std::min<uint32_t>(width, bx + block); ++x) {\n                    size_t i = static_cast<size_t>(y) * width + x;\n                    double py = 0.299 * std::clamp<double>(predicted[4*i], 0.0, 1.0)\n                              + 0.587 * std::clamp<double>(predicted[4*i+1], 0.0, 1.0)\n                              + 0.114 * std::clamp<double>(predicted[4*i+2], 0.0, 1.0);\n                    double ry = (0.299 * reference[4*i] + 0.587 * reference[4*i+1]\n                               + 0.114 * reference[4*i+2]) / 255.0;\n                    meanP += py; meanR += ry; count++;\n                }\n            }\n            if (count < 2) continue;\n            meanP /= count; meanR /= count;\n            double varP = 0.0, varR = 0.0, cov = 0.0;\n            for (uint32_t y = by; y < std::min<uint32_t>(height, by + block); ++y) {\n                for (uint32_t x = bx; x < std::min<uint32_t>(width, bx + block); ++x) {\n                    size_t i = static_cast<size_t>(y) * width + x;\n                    double py = 0.299 * std::clamp<double>(predicted[4*i], 0.0, 1.0)\n                              + 0.587 * std::clamp<double>(predicted[4*i+1], 0.0, 1.0)\n                              + 0.114 * std::clamp<double>(predicted[4*i+2], 0.0, 1.0);\n                    double ry = (0.299 * reference[4*i] + 0.587 * reference[4*i+1]\n                               + 0.114 * reference[4*i+2]) / 255.0;\n                    double dp = py - meanP, dr = ry - meanR;\n                    varP += dp * dp; varR += dr * dr; cov += dp * dr;\n                }\n            }\n            double denom = static_cast<double>(count - 1);\n            varP /= denom; varR /= denom; cov /= denom;\n            double value = ((2.0 * meanP * meanR + c1) * (2.0 * cov + c2))\n                         / ((meanP * meanP + meanR * meanR + c1) * (varP + varR + c2));\n            if (std::isfinite(value)) { total += value; blocks++; }\n        }\n    }\n    return blocks ? total / static_cast<double>(blocks) : NAN;\n}\n\nValidationMetrics validationMetrics(VulkanGSTrainer& trainer,\n                                    VulkanGSRendererUniforms& uniforms,\n                                    VulkanGSPipelineBuffers& buffers,\n                                    const std::string& dataRoot) {\n    ValidationMetrics result;\n    double squaredError = 0.0;\n    size_t samples = 0;\n    double ssimTotal = 0.0;\n    int ssimViews = 0;\n    for (size_t view = 0; view < trainer.num_val(); ++view) {\n        trainer.get_val_camera(view, uniforms);\n        uniforms.active_sh = 3;\n        uniforms.step = trainer.getCompletedTrainingSteps();\n        {\n            auto guard = DeviceGuard(&trainer);\n            forward(trainer, uniforms, buffers, true);\n        }\n        trainer.copyFromDevice(buffers.pixel_state);\n        auto& reference = trainer.get_val_image(view).buffer;\n        size_t pixelCount = static_cast<size_t>(uniforms.image_width) * uniforms.image_height;\n        if (buffers.pixel_state.size() < pixelCount * 4 || reference.size() < pixelCount * 4) continue;\n        for (size_t i = 0; i < pixelCount; ++i) {\n            for (int c = 0; c < 3; ++c) {\n                double predicted = std::clamp(static_cast<double>(buffers.pixel_state[4*i+c]), 0.0, 1.0);\n                double expected = static_cast<double>(reference[4*i+c]) / 255.0;\n                double d = predicted - expected;\n                squaredError += d * d; samples++;\n            }\n        }\n        double ssim = blockSsim(buffers.pixel_state, reference, uniforms.image_width, uniforms.image_height);\n        if (std::isfinite(ssim)) { ssimTotal += ssim; ssimViews++; }\n        writeValidationPpm(dataRoot + "/phase3_holdout_render_" + std::to_string(view) + ".ppm",\n                           uniforms.image_width, uniforms.image_height, buffers.pixel_state);\n        result.views++;\n    }\n    if (samples > 0) {\n        double mse = squaredError / static_cast<double>(samples);\n        result.psnr = mse <= 1e-12 ? 120.0 : 10.0 * std::log10(1.0 / mse);\n    }\n    if (ssimViews > 0) result.ssim = ssimTotal / ssimViews;\n    return result;\n}\n'''
    replace_once(path, old, new, "hold-out PSNR/SSIM renders")
    replace_once(
        path,
        "std::string successJson(size_t count, uint32_t cumulativeSteps, int addedSteps,\n"
        "                        int optimizedSteps, bool resumed, double psnr,\n"
        "                        const std::string& device, size_t peakBytes) {\n",
        "std::string successJson(size_t count, uint32_t cumulativeSteps, int addedSteps,\n"
        "                        int optimizedSteps, bool resumed, double psnr, double ssim,\n"
        "                        int validationViews, const std::string& device, size_t peakBytes) {\n",
        "native success JSON signature",
    )
    replace_once(
        path,
        "    if (std::isfinite(psnr)) out << \",\\\"validation_psnr\\\":\" << psnr;\n"
        "    out << \"}\";\n",
        "    if (std::isfinite(psnr)) out << \",\\\"validation_psnr\\\":\" << psnr;\n"
        "    if (std::isfinite(ssim)) out << \",\\\"validation_ssim\\\":\" << ssim;\n"
        "    out << \",\\\"validation_view_count\\\":\" << validationViews;\n"
        "    out << \"}\";\n",
        "native validation JSON fields",
    )
    replace_once(
        path,
        "        double psnr = validationPsnr(trainer, uniforms, buffers);\n",
        "        ValidationMetrics validation = validationMetrics(trainer, uniforms, buffers, dataRoot);\n"
        "        double psnr = validation.psnr;\n"
        "        double ssim = validation.ssim;\n",
        "native validation call",
    )
    replace_once(
        path,
        "             + \" psnr=\" + (std::isfinite(psnr) ? std::to_string(psnr) : std::string(\"n/a\")));\n",
        "             + \" psnr=\" + (std::isfinite(psnr) ? std::to_string(psnr) : std::string(\"n/a\"))\n"
        "             + \" ssim=\" + (std::isfinite(ssim) ? std::to_string(ssim) : std::string(\"n/a\"))\n"
        "             + \" holdoutViews=\" + std::to_string(validation.views));\n",
        "native validation log",
    )
    replace_once(
        path,
        "        std::string json = successJson(finalCount, cumulativeSteps, steps, optimizedSteps,\n"
        "                                       resumed, psnr, gpu, peakBytes);\n",
        "        std::string json = successJson(finalCount, cumulativeSteps, steps, optimizedSteps,\n"
        "                                       resumed, psnr, ssim, validation.views, gpu, peakBytes);\n",
        "native success JSON call",
    )


def main() -> None:
    patch_phase2_pass_policy()
    patch_exporter()
    patch_native_java()
    patch_job()
    patch_ui()
    patch_native_cpp()
    print("v1.0.9 Phase 3 activation/progressive training patch applied")


if __name__ == "__main__":
    main()
