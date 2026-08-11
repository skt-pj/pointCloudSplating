package com.sktpj.pointcloudsplatting;

import android.content.Context;
import android.content.res.AssetManager;

import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/** JNI bridge for the PCS mobile-first Vulkan differentiable 3D Gaussian Splatting trainer. */
public final class NativeGaussianTrainer {
    private static final String TAG = "Native3DGS";
    private static final String ASSET_ROOT = "vksplat_shader";
    // Kept as an explicit legacy identifier so diagnostics can distinguish a stale v1.0.1 cache.
    private static final String LEGACY_SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v6";
    private static final String SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v8";
    private static final String CHECKPOINT_FILE = "3dgs_checkpoint.bin";
    private static final String PHASE3_STATE_FILE = "phase3_training_state.json";
    private static final String PHASE3_PROFILE = "progressive_low_mid_high_patch";

    // These are defaults for the first mobile run, not a quality ceiling. Additional runs may use
    // any positive step count selected by the user and continue from the persisted optimizer state.
    private static final int BASE_TRAIN_STEPS = 750;

    private static final boolean NATIVE_AVAILABLE;
    private static volatile boolean cumsumConformancePassed;
    static {
        boolean loaded = false;
        try {
            System.loadLibrary("pointcloud3dgs");
            loaded = true;
        } catch (UnsatisfiedLinkError error) {
            DiagnosticLog.e(TAG, "Native 3DGS library failed to load", error);
        }
        NATIVE_AVAILABLE = loaded;
    }

    private NativeGaussianTrainer() {}

    public interface ProgressListener {
        void onProgress(int percent, String message);
    }

    public static final class Result {
        public final boolean success;
        public final String message;
        public final File outputFile;
        public final int gaussianCount;
        /** Cumulative optimizer step stored in the checkpoint after this run. */
        public final int steps;
        public final int addedSteps;
        public final boolean resumed;
        public final double validationPsnr;
        public final double validationSsim;
        public final int validationViewCount;
        public final double peakVramMb;
        public final String device;

        private Result(boolean success, String message, File outputFile, int gaussianCount,
                int steps, int addedSteps, boolean resumed, double validationPsnr,
                double validationSsim, int validationViewCount,
                double peakVramMb, String device) {
            this.success = success;
            this.message = message;
            this.outputFile = outputFile;
            this.gaussianCount = gaussianCount;
            this.steps = steps;
            this.addedSteps = addedSteps;
            this.resumed = resumed;
            this.validationPsnr = validationPsnr;
            this.validationSsim = validationSsim;
            this.validationViewCount = validationViewCount;
            this.peakVramMb = peakVramMb;
            this.device = device;
        }

        static Result fail(String message) {
            return new Result(false, message, null, 0, 0, 0, false,
                    Double.NaN, Double.NaN, 0, Double.NaN, "");
        }
    }

    /** First-run entry point. The frame-count rule chooses only the default initial step count. */
    public static Result train(Context context, File dataset, ProgressListener listener) {
        int frames = readFrameCountForDefault(dataset);
        return train(context, dataset, defaultTrainingSteps(frames), listener);
    }

    /** Initial Phase 3 run: low full-frame -> medium full-frame -> high-resolution patch. */
    public static Result trainProgressiveInitial(Context context, File dataset,
            ProgressListener listener) {
        int frames = readFrameCountForDefault(dataset);
        return trainProgressiveInitial(context, dataset, defaultTrainingSteps(frames), listener);
    }

    public static Result trainProgressiveInitial(Context context, File dataset, int requestedSteps,
            ProgressListener listener) {
        if (requestedSteps < 3) return Result.fail("初回3D学習は3 step以上を指定してください");
        if (context == null || dataset == null || !dataset.isDirectory()) {
            return Result.fail("撮影データが見つかりませんでした");
        }
        int low = Math.max(1, requestedSteps * 30 / 100);
        int mid = Math.max(1, requestedSteps * 40 / 100);
        int high = requestedSteps - low - mid;
        if (high < 1) { high = 1; if (mid > low) mid--; else low--; }
        int[] stageSteps = {low, mid, high};
        JSONObject state = readPhase3State(dataset);
        boolean compatible = state != null
                && PHASE3_PROFILE.equals(state.optString("profile", ""))
                && state.optInt("requested_steps", -1) == requestedSteps;
        if (!compatible) {
            resetInitialPhase3State(dataset);
            state = new JSONObject();
            try {
                state.put("profile", PHASE3_PROFILE);
                state.put("requested_steps", requestedSteps);
                state.put("completed_stage", 0);
            } catch (Exception ignored) {}
            writePhase3State(dataset, state);
        }
        int completedStage = Math.max(0, Math.min(3, state.optInt("completed_stage", 0)));
        Result last = null;
        for (int stage = completedStage + 1; stage <= 3; stage++) {
            try {
                state.put("current_stage", stage);
                state.put("stage_steps", stageSteps[stage - 1]);
                writePhase3State(dataset, state);
            } catch (Exception error) {
                return Result.fail("3D学習状態を保存できませんでした");
            }
            final int currentStage = stage;
            final int progressStart = stage == 1 ? 0 : (stage == 2 ? 30 : 70);
            final int progressSpan = stage == 1 ? 30 : (stage == 2 ? 40 : 30);
            last = train(context, dataset, stageSteps[stage - 1], (percent, message) -> {
                int mapped = progressStart + Math.round(progressSpan * percent / 100f);
                notifyProgress(listener, mapped,
                        "Phase 3 " + currentStage + "/3: " + message);
            });
            if (!last.success) return last;
            try {
                state.put("completed_stage", stage);
                writePhase3State(dataset, state);
            } catch (Exception error) {
                return Result.fail("3D学習状態を更新できませんでした");
            }
        }
        return last == null ? Result.fail("3D学習を開始できませんでした") : last;
    }

    /** Runs exactly requestedSteps more optimizer iterations, resuming a checkpoint when present. */
    public static Result train(Context context, File dataset, int requestedSteps,
            ProgressListener listener) {
        if (!NATIVE_AVAILABLE) return Result.fail("端末内3DGS学習ライブラリを開始できませんでした");
        if (context == null || dataset == null || !dataset.isDirectory()) {
            return Result.fail("撮影データが見つかりませんでした");
        }
        if (requestedSteps <= 0) return Result.fail("学習回数は1以上を指定してください");

        try {
            notifyProgress(listener, 1, "写真・カメラ位置・Depthを学習用に準備しています…");
            ColmapDatasetExporter.Result prepared = ColmapDatasetExporter.prepare(
                    dataset,
                    (percent, message) -> notifyProgress(listener,
                            Math.min(15, Math.max(1, Math.round(percent * 0.33f))), message));
            if (!prepared.success) return Result.fail(prepared.message);

            File checkpoint = new File(prepared.root, CHECKPOINT_FILE);
            boolean checkpointBeforeRun = checkpoint.isFile() && checkpoint.length() > 0L;
            File shaderDir = ensureShaderFiles(context);
            DiagnosticLog.i(TAG,
                    "Using VkSplat shader cache=" + shaderDir.getName()
                            + " legacyCache=" + LEGACY_SHADER_CACHE
                            + " cumsum=glslc256 radix=glslc256/subgroup16/two_stage_no_global_atomic"
                            + " numeric=finite_projection+mcmc_relocation+adam_rollback"
                            + " requestedSteps=" + requestedSteps
                            + " checkpointBefore=" + checkpointBeforeRun);
            logCumsumShaderIdentity(shaderDir, "cumsum_single_pass.spv");
            logCumsumShaderIdentity(shaderDir, "cumsum_block_scan.spv");
            logCumsumShaderIdentity(shaderDir, "cumsum_scan_block_sums.spv");
            logCumsumShaderIdentity(shaderDir, "cumsum_add_block_offsets.spv");
            logRadixShaderIdentity(shaderDir, "upsweep.spv");
            logRadixShaderIdentity(shaderDir, "spine.spv");
            logRadixShaderIdentity(shaderDir, "downsweep.spv");

            String cumsumFailure = ensureCumsumConformance(shaderDir, listener);
            if (cumsumFailure != null) {
                DiagnosticLog.w(TAG, "GPU cumsum conformance failed: " + cumsumFailure);
                return Result.fail(cumsumFailure);
            }

            File output = new File(dataset, "splat.ply");
            File working = prepared.root;
            notifyProgress(listener, 16, checkpointBeforeRun
                    ? "保存した3DGS学習状態から追加学習を始めます…"
                    : "表面形状を初期化して端末向け3DGSを始めます…");

            String raw = nativeTrain(
                    working.getAbsolutePath(),
                    ensureTrailingSlash(prepared.imageDir.getAbsolutePath()),
                    ensureTrailingSlash(prepared.sparseDir.getAbsolutePath()),
                    ensureTrailingSlash(shaderDir.getAbsolutePath()),
                    output.getAbsolutePath(),
                    prepared.frameCount,
                    requestedSteps,
                    (percent, message) -> notifyProgress(listener,
                            16 + Math.round(percent * 0.82f), message));
            JSONObject json = new JSONObject(raw);
            boolean success = json.optBoolean("success", false);
            String message = json.optString("message", success ? "3DGS学習完了" : "3DGS学習に失敗しました");
            if (!success || !output.isFile() || output.length() == 0L) {
                DiagnosticLog.w(TAG, "Mobile training failed result=" + raw);
                return Result.fail(message);
            }
            if (!checkpoint.isFile() || checkpoint.length() == 0L) {
                DiagnosticLog.w(TAG, "Native training completed without resumable checkpoint");
                return Result.fail("3Dモデルは作成されましたが、追加学習用の状態を保存できませんでした");
            }

            int gaussians = json.optInt("gaussian_count", 0);
            int completedSteps = json.optInt("steps", requestedSteps);
            int addedSteps = json.optInt("added_steps", requestedSteps);
            boolean resumed = json.optBoolean("resumed", checkpointBeforeRun);
            double psnr = json.optDouble("validation_psnr", Double.NaN);
            double ssim = json.optDouble("validation_ssim", Double.NaN);
            int validationViews = json.optInt("validation_view_count", 0);
            double peakVramMb = json.optDouble("peak_vram_mb", Double.NaN);
            String device = json.optString("device", "Vulkan");
            int gaussianBudget = json.optInt("gaussian_budget", 120_000);
            String strategy = json.optString("strategy", "bounded_mcmc");
            String initialization = json.optString("initialization", "surface_knn_16_3");

            writeFinalResult(dataset, gaussians, completedSteps, addedSteps, resumed, psnr,
                    ssim, validationViews, peakVramMb, device, gaussianBudget, strategy,
                    initialization, prepared);
            DiagnosticLog.i(TAG, "Mobile 3DGS COMPLETE gaussians=" + gaussians
                    + "/" + gaussianBudget + " steps=" + completedSteps
                    + " addedSteps=" + addedSteps + " resumed=" + resumed
                    + " validationPsnr=" + psnr + " validationSsim=" + ssim
                    + " holdoutViews=" + validationViews + " peakVramMb=" + peakVramMb
                    + " strategy=" + strategy + " init=" + initialization
                    + " device=" + device + " output=" + output.getAbsolutePath());
            notifyProgress(listener, 100, resumed ? "追加学習が完了しました" : "3Dモデルを作成しました");
            return new Result(true, resumed ? "3DGS追加学習完了" : "3DGS学習完了", output,
                    gaussians, completedSteps, addedSteps, resumed, psnr, ssim,
                    validationViews, peakVramMb, device);
        } catch (Throwable error) {
            DiagnosticLog.e(TAG, "Mobile 3DGS training failed", error);
            return Result.fail("端末内3DGS学習に失敗しました: " + error.getClass().getSimpleName());
        }
    }

    private static JSONObject readPhase3State(File dataset) {
        if (dataset == null) return null;
        File file = new File(dataset, PHASE3_STATE_FILE);
        if (!file.isFile()) return null;
        try (InputStream input = new FileInputStream(file)) {
            byte[] bytes = new byte[(int) Math.min(file.length(), 256 * 1024)];
            int offset = 0; int n;
            while (offset < bytes.length && (n = input.read(bytes, offset, bytes.length - offset)) > 0) offset += n;
            return new JSONObject(new String(bytes, 0, offset, StandardCharsets.UTF_8));
        } catch (Exception ignored) { return null; }
    }

    private static int readPhase3Stage(File dataset) {
        JSONObject state = readPhase3State(dataset);
        return state == null ? 0 : state.optInt("current_stage", 0);
    }

    private static void writePhase3State(File dataset, JSONObject state) {
        if (dataset == null || state == null) return;
        try (FileOutputStream out = new FileOutputStream(new File(dataset, PHASE3_STATE_FILE), false)) {
            out.write(state.toString(2).getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        } catch (Exception error) {
            throw new IllegalStateException(error);
        }
    }

    private static void resetInitialPhase3State(File dataset) {
        if (dataset == null) return;
        File[] stale = {
                new File(dataset, "splat.ply"),
                new File(dataset, "3dgs_result.json"),
                new File(dataset, "3dgs_job.json"),
                new File(dataset, "phase3_stage_result.json"),
                new File(new File(dataset, "vksplat_data"), CHECKPOINT_FILE)
        };
        for (File file : stale) if (file.isFile() && !file.delete()) {
            DiagnosticLog.w(TAG, "Could not remove stale Phase 3 artifact " + file.getAbsolutePath());
        }
    }

    public static boolean hasTrainingCheckpoint(File dataset) {
        if (dataset == null) return false;
        File checkpoint = new File(new File(dataset, "vksplat_data"), CHECKPOINT_FILE);
        return checkpoint.isFile() && checkpoint.length() > 0L;
    }

    private static synchronized String ensureCumsumConformance(
            File shaderDir, ProgressListener listener) throws Exception {
        if (cumsumConformancePassed) return null;
        notifyProgress(listener, 16, "GPU演算を検証しています…");
        DiagnosticLog.i(TAG, "cumsum:selftest Java bridge begin");
        String raw = nativeCumsumSelfTest(
                ensureTrailingSlash(shaderDir.getAbsolutePath()),
                (percent, message) -> {
                    if (message != null && message.startsWith("cumsum:selftest")) {
                        DiagnosticLog.i(TAG, message);
                    }
                    notifyProgress(listener, 16, "GPU演算を検証しています…");
                });
        JSONObject json = new JSONObject(raw);
        boolean success = json.optBoolean("success", false);
        String message = json.optString("message",
                success ? "GPU cumsum conformance passed" : "GPU cumsum conformance failed");
        if (!success) return message;
        cumsumConformancePassed = true;
        DiagnosticLog.i(TAG, "cumsum:selftest Java bridge COMPLETE");
        return null;
    }

    private static int defaultTrainingSteps(int frames) {
        int steps = BASE_TRAIN_STEPS;
        if (frames >= 12) steps += 125;
        if (frames >= 24) steps += 125;
        return steps;
    }

    private static int readFrameCountForDefault(File dataset) {
        if (dataset == null) return 0;
        File manifest = new File(dataset, "dataset_manifest.json");
        if (!manifest.isFile()) return 0;
        try (InputStream input = new FileInputStream(manifest)) {
            byte[] bytes = new byte[(int) Math.min(manifest.length(), 1024 * 1024)];
            int offset = 0;
            int n;
            while (offset < bytes.length && (n = input.read(bytes, offset, bytes.length - offset)) > 0) {
                offset += n;
            }
            return new JSONObject(new String(bytes, 0, offset, StandardCharsets.UTF_8))
                    .optInt("frame_count", 0);
        } catch (Exception ignored) {
            return 0;
        }
    }

    private static void writeFinalResult(File dataset, int gaussians, int steps, int addedSteps,
            boolean resumed, double psnr, double ssim, int validationViews, double peakVramMb,
            String device, int gaussianBudget, String strategy, String initialization,
            ColmapDatasetExporter.Result prepared)
            throws Exception {
        int phase3Stage = readPhase3Stage(dataset);
        boolean progressive = phase3Stage > 0;
        boolean finalStage = !progressive || phase3Stage >= 3;
        JSONObject result = new JSONObject();
        result.put("format_version", 9);
        result.put("status", finalStage ? "COMPLETE" : "STAGE_COMPLETE");
        result.put("phase3_pipeline_version", progressive ? 1 : 0);
        if (progressive) {
            result.put("training_profile", PHASE3_PROFILE);
            result.put("phase3_stage", phase3Stage);
        }
        result.put("geometry_source", "phase2_geometry_prior.ply");
        result.put("backend", "pcs_mobile_vulkan_trainer_v1");
        result.put("raster_backend", "VkSplat@41cff93b79145dec314488d4313bc3a6d737038b");
        result.put("output", "splat.ply");
        result.put("frame_count", prepared.frameCount);
        result.put("raw_initial_point_count", prepared.initialPointCount);
        result.put("gaussian_count", gaussians);
        result.put("gaussian_budget", gaussianBudget);
        result.put("training_steps", steps);
        result.put("last_run_added_steps", addedSteps);
        result.put("last_run_resumed", resumed);
        result.put("training_image_policy", progressive
                ? "720px full -> 1000px full -> native-resolution principal-point patch up to 1280x960"
                : "1000px full-frame continuation");
        result.put("loss", "L1 + SSIM");
        result.put("optimized_parameters", "position, quaternion, scale, opacity, SH0-SH3");
        result.put("initialization", initialization);
        result.put("normal_neighbors", 16);
        result.put("scale_neighbors", 3);
        result.put("density_control", true);
        result.put("density_strategy", strategy);
        result.put("densification_pruning", false);
        result.put("bounded_gaussian_budget", true);
        result.put("projection_invariant_checks", true);
        result.put("rasterized_image_loss", true);
        result.put("l1_ssim_backward", true);
        result.put("photometric_optimization", true);
        result.put("final_3dgs", finalStage);
        result.put("resumable_training", true);
        result.put("checkpoint_available", true);
        result.put("checkpoint_file", "vksplat_data/" + CHECKPOINT_FILE);
        result.put("checkpoint_state", "gaussians+adam+rng+step");
        result.put("vulkan_device", device);
        result.put("holdout_view_count", validationViews);
        if (Double.isFinite(psnr)) result.put("validation_psnr", psnr);
        if (Double.isFinite(ssim)) result.put("validation_ssim", ssim);
        if (Double.isFinite(peakVramMb)) result.put("peak_vram_mb", peakVramMb);
        result.put("completed_at_unix_ms", System.currentTimeMillis());
        File resultFile = new File(dataset, finalStage ? "3dgs_result.json" : "phase3_stage_result.json");
        try (FileOutputStream out = new FileOutputStream(resultFile, false)) {
            out.write(result.toString(2).getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }
        if (!finalStage) return;

        File jobFile = new File(dataset, "3dgs_job.json");
        JSONObject job = new JSONObject();
        job.put("format_version", 10);
        job.put("phase3_pipeline_version", progressive ? 1 : 0);
        if (progressive) job.put("training_profile", PHASE3_PROFILE);
        job.put("geometry_source", "phase2_geometry_prior.ply");
        job.put("status", "COMPLETE");
        job.put("backend", "pcs_mobile_vulkan_trainer_v1");
        job.put("raster_backend", "VkSplat@41cff93b79145dec314488d4313bc3a6d737038b");
        job.put("final_output", "splat.ply");
        job.put("final_3dgs", true);
        job.put("photometric_optimization", true);
        job.put("rasterized_image_loss", true);
        job.put("l1_ssim_backward", true);
        job.put("density_control", true);
        job.put("density_strategy", strategy);
        job.put("gaussian_budget", gaussianBudget);
        job.put("training_steps", steps);
        job.put("resumable_training", true);
        job.put("checkpoint_file", "vksplat_data/" + CHECKPOINT_FILE);
        job.put("initialization", initialization);
        try (FileOutputStream out = new FileOutputStream(jobFile)) {
            out.write(job.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    private static File ensureShaderFiles(Context context) throws IOException {
        File root = new File(context.getNoBackupFilesDir(), SHADER_CACHE);
        File marker = new File(root, "VKSPLAT_COMMIT.txt");
        if (marker.isFile() && marker.length() > 0L) return root;
        deleteRecursively(root);
        if (!root.mkdirs() && !root.isDirectory()) throw new IOException("cannot create shader cache");
        copyAssetTree(context.getAssets(), ASSET_ROOT, root);
        if (!marker.isFile()) throw new IOException("VkSplat shader assets missing from APK");
        return root;
    }

    private static void logCumsumShaderIdentity(File shaderDir, String fileName) throws IOException {
        File file = new File(new File(shaderDir, "generated"), fileName);
        logShaderIdentity("cumsum", fileName, file);
    }

    private static void logRadixShaderIdentity(File shaderDir, String fileName) throws IOException {
        File file = new File(new File(shaderDir, "radix_sort"), fileName);
        logShaderIdentity("radix", fileName, file);
    }

    private static void logShaderIdentity(String family, String fileName, File file)
            throws IOException {
        if (!file.isFile() || file.length() <= 0L) {
            throw new IOException("VkSplat " + family + " shader missing: " + fileName);
        }
        DiagnosticLog.i(TAG, "VkSplat " + family + " shader=" + fileName
                + " bytes=" + file.length() + " sha256=" + sha256(file));
    }

    private static String sha256(File file) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream input = new FileInputStream(file)) {
                byte[] buffer = new byte[64 * 1024];
                int n;
                while ((n = input.read(buffer)) != -1) digest.update(buffer, 0, n);
            }
            StringBuilder out = new StringBuilder(64);
            for (byte b : digest.digest()) out.append(String.format("%02x", b & 0xff));
            return out.toString();
        } catch (Exception error) {
            throw new IOException("cannot hash VkSplat shader " + file.getName(), error);
        }
    }

    private static void copyAssetTree(AssetManager assets, String assetPath, File destination)
            throws IOException {
        String[] children = assets.list(assetPath);
        if (children != null && children.length > 0) {
            if (!destination.exists() && !destination.mkdirs()) {
                throw new IOException("cannot create " + destination);
            }
            for (String child : children) {
                copyAssetTree(assets, assetPath + "/" + child, new File(destination, child));
            }
            return;
        }
        try (InputStream input = assets.open(assetPath);
             FileOutputStream output = new FileOutputStream(destination)) {
            byte[] buffer = new byte[64 * 1024];
            int n;
            while ((n = input.read(buffer)) != -1) output.write(buffer, 0, n);
        }
    }

    private static void deleteRecursively(File file) throws IOException {
        if (!file.exists()) return;
        if (file.isDirectory()) {
            File[] children = file.listFiles();
            if (children != null) for (File child : children) deleteRecursively(child);
        }
        if (!file.delete()) throw new IOException("cannot delete " + file);
    }

    private static String ensureTrailingSlash(String path) {
        return path.endsWith(File.separator) ? path : path + File.separator;
    }

    private static void notifyProgress(ProgressListener l, int p, String message) {
        if (l != null) l.onProgress(Math.max(0, Math.min(100, p)), message);
    }

    private static native String nativeCumsumSelfTest(
            String shaderDir,
            ProgressListener listener);

    private static native String nativeTrain(
            String dataRoot,
            String imageDir,
            String sparseDir,
            String shaderDir,
            String outputPly,
            int frameCount,
            int trainSteps,
            ProgressListener listener);
}
