package com.sktpj.pointcloudsplatting;

import android.content.Context;
import android.content.res.AssetManager;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/** JNI bridge for the actual Vulkan differentiable 3D Gaussian Splatting trainer. */
public final class NativeGaussianTrainer {
    private static final String TAG = "Native3DGS";
    private static final String ASSET_ROOT = "vksplat_shader";
    private static final String SHADER_CACHE = "vksplat_shader_41cff93b";
    private static final int DEFAULT_TRAIN_STEPS = 6_000;

    private static final boolean NATIVE_AVAILABLE;
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
        public final int steps;
        public final double validationPsnr;
        public final String device;

        private Result(boolean success, String message, File outputFile, int gaussianCount,
                int steps, double validationPsnr, String device) {
            this.success = success;
            this.message = message;
            this.outputFile = outputFile;
            this.gaussianCount = gaussianCount;
            this.steps = steps;
            this.validationPsnr = validationPsnr;
            this.device = device;
        }

        static Result fail(String message) {
            return new Result(false, message, null, 0, 0, Double.NaN, "");
        }
    }

    public static Result train(Context context, File dataset, ProgressListener listener) {
        if (!NATIVE_AVAILABLE) return Result.fail("端末内3DGS学習ライブラリを開始できませんでした");
        if (context == null || dataset == null || !dataset.isDirectory()) {
            return Result.fail("撮影データが見つかりませんでした");
        }

        try {
            notifyProgress(listener, 1, "写真とカメラ位置を学習用に準備しています…");
            ColmapDatasetExporter.Result prepared = ColmapDatasetExporter.prepare(
                    dataset,
                    (percent, message) -> notifyProgress(listener,
                            Math.min(15, Math.max(1, Math.round(percent * 0.33f))), message));
            if (!prepared.success) return Result.fail(prepared.message);

            File shaderDir = ensureShaderFiles(context);
            File output = new File(dataset, "splat.ply");
            File working = prepared.root;
            int steps = trainingSteps(prepared.frameCount, prepared.initialPointCount);
            notifyProgress(listener, 16, "写真と3Dの形を比較して学習を始めます…");

            String raw = nativeTrain(
                    working.getAbsolutePath(),
                    ensureTrailingSlash(prepared.imageDir.getAbsolutePath()),
                    ensureTrailingSlash(prepared.sparseDir.getAbsolutePath()),
                    ensureTrailingSlash(shaderDir.getAbsolutePath()),
                    output.getAbsolutePath(),
                    prepared.frameCount,
                    steps,
                    (percent, message) -> notifyProgress(listener,
                            16 + Math.round(percent * 0.82f), message));
            JSONObject json = new JSONObject(raw);
            boolean success = json.optBoolean("success", false);
            String message = json.optString("message", success ? "3DGS学習完了" : "3DGS学習に失敗しました");
            if (!success || !output.isFile() || output.length() == 0L) {
                DiagnosticLog.w(TAG, "Native training failed result=" + raw);
                return Result.fail(message);
            }

            int gaussians = json.optInt("gaussian_count", 0);
            int completedSteps = json.optInt("steps", steps);
            double psnr = json.optDouble("validation_psnr", Double.NaN);
            String device = json.optString("device", "Vulkan");
            writeFinalResult(dataset, gaussians, completedSteps, psnr, device, prepared);
            DiagnosticLog.i(TAG, "Full 3DGS COMPLETE gaussians=" + gaussians
                    + " steps=" + completedSteps + " validationPsnr=" + psnr
                    + " device=" + device + " output=" + output.getAbsolutePath());
            notifyProgress(listener, 100, "3Dモデルを作成しました");
            return new Result(true, "3DGS学習完了", output, gaussians,
                    completedSteps, psnr, device);
        } catch (Throwable error) {
            DiagnosticLog.e(TAG, "Native 3DGS training failed", error);
            return Result.fail("端末内3DGS学習に失敗しました: " + error.getClass().getSimpleName());
        }
    }

    private static int trainingSteps(int frames, int points) {
        // Keep the full differentiable optimization long enough to reach SH3 and run multiple
        // densification/pruning cycles, while avoiding an unbounded thermal workload on a phone.
        int base = DEFAULT_TRAIN_STEPS;
        if (frames >= 24) base += 1_500;
        if (points >= 120_000) base += 1_000;
        return Math.min(9_000, base);
    }

    private static void writeFinalResult(File dataset, int gaussians, int steps, double psnr,
            String device, ColmapDatasetExporter.Result prepared) throws Exception {
        JSONObject result = new JSONObject();
        result.put("format_version", 6);
        result.put("status", "COMPLETE");
        result.put("backend", "android_vksplat_vulkan_compute");
        result.put("source_backend", "VkSplat@41cff93b79145dec314488d4313bc3a6d737038b");
        result.put("output", "splat.ply");
        result.put("frame_count", prepared.frameCount);
        result.put("initial_point_count", prepared.initialPointCount);
        result.put("gaussian_count", gaussians);
        result.put("training_steps", steps);
        result.put("training_resolution_scale", 0.25);
        result.put("loss", "L1 + SSIM");
        result.put("optimized_parameters", "position, quaternion, scale, opacity, SH0-SH3");
        result.put("density_control", true);
        result.put("densification_pruning", true);
        result.put("rasterized_image_loss", true);
        result.put("l1_ssim_backward", true);
        result.put("photometric_optimization", true);
        result.put("final_3dgs", true);
        result.put("vulkan_device", device);
        if (Double.isFinite(psnr)) result.put("validation_psnr", psnr);
        result.put("completed_at_unix_ms", System.currentTimeMillis());
        try (FileOutputStream out = new FileOutputStream(new File(dataset, "3dgs_result.json"))) {
            out.write(result.toString(2).getBytes(StandardCharsets.UTF_8));
        }

        File jobFile = new File(dataset, "3dgs_job.json");
        JSONObject job = new JSONObject();
        job.put("format_version", 7);
        job.put("status", "COMPLETE");
        job.put("backend", "android_vksplat_vulkan_compute");
        job.put("final_output", "splat.ply");
        job.put("final_3dgs", true);
        job.put("photometric_optimization", true);
        job.put("rasterized_image_loss", true);
        job.put("l1_ssim_backward", true);
        job.put("density_control", true);
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
