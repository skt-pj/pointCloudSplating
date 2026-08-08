package com.sktpj.pointcloudsplatting;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Validates a saved RGB/Pose dataset, prepares an optional Raw Depth geometry prior, and runs the
 * phone-side high-resolution multi-view appearance refinement. Full final 3DGS is deliberately
 * reserved for a differentiable rasterized L1+SSIM optimizer with density control.
 */
public final class GaussianSplatJob {
    private static final String TAG = "GaussianSplatJob";
    private static final String DEPTH_PRIOR_NAME = "depth_prior.ply";

    private GaussianSplatJob() {}

    public interface ProgressListener {
        void onProgress(int percent, String message);
    }

    public static final class Result {
        /** True only when full differentiable 3DGS optimization has completed. */
        public final boolean success;
        /** True when a depth-derived geometry prior exists. */
        public final boolean priorReady;
        /** True when high-resolution JPEG/Pose/intrinsics appearance refinement exists. */
        public final boolean hqReady;
        public final String message;
        public final int frameCount;
        public final int gaussianCount;
        public final File outputFile;

        private Result(boolean success, boolean priorReady, boolean hqReady, String message,
                int frameCount, int gaussianCount, File outputFile) {
            this.success = success;
            this.priorReady = priorReady;
            this.hqReady = hqReady;
            this.message = message;
            this.frameCount = frameCount;
            this.gaussianCount = gaussianCount;
            this.outputFile = outputFile;
        }

        private static Result fail(String message, int frameCount) {
            return new Result(false, false, false, message, frameCount, 0, null);
        }

        private static Result priorReady(String message, int frameCount, int gaussianCount,
                File priorFile) {
            return new Result(false, true, false, message, frameCount, gaussianCount, priorFile);
        }

        private static Result hqReady(String message, int frameCount, int gaussianCount,
                File outputFile) {
            return new Result(false, true, true, message, frameCount, gaussianCount, outputFile);
        }
    }

    public static Result prepare(File datasetDirectory) {
        return prepare(datasetDirectory, null);
    }

    public static Result prepare(File datasetDirectory, ProgressListener listener) {
        notifyProgress(listener, 2, "撮影データを確認しています…");
        if (datasetDirectory == null || !datasetDirectory.isDirectory()) {
            return Result.fail("撮影データが見つかりませんでした。", 0);
        }
        File transformsFile = new File(datasetDirectory, "transforms.json");
        if (!transformsFile.isFile()) {
            return Result.fail("撮影データの保存が完了していません。", 0);
        }

        try {
            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frames = transforms.getJSONArray("frames");
            int count = frames.length();
            if (count == 0) {
                return Result.fail("保存できた写真がありません。", 0);
            }

            DatasetStats stats = validatePhotometricDataset(datasetDirectory, frames);
            DiagnosticLog.i(TAG,
                    "RGB dataset ready frames=" + count
                            + " maxRgb=" + stats.maxWidth + "x" + stats.maxHeight
                            + " pose+intrinsics=" + stats.validCameraFrames + "/" + count);

            File jobFile = new File(datasetDirectory, "3dgs_job.json");
            JSONObject job = new JSONObject();
            job.put("format_version", 5);
            job.put("status", "PREPARING_GEOMETRY");
            job.put("requested_at_unix_ms", System.currentTimeMillis());
            job.put("frame_count", count);
            job.put("transforms", "transforms.json");
            job.put("rgb_pattern", "frame_*.jpg");
            job.put("rgb_role", "high_resolution_primary_observation");
            job.put("max_rgb_width", stats.maxWidth);
            job.put("max_rgb_height", stats.maxHeight);
            job.put("camera_source", "saved_ARCore_pose_and_intrinsics");
            job.put("depth_prior_pattern", "frame_*.ply_optional");
            job.put("camera_convention", "OpenGL camera-to-world, root-anchor local");
            job.put("target_backend", "android_ndk_vulkan_differentiable_3dgs");
            job.put("initializer_backend", "android_dense_depth_prior_v3");
            job.put("hq_preview_backend", "android_highres_multiview_gaussian_v1");
            job.put("appearance_refinement", false);
            job.put("photometric_optimization", false);
            job.put("rasterized_image_loss", false);
            job.put("l1_ssim_backward", false);
            job.put("density_control", false);
            job.put("final_3dgs", false);
            job.put("note",
                    "COMPLETE is reserved for differentiable rasterized L1+SSIM optimization. "
                            + "The current Java stage is an RGB appearance refinement preview, not training completion.");
            writeJson(jobFile, job);

            File priorFile = new File(datasetDirectory, DEPTH_PRIOR_NAME);
            int priorCount = 0;
            if (!priorFile.isFile()) {
                notifyProgress(listener, 8, "3Dの形を準備しています…");
                job.put("status", "INITIALIZING_DEPTH_PRIOR");
                writeJson(jobFile, job);
                GaussianSplatTrainer.Result initialization = GaussianSplatTrainer.train(
                        datasetDirectory,
                        (percent, message) -> {
                            DiagnosticLog.i(TAG, "depth-prior " + percent + "% " + message);
                            int mapped = 8 + Math.round(percent * 0.27f);
                            notifyProgress(listener, mapped, "3Dの形を準備しています…");
                        });
                if (!initialization.success) {
                    job.put("status", "FAILED_DEPTH_PRIOR");
                    job.put("error", initialization.message);
                    writeJson(jobFile, job);
                    return Result.fail("3Dの形を作るための情報を準備できませんでした。", count);
                }
                priorFile = moveInitializerOutput(datasetDirectory, initialization.outputFile);
                priorCount = initialization.gaussianCount;
                rewriteInitializerResult(datasetDirectory, priorFile, priorCount);
            } else {
                JSONObject existing = readResult(datasetDirectory);
                priorCount = existing == null ? 0 : existing.optInt("gaussian_count", 0);
                DiagnosticLog.i(TAG, "Reusing depth prior " + priorFile.getAbsolutePath());
            }

            notifyProgress(listener, 36, "写真から色と質感を読み取っています…");
            job.put("status", "REFINING_HIGH_RES_RGB");
            job.put("depth_prior_output", priorFile.getName());
            job.put("depth_prior_gaussian_count", priorCount);
            writeJson(jobFile, job);

            HighQualityGaussianTrainer.Result hq = HighQualityGaussianTrainer.train(
                    datasetDirectory,
                    (percent, message) -> {
                        DiagnosticLog.i(TAG, "hq " + percent + "% " + message);
                        int mapped = 36 + Math.round(percent * 0.62f);
                        notifyProgress(listener, mapped, userStageMessage(percent));
                    });
            if (!hq.success) {
                job.put("status", "DEPTH_PRIOR_READY");
                job.put("hq_error", hq.message);
                job.put("appearance_refinement", false);
                job.put("photometric_optimization", false);
                job.put("final_3dgs", false);
                writeJson(jobFile, job);
                DiagnosticLog.w(TAG, "RGB appearance refinement failed after depth prior: " + hq.message);
                return Result.priorReady(
                        "3Dモデルの色と形を仕上げられませんでした。もう一度お試しください。",
                        count, priorCount, priorFile);
            }

            // Important semantic boundary: weighted SH fitting consumes the JPEGs, but it is not
            // differentiable rasterized 3DGS training. Keep this explicit in both job/result files.
            normalizeAppearanceResult(datasetDirectory, hq);
            job.put("status", "HQ_RGB_REFINED");
            job.put("hq_completed_at_unix_ms", System.currentTimeMillis());
            job.put("gaussian_count", hq.gaussianCount);
            job.put("textured_gaussian_count", hq.texturedGaussianCount);
            job.put("hq_output", hq.outputFile.getName());
            job.put("appearance_refinement", true);
            job.put("appearance_fit", "weighted_multiview_SH1_least_squares");
            job.put("photometric_optimization", false);
            job.put("rasterized_image_loss", false);
            job.put("l1_ssim_backward", false);
            job.put("density_control", false);
            job.put("final_3dgs", false);
            if (Double.isFinite(hq.photometricRmse)) {
                job.put("appearance_rmse", hq.photometricRmse);
            }
            writeJson(jobFile, job);

            DiagnosticLog.i(TAG,
                    "RGB appearance refinement ready: " + hq.gaussianCount
                            + " gaussians / " + hq.texturedGaussianCount
                            + " JPEG-observed / appearanceRmse=" + hq.photometricRmse
                            + " full3dgs=false output=" + hq.outputFile.getAbsolutePath());
            notifyProgress(listener, 100, "表示用の3Dモデルを準備しました");
            return Result.hqReady(
                    "表示用の3Dモデルを準備しました。",
                    count, hq.gaussianCount, hq.outputFile);
        } catch (IOException | JSONException | RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to prepare/refine 3DGS dataset", e);
            return Result.fail("3Dモデルを作成できませんでした。", 0);
        }
    }

    private static void normalizeAppearanceResult(
            File datasetDirectory, HighQualityGaussianTrainer.Result hq)
            throws IOException, JSONException {
        File resultFile = new File(datasetDirectory, "3dgs_result.json");
        JSONObject result = resultFile.isFile()
                ? new JSONObject(readText(resultFile)) : new JSONObject();
        result.put("format_version", 4);
        result.put("status", "HQ_RGB_REFINED");
        result.put("backend", "android_highres_multiview_gaussian_v1");
        result.put("output", hq.outputFile.getName());
        result.put("gaussian_count", hq.gaussianCount);
        result.put("textured_gaussian_count", hq.texturedGaussianCount);
        result.put("appearance_refinement", true);
        result.put("appearance_fit", "weighted_multiview_SH1_least_squares");
        if (Double.isFinite(hq.photometricRmse)) {
            result.put("appearance_rmse", hq.photometricRmse);
        }
        // Deliberately overwrite the old misleading flag emitted by the legacy Java refiner.
        result.put("photometric_optimization", false);
        result.put("rasterized_image_loss", false);
        result.put("l1_ssim_backward", false);
        result.put("density_control", false);
        result.put("final_3dgs", false);
        result.put("note",
                "High-resolution JPEGs were sampled and fitted to SH1 appearance. "
                        + "This is a display-quality preview, not differentiable 3DGS training completion.");
        writeJson(resultFile, result);
    }

    private static String userStageMessage(int trainerPercent) {
        if (trainerPercent < 15) return "3Dの形を整えています…";
        if (trainerPercent < 70) return "写真の色を3Dの形に合わせています…";
        if (trainerPercent < 90) return "複数の角度の写真をまとめています…";
        return "表示用の3Dモデルを仕上げています…";
    }

    private static void notifyProgress(ProgressListener listener, int percent, String message) {
        if (listener != null) listener.onProgress(Math.max(0, Math.min(100, percent)), message);
    }

    private static DatasetStats validatePhotometricDataset(
            File datasetDirectory, JSONArray frames) throws JSONException, IOException {
        int maxWidth = 0;
        int maxHeight = 0;
        int validCameraFrames = 0;
        for (int i = 0; i < frames.length(); i++) {
            JSONObject frame = frames.getJSONObject(i);
            String imagePath = frame.getString("file_path");
            File image = new File(datasetDirectory, imagePath);
            if (!image.isFile() || image.length() == 0L) {
                throw new IOException("missing RGB observation: " + imagePath);
            }
            int width = frame.getInt("w");
            int height = frame.getInt("h");
            if (width <= 0 || height <= 0) throw new IOException("invalid RGB dimensions: " + imagePath);
            requireFinitePositive(frame.getDouble("fl_x"), "fl_x");
            requireFinitePositive(frame.getDouble("fl_y"), "fl_y");
            requireFinite(frame.getDouble("cx"), "cx");
            requireFinite(frame.getDouble("cy"), "cy");
            validateTransform(frame.getJSONArray("transform_matrix"));
            maxWidth = Math.max(maxWidth, width);
            maxHeight = Math.max(maxHeight, height);
            validCameraFrames++;
        }
        return new DatasetStats(maxWidth, maxHeight, validCameraFrames);
    }

    private static void validateTransform(JSONArray matrix) throws JSONException, IOException {
        if (matrix.length() != 4) throw new IOException("camera transform must be 4x4");
        for (int row = 0; row < 4; row++) {
            JSONArray values = matrix.getJSONArray(row);
            if (values.length() != 4) throw new IOException("camera transform must be 4x4");
            for (int col = 0; col < 4; col++) requireFinite(values.getDouble(col), "transform_matrix");
        }
    }

    private static void requireFinitePositive(double value, String name) throws IOException {
        if (!Double.isFinite(value) || value <= 0.0) throw new IOException("invalid camera " + name);
    }

    private static void requireFinite(double value, String name) throws IOException {
        if (!Double.isFinite(value)) throw new IOException("invalid camera " + name);
    }

    private static File moveInitializerOutput(File datasetDirectory, File generated) throws IOException {
        if (generated == null || !generated.isFile()) throw new IOException("depth prior initializer produced no PLY");
        File prior = new File(datasetDirectory, DEPTH_PRIOR_NAME);
        if (prior.exists() && !prior.delete()) throw new IOException("cannot replace existing " + DEPTH_PRIOR_NAME);
        if (!generated.renameTo(prior)) {
            copyFile(generated, prior);
            if (!generated.delete()) DiagnosticLog.w(TAG, "Could not delete temporary initializer output " + generated.getAbsolutePath());
        }
        return prior;
    }

    private static void rewriteInitializerResult(File datasetDirectory, File priorFile, int gaussianCount)
            throws IOException, JSONException {
        File resultFile = new File(datasetDirectory, "3dgs_result.json");
        JSONObject result = resultFile.isFile() ? new JSONObject(readText(resultFile)) : new JSONObject();
        result.put("format_version", 4);
        result.put("status", "DEPTH_PRIOR_READY");
        result.put("backend", "android_dense_depth_prior_v3");
        result.put("output", priorFile.getName());
        result.put("gaussian_count", gaussianCount);
        result.put("appearance_refinement", false);
        result.put("photometric_optimization", false);
        result.put("rasterized_image_loss", false);
        result.put("l1_ssim_backward", false);
        result.put("density_control", false);
        result.put("final_3dgs", false);
        result.put("note", "This artifact is geometry initialization only. It is not a completed 3DGS model.");
        writeJson(resultFile, result);
    }

    private static JSONObject readResult(File datasetDirectory) {
        File file = new File(datasetDirectory, "3dgs_result.json");
        if (!file.isFile()) return null;
        try { return new JSONObject(readText(file)); } catch (Exception e) { return null; }
    }

    private static void copyFile(File source, File destination) throws IOException {
        try (FileInputStream in = new FileInputStream(source); FileOutputStream out = new FileOutputStream(destination)) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = in.read(buffer)) != -1) out.write(buffer, 0, read);
        }
    }

    private static void writeJson(File file, JSONObject json) throws IOException, JSONException {
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    private static String readText(File file) throws IOException {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) out.append(line).append('\n');
        }
        return out.toString();
    }

    private static final class DatasetStats {
        final int maxWidth, maxHeight, validCameraFrames;
        DatasetStats(int maxWidth, int maxHeight, int validCameraFrames) {
            this.maxWidth = maxWidth;
            this.maxHeight = maxHeight;
            this.validCameraFrames = validCameraFrames;
        }
    }
}
