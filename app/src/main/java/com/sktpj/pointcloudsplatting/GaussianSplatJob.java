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
 * Validates a saved RGB/Pose/Depth dataset and prepares only the geometry prior.
 *
 * <p>Important: a depth-derived Gaussian PLY is NOT considered completed 3DGS. Completed 3DGS
 * requires photometric optimization against the saved high-resolution JPEG observations.</p>
 */
public final class GaussianSplatJob {
    private static final String TAG = "GaussianSplatJob";
    private static final String DEPTH_PRIOR_NAME = "depth_prior.ply";

    private GaussianSplatJob() {}

    public static final class Result {
        /** True only when real RGB photometric 3DGS optimization has completed. */
        public final boolean success;
        /** True when a depth-derived Gaussian initialization artifact is ready. */
        public final boolean priorReady;
        public final String message;
        public final int frameCount;
        public final int gaussianCount;
        public final File outputFile;

        private Result(
                boolean success,
                boolean priorReady,
                String message,
                int frameCount,
                int gaussianCount,
                File outputFile) {
            this.success = success;
            this.priorReady = priorReady;
            this.message = message;
            this.frameCount = frameCount;
            this.gaussianCount = gaussianCount;
            this.outputFile = outputFile;
        }

        private static Result fail(String message, int frameCount) {
            return new Result(false, false, message, frameCount, 0, null);
        }

        private static Result priorReady(
                String message, int frameCount, int gaussianCount, File priorFile) {
            return new Result(false, true, message, frameCount, gaussianCount, priorFile);
        }
    }

    public static Result prepare(File datasetDirectory) {
        if (datasetDirectory == null || !datasetDirectory.isDirectory()) {
            return Result.fail("dataset directory unavailable", 0);
        }
        File transformsFile = new File(datasetDirectory, "transforms.json");
        if (!transformsFile.isFile()) {
            return Result.fail("transforms.json is missing; press Save first", 0);
        }

        try {
            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frames = transforms.getJSONArray("frames");
            int count = frames.length();
            if (count == 0) {
                return Result.fail("no saved keyframes available for 3DGS", 0);
            }

            DatasetStats stats = validatePhotometricDataset(datasetDirectory, frames);
            DiagnosticLog.i(TAG,
                    "Photometric dataset ready frames=" + count
                            + " maxRgb=" + stats.maxWidth + "x" + stats.maxHeight
                            + " pose+intrinsics=" + stats.validCameraFrames + "/" + count);

            File jobFile = new File(datasetDirectory, "3dgs_job.json");
            JSONObject job = new JSONObject();
            job.put("format_version", 3);
            job.put("status", "INITIALIZING_DEPTH_PRIOR");
            job.put("requested_at_unix_ms", System.currentTimeMillis());
            job.put("frame_count", count);
            job.put("transforms", "transforms.json");
            job.put("rgb_pattern", "frame_*.jpg");
            job.put("rgb_role", "photometric_ground_truth");
            job.put("max_rgb_width", stats.maxWidth);
            job.put("max_rgb_height", stats.maxHeight);
            job.put("camera_source", "saved_ARCore_pose_and_intrinsics");
            job.put("depth_prior_pattern", "frame_*.ply");
            job.put("camera_convention", "OpenGL camera-to-world, root-anchor local");
            job.put("target_backend", "android_ndk_vulkan_photometric_3dgs");
            job.put("initializer_backend", "android_depth_prior_gaussian_v1");
            job.put("photometric_optimization", false);
            job.put("final_3dgs", false);
            job.put("note",
                    "Depth-derived Gaussians are initialization only. COMPLETE is reserved for "
                            + "RGB photometric optimization using the saved JPEG/Pose/intrinsics dataset.");
            writeJson(jobFile, job);

            GaussianSplatTrainer.Result initialization = GaussianSplatTrainer.train(
                    datasetDirectory,
                    (percent, message) -> DiagnosticLog.i(
                            TAG, "depth-prior " + percent + "% " + message));
            if (!initialization.success) {
                job.put("status", "FAILED_DEPTH_PRIOR");
                job.put("error", initialization.message);
                writeJson(jobFile, job);
                return Result.fail(initialization.message, count);
            }

            File priorFile = moveInitializerOutput(datasetDirectory, initialization.outputFile);
            rewriteInitializerResult(datasetDirectory, priorFile, initialization.gaussianCount);

            job.put("status", "DEPTH_PRIOR_READY");
            job.put("depth_prior_completed_at_unix_ms", System.currentTimeMillis());
            job.put("gaussian_count", initialization.gaussianCount);
            job.put("depth_prior_output", priorFile.getName());
            job.put("photometric_optimization", false);
            job.put("final_3dgs", false);
            writeJson(jobFile, job);

            String message = "Depth prior初期化完了: " + initialization.gaussianCount
                    + " Gaussians。高解像度JPEGを使うphotometric 3DGS学習は未完了です。";
            DiagnosticLog.i(TAG, message + " output=" + priorFile.getAbsolutePath());
            return Result.priorReady(message, count, initialization.gaussianCount, priorFile);
        } catch (IOException | JSONException | RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to prepare 3DGS dataset", e);
            return Result.fail(
                    "3DGS preparation failed: " + e.getClass().getSimpleName(), 0);
        }
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
            if (width <= 0 || height <= 0) {
                throw new IOException("invalid RGB dimensions: " + imagePath);
            }
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
        if (matrix.length() != 4) {
            throw new IOException("camera transform must be 4x4");
        }
        for (int row = 0; row < 4; row++) {
            JSONArray values = matrix.getJSONArray(row);
            if (values.length() != 4) {
                throw new IOException("camera transform must be 4x4");
            }
            for (int col = 0; col < 4; col++) {
                requireFinite(values.getDouble(col), "transform_matrix");
            }
        }
    }

    private static void requireFinitePositive(double value, String name) throws IOException {
        if (!Double.isFinite(value) || value <= 0.0) {
            throw new IOException("invalid camera " + name);
        }
    }

    private static void requireFinite(double value, String name) throws IOException {
        if (!Double.isFinite(value)) {
            throw new IOException("invalid camera " + name);
        }
    }

    private static File moveInitializerOutput(File datasetDirectory, File generated)
            throws IOException {
        if (generated == null || !generated.isFile()) {
            throw new IOException("depth prior initializer produced no PLY");
        }
        File prior = new File(datasetDirectory, DEPTH_PRIOR_NAME);
        if (prior.exists() && !prior.delete()) {
            throw new IOException("cannot replace existing " + DEPTH_PRIOR_NAME);
        }
        if (!generated.renameTo(prior)) {
            copyFile(generated, prior);
            if (!generated.delete()) {
                DiagnosticLog.w(TAG, "Could not delete temporary initializer output "
                        + generated.getAbsolutePath());
            }
        }
        return prior;
    }

    private static void rewriteInitializerResult(
            File datasetDirectory, File priorFile, int gaussianCount)
            throws IOException, JSONException {
        File resultFile = new File(datasetDirectory, "3dgs_result.json");
        JSONObject result = resultFile.isFile()
                ? new JSONObject(readText(resultFile)) : new JSONObject();
        result.put("format_version", 2);
        result.put("status", "DEPTH_PRIOR_READY");
        result.put("backend", "android_depth_prior_initializer_v2");
        result.put("output", priorFile.getName());
        result.put("gaussian_count", gaussianCount);
        result.put("photometric_optimization", false);
        result.put("final_3dgs", false);
        result.put("note",
                "This artifact is geometry initialization only. It is not a completed 3DGS model.");
        writeJson(resultFile, result);
    }

    private static void copyFile(File source, File destination) throws IOException {
        try (FileInputStream in = new FileInputStream(source);
             FileOutputStream out = new FileOutputStream(destination)) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
    }

    private static void writeJson(File file, JSONObject json)
            throws IOException, JSONException {
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    private static String readText(File file) throws IOException {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                out.append(line).append('\n');
            }
        }
        return out.toString();
    }

    private static final class DatasetStats {
        final int maxWidth;
        final int maxHeight;
        final int validCameraFrames;

        DatasetStats(int maxWidth, int maxHeight, int validCameraFrames) {
            this.maxWidth = maxWidth;
            this.maxHeight = maxHeight;
            this.validCameraFrames = validCameraFrames;
        }
    }
}
