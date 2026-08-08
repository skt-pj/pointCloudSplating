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

/** Validates a saved dataset and runs the bundled Android depth-prior 3DGS backend. */
public final class GaussianSplatJob {
    private GaussianSplatJob() {}

    public static final class Result {
        public final boolean success;
        public final String message;
        public final int frameCount;
        public final int gaussianCount;
        public final File outputFile;

        private Result(
                boolean success,
                String message,
                int frameCount,
                int gaussianCount,
                File outputFile) {
            this.success = success;
            this.message = message;
            this.frameCount = frameCount;
            this.gaussianCount = gaussianCount;
            this.outputFile = outputFile;
        }
    }

    public static Result prepare(File datasetDirectory) {
        if (datasetDirectory == null || !datasetDirectory.isDirectory()) {
            return new Result(false, "dataset directory unavailable", 0, 0, null);
        }
        File transformsFile = new File(datasetDirectory, "transforms.json");
        if (!transformsFile.isFile()) {
            return new Result(false, "transforms.json is missing; press Save first", 0, 0, null);
        }

        try {
            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frames = transforms.getJSONArray("frames");
            int count = frames.length();
            if (count == 0) {
                return new Result(false,
                        "no saved keyframes available for 3DGS", 0, 0, null);
            }

            File jobFile = new File(datasetDirectory, "3dgs_job.json");
            JSONObject job = new JSONObject();
            job.put("format_version", 2);
            job.put("status", "RUNNING_ANDROID_DEPTH_PRIOR");
            job.put("requested_at_unix_ms", System.currentTimeMillis());
            job.put("frame_count", count);
            job.put("transforms", "transforms.json");
            job.put("rgb_pattern", "frame_*.jpg");
            job.put("depth_prior_pattern", "frame_*.ply");
            job.put("camera_convention", "OpenGL camera-to-world, root-anchor local");
            job.put("backend", "android_depth_prior_gaussian_v1");
            job.put("photometric_optimization", false);
            job.put("note",
                    "This bundled backend fuses synchronized Raw Depth into a standard 3DGS PLY. "
                            + "A future Vulkan optimizer can replace the backend without changing the dataset format.");
            writeJson(jobFile, job);

            GaussianSplatTrainer.Result training = GaussianSplatTrainer.train(
                    datasetDirectory,
                    (percent, message) -> DiagnosticLog.i(
                            "GaussianSplatJob", percent + "% " + message));
            if (!training.success) {
                job.put("status", "FAILED");
                job.put("error", training.message);
                writeJson(jobFile, job);
                return new Result(false, training.message, count, 0, null);
            }

            job.put("status", "COMPLETE");
            job.put("completed_at_unix_ms", System.currentTimeMillis());
            job.put("gaussian_count", training.gaussianCount);
            job.put("output", training.outputFile.getName());
            writeJson(jobFile, job);
            return new Result(
                    true,
                    "3DGS splat generated: " + training.gaussianCount + " Gaussians",
                    count,
                    training.gaussianCount,
                    training.outputFile);
        } catch (IOException | JSONException | RuntimeException e) {
            DiagnosticLog.e("GaussianSplatJob", "Failed to run 3DGS job", e);
            return new Result(false,
                    "3DGS generation failed: " + e.getClass().getSimpleName(), 0, 0, null);
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
}
