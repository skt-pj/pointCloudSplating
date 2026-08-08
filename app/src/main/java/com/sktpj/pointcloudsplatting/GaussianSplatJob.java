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

/** Creates the handoff consumed by the future Android-native 3DGS trainer. */
public final class GaussianSplatJob {
    private GaussianSplatJob() {}

    public static final class Result {
        public final boolean success;
        public final String message;
        public final int frameCount;

        private Result(boolean success, String message, int frameCount) {
            this.success = success;
            this.message = message;
            this.frameCount = frameCount;
        }
    }

    public static Result prepare(File datasetDirectory) {
        if (datasetDirectory == null || !datasetDirectory.isDirectory()) {
            return new Result(false, "dataset directory unavailable", 0);
        }
        File transformsFile = new File(datasetDirectory, "transforms.json");
        if (!transformsFile.isFile()) {
            return new Result(false, "transforms.json is missing; press Save first", 0);
        }

        try {
            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frames = transforms.getJSONArray("frames");
            int count = frames.length();
            if (count < 8) {
                return new Result(false,
                        "need at least 8 keyframes before starting 3DGS", count);
            }

            JSONObject job = new JSONObject();
            job.put("format_version", 1);
            job.put("status", "READY_FOR_ANDROID_NATIVE_TRAINER");
            job.put("requested_at_unix_ms", System.currentTimeMillis());
            job.put("frame_count", count);
            job.put("transforms", "transforms.json");
            job.put("rgb_pattern", "frame_*.jpg");
            job.put("depth_prior_pattern", "frame_*.ply");
            job.put("camera_convention", "OpenGL camera-to-world, root-anchor local");
            job.put("backend", "android_native_backend_pending");
            job.put("note",
                    "The capture/preprocessing handoff is ready. The native Vulkan 3DGS optimizer is not bundled in this build yet.");

            try (FileOutputStream out = new FileOutputStream(
                    new File(datasetDirectory, "3dgs_job.json"))) {
                out.write(job.toString(2).getBytes(StandardCharsets.UTF_8));
            }
            return new Result(true,
                    "3DGS input prepared; native trainer integration is next", count);
        } catch (IOException | JSONException | RuntimeException e) {
            DiagnosticLog.e("GaussianSplatJob", "Failed to prepare 3DGS job", e);
            return new Result(false,
                    "3DGS preparation failed: " + e.getClass().getSimpleName(), 0);
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
