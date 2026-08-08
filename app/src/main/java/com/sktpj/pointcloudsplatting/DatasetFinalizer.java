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
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Finalizes the continuously-spooled capture into a stable dataset for later 3DGS training. */
public final class DatasetFinalizer {
    private DatasetFinalizer() {}

    public static final class Result {
        public final boolean success;
        public final File directory;
        public final int frameCount;
        public final String message;

        private Result(boolean success, File directory, int frameCount, String message) {
            this.success = success;
            this.directory = directory;
            this.frameCount = frameCount;
            this.message = message;
        }

        static Result ok(File directory, int frameCount) {
            return new Result(true, directory, frameCount,
                    "saved " + frameCount + " keyframes");
        }

        static Result fail(File directory, String message) {
            return new Result(false, directory, 0, message);
        }
    }

    /**
     * Writes a Nerfstudio-compatible transforms.json and a small session manifest, then renames
     * capture_tmp_* to dataset_* so the user-visible Save button is the commit point.
     */
    public static Result finalizeDataset(File workingDirectory) {
        if (workingDirectory == null || !workingDirectory.isDirectory()) {
            return Result.fail(workingDirectory, "capture directory unavailable");
        }

        try {
            List<JSONObject> sourceFrames = readFrameMetadata(workingDirectory);
            if (sourceFrames.isEmpty()) {
                return Result.fail(workingDirectory, "no captured keyframes");
            }

            JSONObject transforms = new JSONObject();
            transforms.put("camera_model", "OPENCV");
            transforms.put("k1", 0.0);
            transforms.put("k2", 0.0);
            transforms.put("p1", 0.0);
            transforms.put("p2", 0.0);
            transforms.put("coordinate_system",
                    "ARCore root-anchor local; OpenGL camera convention (+X right,+Y up,-Z forward)");
            transforms.put("source", "pointCloudSplating ARCore SharedCamera");

            JSONArray frames = new JSONArray();
            for (JSONObject source : sourceFrames) {
                frames.put(toNerfstudioFrame(source));
            }
            transforms.put("frames", frames);
            writeJson(new File(workingDirectory, "transforms.json"), transforms);

            JSONObject manifest = new JSONObject();
            manifest.put("format_version", 1);
            manifest.put("state", "saved");
            manifest.put("frame_count", sourceFrames.size());
            manifest.put("transforms", "transforms.json");
            manifest.put("image_pattern", "frame_*.jpg");
            manifest.put("per_frame_metadata_pattern", "frame_*.json");
            manifest.put("raw_depth_point_cloud_pattern", "frame_*.ply");
            manifest.put("capture_strategy",
                    "quality-gated keyframes selected by viewpoint change, focus, exposure and motion");
            manifest.put("note",
                    "Per-frame PLY files are depth priors/initialization data; RGB images and camera poses are the primary 3DGS training observations.");
            writeJson(new File(workingDirectory, "dataset_manifest.json"), manifest);

            try (FileOutputStream out = new FileOutputStream(
                    new File(workingDirectory, ".saved"))) {
                out.write("saved\n".getBytes(StandardCharsets.UTF_8));
            }

            File finalDirectory = renameAsSavedDataset(workingDirectory);
            return Result.ok(finalDirectory, sourceFrames.size());
        } catch (IOException | JSONException | RuntimeException e) {
            DiagnosticLog.e("DatasetFinalizer", "Failed to finalize dataset", e);
            return Result.fail(workingDirectory,
                    "finalize failed: " + e.getClass().getSimpleName());
        }
    }

    private static List<JSONObject> readFrameMetadata(File directory)
            throws IOException, JSONException {
        File[] files = directory.listFiles((dir, name) ->
                name.startsWith("frame_") && name.endsWith(".json"));
        List<JSONObject> frames = new ArrayList<>();
        if (files == null) {
            return frames;
        }
        for (File file : files) {
            frames.add(new JSONObject(readText(file)));
        }
        frames.sort(Comparator.comparingInt(o -> o.optInt("capture_index", Integer.MAX_VALUE)));
        return frames;
    }

    private static JSONObject toNerfstudioFrame(JSONObject source) throws JSONException {
        int jpegWidth = source.getInt("jpeg_width");
        int jpegHeight = source.getInt("jpeg_height");

        JSONObject intrinsics = source.getJSONObject("arcore_image_intrinsics");
        JSONArray focal = intrinsics.getJSONArray("focal_length_px");
        JSONArray principal = intrinsics.getJSONArray("principal_point_px");
        JSONArray dimensions = intrinsics.getJSONArray("image_dimensions");
        double sourceWidth = dimensions.getDouble(0);
        double sourceHeight = dimensions.getDouble(1);
        double scaleX = jpegWidth / sourceWidth;
        double scaleY = jpegHeight / sourceHeight;

        JSONObject out = new JSONObject();
        out.put("file_path", source.getString("image"));
        out.put("depth_point_cloud_path", source.optString("point_cloud", ""));
        out.put("w", jpegWidth);
        out.put("h", jpegHeight);
        out.put("fl_x", focal.getDouble(0) * scaleX);
        out.put("fl_y", focal.getDouble(1) * scaleY);
        out.put("cx", principal.getDouble(0) * scaleX);
        out.put("cy", principal.getDouble(1) * scaleY);
        out.put("capture_index", source.getInt("capture_index"));
        out.put("timestamp_ns", source.getLong("jpeg_sensor_timestamp_ns"));
        out.put("transform_matrix",
                columnMajorToRows(source.getJSONArray("world_from_camera_column_major")));
        return out;
    }

    private static JSONArray columnMajorToRows(JSONArray columnMajor) throws JSONException {
        if (columnMajor.length() != 16) {
            throw new JSONException("Expected 16-value camera transform");
        }
        JSONArray rows = new JSONArray();
        for (int row = 0; row < 4; row++) {
            JSONArray values = new JSONArray();
            for (int col = 0; col < 4; col++) {
                values.put(columnMajor.getDouble(col * 4 + row));
            }
            rows.put(values);
        }
        return rows;
    }

    private static File renameAsSavedDataset(File workingDirectory) {
        String name = workingDirectory.getName();
        if (!name.startsWith("capture_tmp_")) {
            return workingDirectory;
        }
        File parent = workingDirectory.getParentFile();
        if (parent == null) {
            return workingDirectory;
        }
        String suffix = name.substring("capture_tmp_".length());
        File target = new File(parent, "dataset_" + suffix);
        if (target.exists()) {
            target = new File(parent, "dataset_" + suffix + "_saved");
        }
        return workingDirectory.renameTo(target) ? target : workingDirectory;
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

    private static void writeJson(File file, JSONObject json)
            throws IOException, JSONException {
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }
}
