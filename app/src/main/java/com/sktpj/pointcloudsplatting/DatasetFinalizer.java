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

/** Finalizes Phase 1 RGB observations while preserving independent Raw Depth observations. */
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

        static Result ok(File directory, int count) {
            return new Result(true, directory, count, "saved " + count + " keyframes");
        }

        static Result fail(File directory, String message) {
            return new Result(false, directory, 0, message);
        }
    }

    public static Result finalizeDataset(File workingDirectory) {
        if (workingDirectory == null || !workingDirectory.isDirectory()) {
            return Result.fail(workingDirectory, "capture directory unavailable");
        }
        try {
            List<JSONObject> sourceFrames = readFrameMetadata(workingDirectory);
            if (sourceFrames.isEmpty()) {
                return Result.fail(workingDirectory, "no captured keyframes");
            }
            Camera2Calibration legacyCalibration = readCamera2Calibration(workingDirectory);
            int independentDepthObservations = countDepthObservations(workingDirectory);

            JSONObject transforms = new JSONObject();
            transforms.put("camera_model", "OPENCV");
            transforms.put("k1", 0.0);
            transforms.put("k2", 0.0);
            transforms.put("p1", 0.0);
            transforms.put("p2", 0.0);
            transforms.put(
                    "coordinate_system",
                    "ARCore datasetRootAnchor local; OpenGL camera convention (+X right,+Y up,-Z forward)");
            transforms.put("source", "pointCloudSplating Camera2 high-resolution RGB observations");
            transforms.put(
                    "intrinsics_policy",
                    "Prefer per-observation JPEG pixel intrinsics captured from Camera2 calibration/crop; legacy Camera2 mapping and ARCore scaling are compatibility fallbacks");
            transforms.put(
                    "depth_policy",
                    "Independent depth_obs_* observations are geometry prior source-of-truth; frame_*.ply is compatibility-only nearest-depth linkage");

            JSONArray frames = new JSONArray();
            int compatibilityDepthFrames = 0;
            int camera2CalibratedFrames = 0;
            for (JSONObject source : sourceFrames) {
                JSONObject frame = toNerfstudioFrame(source, legacyCalibration);
                frames.put(frame);
                if (frame.has("depth_point_cloud_path")) {
                    compatibilityDepthFrames++;
                }
                if (frame.optString("intrinsics_source", "").startsWith("camera2_")) {
                    camera2CalibratedFrames++;
                }
            }
            transforms.put("frames", frames);
            transforms.put("rgb_frame_count", sourceFrames.size());
            transforms.put("raw_depth_prior_frame_count", compatibilityDepthFrames);
            transforms.put("depth_observation_count", independentDepthObservations);
            transforms.put("camera2_calibrated_rgb_frame_count", camera2CalibratedFrames);
            writeJson(new File(workingDirectory, "transforms.json"), transforms);

            JSONObject manifest = new JSONObject();
            manifest.put("format_version", 5);
            manifest.put("state", "saved");
            manifest.put("phase", "v1_phase1_observation_capture");
            manifest.put("frame_count", sourceFrames.size());
            manifest.put("rgb_frame_count", sourceFrames.size());
            manifest.put("raw_depth_prior_frame_count", compatibilityDepthFrames);
            manifest.put("depth_observation_count", independentDepthObservations);
            manifest.put("camera2_calibrated_rgb_frame_count", camera2CalibratedFrames);
            manifest.put("transforms", "transforms.json");
            manifest.put("image_pattern", "frame_*.jpg");
            manifest.put("per_frame_metadata_pattern", "frame_*.json");
            manifest.put("depth_observation_pattern", "depth_obs_*.json + depth_obs_*.ply");
            manifest.put(
                    "raw_depth_point_cloud_pattern",
                    "frame_*.ply is optional nearest-depth compatibility output; depth_prior_*.ply is legacy trainer fallback");
            manifest.put(
                    "capture_strategy",
                    "Camera2 high-resolution stills selected by viewpoint/motion/focus/exposure; pose correlated by SENSOR_TIMESTAMP to ARCore Android camera timestamps");
            manifest.put(
                    "observation_priority",
                    "High-resolution JPEG + exposure-time pose + JPEG camera model are primary photometric observations. Independent Raw Depth observations are geometry priors and do not gate RGB validity.");
            manifest.put(
                    "intrinsics_priority",
                    "Per-observation jpeg_intrinsics from Camera2 physical calibration/crop, then legacy Camera2 mapping, then ARCore scaled fallback.");
            writeJson(new File(workingDirectory, "dataset_manifest.json"), manifest);

            try (FileOutputStream out = new FileOutputStream(
                    new File(workingDirectory, ".saved"))) {
                out.write("saved\n".getBytes(StandardCharsets.UTF_8));
            }

            DiagnosticLog.i(
                    "DatasetFinalizer",
                    "Finalized Phase1 RGB=" + sourceFrames.size()
                            + " independentDepth=" + independentDepthObservations
                            + " compatibilityDepth=" + compatibilityDepthFrames
                            + " camera2Intrinsics=" + camera2CalibratedFrames);
            File finalDirectory = renameAsSavedDataset(workingDirectory);
            return Result.ok(finalDirectory, sourceFrames.size());
        } catch (IOException | JSONException | RuntimeException e) {
            DiagnosticLog.e("DatasetFinalizer", "Failed to finalize dataset", e);
            return Result.fail(
                    workingDirectory,
                    "finalize failed: " + e.getClass().getSimpleName());
        }
    }

    private static List<JSONObject> readFrameMetadata(File directory)
            throws IOException, JSONException {
        File[] files = directory.listFiles(
                (dir, name) -> name.startsWith("frame_") && name.endsWith(".json"));
        List<JSONObject> frames = new ArrayList<>();
        if (files == null) {
            return frames;
        }
        for (File file : files) {
            frames.add(new JSONObject(readText(file)));
        }
        frames.sort(Comparator.comparingInt(
                object -> object.optInt("capture_index", Integer.MAX_VALUE)));
        return frames;
    }

    private static int countDepthObservations(File directory) {
        File[] files = directory.listFiles(
                (dir, name) -> name.startsWith("depth_obs_") && name.endsWith(".json"));
        return files == null ? 0 : files.length;
    }

    private static Camera2Calibration readCamera2Calibration(File directory) {
        File file = new File(directory, "session_camera.json");
        if (!file.isFile()) {
            return null;
        }
        try {
            JSONObject json = new JSONObject(readText(file));
            JSONArray intrinsics = json.optJSONArray("lens_intrinsic_calibration");
            JSONObject active = json.optJSONObject("active_array");
            if (intrinsics == null || intrinsics.length() < 5 || active == null) {
                return null;
            }
            double fx = intrinsics.getDouble(0);
            double fy = intrinsics.getDouble(1);
            double cx = intrinsics.getDouble(2);
            double cy = intrinsics.getDouble(3);
            double skew = intrinsics.getDouble(4);
            int left = active.getInt("left");
            int top = active.getInt("top");
            int right = active.getInt("right");
            int bottom = active.getInt("bottom");
            if (!finitePositive(fx) || !finitePositive(fy)
                    || right <= left || bottom <= top) {
                return null;
            }
            return new Camera2Calibration(
                    fx, fy, cx, cy, skew, left, top, right, bottom);
        } catch (Exception e) {
            DiagnosticLog.w(
                    "DatasetFinalizer",
                    "Legacy Camera2 calibration unavailable: " + e.getMessage());
            return null;
        }
    }

    private static JSONObject toNerfstudioFrame(
            JSONObject source, Camera2Calibration legacyCalibration)
            throws JSONException {
        int jpegWidth = source.getInt("jpeg_width");
        int jpegHeight = source.getInt("jpeg_height");

        double fx;
        double fy;
        double cx;
        double cy;
        double skew = 0.0;
        String intrinsicsSource;

        JSONObject savedJpegIntrinsics = source.optJSONObject("jpeg_intrinsics");
        if (validSavedJpegIntrinsics(savedJpegIntrinsics)) {
            fx = savedJpegIntrinsics.getDouble("fx");
            fy = savedJpegIntrinsics.getDouble("fy");
            cx = savedJpegIntrinsics.getDouble("cx");
            cy = savedJpegIntrinsics.getDouble("cy");
            skew = savedJpegIntrinsics.optDouble("skew", 0.0);
            intrinsicsSource = savedJpegIntrinsics.optString(
                    "source", "camera2_saved_jpeg_intrinsics");
        } else {
            Camera2MappedIntrinsics mapped = mapCamera2ToJpeg(
                    source, legacyCalibration, jpegWidth, jpegHeight);
            if (mapped != null) {
                fx = mapped.fx;
                fy = mapped.fy;
                cx = mapped.cx;
                cy = mapped.cy;
                skew = mapped.skew;
                intrinsicsSource = "camera2_lens_intrinsic_calibration_legacy_mapping";
            } else {
                JSONObject intrinsics = source.getJSONObject("arcore_image_intrinsics");
                JSONArray focal = intrinsics.getJSONArray("focal_length_px");
                JSONArray principal = intrinsics.getJSONArray("principal_point_px");
                JSONArray dimensions = intrinsics.getJSONArray("image_dimensions");
                double sourceWidth = dimensions.getDouble(0);
                double sourceHeight = dimensions.getDouble(1);
                double scaleX = jpegWidth / sourceWidth;
                double scaleY = jpegHeight / sourceHeight;
                fx = focal.getDouble(0) * scaleX;
                fy = focal.getDouble(1) * scaleY;
                cx = principal.getDouble(0) * scaleX;
                cy = principal.getDouble(1) * scaleY;
                intrinsicsSource = "arcore_image_intrinsics_scaled_fallback";
            }
        }

        JSONObject out = new JSONObject();
        out.put("file_path", source.getString("image"));
        if (!source.isNull("point_cloud")) {
            String pointCloud = source.optString("point_cloud", "");
            if (!pointCloud.isEmpty() && !"null".equals(pointCloud)) {
                out.put("depth_point_cloud_path", pointCloud);
            }
        }
        out.put("w", jpegWidth);
        out.put("h", jpegHeight);
        out.put("fl_x", fx);
        out.put("fl_y", fy);
        out.put("cx", cx);
        out.put("cy", cy);
        out.put("camera2_skew", skew);
        out.put("intrinsics_source", intrinsicsSource);
        out.put("capture_index", source.getInt("capture_index"));
        out.put("timestamp_ns", source.getLong("jpeg_sensor_timestamp_ns"));
        out.put("observation_type", source.optString("observation_type", "rgb"));
        out.put("pose_resolution_method",
                source.optString("pose_resolution_method", "legacy_nearest"));
        out.put("has_raw_depth_prior", source.optBoolean("has_raw_depth_prior", false));
        out.put(
                "transform_matrix",
                columnMajorToRows(source.getJSONArray("world_from_camera_column_major")));
        return out;
    }

    private static boolean validSavedJpegIntrinsics(JSONObject value) {
        if (value == null) {
            return false;
        }
        double fx = value.optDouble("fx", Double.NaN);
        double fy = value.optDouble("fy", Double.NaN);
        double cx = value.optDouble("cx", Double.NaN);
        double cy = value.optDouble("cy", Double.NaN);
        return finitePositive(fx) && finitePositive(fy)
                && Double.isFinite(cx) && Double.isFinite(cy);
    }

    private static Camera2MappedIntrinsics mapCamera2ToJpeg(
            JSONObject source, Camera2Calibration calibration, int width, int height) {
        if (calibration == null) {
            return null;
        }
        try {
            JSONObject capture = source.optJSONObject("camera2_capture");
            if (capture == null) {
                return null;
            }
            int left = calibration.activeLeft;
            int top = calibration.activeTop;
            int right = calibration.activeRight;
            int bottom = calibration.activeBottom;
            JSONObject crop = capture.optJSONObject("crop_region");
            if (crop != null) {
                left = crop.getInt("left");
                top = crop.getInt("top");
                right = crop.getInt("right");
                bottom = crop.getInt("bottom");
            }
            double cropWidth = right - left;
            double cropHeight = bottom - top;
            if (cropWidth <= 0 || cropHeight <= 0) {
                return null;
            }
            double sx = width / cropWidth;
            double sy = height / cropHeight;
            double fx = calibration.fx * sx;
            double fy = calibration.fy * sy;
            double cx = (calibration.cx - left) * sx;
            double cy = (calibration.cy - top) * sy;
            double skew = calibration.skew * sx;
            if (!finitePositive(fx) || !finitePositive(fy)
                    || !Double.isFinite(cx) || !Double.isFinite(cy)) {
                return null;
            }
            if (cx < -width * .25 || cx > width * 1.25
                    || cy < -height * .25 || cy > height * 1.25) {
                return null;
            }
            return new Camera2MappedIntrinsics(fx, fy, cx, cy, skew);
        } catch (Exception e) {
            return null;
        }
    }

    private static boolean finitePositive(double value) {
        return Double.isFinite(value) && value > 0.0;
    }

    private static JSONArray columnMajorToRows(JSONArray columnMajor)
            throws JSONException {
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

    private static final class Camera2Calibration {
        final double fx;
        final double fy;
        final double cx;
        final double cy;
        final double skew;
        final int activeLeft;
        final int activeTop;
        final int activeRight;
        final int activeBottom;

        Camera2Calibration(
                double fx,
                double fy,
                double cx,
                double cy,
                double skew,
                int left,
                int top,
                int right,
                int bottom) {
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
            this.skew = skew;
            activeLeft = left;
            activeTop = top;
            activeRight = right;
            activeBottom = bottom;
        }
    }

    private static final class Camera2MappedIntrinsics {
        final double fx;
        final double fy;
        final double cx;
        final double cy;
        final double skew;

        Camera2MappedIntrinsics(double fx, double fy, double cx, double cy, double skew) {
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
            this.skew = skew;
        }
    }
}
