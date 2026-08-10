package com.sktpj.pointcloudsplatting;

import android.graphics.BitmapFactory;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Phase 1 hard-gate evaluator.
 *
 * <p>This class does not train, build a geometry prior, or judge visual 3DGS quality. It verifies
 * that the captured RGB and Raw Depth observations are structurally and numerically trustworthy
 * enough to hand to Phase 2. Every saved RGB/Depth observation is checked; sampling is not used.
 */
public final class Phase1DatasetEvaluator {
    private static final String TAG = "Phase1Eval";
    private static final long MAX_POSE_DELTA_NS = 75_000_000L;
    private static final long MIN_HIGH_RES_PIXELS = 8_000_000L;
    private static final int PLY_BYTES_PER_POINT = 3 * Float.BYTES + 3 + Float.BYTES;
    private static final double RIGID_TOLERANCE = 0.03;
    private static final double INVERSE_TOLERANCE = 0.002;

    private static final String[] FATAL_SESSION_PATTERNS = {
            "OpenGL/ARCore frame failed; frame loop latched",
            "Shared camera error=",
            "Shared camera disconnected id=",
            "Capture session configuration failed",
            "Failed to start SharedCamera repeating request",
            "Failed to initialize OpenGL resources",
            "Failed to resume ARCore",
            "ARCore resume failed"
    };

    private static final String[] OBSERVATION_WRITE_FAILURE_PATTERNS = {
            "Failed to write RGB observation",
            "Failed to write independent Raw Depth observation"
    };

    private Phase1DatasetEvaluator() {}

    public static final class Result {
        public final boolean passed;
        public final int failureCount;
        public final int warningCount;
        public final File reportFile;
        public final JSONObject report;

        Result(boolean passed, int failureCount, int warningCount, File reportFile, JSONObject report) {
            this.passed = passed;
            this.failureCount = failureCount;
            this.warningCount = warningCount;
            this.reportFile = reportFile;
            this.report = report;
        }
    }

    /** Evaluates all Phase 1 artifacts and always attempts to write phase1_evaluation.json. */
    public static Result evaluate(File dataset) {
        File reportFile = dataset == null ? null : new File(dataset, "phase1_evaluation.json");
        Evaluation state = new Evaluation(dataset);
        try {
            state.evaluateRgb();
            state.evaluateDepth();
            state.evaluateCaptureHealth();
        } catch (Throwable error) {
            state.fail("evaluator_exception", error.getClass().getSimpleName() + ": "
                    + String.valueOf(error.getMessage()));
            DiagnosticLog.e(TAG, "Phase 1 evaluator failed", error);
        }

        JSONObject report;
        try {
            report = state.toJson();
        } catch (JSONException jsonError) {
            DiagnosticLog.e(TAG, "Could not serialize Phase 1 evaluation", jsonError);
            report = new JSONObject();
            try {
                report.put("format_version", 1);
                report.put("phase", "phase1");
                report.put("status", "FAIL");
                report.put("pass", false);
                report.put("failure_count", 1);
                report.put("serialization_error", jsonError.toString());
            } catch (JSONException ignored) {
            }
        }

        if (reportFile != null) {
            try (FileOutputStream out = new FileOutputStream(reportFile)) {
                out.write(report.toString(2).getBytes(StandardCharsets.UTF_8));
            } catch (IOException | JSONException error) {
                DiagnosticLog.e(TAG, "Could not write phase1_evaluation.json", error);
                state.fail("evaluation_report_write_failed", error.toString());
            }
        }

        boolean passed = state.failures.isEmpty();
        String status = passed ? "PASS" : "FAIL";
        DiagnosticLog.i(TAG,
                String.format(Locale.US,
                        "PHASE1_EVAL %s rgb=%d/%d depth=%d/%d poseMaxMs=%.3f "
                                + "interp=%d nearestFallback=%d camera2Intrinsics=%d "
                                + "invalidPose=%d invalidPoint=%d invalidConfidence=%d fatal=%d writeErrors=%d failures=%d warnings=%d",
                        status,
                        state.rgbValidCount,
                        state.rgbCount,
                        state.depthValidCount,
                        state.depthCount,
                        state.maxPoseDeltaNs / 1_000_000.0,
                        state.poseInterpolatedCount,
                        state.poseNearestFallbackCount,
                        state.camera2IntrinsicsCount,
                        state.invalidPoseCount,
                        state.invalidPointCount,
                        state.invalidConfidenceCount,
                        state.fatalSessionErrorCount,
                        state.observationWriteErrorCount,
                        state.failures.size(),
                        state.warnings.size()));
        return new Result(
                passed,
                state.failures.size(),
                state.warnings.size(),
                reportFile,
                report);
    }

    /** Future Phase 2 entry points can use this without reinterpreting the report. */
    public static boolean hasStoredPass(File dataset) {
        if (dataset == null) return false;
        File file = new File(dataset, "phase1_evaluation.json");
        if (!file.isFile()) return false;
        try {
            JSONObject json = new JSONObject(readText(file));
            return json.optBoolean("pass", false) && "PASS".equals(json.optString("status", ""));
        } catch (Exception ignored) {
            return false;
        }
    }

    private static final class Evaluation {
        final File dataset;
        final List<JSONObject> failures = new ArrayList<>();
        final List<JSONObject> warnings = new ArrayList<>();
        final Set<String> rgbIds = new HashSet<>();
        final Set<Long> rgbTimestamps = new HashSet<>();
        final Set<String> depthIds = new HashSet<>();
        final Set<Long> depthTimestamps = new HashSet<>();
        final List<Long> poseDeltasNs = new ArrayList<>();
        final List<String> fatalEvents = new ArrayList<>();
        final List<String> writeErrorEvents = new ArrayList<>();

        int rgbCount;
        int rgbValidCount;
        int rgbWithoutDepthReferenceCount;
        int camera2IntrinsicsCount;
        int fallbackIntrinsicsCount;
        int unavailableIntrinsicsCount;
        int poseInterpolatedCount;
        int poseNearestFallbackCount;
        int poseExactOrSingleCount;
        int invalidPoseCount;
        long maxPoseDeltaNs;
        long minJpegPixels = Long.MAX_VALUE;
        long maxJpegPixels;

        int depthCount;
        int depthValidCount;
        int depthMissingAndroidCameraTimestampCount;
        long depthPointTotal;
        int minDepthPoints = Integer.MAX_VALUE;
        int maxDepthPoints;
        long invalidPointCount;
        long invalidConfidenceCount;
        float minConfidence = Float.POSITIVE_INFINITY;
        float maxConfidence = Float.NEGATIVE_INFINITY;

        boolean diagnosticSessionWindowFound;
        int fatalSessionErrorCount;
        int observationWriteErrorCount;

        Evaluation(File dataset) {
            this.dataset = dataset;
            if (dataset == null || !dataset.isDirectory()) {
                fail("dataset_unavailable", "dataset directory is unavailable");
            }
        }

        void evaluateRgb() throws IOException {
            if (dataset == null || !dataset.isDirectory()) return;
            File[] files = dataset.listFiles(
                    (dir, name) -> name.startsWith("frame_") && name.endsWith(".json"));
            if (files == null) files = new File[0];
            Arrays.sort(files, (a, b) -> a.getName().compareTo(b.getName()));
            rgbCount = files.length;
            if (rgbCount == 0) {
                fail("rgb_observations_missing", "no frame_*.json RGB observations");
                return;
            }

            for (File metadataFile : files) {
                int beforeFailures = failures.size();
                JSONObject json;
                try {
                    json = new JSONObject(readText(metadataFile));
                } catch (Exception error) {
                    fail("rgb_metadata_invalid_json", metadataFile.getName() + ": " + error);
                    continue;
                }
                evaluateOneRgb(metadataFile, json);
                if (failures.size() == beforeFailures) rgbValidCount++;
            }
        }

        private void evaluateOneRgb(File metadataFile, JSONObject json) {
            String label = metadataFile.getName();
            try {
                require("rgb".equals(json.optString("observation_type", "")),
                        "rgb_observation_type", label + " observation_type must be rgb");
                require("photometric_ground_truth".equals(json.optString("observation_role", "")),
                        "rgb_observation_role", label + " is not photometric_ground_truth");
                require("camera2_high_resolution_jpeg".equals(json.optString("source", "")),
                        "rgb_source", label + " is not Camera2 high-resolution JPEG");

                String id = json.optString("rgb_observation_id", "");
                require(!id.isEmpty(), "rgb_id_missing", label + " has no rgb_observation_id");
                if (!id.isEmpty()) {
                    require(rgbIds.add(id), "rgb_id_duplicate", "duplicate RGB id " + id);
                }

                long timestamp = json.optLong("jpeg_sensor_timestamp_ns", -1L);
                require(timestamp > 0L, "rgb_timestamp_missing", label + " has no SENSOR_TIMESTAMP");
                if (timestamp > 0L) {
                    require(rgbTimestamps.add(timestamp), "rgb_timestamp_duplicate",
                            "duplicate RGB SENSOR_TIMESTAMP " + timestamp);
                }

                int width = json.optInt("jpeg_width", -1);
                int height = json.optInt("jpeg_height", -1);
                require(width > 0 && height > 0, "jpeg_dimensions_missing",
                        label + " has invalid JPEG dimensions " + width + "x" + height);

                String imageName = json.optString("image", "");
                File image = new File(dataset, imageName);
                require(!imageName.isEmpty() && image.isFile() && image.length() > 0L,
                        "jpeg_missing", label + " JPEG file missing: " + imageName);
                if (image.isFile() && width > 0 && height > 0) {
                    BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inJustDecodeBounds = true;
                    BitmapFactory.decodeFile(image.getAbsolutePath(), options);
                    require(options.outWidth > 0 && options.outHeight > 0,
                            "jpeg_decode_failed", image.getName() + " cannot be decoded as an image");
                    require(options.outWidth == width && options.outHeight == height,
                            "jpeg_dimension_mismatch",
                            image.getName() + " metadata=" + width + "x" + height
                                    + " decoded=" + options.outWidth + "x" + options.outHeight);
                    long pixels = (long) options.outWidth * options.outHeight;
                    minJpegPixels = Math.min(minJpegPixels, pixels);
                    maxJpegPixels = Math.max(maxJpegPixels, pixels);
                    require(pixels >= MIN_HIGH_RES_PIXELS,
                            "jpeg_not_high_resolution",
                            image.getName() + " is only " + pixels + " pixels; minimum="
                                    + MIN_HIGH_RES_PIXELS);
                }

                evaluateRgbPose(label, json, timestamp);
                evaluateRgbIntrinsics(label, json, width, height);
                evaluateCamera2Metadata(label, json);

                if (json.isNull("nearest_depth_observation_id")) {
                    rgbWithoutDepthReferenceCount++;
                }
                JSONObject gate = json.optJSONObject("capture_quality_gate");
                require(gate != null && !gate.optBoolean(
                                "raw_depth_required_for_rgb_observation", true),
                        "rgb_depth_independence_contract",
                        label + " does not prove Raw Depth is optional for RGB validity");
            } catch (RuntimeException error) {
                fail("rgb_evaluation_exception", label + ": " + error);
            }
        }

        private void evaluateRgbPose(String label, JSONObject json, long timestamp) {
            int before = failures.size();
            String domain = json.optString("pose_timestamp_domain", "");
            require("android_camera_sensor_timestamp".equals(domain),
                    "pose_timestamp_domain", label + " uses timestamp domain " + domain);

            String method = json.optString("pose_resolution_method", "");
            long delta = json.optLong("pose_timestamp_delta_ns", Long.MIN_VALUE);
            if (delta == Long.MIN_VALUE) {
                fail("pose_delta_missing", label + " has no pose_timestamp_delta_ns");
            } else {
                long abs = safeAbs(delta);
                poseDeltasNs.add(abs);
                maxPoseDeltaNs = Math.max(maxPoseDeltaNs, abs);
                require(abs <= MAX_POSE_DELTA_NS,
                        "pose_delta_too_large",
                        label + " pose delta=" + abs + "ns > " + MAX_POSE_DELTA_NS + "ns");
            }

            if ("bracketed_pose_interpolation".equals(method)) {
                poseInterpolatedCount++;
                long beforeTs = json.optLong("pose_before_timestamp_ns", -1L);
                long afterTs = json.optLong("pose_after_timestamp_ns", -1L);
                double t = json.optDouble("pose_interpolation_t", Double.NaN);
                require(timestamp > 0 && beforeTs > 0 && afterTs > 0
                                && beforeTs <= timestamp && timestamp <= afterTs,
                        "pose_interpolation_bracket",
                        label + " is not bracketed by ARCore camera timestamps");
                require(timestamp <= 0 || beforeTs <= 0
                                || timestamp - beforeTs <= MAX_POSE_DELTA_NS,
                        "pose_interpolation_before_delta",
                        label + " before sample exceeds 75ms");
                require(timestamp <= 0 || afterTs <= 0
                                || afterTs - timestamp <= MAX_POSE_DELTA_NS,
                        "pose_interpolation_after_delta",
                        label + " after sample exceeds 75ms");
                require(Double.isFinite(t) && t >= 0.0 && t <= 1.0,
                        "pose_interpolation_t", label + " invalid interpolation t=" + t);
            } else if ("nearest_sample_fallback".equals(method)) {
                poseNearestFallbackCount++;
            } else if ("exact_or_single_sample".equals(method)) {
                poseExactOrSingleCount++;
            } else {
                fail("pose_method_unknown", label + " pose method=" + method);
            }

            long arCameraTs = json.optLong("arcore_pose_android_camera_timestamp_ns", -1L);
            require(arCameraTs > 0L, "arcore_camera_timestamp_missing",
                    label + " has no Camera2-correlated ARCore timestamp");

            double[] worldFromCamera = readMatrix(
                    json.optJSONArray("world_from_camera_column_major"), label + " world_from_camera");
            double[] cameraFromWorld = readMatrix(
                    json.optJSONArray("camera_from_world_column_major"), label + " camera_from_world");
            if (worldFromCamera != null) {
                requireRigid(worldFromCamera, label + " world_from_camera");
                validateTranslationAndQuaternion(label, json, worldFromCamera);
            }
            if (cameraFromWorld != null) requireRigid(cameraFromWorld, label + " camera_from_world");
            if (worldFromCamera != null && cameraFromWorld != null) {
                double inverseError = identityError(multiplyColumnMajor(worldFromCamera, cameraFromWorld));
                require(inverseError <= INVERSE_TOLERANCE,
                        "pose_inverse_mismatch",
                        label + " world/camera inverse error=" + inverseError);
            }
            if (failures.size() > before) invalidPoseCount++;
        }

        private void validateTranslationAndQuaternion(
                String label, JSONObject json, double[] worldFromCamera) {
            JSONArray translation = json.optJSONArray("translation_m");
            require(translation != null && translation.length() == 3,
                    "pose_translation_missing", label + " translation_m missing");
            if (translation != null && translation.length() == 3) {
                try {
                    for (int i = 0; i < 3; i++) {
                        double value = translation.getDouble(i);
                        require(Double.isFinite(value), "pose_translation_nonfinite",
                                label + " translation contains non-finite value");
                        require(Math.abs(value - worldFromCamera[12 + i]) <= 1e-4,
                                "pose_translation_matrix_mismatch",
                                label + " translation does not match transform matrix");
                    }
                } catch (JSONException error) {
                    fail("pose_translation_invalid", label + ": " + error);
                }
            }

            JSONArray quaternion = json.optJSONArray("rotation_quaternion_xyzw");
            require(quaternion != null && quaternion.length() == 4,
                    "pose_quaternion_missing", label + " quaternion missing");
            if (quaternion != null && quaternion.length() == 4) {
                try {
                    double norm2 = 0.0;
                    for (int i = 0; i < 4; i++) {
                        double value = quaternion.getDouble(i);
                        require(Double.isFinite(value), "pose_quaternion_nonfinite",
                                label + " quaternion contains non-finite value");
                        norm2 += value * value;
                    }
                    double norm = Math.sqrt(norm2);
                    require(Math.abs(norm - 1.0) <= 0.02,
                            "pose_quaternion_not_unit", label + " quaternion norm=" + norm);
                } catch (JSONException error) {
                    fail("pose_quaternion_invalid", label + ": " + error);
                }
            }
        }

        private void evaluateRgbIntrinsics(String label, JSONObject json, int width, int height) {
            JSONObject intrinsics = json.optJSONObject("jpeg_intrinsics");
            if (intrinsics == null) {
                unavailableIntrinsicsCount++;
                fail("jpeg_intrinsics_missing", label + " has no jpeg_intrinsics");
                return;
            }
            double fx = intrinsics.optDouble("fx", Double.NaN);
            double fy = intrinsics.optDouble("fy", Double.NaN);
            double cx = intrinsics.optDouble("cx", Double.NaN);
            double cy = intrinsics.optDouble("cy", Double.NaN);
            String source = intrinsics.optString("source", "");
            require(finitePositive(fx) && finitePositive(fy),
                    "jpeg_focal_invalid", label + " invalid fx/fy");
            require(Double.isFinite(cx) && Double.isFinite(cy),
                    "jpeg_principal_invalid", label + " invalid cx/cy");
            if (width > 0 && height > 0 && Double.isFinite(cx) && Double.isFinite(cy)) {
                require(cx >= 0.0 && cx < width && cy >= 0.0 && cy < height,
                        "jpeg_principal_outside_image",
                        label + " principal point outside JPEG: " + cx + "," + cy);
            }
            require(!source.isEmpty() && !"unavailable".equals(source),
                    "jpeg_intrinsics_source_unavailable", label + " intrinsics source unavailable");
            if (source.startsWith("camera2_")) {
                camera2IntrinsicsCount++;
                JSONObject crop = intrinsics.optJSONObject("effective_crop_region");
                require(validRect(crop), "jpeg_effective_crop_missing",
                        label + " Camera2 intrinsics do not include a valid effective crop");
            } else if (source.startsWith("arcore_")) {
                fallbackIntrinsicsCount++;
                warn("jpeg_intrinsics_fallback", label + " uses " + source);
            } else {
                unavailableIntrinsicsCount++;
            }

            JSONArray distortion = intrinsics.optJSONArray("lens_distortion");
            if (distortion == null) {
                warn("jpeg_distortion_unavailable", label + " lens distortion unavailable");
            } else {
                for (int i = 0; i < distortion.length(); i++) {
                    double value = distortion.optDouble(i, Double.NaN);
                    require(Double.isFinite(value), "jpeg_distortion_nonfinite",
                            label + " lens distortion contains non-finite value");
                }
            }
        }

        private void evaluateCamera2Metadata(String label, JSONObject json) {
            JSONObject capture = json.optJSONObject("camera2_capture");
            require(capture != null, "camera2_metadata_missing", label + " camera2_capture missing");
            if (capture == null) return;
            require(capture.optLong("exposure_time_ns", -1L) > 0,
                    "camera2_exposure_missing", label + " exposure_time_ns missing");
            require(capture.optInt("iso", -1) > 0,
                    "camera2_iso_missing", label + " ISO missing");
            require(capture.optInt("jpeg_orientation_degrees", Integer.MIN_VALUE) == 0,
                    "jpeg_orientation_contract", label + " JPEG orientation is not 0");
            require(!capture.optString("camera_id", "").isEmpty(),
                    "camera_id_missing", label + " camera_id missing");
            if (!validRect(capture.optJSONObject("crop_region"))) {
                warn("camera2_crop_unavailable", label + " Camera2 crop_region unavailable");
            }
        }

        void evaluateDepth() throws IOException {
            if (dataset == null || !dataset.isDirectory()) return;
            File[] files = dataset.listFiles(
                    (dir, name) -> name.startsWith("depth_obs_") && name.endsWith(".json"));
            if (files == null) files = new File[0];
            Arrays.sort(files, (a, b) -> a.getName().compareTo(b.getName()));
            depthCount = files.length;
            if (depthCount == 0) {
                fail("depth_observations_missing", "no depth_obs_*.json observations");
                return;
            }

            for (File metadataFile : files) {
                int beforeFailures = failures.size();
                JSONObject json;
                try {
                    json = new JSONObject(readText(metadataFile));
                } catch (Exception error) {
                    fail("depth_metadata_invalid_json", metadataFile.getName() + ": " + error);
                    continue;
                }
                evaluateOneDepth(metadataFile, json);
                if (failures.size() == beforeFailures) depthValidCount++;
            }
            if (minDepthPoints == Integer.MAX_VALUE) minDepthPoints = 0;
            if (minConfidence == Float.POSITIVE_INFINITY) minConfidence = Float.NaN;
            if (maxConfidence == Float.NEGATIVE_INFINITY) maxConfidence = Float.NaN;
        }

        private void evaluateOneDepth(File metadataFile, JSONObject json) {
            String label = metadataFile.getName();
            try {
                require("raw_depth".equals(json.optString("observation_type", "")),
                        "depth_observation_type", label + " observation_type must be raw_depth");
                require("geometry_prior_observation".equals(json.optString("observation_role", "")),
                        "depth_observation_role", label + " wrong observation role");
                require(!json.optBoolean("rgb_pairing_required", true),
                        "depth_rgb_independence_contract", label + " requires RGB pairing");
                require(json.optString("coordinate_system", "").contains("datasetRootAnchor"),
                        "depth_coordinate_system", label + " is not in datasetRootAnchor coordinates");

                String id = json.optString("depth_observation_id", "");
                require(!id.isEmpty(), "depth_id_missing", label + " has no depth_observation_id");
                if (!id.isEmpty()) {
                    require(depthIds.add(id), "depth_id_duplicate", "duplicate Depth id " + id);
                }

                long timestamp = json.optLong("raw_depth_timestamp_ns", -1L);
                require(timestamp > 0L, "depth_timestamp_missing", label + " raw timestamp missing");
                if (timestamp > 0L) {
                    require(depthTimestamps.add(timestamp), "depth_timestamp_duplicate",
                            "duplicate Raw Depth timestamp " + timestamp);
                }
                if (json.isNull("android_camera_timestamp_ns")
                        || json.optLong("android_camera_timestamp_ns", -1L) <= 0L) {
                    depthMissingAndroidCameraTimestampCount++;
                    warn("depth_android_camera_timestamp_missing",
                            label + " has no correlated Android camera timestamp");
                }

                int declaredPoints = json.optInt("point_count", -1);
                require(declaredPoints > 0, "depth_point_count_invalid",
                        label + " point_count=" + declaredPoints);

                double[] rootFromDepthCamera = readMatrix(
                        json.optJSONArray("root_from_depth_camera_column_major"),
                        label + " root_from_depth_camera");
                if (rootFromDepthCamera != null) requireRigid(rootFromDepthCamera, label);

                JSONArray quaternion = json.optJSONArray("depth_camera_rotation_quaternion_xyzw");
                require(quaternion != null && quaternion.length() == 4,
                        "depth_quaternion_missing", label + " depth quaternion missing");
                if (quaternion != null && quaternion.length() == 4) {
                    double norm2 = 0.0;
                    for (int i = 0; i < 4; i++) {
                        double value = quaternion.optDouble(i, Double.NaN);
                        require(Double.isFinite(value), "depth_quaternion_nonfinite",
                                label + " depth quaternion non-finite");
                        norm2 += value * value;
                    }
                    require(Math.abs(Math.sqrt(norm2) - 1.0) <= 0.02,
                            "depth_quaternion_not_unit", label + " depth quaternion not unit");
                }

                String plyName = json.optString("point_cloud", "");
                File ply = new File(dataset, plyName);
                require(!plyName.isEmpty() && ply.isFile() && ply.length() > 0L,
                        "depth_ply_missing", label + " PLY missing: " + plyName);
                if (ply.isFile()) {
                    PlyStats stats = inspectPly(ply);
                    require(stats.layoutValid, "depth_ply_layout", ply.getName() + " invalid PLY layout");
                    require(stats.vertexCount == declaredPoints,
                            "depth_ply_point_count_mismatch",
                            ply.getName() + " json=" + declaredPoints + " ply=" + stats.vertexCount);
                    require(stats.rawDepthTimestampNs == timestamp,
                            "depth_ply_timestamp_mismatch",
                            ply.getName() + " header timestamp=" + stats.rawDepthTimestampNs
                                    + " json=" + timestamp);
                    require(stats.expectedLength == ply.length(),
                            "depth_ply_size_mismatch",
                            ply.getName() + " expected bytes=" + stats.expectedLength
                                    + " actual=" + ply.length());
                    invalidPointCount += stats.invalidPointCount;
                    invalidConfidenceCount += stats.invalidConfidenceCount;
                    depthPointTotal += stats.vertexCount;
                    minDepthPoints = Math.min(minDepthPoints, stats.vertexCount);
                    maxDepthPoints = Math.max(maxDepthPoints, stats.vertexCount);
                    if (Float.isFinite(stats.minConfidence)) {
                        minConfidence = Math.min(minConfidence, stats.minConfidence);
                        maxConfidence = Math.max(maxConfidence, stats.maxConfidence);
                    }
                    require(stats.invalidPointCount == 0,
                            "depth_nonfinite_points",
                            ply.getName() + " invalid XYZ points=" + stats.invalidPointCount);
                    require(stats.invalidConfidenceCount == 0,
                            "depth_invalid_confidence",
                            ply.getName() + " invalid confidence=" + stats.invalidConfidenceCount);
                }
            } catch (Exception error) {
                fail("depth_evaluation_exception", label + ": " + error);
            }
        }

        void evaluateCaptureHealth() {
            if (dataset == null) return;
            String log = DiagnosticLog.snapshot();
            String marker = "Dataset directory=" + dataset.getAbsolutePath();
            int start = log.lastIndexOf(marker);
            diagnosticSessionWindowFound = start >= 0;
            if (!diagnosticSessionWindowFound) {
                fail("diagnostic_session_window_missing",
                        "cannot isolate diagnostic log for " + dataset.getAbsolutePath());
                return;
            }
            String window = log.substring(start);
            for (String pattern : FATAL_SESSION_PATTERNS) {
                collectMatchingLines(window, pattern, fatalEvents);
            }
            for (String pattern : OBSERVATION_WRITE_FAILURE_PATTERNS) {
                collectMatchingLines(window, pattern, writeErrorEvents);
            }
            fatalSessionErrorCount = fatalEvents.size();
            observationWriteErrorCount = writeErrorEvents.size();
            require(fatalSessionErrorCount == 0,
                    "capture_session_fatal_errors",
                    "capture session contains " + fatalSessionErrorCount + " fatal Camera/ARCore errors");
            require(observationWriteErrorCount == 0,
                    "observation_write_errors",
                    "capture session contains " + observationWriteErrorCount + " observation write errors");
        }

        JSONObject toJson() throws JSONException {
            JSONObject root = new JSONObject();
            root.put("format_version", 1);
            root.put("phase", "phase1");
            root.put("hard_gate", true);
            root.put("evaluated_at_unix_ms", System.currentTimeMillis());
            root.put("status", failures.isEmpty() ? "PASS" : "FAIL");
            root.put("pass", failures.isEmpty());
            root.put("failure_count", failures.size());
            root.put("warning_count", warnings.size());

            root.put("rgb_count", rgbCount);
            root.put("rgb_valid_count", rgbValidCount);
            root.put("depth_count", depthCount);
            root.put("depth_valid_count", depthValidCount);
            root.put("pose_delta_max_ms", maxPoseDeltaNs / 1_000_000.0);
            root.put("pose_delta_p95_ms", percentilePoseDeltaNs(0.95) / 1_000_000.0);
            root.put("pose_interpolated_count", poseInterpolatedCount);
            root.put("pose_nearest_fallback_count", poseNearestFallbackCount);
            root.put("pose_exact_or_single_count", poseExactOrSingleCount);
            root.put("camera2_intrinsics_count", camera2IntrinsicsCount);
            root.put("arcore_intrinsics_fallback_count", fallbackIntrinsicsCount);
            root.put("unavailable_intrinsics_count", unavailableIntrinsicsCount);
            root.put("rgb_without_depth_reference_count", rgbWithoutDepthReferenceCount);
            root.put("invalid_pose_count", invalidPoseCount);
            root.put("invalid_point_count", invalidPointCount);
            root.put("invalid_confidence_count", invalidConfidenceCount);
            root.put("fatal_count", fatalSessionErrorCount);
            root.put("observation_write_error_count", observationWriteErrorCount);

            JSONObject rgb = new JSONObject();
            rgb.put("high_resolution_min_pixels_required", MIN_HIGH_RES_PIXELS);
            rgb.put("min_saved_jpeg_pixels", minJpegPixels == Long.MAX_VALUE ? 0 : minJpegPixels);
            rgb.put("max_saved_jpeg_pixels", maxJpegPixels);
            rgb.put("pose_max_delta_ms_required", MAX_POSE_DELTA_NS / 1_000_000.0);
            rgb.put("camera2_intrinsics_count", camera2IntrinsicsCount);
            rgb.put("fallback_intrinsics_count", fallbackIntrinsicsCount);
            rgb.put("without_depth_reference_count", rgbWithoutDepthReferenceCount);
            root.put("rgb", rgb);

            JSONObject depth = new JSONObject();
            depth.put("point_total", depthPointTotal);
            depth.put("min_points_per_observation", minDepthPoints == Integer.MAX_VALUE ? 0 : minDepthPoints);
            depth.put("max_points_per_observation", maxDepthPoints);
            depth.put("missing_android_camera_timestamp_count", depthMissingAndroidCameraTimestampCount);
            depth.put("min_confidence", Float.isFinite(minConfidence) ? minConfidence : JSONObject.NULL);
            depth.put("max_confidence", Float.isFinite(maxConfidence) ? maxConfidence : JSONObject.NULL);
            root.put("depth", depth);

            JSONObject health = new JSONObject();
            health.put("diagnostic_session_window_found", diagnosticSessionWindowFound);
            health.put("fatal_error_count", fatalSessionErrorCount);
            health.put("observation_write_error_count", observationWriteErrorCount);
            health.put("fatal_events", new JSONArray(fatalEvents));
            health.put("write_error_events", new JSONArray(writeErrorEvents));
            root.put("capture_health", health);

            root.put("failures", new JSONArray(failures));
            root.put("warnings", new JSONArray(warnings));
            root.put("next_phase_allowed", failures.isEmpty());
            root.put("next_phase_contract",
                    "Phase 2 may start only when this report has status=PASS and pass=true");
            return root;
        }

        long percentilePoseDeltaNs(double p) {
            if (poseDeltasNs.isEmpty()) return 0L;
            List<Long> sorted = new ArrayList<>(poseDeltasNs);
            Collections.sort(sorted);
            int index = (int) Math.ceil(p * sorted.size()) - 1;
            index = Math.max(0, Math.min(sorted.size() - 1, index));
            return sorted.get(index);
        }

        void require(boolean condition, String code, String message) {
            if (!condition) fail(code, message);
        }

        void fail(String code, String message) {
            failures.add(issue(code, message));
        }

        void warn(String code, String message) {
            warnings.add(issue(code, message));
        }

        JSONObject issue(String code, String message) {
            JSONObject issue = new JSONObject();
            try {
                issue.put("code", code);
                issue.put("message", message == null ? "" : message);
            } catch (JSONException ignored) {
            }
            return issue;
        }

        double[] readMatrix(JSONArray json, String label) {
            if (json == null || json.length() != 16) {
                fail("pose_matrix_missing", label + " must contain 16 values");
                return null;
            }
            double[] values = new double[16];
            for (int i = 0; i < 16; i++) {
                values[i] = json.optDouble(i, Double.NaN);
                if (!Double.isFinite(values[i])) {
                    fail("pose_matrix_nonfinite", label + " contains non-finite value at " + i);
                    return null;
                }
            }
            return values;
        }

        void requireRigid(double[] m, String label) {
            require(Math.abs(m[3]) <= 1e-4 && Math.abs(m[7]) <= 1e-4
                            && Math.abs(m[11]) <= 1e-4 && Math.abs(m[15] - 1.0) <= 1e-4,
                    "pose_matrix_affine_row", label + " invalid affine last row");
            double[] c0 = {m[0], m[1], m[2]};
            double[] c1 = {m[4], m[5], m[6]};
            double[] c2 = {m[8], m[9], m[10]};
            require(Math.abs(norm(c0) - 1.0) <= RIGID_TOLERANCE
                            && Math.abs(norm(c1) - 1.0) <= RIGID_TOLERANCE
                            && Math.abs(norm(c2) - 1.0) <= RIGID_TOLERANCE,
                    "pose_rotation_scale", label + " rotation columns are not unit length");
            require(Math.abs(dot(c0, c1)) <= RIGID_TOLERANCE
                            && Math.abs(dot(c0, c2)) <= RIGID_TOLERANCE
                            && Math.abs(dot(c1, c2)) <= RIGID_TOLERANCE,
                    "pose_rotation_orthogonality", label + " rotation columns are not orthogonal");
            double det = determinant3(c0, c1, c2);
            require(Math.abs(det - 1.0) <= 0.05,
                    "pose_rotation_determinant", label + " rotation determinant=" + det);
        }
    }

    private static PlyStats inspectPly(File file) throws IOException {
        try (BufferedInputStream in = new BufferedInputStream(new FileInputStream(file))) {
            List<String> lines = new ArrayList<>();
            int headerBytes = 0;
            boolean ended = false;
            while (headerBytes < 64 * 1024) {
                AsciiLine line = readAsciiLine(in);
                if (line == null) break;
                headerBytes += line.byteCount;
                lines.add(line.text);
                if ("end_header".equals(line.text)) {
                    ended = true;
                    break;
                }
            }
            if (!ended) throw new IOException("PLY end_header missing: " + file.getName());

            int vertexCount = -1;
            long timestamp = -1L;
            boolean binaryLittleEndian = false;
            List<String> properties = new ArrayList<>();
            boolean inVertex = false;
            for (String line : lines) {
                if ("format binary_little_endian 1.0".equals(line)) binaryLittleEndian = true;
                if (line.startsWith("comment raw_depth_timestamp_ns ")) {
                    try {
                        timestamp = Long.parseLong(line.substring(
                                "comment raw_depth_timestamp_ns ".length()).trim());
                    } catch (NumberFormatException ignored) {
                    }
                }
                if (line.startsWith("element vertex ")) {
                    try {
                        vertexCount = Integer.parseInt(line.substring("element vertex ".length()).trim());
                    } catch (NumberFormatException ignored) {
                    }
                    inVertex = true;
                    continue;
                }
                if (line.startsWith("element ") && !line.startsWith("element vertex ")) {
                    inVertex = false;
                }
                if (inVertex && line.startsWith("property ")) properties.add(line);
            }

            List<String> expectedProperties = Arrays.asList(
                    "property float x",
                    "property float y",
                    "property float z",
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                    "property float confidence");
            boolean layoutValid = binaryLittleEndian
                    && vertexCount >= 0
                    && properties.equals(expectedProperties);
            long expectedLength = vertexCount < 0
                    ? -1L : headerBytes + (long) vertexCount * PLY_BYTES_PER_POINT;

            long invalidPoints = 0L;
            long invalidConfidence = 0L;
            float minConfidence = Float.POSITIVE_INFINITY;
            float maxConfidence = Float.NEGATIVE_INFINITY;
            byte[] record = new byte[PLY_BYTES_PER_POINT];
            ByteBuffer buffer = ByteBuffer.wrap(record).order(ByteOrder.LITTLE_ENDIAN);
            if (vertexCount > 0) {
                for (int i = 0; i < vertexCount; i++) {
                    readFully(in, record);
                    buffer.position(0);
                    float x = buffer.getFloat();
                    float y = buffer.getFloat();
                    float z = buffer.getFloat();
                    buffer.get();
                    buffer.get();
                    buffer.get();
                    float confidence = buffer.getFloat();
                    if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)) {
                        invalidPoints++;
                    }
                    if (!Float.isFinite(confidence) || confidence < 0f || confidence > 1f) {
                        invalidConfidence++;
                    } else {
                        minConfidence = Math.min(minConfidence, confidence);
                        maxConfidence = Math.max(maxConfidence, confidence);
                    }
                }
            }
            return new PlyStats(
                    vertexCount,
                    timestamp,
                    headerBytes,
                    expectedLength,
                    layoutValid,
                    invalidPoints,
                    invalidConfidence,
                    minConfidence,
                    maxConfidence);
        }
    }

    private static AsciiLine readAsciiLine(BufferedInputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int count = 0;
        while (true) {
            int value = in.read();
            if (value < 0) {
                if (count == 0) return null;
                break;
            }
            count++;
            if (value == '\n') break;
            if (value != '\r') out.write(value);
            if (count > 4096) throw new IOException("PLY header line too long");
        }
        return new AsciiLine(new String(out.toByteArray(), StandardCharsets.US_ASCII), count);
    }

    private static void readFully(BufferedInputStream in, byte[] bytes) throws IOException {
        int offset = 0;
        while (offset < bytes.length) {
            int read = in.read(bytes, offset, bytes.length - offset);
            if (read < 0) throw new EOFException("truncated PLY payload");
            offset += read;
        }
    }

    private static void collectMatchingLines(String text, String pattern, List<String> output) {
        int start = 0;
        while (start < text.length()) {
            int end = text.indexOf('\n', start);
            if (end < 0) end = text.length();
            String line = text.substring(start, end);
            if (line.contains(pattern)) output.add(line);
            start = end + 1;
        }
    }

    private static boolean validRect(JSONObject rect) {
        if (rect == null) return false;
        int left = rect.optInt("left", Integer.MAX_VALUE);
        int top = rect.optInt("top", Integer.MAX_VALUE);
        int right = rect.optInt("right", Integer.MIN_VALUE);
        int bottom = rect.optInt("bottom", Integer.MIN_VALUE);
        return left != Integer.MAX_VALUE && top != Integer.MAX_VALUE
                && right > left && bottom > top;
    }

    private static boolean finitePositive(double value) {
        return Double.isFinite(value) && value > 0.0;
    }

    private static long safeAbs(long value) {
        return value == Long.MIN_VALUE ? Long.MAX_VALUE : Math.abs(value);
    }

    private static double norm(double[] v) {
        return Math.sqrt(dot(v, v));
    }

    private static double dot(double[] a, double[] b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    private static double determinant3(double[] c0, double[] c1, double[] c2) {
        return c0[0] * (c1[1] * c2[2] - c1[2] * c2[1])
                - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
                + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1]);
    }

    private static double[] multiplyColumnMajor(double[] a, double[] b) {
        double[] out = new double[16];
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                double value = 0.0;
                for (int k = 0; k < 4; k++) {
                    value += a[k * 4 + row] * b[col * 4 + k];
                }
                out[col * 4 + row] = value;
            }
        }
        return out;
    }

    private static double identityError(double[] m) {
        double max = 0.0;
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                double expected = row == col ? 1.0 : 0.0;
                max = Math.max(max, Math.abs(m[col * 4 + row] - expected));
            }
        }
        return max;
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

    private static final class AsciiLine {
        final String text;
        final int byteCount;

        AsciiLine(String text, int byteCount) {
            this.text = text;
            this.byteCount = byteCount;
        }
    }

    private static final class PlyStats {
        final int vertexCount;
        final long rawDepthTimestampNs;
        final int headerBytes;
        final long expectedLength;
        final boolean layoutValid;
        final long invalidPointCount;
        final long invalidConfidenceCount;
        final float minConfidence;
        final float maxConfidence;

        PlyStats(
                int vertexCount,
                long rawDepthTimestampNs,
                int headerBytes,
                long expectedLength,
                boolean layoutValid,
                long invalidPointCount,
                long invalidConfidenceCount,
                float minConfidence,
                float maxConfidence) {
            this.vertexCount = vertexCount;
            this.rawDepthTimestampNs = rawDepthTimestampNs;
            this.headerBytes = headerBytes;
            this.expectedLength = expectedLength;
            this.layoutValid = layoutValid;
            this.invalidPointCount = invalidPointCount;
            this.invalidConfidenceCount = invalidConfidenceCount;
            this.minConfidence = minConfidence;
            this.maxConfidence = maxConfidence;
        }
    }
}