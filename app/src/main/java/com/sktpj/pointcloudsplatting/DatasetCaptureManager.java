package com.sktpj.pointcloudsplatting;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.TotalCaptureResult;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Environment;
import android.util.Size;

import com.google.ar.core.Camera;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.exceptions.NotYetAvailableException;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Continuously samples ARCore's CPU camera stream and automatically keeps the sharpest keyframe
 * for each short viewpoint window. No Camera2 still capture is issued while scanning. CPU image
 * copying, sharpness evaluation and JPEG encoding are kept off the GL preview thread.
 */
public final class DatasetCaptureManager {
    private static final String TAG = "DatasetCapture";

    private static final long CANDIDATE_INTERVAL_NS = 110_000_000L;
    private static final long SELECTION_WINDOW_NS = 420_000_000L;
    private static final long FORCE_KEYFRAME_AFTER_NS = 1_300_000_000L;
    private static final float NEW_WINDOW_TRANSLATION_METERS = 0.035f;
    private static final float NEW_WINDOW_ROTATION_DEGREES = 3.0f;
    private static final float MIN_SAVED_TRANSLATION_METERS = 0.018f;
    private static final float MIN_SAVED_ROTATION_DEGREES = 1.4f;
    private static final float MOTION_FILTER_ALPHA = 0.35f;

    private static final long MAX_DEPTH_MATCH_DELTA_NS = 300_000_000L;
    private static final int MAX_DEPTH_SAMPLES = 40;
    private static final int MAX_FALLBACK_DEPTH_PRIORS = 6;
    private static final int JPEG_QUALITY = 94;
    private static final int SHARPNESS_SAMPLE_STEP = 3;

    private final File captureRoot;
    private final ExecutorService selector = Executors.newSingleThreadExecutor();
    private final ExecutorService writer = Executors.newSingleThreadExecutor();
    private final AtomicBoolean candidateTaskInFlight = new AtomicBoolean();
    private final AtomicInteger captureSequence = new AtomicInteger();
    private final ArrayDeque<WorldPointCloudSnapshot> depthSamples = new ArrayDeque<>();

    private Pose previousPose;
    private long previousFrameTimestampNs = -1L;
    private float filteredLinearSpeedMps;
    private float filteredAngularSpeedDps;
    private boolean hasFilteredMotion;

    private Candidate bestCandidate;
    private Pose selectionWindowPose;
    private long selectionWindowStartNs = -1L;
    private long lastCandidateTimestampNs = -1L;
    private Pose lastSavedPose;
    private long lastSavedTimestampNs = -1L;

    private boolean cameraActive = true;
    private boolean fallbackDepthPriorsQueued;
    private volatile boolean captureEnabled = true;
    private volatile int savedCount;
    private volatile String lastDecision = "automatic frame selection starting";

    private String cameraId = "unknown";
    private Size legacyJpegSize;

    public DatasetCaptureManager(File externalPicturesDir) {
        if (externalPicturesDir == null) {
            throw new IllegalStateException("External Pictures directory unavailable");
        }
        String sessionName = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        captureRoot = new File(externalPicturesDir, "capture_tmp_" + sessionName);
        if (!captureRoot.mkdirs() && !captureRoot.isDirectory()) {
            throw new IllegalStateException("Failed to create " + captureRoot);
        }
    }

    public static File getPicturesDirectory(android.content.Context context) {
        return context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
    }

    /**
     * ScannerActivity still creates a legacy JPEG surface as part of its SharedCamera session.
     * Continuous capture never targets it, so keep that unused surface as small as possible.
     */
    public static Size chooseJpegSize(CameraCharacteristics characteristics) {
        android.hardware.camera2.params.StreamConfigurationMap map = characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) throw new IllegalStateException("No camera stream configuration map");
        Size[] outputs = map.getOutputSizes(ImageFormat.JPEG);
        if (outputs == null || outputs.length == 0) {
            throw new IllegalStateException("No JPEG output sizes");
        }
        List<Size> sizes = new ArrayList<>();
        java.util.Collections.addAll(sizes, outputs);
        sizes.sort(Comparator.comparingLong(DatasetCaptureManager::pixelCount));
        for (Size size : sizes) {
            if (isFourByThree(size)) return size;
        }
        return sizes.get(0);
    }

    public synchronized void configureCamera(
            String cameraId, Size jpegSize, CameraCharacteristics characteristics) {
        this.cameraId = cameraId;
        this.legacyJpegSize = jpegSize;
        writeSessionCameraInfo(characteristics);
    }

    public int getSavedCount() {
        return savedCount;
    }

    public String getCaptureDirectoryPath() {
        return captureRoot.getAbsolutePath();
    }

    public String getLastDecision() {
        return lastDecision;
    }

    public synchronized void onCameraResumed() {
        cameraActive = true;
        previousPose = null;
        previousFrameTimestampNs = -1L;
        resetMotionFilterLocked();
        bestCandidate = null;
        selectionWindowPose = null;
        selectionWindowStartNs = -1L;
        lastCandidateTimestampNs = -1L;
        if (captureEnabled) lastDecision = "automatic capture running";
        DiagnosticLog.i(TAG, "Continuous ARCore capture state reset on resume");
    }

    public synchronized void onCameraPaused() {
        cameraActive = false;
        flushBestCandidateLocked(true);
        previousPose = null;
        previousFrameTimestampNs = -1L;
        resetMotionFilterLocked();
        selectionWindowPose = null;
        selectionWindowStartNs = -1L;
        lastCandidateTimestampNs = -1L;
        lastDecision = captureEnabled ? "camera paused" : "capture stopped; finalizing";
        DiagnosticLog.i(TAG, "Continuous ARCore capture paused");
    }

    public synchronized void onDepthFrame(DepthData depth, Pose rootPose) {
        if (!cameraActive) return;
        try {
            WorldPointCloudSnapshot snapshot = WorldPointCloudSnapshot.from(depth, rootPose);
            depthSamples.addLast(snapshot);
            while (depthSamples.size() > MAX_DEPTH_SAMPLES) depthSamples.removeFirst();
        } catch (RuntimeException e) {
            DiagnosticLog.w(TAG, "Failed to snapshot Raw Depth: " + e.getMessage());
        }
    }

    /**
     * Acquires the CPU image belonging to the current ARCore frame and immediately hands it to one
     * bounded selector worker. The GL thread never converts YUV, computes sharpness or writes JPEG.
     * Returning false deliberately prevents ScannerActivity's legacy TEMPLATE_STILL_CAPTURE path.
     */
    public synchronized boolean onArFrame(Frame frame, Camera camera, Pose rootPose) {
        if (!cameraActive || !captureEnabled) {
            lastDecision = captureEnabled ? "camera paused" : "capture stopped; finalizing";
            return false;
        }

        final long frameTimestampNs = frame.getTimestamp();
        final long cameraTimestampNs = safeAndroidCameraTimestamp(frame, frameTimestampNs);
        final Pose localPose = rootPose.inverse().compose(camera.getPose());
        final Motion motion = filterMotionLocked(measureMotion(localPose, frameTimestampNs));

        previousPose = localPose;
        previousFrameTimestampNs = frameTimestampNs;

        if (bestCandidate == null && !hasUsefulViewpointChange(localPose, cameraTimestampNs)) {
            lastDecision = String.format(Locale.US,
                    "automatic capture tracking viewpoint %.2f m/s %.1f deg/s",
                    motion.linearSpeedMps, motion.angularSpeedDps);
            return false;
        }
        if (lastCandidateTimestampNs >= 0
                && cameraTimestampNs - lastCandidateTimestampNs < CANDIDATE_INTERVAL_NS) {
            return false;
        }
        if (!candidateTaskInFlight.compareAndSet(false, true)) {
            // Never queue work faster than it can be evaluated. Preview wins over capture density.
            return false;
        }
        lastCandidateTimestampNs = cameraTimestampNs;

        Image image = null;
        boolean handedOff = false;
        try {
            image = frame.acquireCameraImage();
            PoseSample pose = PoseSample.from(cameraTimestampNs, camera, rootPose);
            CandidateSeed seed = new CandidateSeed(
                    cameraTimestampNs,
                    localPose,
                    pose,
                    image,
                    motion.linearSpeedMps,
                    motion.angularSpeedDps);
            selector.execute(() -> processCandidate(seed));
            image = null;
            handedOff = true;
        } catch (NotYetAvailableException e) {
            lastDecision = "automatic capture waiting for CPU camera frame";
        } catch (RejectedExecutionException e) {
            lastDecision = "automatic capture selector stopped";
        } catch (RuntimeException e) {
            lastDecision = "automatic capture frame unavailable";
            DiagnosticLog.w(TAG,
                    "ARCore CPU camera frame unavailable: " + e.getClass().getSimpleName()
                            + ": " + e.getMessage());
        } finally {
            if (image != null) image.close();
            if (!handedOff) candidateTaskInFlight.set(false);
        }
        return false;
    }

    private void processCandidate(CandidateSeed seed) {
        try {
            YuvFrame yuv = copyCameraImage(seed.image);
            double sharpness = calculateSharpness(yuv.luma, yuv.width, yuv.height);
            Motion motion = new Motion(seed.linearSpeedMps, seed.angularSpeedDps);
            double score = rankFrame(sharpness, motion);
            Candidate candidate = new Candidate(
                    seed.timestampNs,
                    seed.localPose,
                    seed.pose,
                    yuv,
                    sharpness,
                    score,
                    seed.linearSpeedMps,
                    seed.angularSpeedDps);

            synchronized (this) {
                if (!cameraActive) return;
                if (bestCandidate != null
                        && shouldCloseSelectionWindow(seed.localPose, seed.timestampNs)) {
                    flushBestCandidateLocked(false);
                }

                if (bestCandidate == null) {
                    bestCandidate = candidate;
                    selectionWindowPose = seed.localPose;
                    selectionWindowStartNs = seed.timestampNs;
                } else if (candidate.score > bestCandidate.score) {
                    bestCandidate = candidate;
                }

                lastDecision = String.format(Locale.US,
                        "automatic capture selecting sharp frame score=%.1f sharpness=%.1f",
                        score, sharpness);

                if (selectionWindowStartNs >= 0
                        && seed.timestampNs - selectionWindowStartNs >= SELECTION_WINDOW_NS) {
                    flushBestCandidateLocked(false);
                }
            }
        } catch (RuntimeException e) {
            synchronized (this) {
                lastDecision = "automatic capture selector failed";
            }
            DiagnosticLog.w(TAG,
                    "Continuous frame selector failed: " + e.getClass().getSimpleName()
                            + ": " + e.getMessage());
        } finally {
            try {
                seed.image.close();
            } catch (RuntimeException ignored) {
            }
            candidateTaskInFlight.set(false);
        }
    }

    // Legacy still-capture callbacks stay source-compatible but are intentionally not used.
    public synchronized void onCaptureRequestFailed(String reason) {
        DiagnosticLog.w(TAG, "Unexpected legacy still failure: " + reason);
    }

    public synchronized void onCaptureCompleted(TotalCaptureResult result) {
        // No-op: continuous frame quality is evaluated directly from the ARCore CPU image.
    }

    public void onJpegAvailable(ImageReader reader) {
        Image image = null;
        try {
            image = reader.acquireLatestImage();
            if (image != null) {
                DiagnosticLog.w(TAG, "Ignoring unexpected legacy JPEG while continuous capture is active");
            }
        } catch (RuntimeException ignored) {
        } finally {
            if (image != null) image.close();
        }
    }

    public boolean stopCaptureAndFlush(long timeoutMs) {
        final long deadline = System.currentTimeMillis() + Math.max(1L, timeoutMs);
        synchronized (this) {
            // Stop scheduling new candidates. Any candidate already handed to the selector is still
            // allowed to finish and can become the final saved keyframe.
            captureEnabled = false;
            lastDecision = "capture stopped; finalizing";
        }

        try {
            long remaining = Math.max(1L, deadline - System.currentTimeMillis());
            selector.submit(() -> {}).get(remaining, TimeUnit.MILLISECONDS);

            synchronized (this) {
                flushBestCandidateLocked(true);
                queueFallbackDepthPriorsLocked();
            }

            remaining = Math.max(1L, deadline - System.currentTimeMillis());
            writer.submit(() -> {}).get(remaining, TimeUnit.MILLISECONDS);
            synchronized (this) {
                lastDecision = "capture stopped; ready to save";
            }
            return true;
        } catch (Exception e) {
            DiagnosticLog.w(TAG, "Timed out flushing continuous capture pipeline: " + e.getMessage());
            synchronized (this) {
                lastDecision = "save timed out flushing files";
            }
            return false;
        }
    }

    public synchronized void resumeCapture() {
        captureEnabled = true;
        lastDecision = "automatic capture running";
    }

    public void shutdown() {
        synchronized (this) {
            captureEnabled = false;
        }
        selector.shutdown();
        try {
            selector.awaitTermination(500L, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        synchronized (this) {
            flushBestCandidateLocked(true);
        }
        writer.shutdown();
    }

    private boolean shouldCloseSelectionWindow(Pose pose, long timestampNs) {
        if (selectionWindowPose == null || selectionWindowStartNs < 0) return false;
        if (timestampNs - selectionWindowStartNs >= SELECTION_WINDOW_NS) return true;
        return translationDistance(selectionWindowPose, pose) >= NEW_WINDOW_TRANSLATION_METERS
                || rotationDegrees(selectionWindowPose, pose) >= NEW_WINDOW_ROTATION_DEGREES;
    }

    private boolean hasUsefulViewpointChange(Pose pose, long timestampNs) {
        if (lastSavedPose == null || lastSavedTimestampNs < 0) return true;
        if (timestampNs - lastSavedTimestampNs >= FORCE_KEYFRAME_AFTER_NS) return true;
        return translationDistance(lastSavedPose, pose) >= MIN_SAVED_TRANSLATION_METERS
                || rotationDegrees(lastSavedPose, pose) >= MIN_SAVED_ROTATION_DEGREES;
    }

    private void flushBestCandidateLocked(boolean force) {
        Candidate candidate = bestCandidate;
        bestCandidate = null;
        selectionWindowPose = null;
        selectionWindowStartNs = -1L;
        if (candidate == null) return;

        if (!force && !hasUsefulViewpointChange(candidate.localPose, candidate.timestampNs)) {
            lastDecision = "automatic capture skipped duplicate viewpoint";
            return;
        }

        WorldPointCloudSnapshot cloud = findNearestDepthLocked(candidate.timestampNs);
        if (cloud != null
                && Math.abs(cloud.getTimestampNs() - candidate.timestampNs) > MAX_DEPTH_MATCH_DELTA_NS) {
            cloud = null;
        }

        final int index = captureSequence.incrementAndGet();
        savedCount = index;
        lastSavedPose = candidate.localPose;
        lastSavedTimestampNs = candidate.timestampNs;
        lastDecision = "saved frame " + index;
        final WorldPointCloudSnapshot selectedCloud = cloud;
        writer.execute(() -> {
            try {
                writeCapture(index, candidate, selectedCloud);
            } catch (IOException | JSONException e) {
                DiagnosticLog.e(TAG, "Failed to write continuous dataset frame", e);
            }
        });
    }

    private WorldPointCloudSnapshot findNearestDepthLocked(long timestampNs) {
        WorldPointCloudSnapshot best = null;
        long bestDelta = Long.MAX_VALUE;
        for (WorldPointCloudSnapshot sample : depthSamples) {
            long delta = Math.abs(sample.getTimestampNs() - timestampNs);
            if (delta < bestDelta) {
                best = sample;
                bestDelta = delta;
            }
        }
        return best;
    }

    private void queueFallbackDepthPriorsLocked() {
        if (fallbackDepthPriorsQueued || depthSamples.isEmpty()) return;
        List<WorldPointCloudSnapshot> available = new ArrayList<>(depthSamples);
        int wanted = Math.min(MAX_FALLBACK_DEPTH_PRIORS, available.size());
        int queued = 0;
        for (int i = 0; i < wanted; i++) {
            int sourceIndex = wanted == 1
                    ? available.size() - 1
                    : Math.round(i * (available.size() - 1f) / (wanted - 1f));
            WorldPointCloudSnapshot snapshot = available.get(sourceIndex);
            if (snapshot == null || snapshot.getPointCount() < 64) continue;
            File output = new File(captureRoot, String.format(Locale.US,
                    "depth_prior_%02d_%d.ply", queued + 1, snapshot.getTimestampNs()));
            writer.execute(() -> {
                try {
                    snapshot.writePly(output);
                    DiagnosticLog.i(TAG,
                            "Saved fallback depth prior=" + output.getName()
                                    + " points=" + snapshot.getPointCount());
                } catch (IOException e) {
                    DiagnosticLog.w(TAG,
                            "Failed to save fallback depth prior " + output.getName()
                                    + ": " + e.getMessage());
                }
            });
            queued++;
        }
        fallbackDepthPriorsQueued = queued > 0;
    }

    private void writeCapture(int index, Candidate candidate, WorldPointCloudSnapshot cloud)
            throws IOException, JSONException {
        String base = String.format(Locale.US,
                "frame_%06d_%d", index, candidate.timestampNs);
        File jpegFile = new File(captureRoot, base + ".jpg");
        File jsonFile = new File(captureRoot, base + ".json");
        File plyFile = cloud == null ? null : new File(captureRoot, base + ".ply");

        byte[] jpeg = encodeJpeg(candidate.yuv);
        try (FileOutputStream out = new FileOutputStream(jpegFile)) {
            out.write(jpeg);
        }
        if (cloud != null) cloud.writePly(plyFile);

        PoseSample pose = candidate.pose;
        JSONObject json = new JSONObject();
        json.put("capture_index", index);
        json.put("image", jpegFile.getName());
        json.put("point_cloud", plyFile == null ? JSONObject.NULL : plyFile.getName());
        json.put("has_raw_depth_prior", cloud != null);
        json.put("jpeg_width", candidate.yuv.width);
        json.put("jpeg_height", candidate.yuv.height);
        json.put("jpeg_sensor_timestamp_ns", candidate.timestampNs);
        json.put("arcore_pose_timestamp_ns", pose.timestampNs);
        json.put("pose_timestamp_delta_ns", pose.timestampNs - candidate.timestampNs);
        json.put("raw_depth_timestamp_ns",
                cloud == null ? JSONObject.NULL : cloud.getTimestampNs());
        json.put("depth_timestamp_delta_ns",
                cloud == null ? JSONObject.NULL : cloud.getTimestampNs() - candidate.timestampNs);
        json.put("point_count", cloud == null ? 0 : cloud.getPointCount());
        json.put("point_cloud_coordinate_system",
                cloud == null ? JSONObject.NULL : "ARCore root-anchor local coordinates");
        json.put("translation_m", array(pose.translation));
        json.put("rotation_quaternion_xyzw", array(pose.rotationQuaternion));
        json.put("world_from_camera_column_major", array(pose.worldFromCamera));
        json.put("camera_from_world_column_major", array(pose.cameraFromWorld));
        json.put("forward_world", array(pose.forwardWorld));
        json.put("up_world", array(pose.upWorld));

        JSONObject imageIntrinsics = new JSONObject();
        imageIntrinsics.put("focal_length_px", array(pose.imageFocalLengthPx));
        imageIntrinsics.put("principal_point_px", array(pose.imagePrincipalPointPx));
        imageIntrinsics.put("image_dimensions", array(pose.imageDimensions));
        json.put("arcore_image_intrinsics", imageIntrinsics);

        JSONObject textureIntrinsics = new JSONObject();
        textureIntrinsics.put("focal_length_px", array(pose.textureFocalLengthPx));
        textureIntrinsics.put("principal_point_px", array(pose.texturePrincipalPointPx));
        textureIntrinsics.put("image_dimensions", array(pose.textureDimensions));
        json.put("arcore_texture_intrinsics", textureIntrinsics);

        JSONObject quality = new JSONObject();
        quality.put("source", "arcore_cpu_yuv_continuous");
        quality.put("selection", "best sharpness within moving viewpoint window");
        quality.put("sharpness_variance_laplacian", candidate.sharpness);
        quality.put("selection_score", candidate.score);
        quality.put("linear_speed_mps", candidate.linearSpeedMps);
        quality.put("angular_speed_dps", candidate.angularSpeedDps);
        quality.put("motion_is_hard_gate", false);
        quality.put("candidate_interval_ns", CANDIDATE_INTERVAL_NS);
        quality.put("selection_window_ns", SELECTION_WINDOW_NS);
        quality.put("jpeg_quality", JPEG_QUALITY);
        quality.put("max_depth_match_delta_ns", MAX_DEPTH_MATCH_DELTA_NS);
        quality.put("quality_processing_thread", "background_selector");
        json.put("capture_quality_gate", quality);
        json.put("camera2_capture", JSONObject.NULL);

        try (FileOutputStream out = new FileOutputStream(jsonFile)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        appendTrajectory(
                index,
                jpegFile.getName(),
                plyFile == null ? "" : plyFile.getName(),
                candidate.timestampNs,
                pose);

        DiagnosticLog.i(TAG,
                String.format(Locale.US,
                        "Saved continuous keyframe=%d image=%s %dx%d sharpness=%.1f score=%.1f motion=%.2fm/s %.1fdeg/s rawDepth=%s",
                        index,
                        jpegFile.getName(),
                        candidate.yuv.width,
                        candidate.yuv.height,
                        candidate.sharpness,
                        candidate.score,
                        candidate.linearSpeedMps,
                        candidate.angularSpeedDps,
                        cloud != null));
    }

    private void appendTrajectory(
            int index, String image, String cloud, long imageTimestampNs, PoseSample pose)
            throws IOException {
        File file = new File(captureRoot, "camera_trajectory.csv");
        boolean writeHeader = !file.exists() || file.length() == 0;
        try (FileOutputStream out = new FileOutputStream(file, true)) {
            if (writeHeader) {
                out.write(("index,image,point_cloud,image_timestamp_ns,pose_timestamp_ns,"
                        + "tx,ty,tz,qx,qy,qz,qw\n").getBytes(StandardCharsets.UTF_8));
            }
            String line = String.format(Locale.US,
                    "%d,%s,%s,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    index, image, cloud, imageTimestampNs, pose.timestampNs,
                    pose.translation[0], pose.translation[1], pose.translation[2],
                    pose.rotationQuaternion[0], pose.rotationQuaternion[1],
                    pose.rotationQuaternion[2], pose.rotationQuaternion[3]);
            out.write(line.getBytes(StandardCharsets.UTF_8));
        }
    }

    private void writeSessionCameraInfo(CameraCharacteristics c) {
        JSONObject json = new JSONObject();
        try {
            json.put("manufacturer", Build.MANUFACTURER);
            json.put("model", Build.MODEL);
            json.put("device", Build.DEVICE);
            json.put("sdk_int", Build.VERSION.SDK_INT);
            json.put("camera_id", cameraId);
            if (legacyJpegSize != null) {
                json.put("legacy_unused_jpeg_surface_width", legacyJpegSize.getWidth());
                json.put("legacy_unused_jpeg_surface_height", legacyJpegSize.getHeight());
            }
            putNullable(json, "sensor_orientation_degrees",
                    c.get(CameraCharacteristics.SENSOR_ORIENTATION));
            putRect(json, "active_array",
                    c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE));
            putFloatArray(json, "available_apertures",
                    c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES));
            putFloatArray(json, "available_focal_lengths_mm",
                    c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS));
            putFloatArray(json, "lens_intrinsic_calibration",
                    c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION));
            if (Build.VERSION.SDK_INT >= 28) {
                putFloatArray(json, "lens_distortion",
                        c.get(CameraCharacteristics.LENS_DISTORTION));
            }
            json.put("dataset_contents",
                    "Automatically selected ARCore CPU RGB keyframes + synchronized root-anchor camera pose; close Raw Depth PLY when available");
            json.put("rgb_observation_policy",
                    "Continuous moving capture; choose the sharpest frame per short viewpoint window; no Camera2 still request");
            json.put("preview_policy",
                    "Never interrupt the ARCore repeating preview for texture photography; frame quality work runs on a background selector");
            try (FileOutputStream out = new FileOutputStream(
                    new File(captureRoot, "session_camera.json"))) {
                out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        } catch (IOException | JSONException e) {
            DiagnosticLog.w(TAG, "Failed to write session camera metadata: " + e.getMessage());
        }
    }

    private static long safeAndroidCameraTimestamp(Frame frame, long fallback) {
        try {
            long value = frame.getAndroidCameraTimestamp();
            return value > 0 ? value : fallback;
        } catch (RuntimeException e) {
            return fallback;
        }
    }

    private Motion filterMotionLocked(Motion raw) {
        if (!hasFilteredMotion) {
            filteredLinearSpeedMps = raw.linearSpeedMps;
            filteredAngularSpeedDps = raw.angularSpeedDps;
            hasFilteredMotion = true;
        } else {
            filteredLinearSpeedMps += MOTION_FILTER_ALPHA
                    * (raw.linearSpeedMps - filteredLinearSpeedMps);
            filteredAngularSpeedDps += MOTION_FILTER_ALPHA
                    * (raw.angularSpeedDps - filteredAngularSpeedDps);
        }
        return new Motion(filteredLinearSpeedMps, filteredAngularSpeedDps);
    }

    private void resetMotionFilterLocked() {
        filteredLinearSpeedMps = 0f;
        filteredAngularSpeedDps = 0f;
        hasFilteredMotion = false;
    }

    private Motion measureMotion(Pose pose, long timestampNs) {
        if (previousPose == null
                || previousFrameTimestampNs < 0
                || timestampNs <= previousFrameTimestampNs) {
            return new Motion(0f, 0f);
        }
        float dt = (timestampNs - previousFrameTimestampNs) / 1_000_000_000f;
        if (dt <= 0f) return new Motion(0f, 0f);
        return new Motion(
                translationDistance(previousPose, pose) / dt,
                rotationDegrees(previousPose, pose) / dt);
    }

    private static double rankFrame(double sharpness, Motion motion) {
        // Pixel sharpness is authoritative. Motion only breaks close ties and never rejects a frame.
        double motionPenalty = 1.0
                + Math.min(1.5, motion.linearSpeedMps * 0.45)
                + Math.min(1.5, motion.angularSpeedDps * 0.012);
        return sharpness / motionPenalty;
    }

    private static double calculateSharpness(byte[] y, int width, int height) {
        if (y == null || width < 5 || height < 5) return 0.0;
        double sum = 0.0;
        double sumSq = 0.0;
        long count = 0L;
        int step = SHARPNESS_SAMPLE_STEP;
        for (int row = step; row < height - step; row += step) {
            int base = row * width;
            for (int col = step; col < width - step; col += step) {
                int c = y[base + col] & 0xff;
                int l = y[base + col - step] & 0xff;
                int r = y[base + col + step] & 0xff;
                int u = y[(row - step) * width + col] & 0xff;
                int d = y[(row + step) * width + col] & 0xff;
                double lap = 4.0 * c - l - r - u - d;
                sum += lap;
                sumSq += lap * lap;
                count++;
            }
        }
        if (count == 0) return 0.0;
        double mean = sum / count;
        return Math.max(0.0, sumSq / count - mean * mean);
    }

    private static YuvFrame copyCameraImage(Image image) {
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException("Expected YUV_420_888 but got " + image.getFormat());
        }
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] luma = new byte[width * height];
        copyLumaPlane(image.getPlanes()[0], width, height, luma);

        int chromaWidth = (width + 1) / 2;
        int chromaHeight = (height + 1) / 2;
        byte[] nv21 = new byte[width * height + 2 * chromaWidth * chromaHeight];
        System.arraycopy(luma, 0, nv21, 0, luma.length);
        copyChromaInterleaved(
                image.getPlanes()[2],
                image.getPlanes()[1],
                chromaWidth,
                chromaHeight,
                nv21,
                luma.length);
        return new YuvFrame(width, height, luma, nv21);
    }

    private static void copyLumaPlane(
            Image.Plane plane, int width, int height, byte[] out) {
        ByteBuffer buffer = plane.getBuffer().duplicate();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        if (pixelStride == 1) {
            for (int row = 0; row < height; row++) {
                int src = row * rowStride;
                if (src + width > buffer.limit()) break;
                buffer.position(src);
                buffer.get(out, row * width, width);
            }
            return;
        }
        for (int row = 0; row < height; row++) {
            int srcRow = row * rowStride;
            int dstRow = row * width;
            for (int col = 0; col < width; col++) {
                int src = srcRow + col * pixelStride;
                if (src >= buffer.limit()) return;
                out[dstRow + col] = buffer.get(src);
            }
        }
    }

    private static void copyChromaInterleaved(
            Image.Plane vPlane,
            Image.Plane uPlane,
            int chromaWidth,
            int chromaHeight,
            byte[] out,
            int offset) {
        ByteBuffer v = vPlane.getBuffer().duplicate();
        ByteBuffer u = uPlane.getBuffer().duplicate();
        int vRowStride = vPlane.getRowStride();
        int uRowStride = uPlane.getRowStride();
        int vPixelStride = vPlane.getPixelStride();
        int uPixelStride = uPlane.getPixelStride();
        int dst = offset;
        for (int row = 0; row < chromaHeight; row++) {
            int vRow = row * vRowStride;
            int uRow = row * uRowStride;
            for (int col = 0; col < chromaWidth; col++) {
                int vi = vRow + col * vPixelStride;
                int ui = uRow + col * uPixelStride;
                out[dst++] = vi < v.limit() ? v.get(vi) : (byte) 128;
                out[dst++] = ui < u.limit() ? u.get(ui) : (byte) 128;
            }
        }
    }

    private static byte[] encodeJpeg(YuvFrame frame) throws IOException {
        YuvImage image = new YuvImage(
                frame.nv21,
                ImageFormat.NV21,
                frame.width,
                frame.height,
                null);
        ByteArrayOutputStream out = new ByteArrayOutputStream(
                Math.max(64 * 1024, frame.width * frame.height / 2));
        if (!image.compressToJpeg(
                new Rect(0, 0, frame.width, frame.height), JPEG_QUALITY, out)) {
            throw new IOException("YUV to JPEG encoding failed");
        }
        return out.toByteArray();
    }

    private static long pixelCount(Size size) {
        return (long) size.getWidth() * size.getHeight();
    }

    private static boolean isFourByThree(Size size) {
        double ratio = (double) Math.max(size.getWidth(), size.getHeight())
                / Math.min(size.getWidth(), size.getHeight());
        return Math.abs(ratio - 4.0 / 3.0) < 0.04;
    }

    private static float translationDistance(Pose a, Pose b) {
        float[] at = a.getTranslation();
        float[] bt = b.getTranslation();
        float dx = at[0] - bt[0];
        float dy = at[1] - bt[1];
        float dz = at[2] - bt[2];
        return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    private static float rotationDegrees(Pose a, Pose b) {
        Pose relative = a.inverse().compose(b);
        float[] q = relative.getRotationQuaternion();
        float w = Math.max(-1f, Math.min(1f, Math.abs(q[3])));
        return (float) Math.toDegrees(2.0 * Math.acos(w));
    }

    private static JSONArray array(float[] values) throws JSONException {
        JSONArray out = new JSONArray();
        for (float value : values) out.put(value);
        return out;
    }

    private static JSONArray array(int[] values) throws JSONException {
        JSONArray out = new JSONArray();
        for (int value : values) out.put(value);
        return out;
    }

    private static void putNullable(JSONObject object, String key, Object value)
            throws JSONException {
        object.put(key, value == null ? JSONObject.NULL : value);
    }

    private static void putFloatArray(JSONObject object, String key, float[] values)
            throws JSONException {
        object.put(key, values == null ? JSONObject.NULL : array(values));
    }

    private static void putRect(JSONObject object, String key, Rect rect) throws JSONException {
        if (rect == null) {
            object.put(key, JSONObject.NULL);
            return;
        }
        JSONObject value = new JSONObject();
        value.put("left", rect.left);
        value.put("top", rect.top);
        value.put("right", rect.right);
        value.put("bottom", rect.bottom);
        object.put(key, value);
    }

    private static final class Motion {
        final float linearSpeedMps;
        final float angularSpeedDps;

        Motion(float linearSpeedMps, float angularSpeedDps) {
            this.linearSpeedMps = linearSpeedMps;
            this.angularSpeedDps = angularSpeedDps;
        }
    }

    private static final class CandidateSeed {
        final long timestampNs;
        final Pose localPose;
        final PoseSample pose;
        final Image image;
        final float linearSpeedMps;
        final float angularSpeedDps;

        CandidateSeed(
                long timestampNs,
                Pose localPose,
                PoseSample pose,
                Image image,
                float linearSpeedMps,
                float angularSpeedDps) {
            this.timestampNs = timestampNs;
            this.localPose = localPose;
            this.pose = pose;
            this.image = image;
            this.linearSpeedMps = linearSpeedMps;
            this.angularSpeedDps = angularSpeedDps;
        }
    }

    private static final class YuvFrame {
        final int width;
        final int height;
        final byte[] luma;
        final byte[] nv21;

        YuvFrame(int width, int height, byte[] luma, byte[] nv21) {
            this.width = width;
            this.height = height;
            this.luma = luma;
            this.nv21 = nv21;
        }
    }

    private static final class Candidate {
        final long timestampNs;
        final Pose localPose;
        final PoseSample pose;
        final YuvFrame yuv;
        final double sharpness;
        final double score;
        final float linearSpeedMps;
        final float angularSpeedDps;

        Candidate(
                long timestampNs,
                Pose localPose,
                PoseSample pose,
                YuvFrame yuv,
                double sharpness,
                double score,
                float linearSpeedMps,
                float angularSpeedDps) {
            this.timestampNs = timestampNs;
            this.localPose = localPose;
            this.pose = pose;
            this.yuv = yuv;
            this.sharpness = sharpness;
            this.score = score;
            this.linearSpeedMps = linearSpeedMps;
            this.angularSpeedDps = angularSpeedDps;
        }
    }

    private static final class PoseSample {
        final long timestampNs;
        final float[] translation;
        final float[] rotationQuaternion;
        final float[] worldFromCamera;
        final float[] cameraFromWorld;
        final float[] forwardWorld;
        final float[] upWorld;
        final float[] imageFocalLengthPx;
        final float[] imagePrincipalPointPx;
        final int[] imageDimensions;
        final float[] textureFocalLengthPx;
        final float[] texturePrincipalPointPx;
        final int[] textureDimensions;

        private PoseSample(
                long timestampNs,
                float[] translation,
                float[] rotationQuaternion,
                float[] worldFromCamera,
                float[] cameraFromWorld,
                float[] forwardWorld,
                float[] upWorld,
                float[] imageFocalLengthPx,
                float[] imagePrincipalPointPx,
                int[] imageDimensions,
                float[] textureFocalLengthPx,
                float[] texturePrincipalPointPx,
                int[] textureDimensions) {
            this.timestampNs = timestampNs;
            this.translation = translation;
            this.rotationQuaternion = rotationQuaternion;
            this.worldFromCamera = worldFromCamera;
            this.cameraFromWorld = cameraFromWorld;
            this.forwardWorld = forwardWorld;
            this.upWorld = upWorld;
            this.imageFocalLengthPx = imageFocalLengthPx;
            this.imagePrincipalPointPx = imagePrincipalPointPx;
            this.imageDimensions = imageDimensions;
            this.textureFocalLengthPx = textureFocalLengthPx;
            this.texturePrincipalPointPx = texturePrincipalPointPx;
            this.textureDimensions = textureDimensions;
        }

        static PoseSample from(long timestampNs, Camera camera, Pose rootPose) {
            Pose pose = rootPose.inverse().compose(camera.getPose());
            float[] worldFromCamera = new float[16];
            float[] cameraFromWorld = new float[16];
            pose.toMatrix(worldFromCamera, 0);
            pose.inverse().toMatrix(cameraFromWorld, 0);
            CameraIntrinsics image = camera.getImageIntrinsics();
            CameraIntrinsics texture = camera.getTextureIntrinsics();
            return new PoseSample(
                    timestampNs,
                    pose.getTranslation().clone(),
                    pose.getRotationQuaternion().clone(),
                    worldFromCamera,
                    cameraFromWorld,
                    pose.rotateVector(new float[] {0f, 0f, -1f}),
                    pose.rotateVector(new float[] {0f, 1f, 0f}),
                    image.getFocalLength().clone(),
                    image.getPrincipalPoint().clone(),
                    image.getImageDimensions().clone(),
                    texture.getFocalLength().clone(),
                    texture.getPrincipalPoint().clone(),
                    texture.getImageDimensions().clone());
        }
    }
}
