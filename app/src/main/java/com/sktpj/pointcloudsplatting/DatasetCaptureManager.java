package com.sktpj.pointcloudsplatting;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Environment;
import android.util.Size;

import com.google.ar.core.Camera;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.ImageMetadata;
import com.google.ar.core.Pose;
import com.google.ar.core.exceptions.MetadataNotFoundException;
import com.google.ar.core.exceptions.NotYetAvailableException;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

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
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Saves high-resolution RGB + camera pose as primary observations and Raw Depth as an optional prior. */
public final class DatasetCaptureManager {
    private static final String TAG = "DatasetCapture";

    private static final long MAX_JPEG_PIXELS = 13_000_000L;
    private static final long MIN_CAPTURE_INTERVAL_NS = 700_000_000L;
    private static final long FORCE_CAPTURE_AFTER_NS = 2_000_000_000L;
    private static final float MIN_VIEW_TRANSLATION_METERS = 0.03f;
    private static final float MIN_VIEW_ROTATION_DEGREES = 2.5f;

    private static final float MAX_LINEAR_SPEED_MPS = 0.25f;
    private static final float MAX_ANGULAR_SPEED_DPS = 20.0f;
    private static final float MOTION_FILTER_ALPHA = 0.35f;
    private static final long MAX_POSE_MATCH_DELTA_NS = 75_000_000L;
    private static final long MAX_DEPTH_MATCH_DELTA_NS = 250_000_000L;
    // Raw Depth is a geometry prior, not a reason to throw away a valid RGB/Pose observation.
    // Give the same-time Depth stream a short grace window, then keep RGB+Pose without per-frame PLY.
    private static final long MAX_DEPTH_WAIT_NS = 450_000_000L;
    private static final long MAX_STILL_EXPOSURE_NS = 8_000_000L; // 1/125 s indoor fallback
    private static final int MAX_STILL_ISO = 6400;

    private static final int MAX_POSE_SAMPLES = 180;
    private static final int MAX_DEPTH_SAMPLES = 32;
    private static final int MAX_FALLBACK_DEPTH_PRIORS = 6;

    private final File captureRoot;
    private final ExecutorService writer = Executors.newSingleThreadExecutor();
    private final AtomicInteger captureSequence = new AtomicInteger();
    private final ArrayDeque<PoseSample> poseSamples = new ArrayDeque<>();
    private final ArrayDeque<WorldPointCloudSnapshot> depthSamples = new ArrayDeque<>();
    private final Map<Long, PendingJpeg> pendingJpegs = new HashMap<>();
    private final Map<Long, StillMetadata> stillMetadata = new HashMap<>();

    private Pose previousPose;
    private long previousFrameTimestampNs = -1L;
    private Pose lastRequestedPose;
    private long lastRequestedTimestampNs = -1L;
    private float filteredLinearSpeedMps;
    private float filteredAngularSpeedDps;
    private boolean hasFilteredMotion;
    private boolean captureInFlight;
    private boolean cameraActive = true;
    private boolean fallbackDepthPriorsQueued;
    private volatile boolean captureEnabled = true;
    private volatile int savedCount;
    private volatile String lastDecision = "waiting for first stable view";

    private String cameraId = "unknown";
    private Size jpegSize;

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

    /** Pixel 10a: prefer the 4:3 binned 12 MP stream rather than the full 48 MP mode. */
    public static Size chooseJpegSize(CameraCharacteristics characteristics) {
        StreamConfigurationMap map = characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) {
            throw new IllegalStateException("No camera stream configuration map");
        }
        Size[] outputs = map.getOutputSizes(ImageFormat.JPEG);
        if (outputs == null || outputs.length == 0) {
            throw new IllegalStateException("No JPEG output sizes");
        }
        List<Size> sizes = new ArrayList<>();
        java.util.Collections.addAll(sizes, outputs);
        sizes.sort(Comparator.comparingLong(DatasetCaptureManager::pixelCount).reversed());
        for (Size size : sizes) {
            if (pixelCount(size) <= MAX_JPEG_PIXELS && isFourByThree(size)) {
                return size;
            }
        }
        for (Size size : sizes) {
            if (pixelCount(size) <= MAX_JPEG_PIXELS) {
                return size;
            }
        }
        return sizes.get(sizes.size() - 1);
    }

    public synchronized void configureCamera(
            String cameraId, Size jpegSize, CameraCharacteristics characteristics) {
        this.cameraId = cameraId;
        this.jpegSize = jpegSize;
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
        captureInFlight = false;
        previousPose = null;
        previousFrameTimestampNs = -1L;
        lastRequestedPose = null;
        lastRequestedTimestampNs = -1L;
        resetMotionFilterLocked();
        poseSamples.clear();
        depthSamples.clear();
        pendingJpegs.clear();
        stillMetadata.clear();
        if (captureEnabled) {
            lastDecision = "camera resumed";
        }
        DiagnosticLog.i(TAG, "Camera synchronization state reset on resume");
    }

    public synchronized void onCameraPaused() {
        int discardedPending = pendingJpegs.size();
        cameraActive = false;
        captureInFlight = false;
        pendingJpegs.clear();
        stillMetadata.clear();
        poseSamples.clear();
        depthSamples.clear();
        previousPose = null;
        previousFrameTimestampNs = -1L;
        lastRequestedPose = null;
        lastRequestedTimestampNs = -1L;
        resetMotionFilterLocked();
        lastDecision = captureEnabled ? "camera paused" : "capture stopped; finalizing";
        DiagnosticLog.i(TAG,
                "Camera synchronization state reset on pause pending=" + discardedPending);
    }

    public synchronized void onDepthFrame(DepthData depth, Pose rootPose) {
        if (!cameraActive) {
            return;
        }
        try {
            WorldPointCloudSnapshot snapshot = WorldPointCloudSnapshot.from(depth, rootPose);
            depthSamples.addLast(snapshot);
            while (depthSamples.size() > MAX_DEPTH_SAMPLES) {
                depthSamples.removeFirst();
            }
            tryFinalizePendingLocked();
        } catch (RuntimeException e) {
            DiagnosticLog.w(TAG, "Failed to snapshot Raw Depth: " + e.getMessage());
        }
    }

    /** Records pose continuously and returns true when a useful, reasonably stable photo is due. */
    public synchronized boolean onArFrame(Frame frame, Camera camera, Pose rootPose) {
        if (!cameraActive) {
            lastDecision = "camera paused";
            return false;
        }
        long timestampNs = frame.getTimestamp();
        Pose pose = rootPose.inverse().compose(camera.getPose());
        Motion motion = filterMotionLocked(measureMotion(pose, timestampNs));

        poseSamples.addLast(PoseSample.from(timestampNs, camera, rootPose));
        while (poseSamples.size() > MAX_POSE_SAMPLES) {
            poseSamples.removeFirst();
        }

        previousPose = pose;
        previousFrameTimestampNs = timestampNs;
        tryFinalizePendingLocked();

        if (!captureEnabled) {
            lastDecision = "capture stopped; finalizing";
            return false;
        }
        if (captureInFlight) {
            lastDecision = "photo capture in flight";
            return false;
        }
        if (lastRequestedTimestampNs >= 0
                && timestampNs - lastRequestedTimestampNs < MIN_CAPTURE_INTERVAL_NS) {
            lastDecision = "waiting for next viewpoint";
            return false;
        }
        if (!hasUsefulViewpointChange(pose, timestampNs)) {
            lastDecision = "move to a new viewpoint";
            return false;
        }

        ArFrameMetadata metadata = readArFrameMetadata(frame);
        if (motion.linearSpeedMps > MAX_LINEAR_SPEED_MPS
                || motion.angularSpeedDps > MAX_ANGULAR_SPEED_DPS) {
            lastDecision = String.format(Locale.US,
                    "waiting for lower blur: %.2f m/s %.1f deg/s",
                    motion.linearSpeedMps, motion.angularSpeedDps);
            return false;
        }
        if (metadata.lensState != null
                && metadata.lensState != CaptureResult.LENS_STATE_STATIONARY) {
            lastDecision = "waiting for lens to stop";
            return false;
        }
        if (!isFocusedState(metadata.afState)) {
            lastDecision = "waiting for focus";
            return false;
        }

        captureInFlight = true;
        lastRequestedPose = pose;
        lastRequestedTimestampNs = timestampNs;
        lastDecision = "requesting texture photo";
        return true;
    }

    public synchronized void onCaptureRequestFailed(String reason) {
        captureInFlight = false;
        lastDecision = reason == null ? "photo request failed" : reason;
    }

    public synchronized void onCaptureCompleted(TotalCaptureResult result) {
        if (!cameraActive) {
            return;
        }
        Long timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP);
        if (timestamp == null) {
            captureInFlight = false;
            lastDecision = "photo missing sensor timestamp";
            return;
        }
        stillMetadata.put(timestamp, StillMetadata.from(result));
        trimMetadataLocked();
        tryFinalizePendingLocked();
    }

    public void onJpegAvailable(ImageReader reader) {
        Image image = null;
        try {
            image = reader.acquireNextImage();
            if (image == null) {
                return;
            }
            ByteBuffer buffer = image.getPlanes()[0].getBuffer();
            byte[] jpeg = new byte[buffer.remaining()];
            buffer.get(jpeg);
            long timestamp = image.getTimestamp();
            synchronized (this) {
                if (!cameraActive) {
                    return;
                }
                pendingJpegs.put(timestamp, new PendingJpeg(timestamp, jpeg));
                captureInFlight = false;
                tryFinalizePendingLocked();
            }
        } catch (RuntimeException e) {
            synchronized (this) {
                captureInFlight = false;
                lastDecision = "failed to read JPEG";
            }
            DiagnosticLog.e(TAG, "Failed to read JPEG", e);
        } finally {
            if (image != null) {
                image.close();
            }
        }
    }

    /** Stops requesting new frames and waits for the current Camera2/JPEG + disk writes. */
    public boolean stopCaptureAndFlush(long timeoutMs) {
        long deadline = System.currentTimeMillis() + Math.max(1L, timeoutMs);
        synchronized (this) {
            captureEnabled = false;
            lastDecision = "capture stopped; finalizing";
            queueFallbackDepthPriorsLocked();
        }

        while (true) {
            boolean settled;
            synchronized (this) {
                tryFinalizePendingLocked();
                settled = !captureInFlight && pendingJpegs.isEmpty();
            }
            if (settled) {
                break;
            }
            if (System.currentTimeMillis() >= deadline) {
                synchronized (this) {
                    lastDecision = "save timed out waiting for Camera2 frame";
                }
                return false;
            }
            try {
                Thread.sleep(20L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }

        long remaining = Math.max(1L, deadline - System.currentTimeMillis());
        try {
            writer.submit(() -> {}).get(remaining, TimeUnit.MILLISECONDS);
            synchronized (this) {
                lastDecision = "capture stopped; ready to save";
            }
            return true;
        } catch (Exception e) {
            DiagnosticLog.w(TAG, "Timed out flushing dataset writer: " + e.getMessage());
            synchronized (this) {
                lastDecision = "save timed out flushing files";
            }
            return false;
        }
    }

    public synchronized void resumeCapture() {
        captureEnabled = true;
        lastDecision = "capture resumed";
    }

    /**
     * Saves a small set of root-anchor Raw Depth snapshots for 3DGS initialization. These are
     * intentionally independent of JPEG timestamp matching: a root-local geometry prior does not
     * need to be the same instant as a photometric observation. Per-frame PLY remains synchronized
     * when available; these files only prevent valid rolling depth from being discarded at Save.
     */
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
        DiagnosticLog.i(TAG,
                "Queued fallback depth priors=" + queued
                        + " rollingDepthSnapshots=" + available.size());
    }

    public void shutdown() {
        writer.shutdown();
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
        if (dt <= 0f) {
            return new Motion(0f, 0f);
        }
        return new Motion(
                translationDistance(previousPose, pose) / dt,
                rotationDegrees(previousPose, pose) / dt);
    }

    private boolean hasUsefulViewpointChange(Pose pose, long timestampNs) {
        if (lastRequestedPose == null || lastRequestedTimestampNs < 0) {
            return true;
        }
        if (timestampNs - lastRequestedTimestampNs >= FORCE_CAPTURE_AFTER_NS) {
            return true;
        }
        return translationDistance(lastRequestedPose, pose) >= MIN_VIEW_TRANSLATION_METERS
                || rotationDegrees(lastRequestedPose, pose) >= MIN_VIEW_ROTATION_DEGREES;
    }

    private void tryFinalizePendingLocked() {
        if (pendingJpegs.isEmpty() || poseSamples.isEmpty()) {
            return;
        }
        long newestPoseTimestamp = poseSamples.peekLast().timestampNs;
        Iterator<Map.Entry<Long, PendingJpeg>> iterator = pendingJpegs.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Long, PendingJpeg> entry = iterator.next();
            long imageTimestamp = entry.getKey();
            StillMetadata metadata = stillMetadata.get(imageTimestamp);
            if (metadata == null || newestPoseTimestamp < imageTimestamp) {
                continue;
            }

            if (!isStillSharpEnough(metadata)) {
                iterator.remove();
                stillMetadata.remove(imageTimestamp);
                lastDecision = "discarded photo: exposure/focus quality gate";
                DiagnosticLog.w(TAG,
                        "Discarding texture photo quality gate ts=" + imageTimestamp
                                + " exp=" + metadata.exposureTimeNs + " iso=" + metadata.iso);
                continue;
            }

            PoseSample pose = findNearestPoseLocked(imageTimestamp);
            if (pose == null
                    || Math.abs(pose.timestampNs - imageTimestamp) > MAX_POSE_MATCH_DELTA_NS) {
                iterator.remove();
                stillMetadata.remove(imageTimestamp);
                lastDecision = "discarded photo: no synchronized ARCore pose";
                long deltaNs = pose == null ? Long.MAX_VALUE : pose.timestampNs - imageTimestamp;
                DiagnosticLog.w(TAG,
                        "No close pose for JPEG ts=" + imageTimestamp
                                + " nearestDeltaMs=" + formatDeltaMs(deltaNs));
                continue;
            }

            WorldPointCloudSnapshot nearestDepth = findNearestDepthLocked(imageTimestamp);
            long depthDeltaNs = nearestDepth == null
                    ? Long.MAX_VALUE : nearestDepth.getTimestampNs() - imageTimestamp;
            boolean closeDepth = nearestDepth != null
                    && Math.abs(depthDeltaNs) <= MAX_DEPTH_MATCH_DELTA_NS;

            if (!closeDepth && newestPoseTimestamp < imageTimestamp + MAX_DEPTH_WAIT_NS) {
                lastDecision = "waiting briefly for Raw Depth prior";
                continue;
            }

            WorldPointCloudSnapshot cloud = closeDepth ? nearestDepth : null;
            if (cloud == null) {
                DiagnosticLog.i(TAG,
                        "Saving RGB/Pose keyframe without close Raw Depth ts=" + imageTimestamp
                                + " nearestDeltaMs=" + formatDeltaMs(depthDeltaNs));
            }

            iterator.remove();
            stillMetadata.remove(imageTimestamp);
            PendingJpeg jpeg = entry.getValue();
            int index = captureSequence.incrementAndGet();
            savedCount = index;
            lastDecision = "saved frame " + index;
            writer.execute(() -> {
                try {
                    writeCapture(index, jpeg, pose, metadata, cloud);
                } catch (IOException | JSONException e) {
                    DiagnosticLog.e(TAG, "Failed to write dataset frame", e);
                }
            });
        }
    }

    private PoseSample findNearestPoseLocked(long timestampNs) {
        PoseSample best = null;
        long bestDelta = Long.MAX_VALUE;
        for (PoseSample sample : poseSamples) {
            long delta = Math.abs(sample.timestampNs - timestampNs);
            if (delta < bestDelta) {
                best = sample;
                bestDelta = delta;
            }
        }
        return best;
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

    private void trimMetadataLocked() {
        if (stillMetadata.size() <= 12) {
            return;
        }
        long oldestPose = poseSamples.isEmpty()
                ? Long.MIN_VALUE : poseSamples.peekFirst().timestampNs;
        stillMetadata.entrySet().removeIf(
                e -> e.getKey() < oldestPose - MAX_POSE_MATCH_DELTA_NS);
    }

    private void writeCapture(
            int index,
            PendingJpeg jpeg,
            PoseSample pose,
            StillMetadata metadata,
            WorldPointCloudSnapshot cloud)
            throws IOException, JSONException {
        String base = String.format(Locale.US, "frame_%06d_%d", index, jpeg.timestampNs);
        File jpegFile = new File(captureRoot, base + ".jpg");
        File jsonFile = new File(captureRoot, base + ".json");
        File plyFile = cloud == null ? null : new File(captureRoot, base + ".ply");

        try (FileOutputStream out = new FileOutputStream(jpegFile)) {
            out.write(jpeg.bytes);
        }
        if (cloud != null) {
            cloud.writePly(plyFile);
        }

        JSONObject json = new JSONObject();
        json.put("capture_index", index);
        json.put("image", jpegFile.getName());
        json.put("point_cloud", plyFile == null ? JSONObject.NULL : plyFile.getName());
        json.put("has_raw_depth_prior", cloud != null);
        json.put("jpeg_width", jpegSize == null ? JSONObject.NULL : jpegSize.getWidth());
        json.put("jpeg_height", jpegSize == null ? JSONObject.NULL : jpegSize.getHeight());
        json.put("jpeg_sensor_timestamp_ns", jpeg.timestampNs);
        json.put("arcore_pose_timestamp_ns", pose.timestampNs);
        json.put("pose_timestamp_delta_ns", pose.timestampNs - jpeg.timestampNs);
        json.put("raw_depth_timestamp_ns",
                cloud == null ? JSONObject.NULL : cloud.getTimestampNs());
        json.put("depth_timestamp_delta_ns",
                cloud == null ? JSONObject.NULL : cloud.getTimestampNs() - jpeg.timestampNs);
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
        json.put("camera2_capture", metadata.toJson());

        JSONObject quality = new JSONObject();
        quality.put("target_shutter", "1/500 s");
        quality.put("max_saved_exposure_ns", MAX_STILL_EXPOSURE_NS);
        quality.put("max_saved_iso", MAX_STILL_ISO);
        quality.put("max_linear_speed_mps", MAX_LINEAR_SPEED_MPS);
        quality.put("max_angular_speed_dps", MAX_ANGULAR_SPEED_DPS);
        quality.put("max_pose_match_delta_ns", MAX_POSE_MATCH_DELTA_NS);
        quality.put("max_depth_match_delta_ns", MAX_DEPTH_MATCH_DELTA_NS);
        quality.put("depth_wait_grace_ns", MAX_DEPTH_WAIT_NS);
        quality.put("raw_depth_required_for_rgb_keyframe", false);
        json.put("capture_quality_gate", quality);

        try (FileOutputStream out = new FileOutputStream(jsonFile)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        appendTrajectory(
                index,
                jpegFile.getName(),
                plyFile == null ? "" : plyFile.getName(),
                jpeg.timestampNs,
                pose);
        DiagnosticLog.i(TAG,
                "Saved dataset frame=" + index + " photo=" + jpegFile.getName()
                        + " points=" + (cloud == null ? 0 : cloud.getPointCount())
                        + " rawDepth=" + (cloud != null));
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
            if (jpegSize != null) {
                json.put("jpeg_width", jpegSize.getWidth());
                json.put("jpeg_height", jpegSize.getHeight());
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
                    "Each RGB frame has .jpg + synchronized root-anchor camera pose JSON; close Raw Depth .ply is attached when available");
            json.put("rgb_observation_policy",
                    "Valid high-resolution JPEG + pose is primary and is not discarded only because Raw Depth is late/missing");
            json.put("capture_policy",
                    "Prefer 1/500 s; indoor fallback no slower than 1/125 s and no higher than ISO 6400");
            json.put("stabilization_policy", "EIS OFF; OIS OFF for saved stills");
            try (FileOutputStream out = new FileOutputStream(
                    new File(captureRoot, "session_camera.json"))) {
                out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        } catch (IOException | JSONException e) {
            DiagnosticLog.w(TAG, "Failed to write session camera metadata: " + e.getMessage());
        }
    }

    private static boolean isStillSharpEnough(StillMetadata metadata) {
        if (metadata.exposureTimeNs == null || metadata.exposureTimeNs > MAX_STILL_EXPOSURE_NS) {
            return false;
        }
        if (metadata.iso == null || metadata.iso > MAX_STILL_ISO) {
            return false;
        }
        if (metadata.lensState != null
                && metadata.lensState != CaptureResult.LENS_STATE_STATIONARY) {
            return false;
        }
        if (metadata.oisMode != null
                && metadata.oisMode != CaptureResult.LENS_OPTICAL_STABILIZATION_MODE_OFF) {
            return false;
        }
        if (metadata.videoStabilizationMode != null
                && metadata.videoStabilizationMode
                != CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE_OFF) {
            return false;
        }
        return isFocusedState(metadata.afState);
    }

    private static boolean isFocusedState(Integer afState) {
        if (afState == null) {
            return true;
        }
        return afState == CaptureResult.CONTROL_AF_STATE_INACTIVE
                || afState == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED
                || afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED;
    }

    private static ArFrameMetadata readArFrameMetadata(Frame frame) {
        ArFrameMetadata out = new ArFrameMetadata();
        try {
            ImageMetadata metadata = frame.getImageMetadata();
            Byte lens = getByte(metadata, ImageMetadata.LENS_STATE);
            out.lensState = lens == null ? null : lens & 0xff;
            Byte af = getByte(metadata, ImageMetadata.CONTROL_AF_STATE);
            out.afState = af == null ? null : af & 0xff;
        } catch (NotYetAvailableException ignored) {
        }
        return out;
    }

    private static Byte getByte(ImageMetadata metadata, int key) {
        try {
            return metadata.getByte(key);
        } catch (MetadataNotFoundException | IllegalArgumentException e) {
            return null;
        }
    }

    private static String formatDeltaMs(long deltaNs) {
        return deltaNs == Long.MAX_VALUE
                ? "none"
                : String.format(Locale.US, "%.3f", deltaNs / 1_000_000.0);
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
        for (float value : values) {
            out.put(value);
        }
        return out;
    }

    private static JSONArray array(int[] values) throws JSONException {
        JSONArray out = new JSONArray();
        for (int value : values) {
            out.put(value);
        }
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

    private static final class ArFrameMetadata {
        Integer lensState;
        Integer afState;
    }

    private static final class PendingJpeg {
        final long timestampNs;
        final byte[] bytes;

        PendingJpeg(long timestampNs, byte[] bytes) {
            this.timestampNs = timestampNs;
            this.bytes = bytes;
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

    private static final class StillMetadata {
        final Long exposureTimeNs;
        final Integer iso;
        final Float aperture;
        final Float focalLengthMm;
        final Float focusDistanceDiopters;
        final Integer afMode;
        final Integer afState;
        final Integer aeState;
        final Integer awbState;
        final Integer lensState;
        final Integer oisMode;
        final Integer videoStabilizationMode;
        final Long rollingShutterSkewNs;
        final Long frameDurationNs;
        final Rect cropRegion;
        final Integer jpegOrientationDegrees;

        StillMetadata(TotalCaptureResult result) {
            exposureTimeNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
            iso = result.get(CaptureResult.SENSOR_SENSITIVITY);
            aperture = result.get(CaptureResult.LENS_APERTURE);
            focalLengthMm = result.get(CaptureResult.LENS_FOCAL_LENGTH);
            focusDistanceDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE);
            afMode = result.get(CaptureResult.CONTROL_AF_MODE);
            afState = result.get(CaptureResult.CONTROL_AF_STATE);
            aeState = result.get(CaptureResult.CONTROL_AE_STATE);
            awbState = result.get(CaptureResult.CONTROL_AWB_STATE);
            lensState = result.get(CaptureResult.LENS_STATE);
            oisMode = result.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE);
            videoStabilizationMode = result.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE);
            rollingShutterSkewNs = result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW);
            frameDurationNs = result.get(CaptureResult.SENSOR_FRAME_DURATION);
            Rect crop = result.get(CaptureResult.SCALER_CROP_REGION);
            cropRegion = crop == null ? null : new Rect(crop);
            jpegOrientationDegrees = result.getRequest().get(CaptureRequest.JPEG_ORIENTATION);
        }

        static StillMetadata from(TotalCaptureResult result) {
            return new StillMetadata(result);
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            putNullable(json, "exposure_time_ns", exposureTimeNs);
            putNullable(json, "iso", iso);
            putNullable(json, "aperture_f_number", aperture);
            putNullable(json, "focal_length_mm", focalLengthMm);
            putNullable(json, "focus_distance_diopters", focusDistanceDiopters);
            putNullable(json, "af_mode", afMode);
            putNullable(json, "af_state", afState);
            putNullable(json, "ae_state", aeState);
            putNullable(json, "awb_state", awbState);
            putNullable(json, "lens_state", lensState);
            putNullable(json, "ois_mode", oisMode);
            putNullable(json, "video_stabilization_mode", videoStabilizationMode);
            putNullable(json, "rolling_shutter_skew_ns", rollingShutterSkewNs);
            putNullable(json, "frame_duration_ns", frameDurationNs);
            putNullable(json, "jpeg_orientation_degrees", jpegOrientationDegrees);
            putRect(json, "crop_region", cropRegion);
            return json;
        }
    }
}
