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

/**
 * Phase 1 observation recorder.
 *
 * <p>High-resolution Camera2 JPEGs are the RGB observations of record. Raw Depth observations are
 * saved independently in the dataset root coordinate system. A nearby depth observation may also
 * be emitted as a legacy frame_*.ply compatibility artifact, but RGB validity never depends on it.
 */
public final class DatasetCaptureManager {
    private static final String TAG = "DatasetCapture";

    private static final long MAX_JPEG_PIXELS = 13_000_000L;
    private static final long MIN_CAPTURE_INTERVAL_NS = 700_000_000L;
    private static final long FORCE_CAPTURE_AFTER_NS = 2_000_000_000L;
    private static final float MIN_VIEW_TRANSLATION_METERS = 0.03f;
    private static final float MIN_VIEW_ROTATION_DEGREES = 2.5f;
    private static final float MAX_LINEAR_SPEED_MPS = 0.25f;
    private static final float MAX_ANGULAR_SPEED_DPS = 20.0f;

    private static final long MAX_POSE_MATCH_DELTA_NS = 75_000_000L;
    private static final long MAX_DEPTH_REFERENCE_DELTA_NS = 250_000_000L;
    private static final long MIN_DEPTH_OBSERVATION_INTERVAL_NS = 250_000_000L;
    private static final int MAX_POSE_SAMPLES = 180;
    private static final int MAX_DEPTH_SAMPLES = 48;
    private static final int MAX_FALLBACK_DEPTH_PRIORS = 6;

    // ScannerActivity applies the selected indoor/outdoor limit before issuing the request. These
    // are the broad absolute save guards so an unexpectedly slow/noisy result is still rejected.
    private static final long MAX_STILL_EXPOSURE_NS = 8_000_000L;
    private static final int MAX_STILL_ISO = 6400;

    private final File captureRoot;
    private final ExecutorService writer = Executors.newSingleThreadExecutor();
    private final AtomicInteger captureSequence = new AtomicInteger();
    private final AtomicInteger depthSequence = new AtomicInteger();
    private final ArrayDeque<PoseSample> poseSamples = new ArrayDeque<>();
    private final ArrayDeque<DepthObservation> depthSamples = new ArrayDeque<>();
    private final Map<Long, PendingJpeg> pendingJpegs = new HashMap<>();
    private final Map<Long, StillMetadata> stillMetadata = new HashMap<>();

    private Pose previousPose;
    private long previousMotionTimestampNs = -1L;
    private Pose lastRequestedPose;
    private long lastRequestedTimestampNs = -1L;
    private long lastSavedDepthTimestampNs = -1L;
    private boolean captureInFlight;
    private boolean cameraActive = true;
    private boolean fallbackDepthPriorsQueued;
    private volatile boolean captureEnabled = true;
    private volatile int savedCount;
    private volatile int savedDepthCount;
    private volatile String lastDecision = "waiting for first stable view";

    private String cameraId = "unknown";
    private Size jpegSize;
    private Rect activeArray;
    private Rect preCorrectionActiveArray;
    private float[] lensIntrinsicCalibration;
    private float[] lensDistortion;
    private Integer sensorOrientationDegrees;

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

    /** Prefer the largest binned-quality JPEG up to about 13 MP; do not couple it to ARCore size. */
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
        Rect active = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        Rect pre = characteristics.get(
                CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE);
        activeArray = active == null ? null : new Rect(active);
        preCorrectionActiveArray = pre == null ? null : new Rect(pre);
        float[] intr = characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
        lensIntrinsicCalibration = intr == null ? null : intr.clone();
        if (Build.VERSION.SDK_INT >= 28) {
            float[] distortion = characteristics.get(CameraCharacteristics.LENS_DISTORTION);
            lensDistortion = distortion == null ? null : distortion.clone();
        } else {
            lensDistortion = null;
        }
        sensorOrientationDegrees = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        writeSessionCameraInfo(characteristics);
    }

    public int getSavedCount() {
        return savedCount;
    }

    public int getSavedDepthObservationCount() {
        return savedDepthCount;
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
        previousMotionTimestampNs = -1L;
        lastRequestedPose = null;
        lastRequestedTimestampNs = -1L;
        poseSamples.clear();
        depthSamples.clear();
        pendingJpegs.clear();
        stillMetadata.clear();
        if (captureEnabled) {
            lastDecision = "camera resumed";
        }
        DiagnosticLog.i(TAG, "Camera observation synchronization state reset on resume");
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
        previousMotionTimestampNs = -1L;
        lastRequestedPose = null;
        lastRequestedTimestampNs = -1L;
        lastDecision = captureEnabled ? "camera paused" : "capture stopped; finalizing";
        DiagnosticLog.i(TAG,
                "Camera observation synchronization state reset on pause pending="
                        + discardedPending);
    }

    /** Compatibility overload. New callers should pass Frame.getAndroidCameraTimestamp(). */
    public synchronized void onDepthFrame(DepthData depth, Pose rootPose) {
        onDepthFrame(depth, rootPose, -1L);
    }

    /** Saves Raw Depth as an independent geometry observation, not as an RGB-frame requirement. */
    public synchronized void onDepthFrame(
            DepthData depth, Pose rootPose, long androidCameraTimestampNs) {
        if (!cameraActive || depth == null || rootPose == null) {
            return;
        }
        try {
            WorldPointCloudSnapshot snapshot = WorldPointCloudSnapshot.from(depth, rootPose);
            Pose rootFromDepthCamera = rootPose.inverse().compose(depth.getCameraPose());
            long effectiveTimestamp = androidCameraTimestampNs > 0
                    ? androidCameraTimestampNs : depth.getTimestamp();

            if (lastSavedDepthTimestampNs > 0
                    && effectiveTimestamp > lastSavedDepthTimestampNs
                    && effectiveTimestamp - lastSavedDepthTimestampNs
                    < MIN_DEPTH_OBSERVATION_INTERVAL_NS) {
                return;
            }

            int index = depthSequence.incrementAndGet();
            DepthObservation observation = new DepthObservation(
                    index,
                    depth.getTimestamp(),
                    androidCameraTimestampNs,
                    rootFromDepthCamera,
                    snapshot);
            depthSamples.addLast(observation);
            while (depthSamples.size() > MAX_DEPTH_SAMPLES) {
                depthSamples.removeFirst();
            }
            lastSavedDepthTimestampNs = effectiveTimestamp;
            savedDepthCount = index;

            if (captureEnabled) {
                writer.execute(() -> {
                    try {
                        writeDepthObservation(observation);
                    } catch (IOException | JSONException e) {
                        DiagnosticLog.e(TAG, "Failed to write independent Raw Depth observation", e);
                    }
                });
            }
        } catch (RuntimeException e) {
            DiagnosticLog.w(TAG, "Failed to snapshot Raw Depth: " + e.getMessage());
        }
    }

    /** Records ARCore poses continuously and requests a high-resolution still when quality permits. */
    public synchronized boolean onArFrame(Frame frame, Camera camera, Pose rootPose) {
        if (!cameraActive) {
            lastDecision = "camera paused";
            return false;
        }

        long arFrameTimestampNs = frame.getTimestamp();
        long androidCameraTimestampNs = safeAndroidCameraTimestamp(frame);
        long correlationTimestampNs = androidCameraTimestampNs > 0
                ? androidCameraTimestampNs : arFrameTimestampNs;
        Pose rootFromCamera = rootPose.inverse().compose(camera.getPose());
        Motion motion = measureMotion(rootFromCamera, correlationTimestampNs);

        poseSamples.addLast(PoseSample.from(
                androidCameraTimestampNs,
                arFrameTimestampNs,
                camera,
                rootPose));
        while (poseSamples.size() > MAX_POSE_SAMPLES) {
            poseSamples.removeFirst();
        }

        previousPose = rootFromCamera;
        previousMotionTimestampNs = correlationTimestampNs;
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
                && correlationTimestampNs - lastRequestedTimestampNs < MIN_CAPTURE_INTERVAL_NS) {
            lastDecision = "waiting for next viewpoint";
            return false;
        }
        if (!hasUsefulViewpointChange(rootFromCamera, correlationTimestampNs)) {
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
        lastRequestedPose = rootFromCamera;
        lastRequestedTimestampNs = correlationTimestampNs;
        lastDecision = "requesting high-resolution photo";
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
                lastDecision = "failed to read high-resolution JPEG";
            }
            DiagnosticLog.e(TAG, "Failed to read high-resolution JPEG", e);
        } finally {
            if (image != null) {
                image.close();
            }
        }
    }

    /** Stops new capture requests and flushes Camera2/JPEG and independent Depth writes. */
    public boolean stopCaptureAndFlush(long timeoutMs) {
        long deadline = System.currentTimeMillis() + Math.max(1L, timeoutMs);
        synchronized (this) {
            captureEnabled = false;
            lastDecision = "capture stopped; finalizing";
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

        synchronized (this) {
            queueFallbackDepthPriorsLocked();
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
        fallbackDepthPriorsQueued = false;
        lastDecision = "capture resumed";
    }

    public void shutdown() {
        writer.shutdown();
    }

    private Motion measureMotion(Pose pose, long timestampNs) {
        if (previousPose == null
                || previousMotionTimestampNs < 0
                || timestampNs <= previousMotionTimestampNs) {
            return new Motion(0f, 0f);
        }
        float dt = (timestampNs - previousMotionTimestampNs) / 1_000_000_000f;
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
        PoseSample newest = poseSamples.peekLast();
        long newestCameraTimestamp = newest == null ? -1L : newest.correlationTimestampNs();

        Iterator<Map.Entry<Long, PendingJpeg>> iterator = pendingJpegs.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<Long, PendingJpeg> entry = iterator.next();
            long imageTimestamp = entry.getKey();
            StillMetadata metadata = stillMetadata.get(imageTimestamp);
            if (metadata == null) {
                continue;
            }

            if (!isStillSharpEnough(metadata)) {
                iterator.remove();
                stillMetadata.remove(imageTimestamp);
                lastDecision = "discarded photo: exposure/focus quality gate";
                DiagnosticLog.w(TAG,
                        "Discarding high-resolution JPEG quality gate ts=" + imageTimestamp
                                + " exp=" + metadata.exposureTimeNs + " iso=" + metadata.iso);
                continue;
            }

            ResolvedPose pose = resolvePoseLocked(imageTimestamp);
            if (pose == null) {
                if (newestCameraTimestamp > 0
                        && newestCameraTimestamp < imageTimestamp + MAX_POSE_MATCH_DELTA_NS) {
                    lastDecision = "waiting for exposure-time pose";
                    continue;
                }
                iterator.remove();
                stillMetadata.remove(imageTimestamp);
                lastDecision = "discarded photo: no synchronized ARCore camera pose";
                DiagnosticLog.w(TAG,
                        "No Camera2-correlated ARCore pose for JPEG ts=" + imageTimestamp);
                continue;
            }

            DepthObservation depthReference = findNearestDepthLocked(imageTimestamp);
            long depthDeltaNs = depthReference == null
                    ? Long.MAX_VALUE
                    : depthReference.correlationTimestampNs() - imageTimestamp;
            if (depthReference != null
                    && Math.abs(depthDeltaNs) > MAX_DEPTH_REFERENCE_DELTA_NS) {
                depthReference = null;
                depthDeltaNs = Long.MAX_VALUE;
            }

            iterator.remove();
            stillMetadata.remove(imageTimestamp);
            PendingJpeg jpeg = entry.getValue();
            int index = captureSequence.incrementAndGet();
            savedCount = index;
            lastDecision = "saved high-resolution frame " + index;
            DepthObservation finalDepthReference = depthReference;
            long finalDepthDeltaNs = depthDeltaNs;
            writer.execute(() -> {
                try {
                    writeCapture(index, jpeg, pose, metadata, finalDepthReference, finalDepthDeltaNs);
                } catch (IOException | JSONException e) {
                    DiagnosticLog.e(TAG, "Failed to write RGB observation", e);
                }
            });
        }
    }

    private ResolvedPose resolvePoseLocked(long sensorTimestampNs) {
        PoseSample before = null;
        PoseSample after = null;
        PoseSample nearest = null;
        long nearestDelta = Long.MAX_VALUE;

        for (PoseSample sample : poseSamples) {
            long ts = sample.correlationTimestampNs();
            if (ts <= 0) {
                continue;
            }
            long absDelta = Math.abs(ts - sensorTimestampNs);
            if (absDelta < nearestDelta) {
                nearest = sample;
                nearestDelta = absDelta;
            }
            if (ts <= sensorTimestampNs) {
                before = sample;
            }
            if (ts >= sensorTimestampNs) {
                after = sample;
                break;
            }
        }

        if (before != null && after != null) {
            long beforeDelta = sensorTimestampNs - before.correlationTimestampNs();
            long afterDelta = after.correlationTimestampNs() - sensorTimestampNs;
            if (beforeDelta <= MAX_POSE_MATCH_DELTA_NS
                    && afterDelta <= MAX_POSE_MATCH_DELTA_NS) {
                if (before == after || before.correlationTimestampNs() == after.correlationTimestampNs()) {
                    return ResolvedPose.fromSample(before, "exact_or_single_sample", sensorTimestampNs);
                }
                float t = (float) ((double) beforeDelta
                        / (double) (after.correlationTimestampNs()
                        - before.correlationTimestampNs()));
                t = Math.max(0f, Math.min(1f, t));
                Pose interpolated = Pose.makeInterpolated(
                        before.rootFromCamera,
                        after.rootFromCamera,
                        t);
                PoseSample intrinsicsSample = beforeDelta <= afterDelta ? before : after;
                return ResolvedPose.interpolated(
                        interpolated,
                        intrinsicsSample,
                        before,
                        after,
                        sensorTimestampNs,
                        t);
            }
        }

        if (nearest != null && nearestDelta <= MAX_POSE_MATCH_DELTA_NS) {
            return ResolvedPose.fromSample(nearest, "nearest_sample_fallback", sensorTimestampNs);
        }
        return null;
    }

    private DepthObservation findNearestDepthLocked(long sensorTimestampNs) {
        DepthObservation best = null;
        long bestDelta = Long.MAX_VALUE;
        for (DepthObservation sample : depthSamples) {
            long ts = sample.correlationTimestampNs();
            if (ts <= 0) {
                continue;
            }
            long delta = Math.abs(ts - sensorTimestampNs);
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
        PoseSample oldest = poseSamples.peekFirst();
        long oldestCameraTimestamp = oldest == null ? Long.MIN_VALUE : oldest.correlationTimestampNs();
        if (oldestCameraTimestamp <= 0) {
            return;
        }
        stillMetadata.entrySet().removeIf(
                e -> e.getKey() < oldestCameraTimestamp - MAX_POSE_MATCH_DELTA_NS);
    }

    private void writeCapture(
            int index,
            PendingJpeg jpeg,
            ResolvedPose pose,
            StillMetadata metadata,
            DepthObservation depthReference,
            long depthDeltaNs)
            throws IOException, JSONException {
        String base = String.format(Locale.US, "frame_%06d_%d", index, jpeg.timestampNs);
        File jpegFile = new File(captureRoot, base + ".jpg");
        File jsonFile = new File(captureRoot, base + ".json");
        File compatibilityPly = depthReference == null
                ? null : new File(captureRoot, base + ".ply");

        try (FileOutputStream out = new FileOutputStream(jpegFile)) {
            out.write(jpeg.bytes);
        }
        if (compatibilityPly != null) {
            depthReference.snapshot.writePly(compatibilityPly);
        }

        JpegIntrinsics jpegIntrinsics = resolveJpegIntrinsics(metadata, pose.intrinsicsSample);
        Pose rootFromCamera = pose.rootFromCamera;
        float[] worldFromCamera = new float[16];
        float[] cameraFromWorld = new float[16];
        rootFromCamera.toMatrix(worldFromCamera, 0);
        rootFromCamera.inverse().toMatrix(cameraFromWorld, 0);

        JSONObject json = new JSONObject();
        json.put("observation_type", "rgb");
        json.put("observation_role", "photometric_ground_truth");
        json.put("capture_index", index);
        json.put("rgb_observation_id", String.format(Locale.US, "rgb_%06d", index));
        json.put("source", "camera2_high_resolution_jpeg");
        json.put("image", jpegFile.getName());
        json.put("point_cloud",
                compatibilityPly == null ? JSONObject.NULL : compatibilityPly.getName());
        json.put("has_raw_depth_prior", depthReference != null);
        json.put("jpeg_width", jpegSize == null ? JSONObject.NULL : jpegSize.getWidth());
        json.put("jpeg_height", jpegSize == null ? JSONObject.NULL : jpegSize.getHeight());
        json.put("jpeg_sensor_timestamp_ns", jpeg.timestampNs);
        json.put("pose_timestamp_domain", "android_camera_sensor_timestamp");
        json.put("pose_resolution_method", pose.method);
        json.put("pose_timestamp_delta_ns", pose.nearestDeltaNs);
        json.put("pose_before_timestamp_ns",
                pose.before == null ? JSONObject.NULL : pose.before.correlationTimestampNs());
        json.put("pose_after_timestamp_ns",
                pose.after == null ? JSONObject.NULL : pose.after.correlationTimestampNs());
        json.put("pose_interpolation_t",
                pose.interpolationT == null ? JSONObject.NULL : pose.interpolationT);
        json.put("arcore_pose_android_camera_timestamp_ns",
                pose.intrinsicsSample == null
                        ? JSONObject.NULL : pose.intrinsicsSample.androidCameraTimestampNs);
        json.put("arcore_frame_timestamp_ns",
                pose.intrinsicsSample == null
                        ? JSONObject.NULL : pose.intrinsicsSample.arFrameTimestampNs);

        json.put("translation_m", array(rootFromCamera.getTranslation()));
        json.put("rotation_quaternion_xyzw", array(rootFromCamera.getRotationQuaternion()));
        json.put("world_from_camera_column_major", array(worldFromCamera));
        json.put("camera_from_world_column_major", array(cameraFromWorld));
        json.put("forward_world", array(rootFromCamera.rotateVector(new float[] {0f, 0f, -1f})));
        json.put("up_world", array(rootFromCamera.rotateVector(new float[] {0f, 1f, 0f})));

        if (pose.intrinsicsSample != null) {
            JSONObject imageIntrinsics = new JSONObject();
            imageIntrinsics.put("focal_length_px", array(pose.intrinsicsSample.imageFocalLengthPx));
            imageIntrinsics.put("principal_point_px", array(pose.intrinsicsSample.imagePrincipalPointPx));
            imageIntrinsics.put("image_dimensions", array(pose.intrinsicsSample.imageDimensions));
            json.put("arcore_image_intrinsics", imageIntrinsics);

            JSONObject textureIntrinsics = new JSONObject();
            textureIntrinsics.put("focal_length_px", array(pose.intrinsicsSample.textureFocalLengthPx));
            textureIntrinsics.put("principal_point_px", array(pose.intrinsicsSample.texturePrincipalPointPx));
            textureIntrinsics.put("image_dimensions", array(pose.intrinsicsSample.textureDimensions));
            json.put("arcore_texture_intrinsics", textureIntrinsics);
        }

        json.put("jpeg_intrinsics", jpegIntrinsics.toJson());
        json.put("camera2_capture", metadata.toJson());

        if (depthReference != null) {
            json.put("nearest_depth_observation_id", depthReference.observationId());
            json.put("nearest_depth_observation_ply", depthReference.plyFileName());
            json.put("nearest_depth_android_camera_timestamp_ns",
                    depthReference.androidCameraTimestampNs > 0
                            ? depthReference.androidCameraTimestampNs : JSONObject.NULL);
            json.put("nearest_depth_raw_timestamp_ns", depthReference.depthTimestampNs);
            json.put("nearest_depth_timestamp_delta_ns", depthDeltaNs);
        } else {
            json.put("nearest_depth_observation_id", JSONObject.NULL);
            json.put("nearest_depth_observation_ply", JSONObject.NULL);
            json.put("nearest_depth_timestamp_delta_ns", JSONObject.NULL);
        }

        JSONObject quality = new JSONObject();
        quality.put("target_shutter", "1/500 s");
        quality.put("absolute_max_saved_exposure_ns", MAX_STILL_EXPOSURE_NS);
        quality.put("absolute_max_saved_iso", MAX_STILL_ISO);
        quality.put("max_linear_speed_mps", MAX_LINEAR_SPEED_MPS);
        quality.put("max_angular_speed_dps", MAX_ANGULAR_SPEED_DPS);
        quality.put("max_pose_match_delta_ns", MAX_POSE_MATCH_DELTA_NS);
        quality.put("max_depth_reference_delta_ns", MAX_DEPTH_REFERENCE_DELTA_NS);
        quality.put("raw_depth_required_for_rgb_observation", false);
        json.put("capture_quality_gate", quality);

        try (FileOutputStream out = new FileOutputStream(jsonFile)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        appendTrajectory(index, jpegFile.getName(), jpeg.timestampNs, pose);
        DiagnosticLog.i(TAG,
                "Saved RGB observation=" + index
                        + " JPEG=" + jpegFile.getName()
                        + " pose=" + pose.method
                        + " depthRef="
                        + (depthReference == null ? "none" : depthReference.observationId()));
    }

    private void writeDepthObservation(DepthObservation observation)
            throws IOException, JSONException {
        File plyFile = new File(captureRoot, observation.plyFileName());
        File jsonFile = new File(captureRoot, observation.jsonFileName());
        observation.snapshot.writePly(plyFile);

        float[] rootFromDepthCameraMatrix = new float[16];
        observation.rootFromDepthCamera.toMatrix(rootFromDepthCameraMatrix, 0);

        JSONObject json = new JSONObject();
        json.put("observation_type", "raw_depth");
        json.put("observation_role", "geometry_prior_observation");
        json.put("depth_observation_index", observation.index);
        json.put("depth_observation_id", observation.observationId());
        json.put("point_cloud", plyFile.getName());
        json.put("raw_depth_timestamp_ns", observation.depthTimestampNs);
        json.put("android_camera_timestamp_ns",
                observation.androidCameraTimestampNs > 0
                        ? observation.androidCameraTimestampNs : JSONObject.NULL);
        json.put("point_count", observation.snapshot.getPointCount());
        json.put("coordinate_system", "datasetRootAnchor local coordinates");
        json.put("root_from_depth_camera_column_major", array(rootFromDepthCameraMatrix));
        json.put("depth_camera_translation_m",
                array(observation.rootFromDepthCamera.getTranslation()));
        json.put("depth_camera_rotation_quaternion_xyzw",
                array(observation.rootFromDepthCamera.getRotationQuaternion()));
        json.put("rgb_pairing_required", false);

        try (FileOutputStream out = new FileOutputStream(jsonFile)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        DiagnosticLog.i(TAG,
                "Saved independent Depth observation=" + observation.index
                        + " points=" + observation.snapshot.getPointCount());
    }

    private void appendTrajectory(
            int index, String image, long imageTimestampNs, ResolvedPose pose)
            throws IOException {
        File file = new File(captureRoot, "camera_trajectory.csv");
        boolean writeHeader = !file.exists() || file.length() == 0;
        try (FileOutputStream out = new FileOutputStream(file, true)) {
            if (writeHeader) {
                out.write(("index,image,image_timestamp_ns,pose_method,pose_delta_ns,"
                        + "tx,ty,tz,qx,qy,qz,qw\n").getBytes(StandardCharsets.UTF_8));
            }
            float[] t = pose.rootFromCamera.getTranslation();
            float[] q = pose.rootFromCamera.getRotationQuaternion();
            String line = String.format(Locale.US,
                    "%d,%s,%d,%s,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f\n",
                    index,
                    image,
                    imageTimestampNs,
                    pose.method,
                    pose.nearestDeltaNs,
                    t[0], t[1], t[2], q[0], q[1], q[2], q[3]);
            out.write(line.getBytes(StandardCharsets.UTF_8));
        }
    }

    private JpegIntrinsics resolveJpegIntrinsics(
            StillMetadata metadata, PoseSample fallbackSample) {
        if (jpegSize != null
                && lensIntrinsicCalibration != null
                && lensIntrinsicCalibration.length >= 5
                && activeArray != null) {
            try {
                double fx = lensIntrinsicCalibration[0];
                double fy = lensIntrinsicCalibration[1];
                double cx = lensIntrinsicCalibration[2];
                double cy = lensIntrinsicCalibration[3];
                double skew = lensIntrinsicCalibration[4];

                // Camera2 calibration is defined in the pre-correction active-array coordinate
                // system. Map it to active-array coordinates before applying the capture crop.
                Rect pre = preCorrectionActiveArray;
                if (pre != null && pre.width() > 0 && pre.height() > 0
                        && (pre.left != activeArray.left
                        || pre.top != activeArray.top
                        || pre.width() != activeArray.width()
                        || pre.height() != activeArray.height())) {
                    double sx = (double) activeArray.width() / pre.width();
                    double sy = (double) activeArray.height() / pre.height();
                    fx *= sx;
                    fy *= sy;
                    skew *= sx;
                    cx = activeArray.left + (cx - pre.left) * sx;
                    cy = activeArray.top + (cy - pre.top) * sy;
                }

                Rect crop = metadata.cropRegion == null
                        ? new Rect(activeArray) : new Rect(metadata.cropRegion);
                Rect effectiveCrop = centerCropToAspect(
                        crop, (double) jpegSize.getWidth() / jpegSize.getHeight());
                if (effectiveCrop.width() > 0 && effectiveCrop.height() > 0) {
                    double sx = (double) jpegSize.getWidth() / effectiveCrop.width();
                    double sy = (double) jpegSize.getHeight() / effectiveCrop.height();
                    double outFx = fx * sx;
                    double outFy = fy * sy;
                    double outCx = (cx - effectiveCrop.left) * sx;
                    double outCy = (cy - effectiveCrop.top) * sy;
                    double outSkew = skew * sx;
                    if (finitePositive(outFx) && finitePositive(outFy)
                            && Double.isFinite(outCx) && Double.isFinite(outCy)) {
                        return new JpegIntrinsics(
                                outFx,
                                outFy,
                                outCx,
                                outCy,
                                outSkew,
                                "camera2_lens_intrinsic_calibration_crop_mapped_v1",
                                effectiveCrop,
                                lensDistortion);
                    }
                }
            } catch (RuntimeException ignored) {
            }
        }

        if (jpegSize != null && fallbackSample != null
                && fallbackSample.imageDimensions[0] > 0
                && fallbackSample.imageDimensions[1] > 0) {
            double sx = (double) jpegSize.getWidth() / fallbackSample.imageDimensions[0];
            double sy = (double) jpegSize.getHeight() / fallbackSample.imageDimensions[1];
            return new JpegIntrinsics(
                    fallbackSample.imageFocalLengthPx[0] * sx,
                    fallbackSample.imageFocalLengthPx[1] * sy,
                    fallbackSample.imagePrincipalPointPx[0] * sx,
                    fallbackSample.imagePrincipalPointPx[1] * sy,
                    0.0,
                    "arcore_image_intrinsics_scaled_fallback",
                    null,
                    lensDistortion);
        }

        return new JpegIntrinsics(
                Double.NaN,
                Double.NaN,
                Double.NaN,
                Double.NaN,
                0.0,
                "unavailable",
                null,
                lensDistortion);
    }

    private void queueFallbackDepthPriorsLocked() {
        if (fallbackDepthPriorsQueued || depthSamples.isEmpty()) {
            return;
        }
        fallbackDepthPriorsQueued = true;
        List<DepthObservation> candidates = new ArrayList<>(depthSamples);
        int start = Math.max(0, candidates.size() - MAX_FALLBACK_DEPTH_PRIORS);
        int outIndex = 0;
        for (int i = start; i < candidates.size(); i++) {
            DepthObservation observation = candidates.get(i);
            int fallbackIndex = ++outIndex;
            writer.execute(() -> {
                File file = new File(captureRoot, String.format(
                        Locale.US,
                        "depth_prior_%02d_%d.ply",
                        fallbackIndex,
                        observation.depthTimestampNs));
                try {
                    observation.snapshot.writePly(file);
                } catch (IOException e) {
                    DiagnosticLog.w(TAG,
                            "Failed to write compatibility Depth prior: " + e.getMessage());
                }
            });
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
                json.put("jpeg_megapixels", pixelCount(jpegSize) / 1_000_000.0);
            }
            putNullable(json, "sensor_orientation_degrees", sensorOrientationDegrees);
            putRect(json, "active_array", activeArray);
            putRect(json, "pre_correction_active_array", preCorrectionActiveArray);
            putFloatArray(json, "available_apertures",
                    c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES));
            putFloatArray(json, "available_focal_lengths_mm",
                    c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS));
            putFloatArray(json, "lens_intrinsic_calibration", lensIntrinsicCalibration);
            putFloatArray(json, "lens_distortion", lensDistortion);
            json.put("phase", "v1_phase1_observation_capture");
            json.put("rgb_observation_policy",
                    "Camera2 high-resolution JPEG + exposure-time ARCore pose + JPEG camera model is the photometric observation of record");
            json.put("depth_observation_policy",
                    "Raw Depth is saved independently with its own timestamp/pose in datasetRootAnchor coordinates; RGB pairing is optional reference only");
            json.put("arcore_cpu_image_policy",
                    "tracking/diagnostic/depth-color helper only; never 3DGS photometric ground truth");
            json.put("pose_correlation_policy",
                    "Camera2 SENSOR_TIMESTAMP correlated against ARCore Frame.getAndroidCameraTimestamp; interpolate bracketed poses when possible");
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

    private static long safeAndroidCameraTimestamp(Frame frame) {
        try {
            long value = frame.getAndroidCameraTimestamp();
            return value > 0 ? value : -1L;
        } catch (RuntimeException e) {
            return -1L;
        }
    }

    private static Rect centerCropToAspect(Rect source, double targetAspect) {
        if (source == null || source.width() <= 0 || source.height() <= 0
                || !Double.isFinite(targetAspect) || targetAspect <= 0.0) {
            return source == null ? null : new Rect(source);
        }
        double sourceAspect = (double) source.width() / source.height();
        if (Math.abs(sourceAspect - targetAspect) < 1e-6) {
            return new Rect(source);
        }
        if (sourceAspect > targetAspect) {
            int targetWidth = Math.max(1, (int) Math.round(source.height() * targetAspect));
            int inset = Math.max(0, (source.width() - targetWidth) / 2);
            return new Rect(source.left + inset, source.top,
                    source.right - inset, source.bottom);
        }
        int targetHeight = Math.max(1, (int) Math.round(source.width() / targetAspect));
        int inset = Math.max(0, (source.height() - targetHeight) / 2);
        return new Rect(source.left, source.top + inset,
                source.right, source.bottom - inset);
    }

    private static long pixelCount(Size size) {
        return (long) size.getWidth() * size.getHeight();
    }

    private static boolean isFourByThree(Size size) {
        double ratio = (double) Math.max(size.getWidth(), size.getHeight())
                / Math.min(size.getWidth(), size.getHeight());
        return Math.abs(ratio - 4.0 / 3.0) < 0.04;
    }

    private static boolean finitePositive(double value) {
        return Double.isFinite(value) && value > 0.0;
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
        final long androidCameraTimestampNs;
        final long arFrameTimestampNs;
        final Pose rootFromCamera;
        final float[] imageFocalLengthPx;
        final float[] imagePrincipalPointPx;
        final int[] imageDimensions;
        final float[] textureFocalLengthPx;
        final float[] texturePrincipalPointPx;
        final int[] textureDimensions;

        private PoseSample(
                long androidCameraTimestampNs,
                long arFrameTimestampNs,
                Pose rootFromCamera,
                float[] imageFocalLengthPx,
                float[] imagePrincipalPointPx,
                int[] imageDimensions,
                float[] textureFocalLengthPx,
                float[] texturePrincipalPointPx,
                int[] textureDimensions) {
            this.androidCameraTimestampNs = androidCameraTimestampNs;
            this.arFrameTimestampNs = arFrameTimestampNs;
            this.rootFromCamera = rootFromCamera;
            this.imageFocalLengthPx = imageFocalLengthPx;
            this.imagePrincipalPointPx = imagePrincipalPointPx;
            this.imageDimensions = imageDimensions;
            this.textureFocalLengthPx = textureFocalLengthPx;
            this.texturePrincipalPointPx = texturePrincipalPointPx;
            this.textureDimensions = textureDimensions;
        }

        long correlationTimestampNs() {
            return androidCameraTimestampNs > 0 ? androidCameraTimestampNs : arFrameTimestampNs;
        }

        static PoseSample from(
                long androidCameraTimestampNs,
                long arFrameTimestampNs,
                Camera camera,
                Pose rootPose) {
            Pose pose = rootPose.inverse().compose(camera.getPose());
            CameraIntrinsics image = camera.getImageIntrinsics();
            CameraIntrinsics texture = camera.getTextureIntrinsics();
            return new PoseSample(
                    androidCameraTimestampNs,
                    arFrameTimestampNs,
                    pose,
                    image.getFocalLength().clone(),
                    image.getPrincipalPoint().clone(),
                    image.getImageDimensions().clone(),
                    texture.getFocalLength().clone(),
                    texture.getPrincipalPoint().clone(),
                    texture.getImageDimensions().clone());
        }
    }

    private static final class ResolvedPose {
        final Pose rootFromCamera;
        final PoseSample intrinsicsSample;
        final PoseSample before;
        final PoseSample after;
        final String method;
        final long nearestDeltaNs;
        final Float interpolationT;

        private ResolvedPose(
                Pose rootFromCamera,
                PoseSample intrinsicsSample,
                PoseSample before,
                PoseSample after,
                String method,
                long nearestDeltaNs,
                Float interpolationT) {
            this.rootFromCamera = rootFromCamera;
            this.intrinsicsSample = intrinsicsSample;
            this.before = before;
            this.after = after;
            this.method = method;
            this.nearestDeltaNs = nearestDeltaNs;
            this.interpolationT = interpolationT;
        }

        static ResolvedPose fromSample(
                PoseSample sample, String method, long targetTimestampNs) {
            return new ResolvedPose(
                    sample.rootFromCamera,
                    sample,
                    sample,
                    sample,
                    method,
                    sample.correlationTimestampNs() - targetTimestampNs,
                    null);
        }

        static ResolvedPose interpolated(
                Pose pose,
                PoseSample intrinsicsSample,
                PoseSample before,
                PoseSample after,
                long targetTimestampNs,
                float t) {
            long beforeDelta = before.correlationTimestampNs() - targetTimestampNs;
            long afterDelta = after.correlationTimestampNs() - targetTimestampNs;
            long nearestDelta = Math.abs(beforeDelta) <= Math.abs(afterDelta)
                    ? beforeDelta : afterDelta;
            return new ResolvedPose(
                    pose,
                    intrinsicsSample,
                    before,
                    after,
                    "bracketed_pose_interpolation",
                    nearestDelta,
                    t);
        }
    }

    private static final class DepthObservation {
        final int index;
        final long depthTimestampNs;
        final long androidCameraTimestampNs;
        final Pose rootFromDepthCamera;
        final WorldPointCloudSnapshot snapshot;

        DepthObservation(
                int index,
                long depthTimestampNs,
                long androidCameraTimestampNs,
                Pose rootFromDepthCamera,
                WorldPointCloudSnapshot snapshot) {
            this.index = index;
            this.depthTimestampNs = depthTimestampNs;
            this.androidCameraTimestampNs = androidCameraTimestampNs;
            this.rootFromDepthCamera = rootFromDepthCamera;
            this.snapshot = snapshot;
        }

        long correlationTimestampNs() {
            return androidCameraTimestampNs > 0 ? androidCameraTimestampNs : depthTimestampNs;
        }

        String observationId() {
            return String.format(Locale.US, "depth_%06d", index);
        }

        String baseName() {
            return String.format(Locale.US,
                    "depth_obs_%06d_%d", index, depthTimestampNs);
        }

        String plyFileName() {
            return baseName() + ".ply";
        }

        String jsonFileName() {
            return baseName() + ".json";
        }
    }

    private static final class JpegIntrinsics {
        final double fx;
        final double fy;
        final double cx;
        final double cy;
        final double skew;
        final String source;
        final Rect effectiveCrop;
        final float[] distortion;

        JpegIntrinsics(
                double fx,
                double fy,
                double cx,
                double cy,
                double skew,
                String source,
                Rect effectiveCrop,
                float[] distortion) {
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
            this.skew = skew;
            this.source = source;
            this.effectiveCrop = effectiveCrop == null ? null : new Rect(effectiveCrop);
            this.distortion = distortion == null ? null : distortion.clone();
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("source", source);
            json.put("fx", Double.isFinite(fx) ? fx : JSONObject.NULL);
            json.put("fy", Double.isFinite(fy) ? fy : JSONObject.NULL);
            json.put("cx", Double.isFinite(cx) ? cx : JSONObject.NULL);
            json.put("cy", Double.isFinite(cy) ? cy : JSONObject.NULL);
            json.put("skew", Double.isFinite(skew) ? skew : JSONObject.NULL);
            putRect(json, "effective_crop_region", effectiveCrop);
            putFloatArray(json, "lens_distortion", distortion);
            return json;
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
        final String activePhysicalCameraId;

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
            if (Build.VERSION.SDK_INT >= 29) {
                activePhysicalCameraId = result.get(
                        CaptureResult.LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID);
            } else {
                activePhysicalCameraId = null;
            }
        }

        static StillMetadata from(TotalCaptureResult result) {
            return new StillMetadata(result);
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            putNullable(json, "camera_id", null);
            putNullable(json, "active_physical_camera_id", activePhysicalCameraId);
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
