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
import android.util.Log;
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
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Stores photogrammetry texture frames and matches each Camera2 JPEG sensor timestamp to the
 * nearest ARCore camera pose.
 *
 * <p>ARCore is used for tracking/depth/pose. Image quality decisions are made for downstream
 * SfM/MVS/3DGS: sufficient viewpoint change, low phone motion, stable focus, short still exposure,
 * bounded ISO, and disabled image stabilization.
 */
public final class HighResPhotoCaptureManager {
    private static final String TAG = "HighResPhotoCapture";

    // SharedCamera is much more reliable with a binned ~12 MP JPEG stream than with full 48 MP.
    private static final long MAX_JPEG_PIXELS = 13_000_000L;

    private static final long MIN_CAPTURE_INTERVAL_NS = 650_000_000L;
    private static final long FORCE_CAPTURE_AFTER_NS = 3_000_000_000L;
    private static final float MIN_VIEW_TRANSLATION_METERS = 0.035f;
    private static final float MIN_VIEW_ROTATION_DEGREES = 3.0f;
    private static final long MAX_POSE_MATCH_DELTA_NS = 50_000_000L;
    private static final int MAX_POSE_SAMPLES = 120;

    // Photogrammetry quality limits. MainActivity targets 1/500 s; 1/300 s is the hard save limit.
    private static final long MAX_STILL_EXPOSURE_NS = 3_333_333L;
    private static final int MAX_STILL_ISO = 1600;

    private final File captureRoot;
    private final ExecutorService writer = Executors.newSingleThreadExecutor();
    private final AtomicInteger captureSequence = new AtomicInteger();

    private final boolean pixel10a;
    private final float maxLinearSpeedMps;
    private final float maxAngularSpeedDps;

    private final ArrayDeque<PoseSample> poseSamples = new ArrayDeque<>();
    private final Map<Long, PendingJpeg> pendingJpegs = new HashMap<>();
    private final Map<Long, StillMetadata> stillMetadata = new HashMap<>();

    private Pose previousPose;
    private long previousFrameTimestampNs = -1L;
    private Pose lastRequestedPose;
    private long lastRequestedTimestampNs = -1L;
    private boolean captureInFlight;
    private volatile int savedCount;

    private Size jpegSize;
    private String cameraId = "unknown";

    public HighResPhotoCaptureManager(File externalPicturesDir) {
        if (externalPicturesDir == null) {
            throw new IllegalStateException("External Pictures directory is unavailable");
        }

        String model = Build.MODEL == null ? "" : Build.MODEL.toLowerCase(Locale.US);
        pixel10a = model.contains("pixel 10a");
        if (pixel10a) {
            // Pose gating is intentionally strict even though the still shutter is fast.
            maxLinearSpeedMps = 0.08f;
            maxAngularSpeedDps = 6.0f;
        } else {
            maxLinearSpeedMps = 0.10f;
            maxAngularSpeedDps = 8.0f;
        }

        String sessionName =
                new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        captureRoot = new File(externalPicturesDir, "photogrammetry_" + sessionName);
        if (!captureRoot.mkdirs() && !captureRoot.isDirectory()) {
            throw new IllegalStateException("Failed to create " + captureRoot);
        }
    }

    public static File getPicturesDirectory(android.content.Context context) {
        return context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
    }

    /** Chooses the largest roughly 4:3 JPEG output at or below 13 MP. */
    public static Size chooseJpegSize(CameraCharacteristics characteristics) {
        StreamConfigurationMap map =
                characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) {
            throw new IllegalStateException("Camera has no stream configuration map");
        }
        Size[] outputs = map.getOutputSizes(ImageFormat.JPEG);
        if (outputs == null || outputs.length == 0) {
            throw new IllegalStateException("Camera has no JPEG output sizes");
        }

        List<Size> sorted = new ArrayList<>();
        java.util.Collections.addAll(sorted, outputs);
        sorted.sort(Comparator.comparingLong(HighResPhotoCaptureManager::pixelCount).reversed());

        for (Size size : sorted) {
            if (pixelCount(size) <= MAX_JPEG_PIXELS && isFourByThree(size)) {
                return size;
            }
        }
        for (Size size : sorted) {
            if (pixelCount(size) <= MAX_JPEG_PIXELS) {
                return size;
            }
        }
        return sorted.get(sorted.size() - 1);
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

    public String getProfileName() {
        return pixel10a ? "Pixel 10a photogrammetry" : "photogrammetry";
    }

    /**
     * Records the current ARCore pose and returns true when the viewpoint and motion are suitable
     * for requesting a high-resolution Camera2 still.
     */
    public synchronized boolean onArFrame(Frame frame, Camera camera) {
        long timestampNs = frame.getTimestamp();
        Pose pose = camera.getPose();
        Motion motion = measureMotion(pose, timestampNs);

        PoseSample sample = PoseSample.from(timestampNs, camera);
        poseSamples.addLast(sample);
        while (poseSamples.size() > MAX_POSE_SAMPLES) {
            poseSamples.removeFirst();
        }

        previousPose = pose;
        previousFrameTimestampNs = timestampNs;
        tryFinalizePendingLocked();

        if (captureInFlight) {
            return false;
        }
        if (lastRequestedTimestampNs >= 0
                && timestampNs - lastRequestedTimestampNs < MIN_CAPTURE_INTERVAL_NS) {
            return false;
        }
        if (!hasUsefulViewpointChange(pose, timestampNs)) {
            return false;
        }

        ArFrameMetadata metadata = readArFrameMetadata(frame);
        if (!isPoseAndFocusStable(metadata, motion)) {
            return false;
        }

        captureInFlight = true;
        lastRequestedPose = pose;
        lastRequestedTimestampNs = timestampNs;
        return true;
    }

    public synchronized void onCaptureRequestFailed() {
        captureInFlight = false;
    }

    public synchronized void onCaptureCompleted(TotalCaptureResult result) {
        Long timestamp = result.get(CaptureResult.SENSOR_TIMESTAMP);
        if (timestamp == null) {
            captureInFlight = false;
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
                pendingJpegs.put(timestamp, new PendingJpeg(timestamp, jpeg));
                captureInFlight = false;
                tryFinalizePendingLocked();
            }
        } catch (RuntimeException e) {
            Log.e(TAG, "Failed to read high-resolution JPEG", e);
            synchronized (this) {
                captureInFlight = false;
            }
        } finally {
            if (image != null) {
                image.close();
            }
        }
    }

    public void shutdown() {
        writer.shutdown();
    }

    private Motion measureMotion(Pose pose, long timestampNs) {
        if (previousPose == null
                || previousFrameTimestampNs < 0
                || timestampNs <= previousFrameTimestampNs) {
            return new Motion(0f, 0f);
        }
        float dt = (timestampNs - previousFrameTimestampNs) / 1_000_000_000f;
        float translation = translationDistance(previousPose, pose);
        float rotation = rotationDegrees(previousPose, pose);
        return new Motion(translation / dt, rotation / dt);
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

    private boolean isPoseAndFocusStable(ArFrameMetadata metadata, Motion motion) {
        if (motion.linearSpeedMps > maxLinearSpeedMps
                || motion.angularSpeedDps > maxAngularSpeedDps) {
            return false;
        }
        if (metadata.lensState != null && metadata.lensState != 0) {
            return false;
        }
        return isFocusedState(metadata.afState);
    }

    private static boolean isStillSharpEnough(StillMetadata metadata) {
        if (metadata.exposureTimeNs == null
                || metadata.exposureTimeNs > MAX_STILL_EXPOSURE_NS) {
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
            if (metadata == null) {
                continue;
            }
            if (newestPoseTimestamp < imageTimestamp) {
                continue;
            }

            iterator.remove();
            stillMetadata.remove(imageTimestamp);
            if (!isStillSharpEnough(metadata)) {
                Log.d(TAG, "Discarding still that failed photogrammetry quality gate: "
                        + imageTimestamp);
                continue;
            }

            PoseSample pose = findNearestPoseLocked(imageTimestamp);
            if (pose == null
                    || Math.abs(pose.timestampNs - imageTimestamp) > MAX_POSE_MATCH_DELTA_NS) {
                Log.w(TAG, "Discarding JPEG without close ARCore pose: " + imageTimestamp);
                continue;
            }

            PendingJpeg jpeg = entry.getValue();
            int index = captureSequence.incrementAndGet();
            savedCount = index;
            writer.execute(() -> {
                try {
                    writeCapture(index, jpeg, pose, metadata);
                } catch (IOException | JSONException e) {
                    Log.e(TAG, "Failed to write photogrammetry frame", e);
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

    private void trimMetadataLocked() {
        if (stillMetadata.size() <= 8) {
            return;
        }
        long oldestPose = poseSamples.isEmpty() ? Long.MIN_VALUE : poseSamples.peekFirst().timestampNs;
        stillMetadata.entrySet().removeIf(e -> e.getKey() < oldestPose - MAX_POSE_MATCH_DELTA_NS);
    }

    private void writeCapture(
            int index, PendingJpeg jpeg, PoseSample pose, StillMetadata metadata)
            throws IOException, JSONException {
        String base = String.format(Locale.US, "frame_%06d_%d", index, jpeg.timestampNs);
        File jpegFile = new File(captureRoot, base + ".jpg");
        File jsonFile = new File(captureRoot, base + ".json");

        try (FileOutputStream out = new FileOutputStream(jpegFile)) {
            out.write(jpeg.bytes);
        }

        JSONObject json = new JSONObject();
        json.put("capture_index", index);
        json.put("image", jpegFile.getName());
        json.put("jpeg_width", jpegSize == null ? JSONObject.NULL : jpegSize.getWidth());
        json.put("jpeg_height", jpegSize == null ? JSONObject.NULL : jpegSize.getHeight());
        json.put("jpeg_sensor_timestamp_ns", jpeg.timestampNs);
        json.put("arcore_pose_timestamp_ns", pose.timestampNs);
        json.put("pose_timestamp_delta_ns", pose.timestampNs - jpeg.timestampNs);
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

        JSONObject camera = metadata.toJson();
        camera.put("camera_id", cameraId);
        json.put("camera2_capture", camera);

        JSONObject quality = new JSONObject();
        quality.put("profile", pixel10a ? "pixel_10a_photogrammetry" : "photogrammetry");
        quality.put("target_shutter", "1/500 s");
        quality.put("max_saved_exposure_ns", MAX_STILL_EXPOSURE_NS);
        quality.put("max_saved_iso", MAX_STILL_ISO);
        quality.put("max_linear_speed_mps", maxLinearSpeedMps);
        quality.put("max_angular_speed_dps", maxAngularSpeedDps);
        quality.put("max_pose_match_delta_ns", MAX_POSE_MATCH_DELTA_NS);
        json.put("capture_quality_gate", quality);

        try (FileOutputStream out = new FileOutputStream(jsonFile)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    private void writeSessionCameraInfo(CameraCharacteristics c) {
        JSONObject json = new JSONObject();
        try {
            json.put("manufacturer", Build.MANUFACTURER);
            json.put("brand", Build.BRAND);
            json.put("model", Build.MODEL);
            json.put("device", Build.DEVICE);
            json.put("sdk_int", Build.VERSION.SDK_INT);
            json.put("profile", pixel10a ? "pixel_10a_photogrammetry" : "photogrammetry");
            json.put("camera_id", cameraId);
            if (jpegSize != null) {
                json.put("jpeg_width", jpegSize.getWidth());
                json.put("jpeg_height", jpegSize.getHeight());
                json.put("jpeg_megapixels", pixelCount(jpegSize) / 1_000_000.0);
            }
            putNullable(json, "sensor_orientation_degrees",
                    c.get(CameraCharacteristics.SENSOR_ORIENTATION));
            putRect(json, "active_array", c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE));
            putFloatArray(json, "available_apertures",
                    c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES));
            putFloatArray(json, "available_focal_lengths_mm",
                    c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS));
            putIntArray(json, "available_af_modes",
                    c.get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES));
            putIntArray(json, "available_ois_modes",
                    c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION));
            putFloatArray(json, "lens_intrinsic_calibration",
                    c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION));
            if (Build.VERSION.SDK_INT >= 28) {
                putFloatArray(json, "lens_distortion",
                        c.get(CameraCharacteristics.LENS_DISTORTION));
            }
            json.put("capture_policy",
                    "Camera2 still: target 1/500 s, never save slower than 1/300 s, ISO <= 1600");
            json.put("focus_policy",
                    "ARCore/Camera2 autofocus for acquisition; lock measured focus distance for still");
            json.put("stabilization_policy", "EIS OFF and OIS OFF for saved stills");
            json.put("aperture_policy", "do not change physical aperture; record actual f-number");

            try (FileOutputStream out =
                         new FileOutputStream(new File(captureRoot, "session_camera.json"))) {
                out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        } catch (IOException | JSONException e) {
            Log.w(TAG, "Failed to write camera session metadata", e);
        }
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

    private static Byte getByte(ImageMetadata metadata, int key) {
        try {
            return metadata.getByte(key);
        } catch (MetadataNotFoundException | IllegalArgumentException e) {
            return null;
        }
    }

    private static JSONArray array(float[] values) throws JSONException {
        JSONArray array = new JSONArray();
        for (float value : values) {
            array.put(value);
        }
        return array;
    }

    private static JSONArray array(int[] values) throws JSONException {
        JSONArray array = new JSONArray();
        for (int value : values) {
            array.put(value);
        }
        return array;
    }

    private static void putNullable(JSONObject object, String key, Object value)
            throws JSONException {
        object.put(key, value == null ? JSONObject.NULL : value);
    }

    private static void putFloatArray(JSONObject object, String key, float[] values)
            throws JSONException {
        if (values == null) {
            object.put(key, JSONObject.NULL);
        } else {
            object.put(key, array(values));
        }
    }

    private static void putIntArray(JSONObject object, String key, int[] values)
            throws JSONException {
        if (values == null) {
            object.put(key, JSONObject.NULL);
        } else {
            object.put(key, array(values));
        }
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

        static PoseSample from(long timestampNs, Camera camera) {
            Pose pose = camera.getPose();
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

        private StillMetadata(
                Long exposureTimeNs,
                Integer iso,
                Float aperture,
                Float focalLengthMm,
                Float focusDistanceDiopters,
                Integer afMode,
                Integer afState,
                Integer aeState,
                Integer awbState,
                Integer lensState,
                Integer oisMode,
                Integer videoStabilizationMode,
                Long rollingShutterSkewNs,
                Long frameDurationNs,
                Rect cropRegion,
                Integer jpegOrientationDegrees) {
            this.exposureTimeNs = exposureTimeNs;
            this.iso = iso;
            this.aperture = aperture;
            this.focalLengthMm = focalLengthMm;
            this.focusDistanceDiopters = focusDistanceDiopters;
            this.afMode = afMode;
            this.afState = afState;
            this.aeState = aeState;
            this.awbState = awbState;
            this.lensState = lensState;
            this.oisMode = oisMode;
            this.videoStabilizationMode = videoStabilizationMode;
            this.rollingShutterSkewNs = rollingShutterSkewNs;
            this.frameDurationNs = frameDurationNs;
            this.cropRegion = cropRegion == null ? null : new Rect(cropRegion);
            this.jpegOrientationDegrees = jpegOrientationDegrees;
        }

        static StillMetadata from(TotalCaptureResult result) {
            return new StillMetadata(
                    result.get(CaptureResult.SENSOR_EXPOSURE_TIME),
                    result.get(CaptureResult.SENSOR_SENSITIVITY),
                    result.get(CaptureResult.LENS_APERTURE),
                    result.get(CaptureResult.LENS_FOCAL_LENGTH),
                    result.get(CaptureResult.LENS_FOCUS_DISTANCE),
                    result.get(CaptureResult.CONTROL_AF_MODE),
                    result.get(CaptureResult.CONTROL_AF_STATE),
                    result.get(CaptureResult.CONTROL_AE_STATE),
                    result.get(CaptureResult.CONTROL_AWB_STATE),
                    result.get(CaptureResult.LENS_STATE),
                    result.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE),
                    result.get(CaptureResult.CONTROL_VIDEO_STABILIZATION_MODE),
                    result.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW),
                    result.get(CaptureResult.SENSOR_FRAME_DURATION),
                    result.get(CaptureResult.SCALER_CROP_REGION),
                    result.getRequest().get(CaptureRequest.JPEG_ORIENTATION));
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
