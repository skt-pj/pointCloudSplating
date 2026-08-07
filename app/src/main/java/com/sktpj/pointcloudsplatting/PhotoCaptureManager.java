package com.sktpj.pointcloudsplatting;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Environment;
import android.util.Log;

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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Periodically stores texture images suitable for photogrammetry together with the exact ARCore
 * camera pose and camera metadata for the frame.
 *
 * <p>The capture gate favors sharp frames: short exposure, low device motion, stationary lens and
 * a focused AF state when those metadata are available. Images are written on a background thread.
 */
public final class PhotoCaptureManager {
    private static final String TAG = "PhotoCaptureManager";

    // Photogrammetry prefers overlap, but not dozens of identical images while the phone is still.
    private static final long MIN_CAPTURE_INTERVAL_NS = 750_000_000L;
    private static final long FORCE_CAPTURE_AFTER_NS = 3_000_000_000L;
    private static final float MIN_VIEW_TRANSLATION_METERS = 0.04f;
    private static final float MIN_VIEW_ROTATION_DEGREES = 3.0f;

    // Hand-held sharpness gate. 1/60 s is treated as the longest preferred exposure.
    private static final long MAX_EXPOSURE_NS = 16_666_667L;
    private static final float MAX_LINEAR_SPEED_MPS = 0.10f;
    private static final float MAX_ANGULAR_SPEED_DPS = 8.0f;

    private static final int JPEG_QUALITY = 95;
    private static final int MAX_PENDING_WRITES = 2;

    private final File captureRoot;
    private final ExecutorService writer = Executors.newSingleThreadExecutor();
    private final AtomicInteger pendingWrites = new AtomicInteger();

    private Pose previousPose;
    private long previousFrameTimestampNs = -1L;
    private Pose lastSavedPose;
    private long lastSavedTimestampNs = -1L;
    private int savedCount;

    public PhotoCaptureManager(File externalFilesDir) {
        File pictures = externalFilesDir;
        if (pictures == null) {
            throw new IllegalStateException("External files directory is unavailable");
        }
        String session = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        captureRoot = new File(pictures, "photogrammetry_" + session);
        if (!captureRoot.mkdirs() && !captureRoot.isDirectory()) {
            throw new IllegalStateException("Failed to create capture directory: " + captureRoot);
        }
    }

    public static File getPicturesDirectory(android.content.Context context) {
        return context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
    }

    /** Returns the number of texture frames successfully queued for storage. */
    public int getSavedCount() {
        return savedCount;
    }

    public String getCaptureDirectoryPath() {
        return captureRoot.getAbsolutePath();
    }

    /**
     * Evaluates the current frame and stores it if it is a useful, low-blur photogrammetry view.
     * Must be called for the latest ARCore frame while tracking.
     */
    public void consider(Frame frame, Camera camera) {
        long timestampNs = frame.getTimestamp();
        Pose pose = camera.getPose();

        Motion motion = measureMotion(pose, timestampNs);
        previousPose = pose;
        previousFrameTimestampNs = timestampNs;

        if (lastSavedTimestampNs >= 0 && timestampNs - lastSavedTimestampNs < MIN_CAPTURE_INTERVAL_NS) {
            return;
        }
        if (pendingWrites.get() >= MAX_PENDING_WRITES) {
            return;
        }

        CameraMetadata metadata = readMetadata(frame);
        if (!isSharpEnough(metadata, motion)) {
            return;
        }
        if (!hasUsefulViewpointChange(pose, timestampNs)) {
            return;
        }

        try (Image image = frame.acquireCameraImage()) {
            CapturePacket packet = copyFrame(image, camera, pose, timestampNs, metadata, motion);
            lastSavedPose = pose;
            lastSavedTimestampNs = timestampNs;
            savedCount++;
            pendingWrites.incrementAndGet();
            writer.execute(() -> {
                try {
                    writeCapture(packet);
                } catch (IOException | JSONException e) {
                    Log.e(TAG, "Failed to store texture capture", e);
                } finally {
                    pendingWrites.decrementAndGet();
                }
            });
        } catch (NotYetAvailableException e) {
            // A CPU image is not guaranteed for every frame. Try again on a later sharp frame.
        }
    }

    public void shutdown() {
        writer.shutdown();
    }

    private Motion measureMotion(Pose pose, long timestampNs) {
        if (previousPose == null || previousFrameTimestampNs < 0 || timestampNs <= previousFrameTimestampNs) {
            return new Motion(0f, 0f);
        }
        float dt = (timestampNs - previousFrameTimestampNs) / 1_000_000_000f;
        float translation = translationDistance(previousPose, pose);
        float rotation = rotationDegrees(previousPose, pose);
        return new Motion(translation / dt, rotation / dt);
    }

    private boolean hasUsefulViewpointChange(Pose pose, long timestampNs) {
        if (lastSavedPose == null || lastSavedTimestampNs < 0) {
            return true;
        }
        if (timestampNs - lastSavedTimestampNs >= FORCE_CAPTURE_AFTER_NS) {
            return true;
        }
        return translationDistance(lastSavedPose, pose) >= MIN_VIEW_TRANSLATION_METERS
                || rotationDegrees(lastSavedPose, pose) >= MIN_VIEW_ROTATION_DEGREES;
    }

    private static boolean isSharpEnough(CameraMetadata metadata, Motion motion) {
        if (motion.linearSpeedMps > MAX_LINEAR_SPEED_MPS
                || motion.angularSpeedDps > MAX_ANGULAR_SPEED_DPS) {
            return false;
        }
        if (metadata.exposureTimeNs != null && metadata.exposureTimeNs > MAX_EXPOSURE_NS) {
            return false;
        }
        if (metadata.lensState != null && metadata.lensState != 0) {
            // Camera2/NDK LENS_STATE_STATIONARY == 0, MOVING == 1.
            return false;
        }
        if (metadata.afState != null) {
            int af = metadata.afState & 0xff;
            // PASSIVE_FOCUSED == 2, FOCUSED_LOCKED == 4. INACTIVE == 0 is accepted for fixed-focus
            // cameras where no AF mechanism is moving.
            if (af != 0 && af != 2 && af != 4) {
                return false;
            }
        }
        return true;
    }

    private static CameraMetadata readMetadata(Frame frame) {
        CameraMetadata out = new CameraMetadata();
        try {
            ImageMetadata metadata = frame.getImageMetadata();
            out.exposureTimeNs = getLong(metadata, ImageMetadata.SENSOR_EXPOSURE_TIME);
            out.iso = getInt(metadata, ImageMetadata.SENSOR_SENSITIVITY);
            out.aperture = getFloat(metadata, ImageMetadata.LENS_APERTURE);
            out.focalLengthMm = getFloat(metadata, ImageMetadata.LENS_FOCAL_LENGTH);
            out.focusDistanceDiopters = getFloat(metadata, ImageMetadata.LENS_FOCUS_DISTANCE);
            out.focusRangeDiopters = getFloatArray(metadata, ImageMetadata.LENS_FOCUS_RANGE);
            out.lensState = getByte(metadata, ImageMetadata.LENS_STATE);
            out.afState = getByte(metadata, ImageMetadata.CONTROL_AF_STATE);
            out.oisMode = getByte(metadata, ImageMetadata.LENS_OPTICAL_STABILIZATION_MODE);
            out.rollingShutterSkewNs = getLong(metadata, ImageMetadata.SENSOR_ROLLING_SHUTTER_SKEW);
        } catch (NotYetAvailableException ignored) {
            // Metadata is absent on some startup frames. Pose-based sharpness gating still works.
        }
        return out;
    }

    private static CapturePacket copyFrame(
            Image image,
            Camera camera,
            Pose pose,
            long timestampNs,
            CameraMetadata metadata,
            Motion motion) {
        byte[] nv21 = yuv420888ToNv21(image);
        CameraIntrinsics intrinsics = camera.getImageIntrinsics();
        return new CapturePacket(
                nv21,
                image.getWidth(),
                image.getHeight(),
                timestampNs,
                pose.getTranslation().clone(),
                pose.getRotationQuaternion().clone(),
                intrinsics.getFocalLength().clone(),
                intrinsics.getPrincipalPoint().clone(),
                intrinsics.getImageDimensions().clone(),
                metadata,
                motion);
    }

    private void writeCapture(CapturePacket packet) throws IOException, JSONException {
        String base = String.format(Locale.US, "frame_%06d_%d", savedCount, packet.timestampNs);
        File jpegFile = new File(captureRoot, base + ".jpg");
        File jsonFile = new File(captureRoot, base + ".json");

        YuvImage yuv = new YuvImage(packet.nv21, ImageFormat.NV21, packet.width, packet.height, null);
        try (FileOutputStream output = new FileOutputStream(jpegFile)) {
            if (!yuv.compressToJpeg(new Rect(0, 0, packet.width, packet.height), JPEG_QUALITY, output)) {
                throw new IOException("YUV to JPEG conversion failed");
            }
        }

        JSONObject json = new JSONObject();
        json.put("timestamp_ns", packet.timestampNs);
        json.put("image", jpegFile.getName());
        json.put("width", packet.width);
        json.put("height", packet.height);
        json.put("translation_m", array(packet.translation));
        json.put("rotation_quaternion_xyzw", array(packet.rotationQuaternion));

        Pose pose = new Pose(packet.translation, packet.rotationQuaternion);
        json.put("forward_world", array(pose.rotateVector(new float[] {0f, 0f, -1f})));
        json.put("up_world", array(pose.rotateVector(new float[] {0f, 1f, 0f})));

        JSONObject intrinsics = new JSONObject();
        intrinsics.put("focal_length_px", array(packet.focalLengthPx));
        intrinsics.put("principal_point_px", array(packet.principalPointPx));
        intrinsics.put("image_dimensions", array(packet.intrinsicsDimensions));
        json.put("intrinsics", intrinsics);

        JSONObject cameraMetadata = new JSONObject();
        putNullable(cameraMetadata, "exposure_time_ns", packet.metadata.exposureTimeNs);
        putNullable(cameraMetadata, "iso", packet.metadata.iso);
        putNullable(cameraMetadata, "aperture_f_number", packet.metadata.aperture);
        putNullable(cameraMetadata, "focal_length_mm", packet.metadata.focalLengthMm);
        putNullable(cameraMetadata, "focus_distance_diopters", packet.metadata.focusDistanceDiopters);
        if (packet.metadata.focusRangeDiopters != null) {
            cameraMetadata.put("focus_range_diopters", array(packet.metadata.focusRangeDiopters));
        }
        putNullable(cameraMetadata, "lens_state", packet.metadata.lensState);
        putNullable(cameraMetadata, "af_state", packet.metadata.afState);
        putNullable(cameraMetadata, "ois_mode", packet.metadata.oisMode);
        putNullable(cameraMetadata, "rolling_shutter_skew_ns", packet.metadata.rollingShutterSkewNs);
        json.put("camera_metadata", cameraMetadata);

        JSONObject quality = new JSONObject();
        quality.put("linear_speed_mps", packet.motion.linearSpeedMps);
        quality.put("angular_speed_dps", packet.motion.angularSpeedDps);
        quality.put("max_exposure_ns", MAX_EXPOSURE_NS);
        quality.put("max_linear_speed_mps", MAX_LINEAR_SPEED_MPS);
        quality.put("max_angular_speed_dps", MAX_ANGULAR_SPEED_DPS);
        json.put("capture_quality_gate", quality);

        try (FileOutputStream output = new FileOutputStream(jsonFile)) {
            output.write(json.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
        }
    }

    /** Converts an Android YUV_420_888 CPU image to tightly packed NV21 with stride handling. */
    private static byte[] yuv420888ToNv21(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] out = new byte[width * height * 3 / 2];
        Image.Plane[] planes = image.getPlanes();

        copyPlane(planes[0], width, height, out, 0, 1);

        int chromaOffset = width * height;
        int chromaWidth = width / 2;
        int chromaHeight = height / 2;
        copyPlane(planes[2], chromaWidth, chromaHeight, out, chromaOffset, 2);     // V
        copyPlane(planes[1], chromaWidth, chromaHeight, out, chromaOffset + 1, 2); // U
        return out;
    }

    private static void copyPlane(
            Image.Plane plane,
            int width,
            int height,
            byte[] output,
            int outputOffset,
            int outputPixelStride) {
        ByteBuffer buffer = plane.getBuffer().duplicate();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        for (int row = 0; row < height; row++) {
            int rowStart = row * rowStride;
            for (int col = 0; col < width; col++) {
                int sourceIndex = rowStart + col * pixelStride;
                if (sourceIndex >= 0 && sourceIndex < buffer.limit()) {
                    output[outputOffset + (row * width + col) * outputPixelStride] =
                            buffer.get(sourceIndex);
                }
            }
        }
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

    private static Long getLong(ImageMetadata metadata, int key) {
        try { return metadata.getLong(key); } catch (MetadataNotFoundException e) { return null; }
    }

    private static Integer getInt(ImageMetadata metadata, int key) {
        try { return metadata.getInt(key); } catch (MetadataNotFoundException e) { return null; }
    }

    private static Float getFloat(ImageMetadata metadata, int key) {
        try { return metadata.getFloat(key); } catch (MetadataNotFoundException e) { return null; }
    }

    private static float[] getFloatArray(ImageMetadata metadata, int key) {
        try { return metadata.getFloatArray(key); } catch (MetadataNotFoundException e) { return null; }
    }

    private static Byte getByte(ImageMetadata metadata, int key) {
        try { return metadata.getByte(key); } catch (MetadataNotFoundException e) { return null; }
    }

    private static JSONArray array(float[] values) {
        JSONArray array = new JSONArray();
        for (float value : values) array.put(value);
        return array;
    }

    private static JSONArray array(int[] values) {
        JSONArray array = new JSONArray();
        for (int value : values) array.put(value);
        return array;
    }

    private static void putNullable(JSONObject json, String key, Object value) throws JSONException {
        json.put(key, value == null ? JSONObject.NULL : value);
    }

    private static final class Motion {
        final float linearSpeedMps;
        final float angularSpeedDps;

        Motion(float linearSpeedMps, float angularSpeedDps) {
            this.linearSpeedMps = linearSpeedMps;
            this.angularSpeedDps = angularSpeedDps;
        }
    }

    private static final class CameraMetadata {
        Long exposureTimeNs;
        Integer iso;
        Float aperture;
        Float focalLengthMm;
        Float focusDistanceDiopters;
        float[] focusRangeDiopters;
        Byte lensState;
        Byte afState;
        Byte oisMode;
        Long rollingShutterSkewNs;
    }

    private static final class CapturePacket {
        final byte[] nv21;
        final int width;
        final int height;
        final long timestampNs;
        final float[] translation;
        final float[] rotationQuaternion;
        final float[] focalLengthPx;
        final float[] principalPointPx;
        final int[] intrinsicsDimensions;
        final CameraMetadata metadata;
        final Motion motion;

        CapturePacket(
                byte[] nv21,
                int width,
                int height,
                long timestampNs,
                float[] translation,
                float[] rotationQuaternion,
                float[] focalLengthPx,
                float[] principalPointPx,
                int[] intrinsicsDimensions,
                CameraMetadata metadata,
                Motion motion) {
            this.nv21 = nv21;
            this.width = width;
            this.height = height;
            this.timestampNs = timestampNs;
            this.translation = translation;
            this.rotationQuaternion = rotationQuaternion;
            this.focalLengthPx = focalLengthPx;
            this.principalPointPx = principalPointPx;
            this.intrinsicsDimensions = intrinsicsDimensions;
            this.metadata = metadata;
            this.motion = motion;
        }
    }
}
