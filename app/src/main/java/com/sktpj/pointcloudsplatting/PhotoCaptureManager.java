package com.sktpj.pointcloudsplatting;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Build;
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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Periodically stores photogrammetry texture images together with the exact ARCore camera pose,
 * intrinsics and per-frame Camera2 metadata exposed by ARCore.
 *
 * <p>The capture gate favors sharp frames: short exposure, low device motion, stationary lens and
 * a focused AF state. Pixel 10a uses a stricter hand-held profile tuned for its f/1.68 OIS main
 * camera. Images are written on a background thread so point-cloud acquisition is not blocked by
 * JPEG compression and file I/O.
 */
public final class PhotoCaptureManager {
    private static final String TAG = "PhotoCaptureManager";

    private static final long MIN_CAPTURE_INTERVAL_NS = 650_000_000L;
    private static final long FORCE_CAPTURE_AFTER_NS = 2_500_000_000L;
    private static final float MIN_VIEW_TRANSLATION_METERS = 0.035f;
    private static final float MIN_VIEW_ROTATION_DEGREES = 3.0f;

    private static final int JPEG_QUALITY = 95;
    private static final int MAX_PENDING_WRITES = 2;

    private final File captureRoot;
    private final ExecutorService writer = Executors.newSingleThreadExecutor();
    private final AtomicInteger pendingWrites = new AtomicInteger();
    private final AtomicInteger captureSequence = new AtomicInteger();

    private final boolean pixel10a;
    private final long maxExposureNs;
    private final float maxLinearSpeedMps;
    private final float maxAngularSpeedDps;

    private Pose previousPose;
    private long previousFrameTimestampNs = -1L;
    private Pose lastSavedPose;
    private long lastSavedTimestampNs = -1L;
    private volatile int savedCount;

    public PhotoCaptureManager(File externalFilesDir) {
        if (externalFilesDir == null) {
            throw new IllegalStateException("External files directory is unavailable");
        }

        String model = Build.MODEL == null ? "" : Build.MODEL.toLowerCase(Locale.US);
        pixel10a = model.contains("pixel 10a");

        if (pixel10a) {
            // Pixel 10a main camera is f/1.68 with OIS. Prefer 1/100 s or faster and very low phone
            // motion to keep natural image detail for feature matching rather than relying on EIS.
            maxExposureNs = 10_000_000L;
            maxLinearSpeedMps = 0.08f;
            maxAngularSpeedDps = 6.0f;
        } else {
            maxExposureNs = 12_500_000L; // 1/80 s.
            maxLinearSpeedMps = 0.10f;
            maxAngularSpeedDps = 8.0f;
        }

        String session = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        captureRoot = new File(externalFilesDir, "photogrammetry_" + session);
        if (!captureRoot.mkdirs() && !captureRoot.isDirectory()) {
            throw new IllegalStateException("Failed to create capture directory: " + captureRoot);
        }

        writeDeviceInfo();
    }

    public static File getPicturesDirectory(android.content.Context context) {
        return context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
    }

    public int getSavedCount() {
        return savedCount;
    }

    public String getCaptureDirectoryPath() {
        return captureRoot.getAbsolutePath();
    }

    public boolean isPixel10aProfile() {
        return pixel10a;
    }

    public void consider(Frame frame, Camera camera) {
        long frameTimestampNs = frame.getTimestamp();
        Pose pose = camera.getPose();

        Motion motion = measureMotion(pose, frameTimestampNs);
        previousPose = pose;
        previousFrameTimestampNs = frameTimestampNs;

        if (lastSavedTimestampNs >= 0
                && frameTimestampNs - lastSavedTimestampNs < MIN_CAPTURE_INTERVAL_NS) {
            return;
        }
        if (pendingWrites.get() >= MAX_PENDING_WRITES) {
            return;
        }

        CameraMetadata metadata = readMetadata(frame);
        if (!isSharpEnough(metadata, motion)) {
            return;
        }
        if (!hasUsefulViewpointChange(pose, frameTimestampNs)) {
            return;
        }

        try (Image image = frame.acquireCameraImage()) {
            int captureIndex = captureSequence.incrementAndGet();
            CapturePacket packet = copyFrame(
                    image, camera, pose, frameTimestampNs, metadata, motion, captureIndex);

            lastSavedPose = pose;
            lastSavedTimestampNs = frameTimestampNs;
            savedCount = captureIndex;
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
            // A CPU image is not guaranteed for every ARCore frame. Try again on a later sharp frame.
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
        if (lastSavedPose == null || lastSavedTimestampNs < 0) {
            return true;
        }
        if (timestampNs - lastSavedTimestampNs >= FORCE_CAPTURE_AFTER_NS) {
            return true;
        }
        return translationDistance(lastSavedPose, pose) >= MIN_VIEW_TRANSLATION_METERS
                || rotationDegrees(lastSavedPose, pose) >= MIN_VIEW_ROTATION_DEGREES;
    }

    private boolean isSharpEnough(CameraMetadata metadata, Motion motion) {
        if (motion.linearSpeedMps > maxLinearSpeedMps
                || motion.angularSpeedDps > maxAngularSpeedDps) {
            return false;
        }
        if (metadata.exposureTimeNs != null && metadata.exposureTimeNs > maxExposureNs) {
            return false;
        }
        if (metadata.lensState != null && metadata.lensState != 0) {
            return false;
        }
        if (metadata.afState != null) {
            int af = metadata.afState & 0xff;
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
            out.sensorTimestampNs = getLong(metadata, ImageMetadata.SENSOR_TIMESTAMP);
            out.exposureTimeNs = getLong(metadata, ImageMetadata.SENSOR_EXPOSURE_TIME);
            out.frameDurationNs = getLong(metadata, ImageMetadata.SENSOR_FRAME_DURATION);
            out.rollingShutterSkewNs = getLong(metadata, ImageMetadata.SENSOR_ROLLING_SHUTTER_SKEW);
            out.iso = getInt(metadata, ImageMetadata.SENSOR_SENSITIVITY);
            out.aperture = getFloat(metadata, ImageMetadata.LENS_APERTURE);
            out.focalLengthMm = getFloat(metadata, ImageMetadata.LENS_FOCAL_LENGTH);
            out.focusDistanceDiopters = getFloat(metadata, ImageMetadata.LENS_FOCUS_DISTANCE);
            out.focusRangeDiopters = getFloatArray(metadata, ImageMetadata.LENS_FOCUS_RANGE);
            out.lensState = getByte(metadata, ImageMetadata.LENS_STATE);
            out.afMode = getByte(metadata, ImageMetadata.CONTROL_AF_MODE);
            out.afState = getByte(metadata, ImageMetadata.CONTROL_AF_STATE);
            out.aeState = getByte(metadata, ImageMetadata.CONTROL_AE_STATE);
            out.awbState = getByte(metadata, ImageMetadata.CONTROL_AWB_STATE);
            out.oisMode = getByte(metadata, ImageMetadata.LENS_OPTICAL_STABILIZATION_MODE);
            out.videoStabilizationMode =
                    getByte(metadata, ImageMetadata.CONTROL_VIDEO_STABILIZATION_MODE);
            out.intrinsicCalibration =
                    getFloatArray(metadata, ImageMetadata.LENS_INTRINSIC_CALIBRATION);
            out.radialDistortion = getFloatArray(metadata, ImageMetadata.LENS_RADIAL_DISTORTION);
        } catch (NotYetAvailableException ignored) {
        }
        return out;
    }

    private static CapturePacket copyFrame(
            Image image,
            Camera camera,
            Pose pose,
            long frameTimestampNs,
            CameraMetadata metadata,
            Motion motion,
            int captureIndex) {
        byte[] nv21 = yuv420888ToNv21(image);
        CameraIntrinsics intrinsics = camera.getImageIntrinsics();

        float[] worldFromCamera = new float[16];
        float[] cameraFromWorld = new float[16];
        pose.toMatrix(worldFromCamera, 0);
        pose.inverse().toMatrix(cameraFromWorld, 0);

        return new CapturePacket(
                captureIndex,
                nv21,
                image.getWidth(),
                image.getHeight(),
                frameTimestampNs,
                image.getTimestamp(),
                pose.getTranslation().clone(),
                pose.getRotationQuaternion().clone(),
                worldFromCamera,
                cameraFromWorld,
                intrinsics.getFocalLength().clone(),
                intrinsics.getPrincipalPoint().clone(),
                intrinsics.getImageDimensions().clone(),
                metadata,
                motion);
    }

    private void writeCapture(CapturePacket packet) throws IOException, JSONException {
        String base = String.format(
                Locale.US, "frame_%06d_%d", packet.captureIndex, packet.imageTimestampNs);
        File jpegFile = new File(captureRoot, base + ".jpg");
        File jsonFile = new File(captureRoot, base + ".json");

        YuvImage yuv = new YuvImage(
                packet.nv21, ImageFormat.NV21, packet.width, packet.height, null);
        try (FileOutputStream output = new FileOutputStream(jpegFile)) {
            if (!yuv.compressToJpeg(
                    new Rect(0, 0, packet.width, packet.height), JPEG_QUALITY, output)) {
                throw new IOException("YUV to JPEG conversion failed");
            }
        }

        JSONObject json = new JSONObject();
        json.put("capture_index", packet.captureIndex);
        json.put("image", jpegFile.getName());
        json.put("width", packet.width);
        json.put("height", packet.height);
        json.put("frame_timestamp_ns", packet.frameTimestampNs);
        json.put("image_timestamp_ns", packet.imageTimestampNs);
        json.put("translation_m", array(packet.translation));
        json.put("rotation_quaternion_xyzw", array(packet.rotationQuaternion));
        json.put("world_from_camera_column_major", array(packet.worldFromCamera));
        json.put("camera_from_world_column_major", array(packet.cameraFromWorld));

        Pose pose = new Pose(packet.translation, packet.rotationQuaternion);
        json.put("forward_world", array(pose.rotateVector(new float[] {0f, 0f, -1f})));
        json.put("up_world", array(pose.rotateVector(new float[] {0f, 1f, 0f})));

        JSONObject intrinsics = new JSONObject();
        intrinsics.put("focal_length_px", array(packet.focalLengthPx));
        intrinsics.put("principal_point_px", array(packet.principalPointPx));
        intrinsics.put("image_dimensions", array(packet.intrinsicsDimensions));
        json.put("intrinsics", intrinsics);

        JSONObject cameraMetadata = new JSONObject();
        putNullable(cameraMetadata, "sensor_timestamp_ns", packet.metadata.sensorTimestampNs);
        putNullable(cameraMetadata, "exposure_time_ns", packet.metadata.exposureTimeNs);
        putNullable(cameraMetadata, "frame_duration_ns", packet.metadata.frameDurationNs);
        putNullable(cameraMetadata, "rolling_shutter_skew_ns", packet.metadata.rollingShutterSkewNs);
        putNullable(cameraMetadata, "iso", packet.metadata.iso);
        putNullable(cameraMetadata, "aperture_f_number", packet.metadata.aperture);
        putNullable(cameraMetadata, "focal_length_mm", packet.metadata.focalLengthMm);
        putNullable(
                cameraMetadata, "focus_distance_diopters", packet.metadata.focusDistanceDiopters);
        if (packet.metadata.focusRangeDiopters != null) {
            cameraMetadata.put(
                    "focus_range_diopters", array(packet.metadata.focusRangeDiopters));
        }
        putNullable(cameraMetadata, "lens_state", packet.metadata.lensState);
        putNullable(cameraMetadata, "af_mode", packet.metadata.afMode);
        putNullable(cameraMetadata, "af_state", packet.metadata.afState);
        putNullable(cameraMetadata, "ae_state", packet.metadata.aeState);
        putNullable(cameraMetadata, "awb_state", packet.metadata.awbState);
        putNullable(cameraMetadata, "ois_mode", packet.metadata.oisMode);
        putNullable(
                cameraMetadata,
                "video_stabilization_mode",
                packet.metadata.videoStabilizationMode);
        if (packet.metadata.intrinsicCalibration != null) {
            cameraMetadata.put(
                    "lens_intrinsic_calibration", array(packet.metadata.intrinsicCalibration));
        }
        if (packet.metadata.radialDistortion != null) {
            cameraMetadata.put("lens_radial_distortion", array(packet.metadata.radialDistortion));
        }
        json.put("camera_metadata", cameraMetadata);

        JSONObject quality = new JSONObject();
        quality.put("profile", pixel10a ? "pixel_10a" : "generic");
        quality.put("linear_speed_mps", packet.motion.linearSpeedMps);
        quality.put("angular_speed_dps", packet.motion.angularSpeedDps);
        quality.put("max_exposure_ns", maxExposureNs);
        quality.put("max_linear_speed_mps", maxLinearSpeedMps);
        quality.put("max_angular_speed_dps", maxAngularSpeedDps);
        quality.put("jpeg_quality", JPEG_QUALITY);
        json.put("capture_quality_gate", quality);

        try (FileOutputStream output = new FileOutputStream(jsonFile)) {
            output.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    private void writeDeviceInfo() {
        JSONObject json = new JSONObject();
        try {
            json.put("manufacturer", Build.MANUFACTURER);
            json.put("brand", Build.BRAND);
            json.put("model", Build.MODEL);
            json.put("device", Build.DEVICE);
            json.put("sdk_int", Build.VERSION.SDK_INT);
            json.put("capture_profile", pixel10a ? "pixel_10a" : "generic");
            json.put("eis_policy", "OFF for stable photogrammetry geometry");
            json.put("focus_policy", "ARCore AUTO; save only stationary/focused lens frames");
            json.put("max_exposure_ns", maxExposureNs);
            json.put("max_linear_speed_mps", maxLinearSpeedMps);
            json.put("max_angular_speed_dps", maxAngularSpeedDps);
            try (FileOutputStream output =
                         new FileOutputStream(new File(captureRoot, "session_device.json"))) {
                output.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        } catch (IOException | JSONException e) {
            Log.w(TAG, "Failed to write session device metadata", e);
        }
    }

    private static byte[] yuv420888ToNv21(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        byte[] out = new byte[width * height * 3 / 2];
        Image.Plane[] planes = image.getPlanes();

        copyPlane(planes[0], width, height, out, 0, 1);

        int chromaOffset = width * height;
        int chromaWidth = width / 2;
        int chromaHeight = height / 2;
        copyPlane(planes[2], chromaWidth, chromaHeight, out, chromaOffset, 2);
        copyPlane(planes[1], chromaWidth, chromaHeight, out, chromaOffset + 1, 2);
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
        try {
            return metadata.getLong(key);
        } catch (MetadataNotFoundException | IllegalArgumentException e) {
            return null;
        }
    }

    private static Integer getInt(ImageMetadata metadata, int key) {
        try {
            return metadata.getInt(key);
        } catch (MetadataNotFoundException | IllegalArgumentException e) {
            return null;
        }
    }

    private static Float getFloat(ImageMetadata metadata, int key) {
        try {
            return metadata.getFloat(key);
        } catch (MetadataNotFoundException | IllegalArgumentException e) {
            return null;
        }
    }

    private static float[] getFloatArray(ImageMetadata metadata, int key) {
        try {
            return metadata.getFloatArray(key);
        } catch (MetadataNotFoundException | IllegalArgumentException e) {
            return null;
        }
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
        Long sensorTimestampNs;
        Long exposureTimeNs;
        Long frameDurationNs;
        Long rollingShutterSkewNs;
        Integer iso;
        Float aperture;
        Float focalLengthMm;
        Float focusDistanceDiopters;
        float[] focusRangeDiopters;
        Byte lensState;
        Byte afMode;
        Byte afState;
        Byte aeState;
        Byte awbState;
        Byte oisMode;
        Byte videoStabilizationMode;
        float[] intrinsicCalibration;
        float[] radialDistortion;
    }

    private static final class CapturePacket {
        final int captureIndex;
        final byte[] nv21;
        final int width;
        final int height;
        final long frameTimestampNs;
        final long imageTimestampNs;
        final float[] translation;
        final float[] rotationQuaternion;
        final float[] worldFromCamera;
        final float[] cameraFromWorld;
        final float[] focalLengthPx;
        final float[] principalPointPx;
        final int[] intrinsicsDimensions;
        final CameraMetadata metadata;
        final Motion motion;

        CapturePacket(
                int captureIndex,
                byte[] nv21,
                int width,
                int height,
                long frameTimestampNs,
                long imageTimestampNs,
                float[] translation,
                float[] rotationQuaternion,
                float[] worldFromCamera,
                float[] cameraFromWorld,
                float[] focalLengthPx,
                float[] principalPointPx,
                int[] intrinsicsDimensions,
                CameraMetadata metadata,
                Motion motion) {
            this.captureIndex = captureIndex;
            this.nv21 = nv21;
            this.width = width;
            this.height = height;
            this.frameTimestampNs = frameTimestampNs;
            this.imageTimestampNs = imageTimestampNs;
            this.translation = translation;
            this.rotationQuaternion = rotationQuaternion;
            this.worldFromCamera = worldFromCamera;
            this.cameraFromWorld = cameraFromWorld;
            this.focalLengthPx = focalLengthPx;
            this.principalPointPx = principalPointPx;
            this.intrinsicsDimensions = intrinsicsDimensions;
            this.metadata = metadata;
            this.motion = motion;
        }
    }
}
