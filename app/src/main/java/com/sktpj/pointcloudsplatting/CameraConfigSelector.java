package com.sktpj.pointcloudsplatting;

import android.util.Size;

import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.Session;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;

/** Selects a stream-safe 30 fps SharedCamera config while preserving headroom for high-resolution JPEG stills. */
public final class CameraConfigSelector {
    private static final String TAG = "CameraConfigSelector";

    private CameraConfigSelector() {}

    public static CameraConfig selectPhotogrammetry30Fps(Session session) {
        CameraConfig current = session.getCameraConfig();
        String currentCameraId = current.getCameraId();

        CameraConfigFilter filter = new CameraConfigFilter(session);
        filter.setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30));
        filter.setDepthSensorUsage(EnumSet.of(CameraConfig.DepthSensorUsage.DO_NOT_USE));

        List<CameraConfig> configs = session.getSupportedCameraConfigs(filter);
        if (configs.isEmpty()) {
            DiagnosticLog.w(TAG, "No 30 fps SharedCamera configs; keeping default config");
            return current;
        }

        // Keep the physical camera ARCore selected by default. Switching camera IDs would make the
        // saved Camera2 calibration and the ARCore tracking/depth camera refer to different lenses.
        List<CameraConfig> sameCamera = new ArrayList<>();
        for (CameraConfig config : configs) {
            if (currentCameraId.equals(config.getCameraId())) {
                sameCamera.add(config);
            }
        }
        List<CameraConfig> candidates = sameCamera.isEmpty() ? configs : sameCamera;

        // High-resolution Camera2 JPEG is the RGB observation of record. ARCore's CPU stream is
        // therefore kept at the lowest supported load; among equally light CPU configs, retain the
        // largest GPU preview. This is intentionally independent of JPEG aspect ratio/resolution.
        long minCpuPixels = candidates.stream()
                .mapToLong(CameraConfigSelector::cpuPixelCount)
                .min()
                .orElse(cpuPixelCount(current));

        CameraConfig best = candidates.stream()
                .filter(config -> cpuPixelCount(config) == minCpuPixels)
                .max(Comparator.comparingLong(CameraConfigSelector::gpuPixelCount))
                .orElse(current);

        Size cpu = best.getImageSize();
        Size gpu = best.getTextureSize();
        DiagnosticLog.i(TAG,
                "Selected stream-safe SharedCamera config CPU="
                        + cpu.getWidth() + "x" + cpu.getHeight()
                        + " GPU=" + gpu.getWidth() + "x" + gpu.getHeight()
                        + " cameraId=" + best.getCameraId()
                        + " depthSensorUsage=" + best.getDepthSensorUsage());
        return best;
    }

    private static long cpuPixelCount(CameraConfig config) {
        Size size = config.getImageSize();
        return (long) size.getWidth() * size.getHeight();
    }

    private static long gpuPixelCount(CameraConfig config) {
        Size size = config.getTextureSize();
        return (long) size.getWidth() * size.getHeight();
    }
}
