package com.sktpj.pointcloudsplatting;

import android.util.Size;

import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.Session;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;

/** Selects a 30 fps SharedCamera config for continuous ARCore CPU-image keyframe capture. */
public final class CameraConfigSelector {
    private static final String TAG = "CameraConfigSelector";
    private static final long MAX_CONTINUOUS_CPU_PIXELS = 2_500_000L;

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

        List<CameraConfig> sameCamera = new ArrayList<>();
        for (CameraConfig config : configs) {
            if (currentCameraId.equals(config.getCameraId())) sameCamera.add(config);
        }
        List<CameraConfig> candidates = sameCamera.isEmpty() ? configs : sameCamera;

        // The scanner no longer inserts high-resolution Camera2 still captures. Prefer the largest
        // ARCore CPU image that remains in a phone-friendly continuous 30 fps budget, because this
        // exact YUV frame is what DatasetCaptureManager ranks and saves without interrupting preview.
        CameraConfig best = candidates.stream()
                .filter(config -> cpuPixelCount(config) <= MAX_CONTINUOUS_CPU_PIXELS)
                .max(Comparator
                        .comparingLong(CameraConfigSelector::cpuPixelCount)
                        .thenComparingLong(CameraConfigSelector::gpuPixelCount))
                .orElseGet(() -> candidates.stream()
                        .min(Comparator.comparingLong(CameraConfigSelector::cpuPixelCount))
                        .orElse(current));

        Size cpu = best.getImageSize();
        Size gpu = best.getTextureSize();
        DiagnosticLog.i(TAG,
                "Selected continuous-capture SharedCamera config CPU="
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
