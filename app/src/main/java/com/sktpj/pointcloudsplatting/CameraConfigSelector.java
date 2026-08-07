package com.sktpj.pointcloudsplatting;

import android.util.Log;
import android.util.Size;

import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.Session;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;

/** Selects a 30 fps SharedCamera config that leaves stream bandwidth for texture JPEG capture. */
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

        // Keep the same physical camera ARCore selected by default. Switching camera IDs here can
        // silently move to a different rear lens and invalidate the intended photogrammetry setup.
        List<CameraConfig> sameCamera = new ArrayList<>();
        for (CameraConfig config : configs) {
            if (currentCameraId.equals(config.getCameraId())) {
                sameCamera.add(config);
            }
        }
        List<CameraConfig> candidates = sameCamera.isEmpty() ? configs : sameCamera;

        // ARCore always needs a tracking CPU stream and a GPU stream. A high CPU image config can
        // introduce another ARCore surface. We intentionally choose the smallest CPU image size so
        // the Pixel 10a still has stream bandwidth for the occasional ~12 MP JPEG surface. Among
        // equally small CPU configs, keep the largest GPU preview resolution.
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
