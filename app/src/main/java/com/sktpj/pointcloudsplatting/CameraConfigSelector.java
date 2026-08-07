package com.sktpj.pointcloudsplatting;

import android.util.Log;
import android.util.Size;

import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.Session;

import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;

/** Selects the best 30 fps ARCore config that is compatible with SharedCamera. */
public final class CameraConfigSelector {
    private static final String TAG = "CameraConfigSelector";

    private CameraConfigSelector() {}

    public static CameraConfig selectHighestResolution30Fps(Session session) {
        CameraConfigFilter filter = new CameraConfigFilter(session);
        filter.setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30));

        // SharedCamera cannot use an ARCore hardware depth sensor. Raw Depth may still be provided
        // by ARCore's motion/software depth pipeline when the selected device/camera supports it.
        filter.setDepthSensorUsage(EnumSet.of(CameraConfig.DepthSensorUsage.DO_NOT_USE));

        List<CameraConfig> configs = session.getSupportedCameraConfigs(filter);
        if (configs.isEmpty()) {
            Log.w(TAG, "No SharedCamera-compatible 30 fps configs; keeping ARCore default config");
            return session.getCameraConfig();
        }

        CameraConfig best = configs.stream()
                .max(Comparator.comparingLong(CameraConfigSelector::cpuPixelCount)
                        .thenComparingLong(CameraConfigSelector::gpuPixelCount))
                .orElse(configs.get(0));

        Size cpu = best.getImageSize();
        Size gpu = best.getTextureSize();
        Log.i(TAG, "Selected SharedCamera config CPU=" + cpu.getWidth() + "x" + cpu.getHeight()
                + " GPU=" + gpu.getWidth() + "x" + gpu.getHeight()
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
