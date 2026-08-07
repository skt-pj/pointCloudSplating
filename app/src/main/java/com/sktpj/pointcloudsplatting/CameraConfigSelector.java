package com.sktpj.pointcloudsplatting;

import android.util.Log;
import android.util.Size;

import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.Session;

import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;

/** Selects a 30 fps ARCore camera configuration with the largest available CPU image stream. */
public final class CameraConfigSelector {
    private static final String TAG = "CameraConfigSelector";

    private CameraConfigSelector() {}

    public static CameraConfig selectHighestResolution30Fps(Session session) {
        CameraConfigFilter filter = new CameraConfigFilter(session);
        filter.setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30));
        List<CameraConfig> configs = session.getSupportedCameraConfigs(filter);
        if (configs.isEmpty()) {
            Log.w(TAG, "No 30 fps camera configs; keeping ARCore default config");
            return session.getCameraConfig();
        }

        CameraConfig best = configs.stream()
                .max(Comparator.comparingLong(CameraConfigSelector::cpuPixelCount)
                        .thenComparingLong(CameraConfigSelector::gpuPixelCount))
                .orElse(configs.get(0));

        Size cpu = best.getImageSize();
        Size gpu = best.getTextureSize();
        Log.i(TAG, "Selected camera config CPU=" + cpu.getWidth() + "x" + cpu.getHeight()
                + " GPU=" + gpu.getWidth() + "x" + gpu.getHeight());
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
