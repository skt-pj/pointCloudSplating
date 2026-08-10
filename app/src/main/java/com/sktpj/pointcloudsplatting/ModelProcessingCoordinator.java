package com.sktpj.pointcloudsplatting;

import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.os.Looper;

import java.lang.ref.WeakReference;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** Coordinates an opaque processing Activity so scanner camera/AR/GL resources are paused first. */
final class ModelProcessingCoordinator {
    private static final String TAG = "ModelProcessing";
    /**
     * v1.0.5 is a Phase 2 measurement build. Keep the old Phase 3 trainer physically present for
     * regression checks, but do not allow UI actions to start it until Phase 2 has a real-device
     * PASS and Phase 3 work is explicitly enabled in a later build.
     */
    private static final boolean PHASE3_PROCESSING_ENABLED = false;
    private static final Handler MAIN = new Handler(Looper.getMainLooper());
    private static final Object LOCK = new Object();

    private static volatile boolean active;
    private static volatile int latestProgress;
    private static volatile String latestMessage = "3Dモデルを準備しています…";
    private static CountDownLatch resumedLatch;
    private static WeakReference<ModelProcessingActivity> activityRef = new WeakReference<>(null);

    private ModelProcessingCoordinator() {}

    static boolean enter(Context context) {
        if (context == null) return false;
        if (!PHASE3_PROCESSING_ENABLED) {
            DiagnosticLog.w(
                    TAG,
                    "Phase 3 model processing blocked: current build is Phase 2 evaluation only");
            return false;
        }

        final CountDownLatch latch = new CountDownLatch(1);
        final AtomicBoolean launchFailed = new AtomicBoolean(false);
        synchronized (LOCK) {
            active = true;
            latestProgress = 0;
            latestMessage = "撮影を停止して変換用メモリを確保しています…";
            resumedLatch = latch;
        }

        MAIN.post(() -> {
            try {
                Intent intent = new Intent(context, ModelProcessingActivity.class);
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_NO_ANIMATION);
                context.startActivity(intent);
            } catch (RuntimeException error) {
                launchFailed.set(true);
                DiagnosticLog.e(TAG, "Failed to open processing Activity", error);
                latch.countDown();
            }
        });

        boolean resumed = false;
        try {
            resumed = latch.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        }

        if (!resumed || launchFailed.get()) {
            DiagnosticLog.e(TAG, "Processing Activity did not become foreground; native training aborted");
            exit();
            return false;
        }

        // The new Activity cannot reach onResume until ScannerActivity.onPause() has run. At this
        // point Camera2 is closed, ARCore is paused, and the scanner GLSurfaceView is paused.
        int clearedDepthFrames = PointCloudRenderer.clearAllFramesForModelProcessing();
        long beforeGc = usedJavaBytes();
        System.gc();
        try {
            Thread.sleep(250L); // allow asynchronous camera-close callbacks/driver buffers to settle
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        }
        long afterGc = usedJavaBytes();
        DiagnosticLog.i(TAG,
                "Scanner suspended before 3DGS: clearedDepthFrames=" + clearedDepthFrames
                        + " javaUsedBeforeGcMB=" + toMb(beforeGc)
                        + " javaUsedAfterGcMB=" + toMb(afterGc));
        publishProgress(1, "カメラを停止しました。3Dモデルを変換しています…");
        return true;
    }

    static void attach(ModelProcessingActivity activity) {
        synchronized (LOCK) {
            activityRef = new WeakReference<>(activity);
            if (resumedLatch != null) resumedLatch.countDown();
        }
        activity.updateProgress(latestProgress, latestMessage, active);
    }

    static void detach(ModelProcessingActivity activity) {
        synchronized (LOCK) {
            ModelProcessingActivity current = activityRef.get();
            if (current == activity) activityRef = new WeakReference<>(null);
        }
    }

    static void publishProgress(int progress, String message) {
        latestProgress = Math.max(0, Math.min(100, progress));
        if (message != null && !message.isEmpty()) latestMessage = message;
        MAIN.post(() -> {
            ModelProcessingActivity activity = activityRef.get();
            if (activity != null) {
                activity.updateProgress(latestProgress, latestMessage, active);
            }
        });
    }

    static boolean isActive() {
        return active;
    }

    static void exit() {
        active = false;
        MAIN.post(() -> {
            ModelProcessingActivity activity = activityRef.get();
            if (activity != null && !activity.isFinishing()) activity.finish();
        });
    }

    private static long usedJavaBytes() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static long toMb(long bytes) {
        return Math.max(0L, bytes / (1024L * 1024L));
    }
}
