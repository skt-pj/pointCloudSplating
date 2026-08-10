package com.sktpj.pointcloudsplatting;

import android.content.Context;
import android.os.Environment;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Runs Phase 2 only after a saved dataset has a stored Phase 1 PASS.
 *
 * <p>The coordinator deliberately stays away from actively changing capture_tmp directories. This
 * lets v1.0.5 evaluate a newly finalized dataset automatically without competing with Camera2 /
 * ARCore capture. Existing Phase 3 trainer code is not called from here.
 */
public final class Phase2EvaluationCoordinator {
    private static final String TAG = "Phase2Coordinator";
    private static final long SCAN_INTERVAL_SECONDS = 4L;
    private static final long ACTIVE_CAPTURE_GRACE_MS = 8_000L;

    private static final ScheduledExecutorService EXECUTOR =
            Executors.newSingleThreadScheduledExecutor(runnable -> {
                Thread thread = new Thread(runnable, "Phase2Evaluation");
                thread.setDaemon(true);
                return thread;
            });
    private static final AtomicBoolean RUNNING = new AtomicBoolean(false);
    private static volatile Context appContext;

    private Phase2EvaluationCoordinator() {}

    public static void initialize(Context context) {
        if (context == null || appContext != null) return;
        appContext = context.getApplicationContext();
        EXECUTOR.scheduleWithFixedDelay(
                Phase2EvaluationCoordinator::scanOnce,
                4L,
                SCAN_INTERVAL_SECONDS,
                TimeUnit.SECONDS);
    }

    private static void scanOnce() {
        Context context = appContext;
        if (context == null || !RUNNING.compareAndSet(false, true)) return;
        try {
            File pictures = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
            if (pictures == null || !pictures.isDirectory()) return;
            if (hasActivelyChangingCapture(pictures)) return;

            File dataset = newestPhase2Candidate(pictures);
            if (dataset == null) return;

            DriveDiagnosticLogStore.setPrimaryDataset(dataset);
            DiagnosticLog.i(TAG, "Starting Phase 2 evaluation dataset=" + dataset.getAbsolutePath());
            Phase2DatasetEvaluator.Result result = Phase2DatasetEvaluator.evaluate(dataset);
            DiagnosticLog.i(
                    TAG,
                    "Phase 2 evaluation complete status="
                            + result.report.optString("status", "FAIL")
                            + " hardGate=" + result.hardGatePassed
                            + " reviewRequired=" + result.reviewRequired
                            + " failures=" + result.failureCount
                            + " warnings=" + result.warningCount);
            DriveDiagnosticLogStore.requestOverwrite();
        } catch (Throwable error) {
            DiagnosticLog.e(TAG, "Phase 2 background evaluation failed", error);
        } finally {
            RUNNING.set(false);
        }
    }

    private static boolean hasActivelyChangingCapture(File pictures) {
        File[] captures = pictures.listFiles(file ->
                file.isDirectory() && file.getName().startsWith("capture_tmp_"));
        if (captures == null || captures.length == 0) return false;
        long now = System.currentTimeMillis();
        for (File capture : captures) {
            if (now - capture.lastModified() <= ACTIVE_CAPTURE_GRACE_MS) return true;
        }
        return false;
    }

    private static File newestPhase2Candidate(File pictures) {
        File[] datasets = pictures.listFiles(file ->
                file.isDirectory()
                        && file.getName().startsWith("dataset_")
                        && new File(file, ".saved").isFile());
        if (datasets == null || datasets.length == 0) return null;
        Arrays.sort(datasets, Comparator.comparingLong(File::lastModified).reversed());
        for (File dataset : datasets) {
            if (!Phase1DatasetEvaluator.hasStoredPass(dataset)) continue;
            if (Phase2DatasetEvaluator.needsEvaluation(dataset)) return dataset;
        }
        return null;
    }
}
