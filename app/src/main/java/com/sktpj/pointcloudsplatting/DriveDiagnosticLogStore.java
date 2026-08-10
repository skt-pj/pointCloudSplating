package com.sktpj.pointcloudsplatting;

import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Replaces one user-selected Drive/SAF diagnostics document with the latest report.
 *
 * <p>The remote document is never appended. The dataset section prefers an explicitly selected
 * finalized dataset, then the newest finalized {@code dataset_*} carrying {@code .saved}. A newer
 * {@code capture_tmp_*} must never displace a completed dataset's Phase evaluation report.
 */
public final class DriveDiagnosticLogStore {
    public static final String FILE_NAME = "pointCloudSplating-current-diagnostics.txt";

    private static final String TAG = "DriveDiagnosticLog";
    private static final String PREFS_NAME = "drive-diagnostic-log";
    private static final String PREF_DOCUMENT_URI = "document_uri";
    private static final String PREF_PRIMARY_DATASET_PATH = "primary_dataset_path";
    private static final long AUTO_SYNC_DELAY_SECONDS = 3L;
    private static final int MAX_AUXILIARY_FILE_BYTES = 512_000;
    private static final String[] AUXILIARY_FILES = {
            "phase1_evaluation.json",
            "phase2_evaluation.json",
            "phase3_evaluation.json",
            "dataset_manifest.json",
            "3dgs_result.json"
    };

    private static final ScheduledExecutorService EXECUTOR =
            Executors.newSingleThreadScheduledExecutor(runnable -> {
                Thread thread = new Thread(runnable, "DriveDiagnosticLog");
                thread.setDaemon(true);
                return thread;
            });
    private static final AtomicBoolean AUTO_SYNC_QUEUED = new AtomicBoolean(false);
    private static final SimpleDateFormat DATE_TIME =
            new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS Z", Locale.US);

    private static volatile Context appContext;

    private DriveDiagnosticLogStore() {}

    public interface Completion {
        void onComplete(boolean success, String message);
    }

    public static void initialize(Context context) {
        if (context == null) return;
        appContext = context.getApplicationContext();
    }

    public static Intent createDestinationIntent() {
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_TITLE, FILE_NAME);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        return intent;
    }

    public static boolean hasDestination(Context context) {
        Context value = context == null ? appContext : context.getApplicationContext();
        return value != null && !readStoredUri(value).isEmpty();
    }

    public static boolean registerDestination(Context context, Intent resultData) {
        if (context == null || resultData == null || resultData.getData() == null) return false;
        Context value = context.getApplicationContext();
        Uri uri = resultData.getData();
        int takeFlags = resultData.getFlags()
                & (Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        if ((takeFlags & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) == 0) return false;
        try {
            value.getContentResolver().takePersistableUriPermission(uri, takeFlags);
            preferences(value).edit().putString(PREF_DOCUMENT_URI, uri.toString()).commit();
            appContext = value;
            return true;
        } catch (RuntimeException error) {
            Log.e(TAG, "Could not persist diagnostics document URI", error);
            return false;
        }
    }

    /** Makes a finalized dataset the source-of-truth for the Drive report. */
    public static void setPrimaryDataset(File dataset) {
        Context context = appContext;
        if (context == null || !isFinalizedDataset(dataset)) return;
        preferences(context)
                .edit()
                .putString(PREF_PRIMARY_DATASET_PATH, dataset.getAbsolutePath())
                .apply();
    }

    public static void requestOverwrite() {
        Context context = appContext;
        if (context == null || !hasDestination(context)) return;
        if (!AUTO_SYNC_QUEUED.compareAndSet(false, true)) return;
        EXECUTOR.schedule(() -> {
            AUTO_SYNC_QUEUED.set(false);
            overwriteInternal(context);
        }, AUTO_SYNC_DELAY_SECONDS, TimeUnit.SECONDS);
    }

    public static void overwriteNow(Completion completion) {
        Context context = appContext;
        if (context == null) {
            if (completion != null) completion.onComplete(false, "app context unavailable");
            return;
        }
        EXECUTOR.execute(() -> {
            WriteResult result = overwriteInternal(context);
            if (completion != null) completion.onComplete(result.success, result.message);
        });
    }

    private static WriteResult overwriteInternal(Context context) {
        String stored = readStoredUri(context);
        if (stored.isEmpty()) return WriteResult.fail("Driveログの保存先が未設定です");

        Uri uri;
        try {
            uri = Uri.parse(stored);
        } catch (RuntimeException error) {
            clearStoredUri(context);
            return WriteResult.fail("Driveログの保存先を読み取れませんでした");
        }

        String report;
        try {
            report = buildReport(context);
        } catch (RuntimeException error) {
            Log.e(TAG, "Could not build Drive diagnostics report", error);
            return WriteResult.fail("診断ログを準備できませんでした");
        }

        ContentResolver resolver = context.getContentResolver();
        try (OutputStream out = resolver.openOutputStream(uri, "wt")) {
            if (out == null) throw new IOException("output stream unavailable");
            out.write(report.getBytes(StandardCharsets.UTF_8));
            out.flush();
            return WriteResult.ok("Driveのログを上書きしました");
        } catch (SecurityException error) {
            Log.e(TAG, "Drive diagnostics permission lost", error);
            clearStoredUri(context);
            return WriteResult.fail("Driveの保存権限が失われました。保存先をもう一度設定してください");
        } catch (IOException | RuntimeException error) {
            Log.e(TAG, "Could not overwrite Drive diagnostics", error);
            return WriteResult.fail("Driveのログを更新できませんでした");
        }
    }

    private static String buildReport(Context context) {
        File dataset = findLatestDataset(context);
        String datasetState = dataset == null
                ? "unavailable"
                : (isFinalizedDataset(dataset) ? "finalized" : "temporary_fallback");

        StringBuilder out = new StringBuilder();
        out.append("pointCloudSplating Drive diagnostics\n")
                .append("version=").append(BuildConfig.VERSION_NAME).append('\n')
                .append("manufacturer=").append(Build.MANUFACTURER).append('\n')
                .append("model=").append(Build.MODEL).append('\n')
                .append("device=").append(Build.DEVICE).append('\n')
                .append("sdk=").append(Build.VERSION.SDK_INT).append('\n')
                .append("updatedAt=").append(DATE_TIME.format(new Date())).append('\n')
                .append("driveFile=").append(FILE_NAME).append('\n')
                .append("driveWriteMode=truncate_replace_not_append\n")
                .append("logScope=current_process_only\n")
                .append("datasetSelection=last_finalized_preferred\n")
                .append("datasetState=").append(datasetState).append('\n')
                .append("dataset=").append(dataset == null ? "unavailable" : dataset.getAbsolutePath())
                .append("\n\n=== current process log ===\n")
                .append(DiagnosticLog.currentProcessSnapshot());

        if (dataset != null) {
            for (String name : AUXILIARY_FILES) {
                File file = new File(dataset, name);
                if (!file.isFile()) continue;
                out.append("\n=== ").append(name).append(" ===\n");
                try {
                    out.append(readUtf8Bounded(file, MAX_AUXILIARY_FILE_BYTES));
                } catch (IOException error) {
                    out.append("[unavailable: ").append(error).append("]\n");
                }
            }
        }
        return out.toString();
    }

    /**
     * Primary selection policy:
     * 1. Persisted finalized dataset selected by evaluator/finalizer.
     * 2. Newest finalized dataset_* with .saved.
     * 3. Only when no finalized dataset exists, newest capture_tmp_* as a diagnostic fallback.
     */
    private static File findLatestDataset(Context context) {
        String preferredPath = preferences(context).getString(PREF_PRIMARY_DATASET_PATH, "");
        if (!preferredPath.isEmpty()) {
            File preferred = new File(preferredPath);
            if (isFinalizedDataset(preferred)) return preferred;
            preferences(context).edit().remove(PREF_PRIMARY_DATASET_PATH).apply();
        }

        File pictures = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        if (pictures == null || !pictures.isDirectory()) return null;

        File[] finalized = pictures.listFiles(DriveDiagnosticLogStore::isFinalizedDataset);
        if (finalized != null && finalized.length > 0) {
            Arrays.sort(finalized, Comparator.comparingLong(File::lastModified).reversed());
            File newest = finalized[0];
            preferences(context)
                    .edit()
                    .putString(PREF_PRIMARY_DATASET_PATH, newest.getAbsolutePath())
                    .apply();
            return newest;
        }

        File[] temporary = pictures.listFiles(file ->
                file.isDirectory() && file.getName().startsWith("capture_tmp_"));
        if (temporary == null || temporary.length == 0) return null;
        Arrays.sort(temporary, Comparator.comparingLong(File::lastModified).reversed());
        return temporary[0];
    }

    private static boolean isFinalizedDataset(File file) {
        return file != null
                && file.isDirectory()
                && file.getName().startsWith("dataset_")
                && new File(file, ".saved").isFile();
    }

    private static String readUtf8Bounded(File file, int maxBytes) throws IOException {
        try (FileInputStream in = new FileInputStream(file);
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int remaining = maxBytes;
            while (remaining > 0) {
                int read = in.read(buffer, 0, Math.min(buffer.length, remaining));
                if (read < 0) break;
                bytes.write(buffer, 0, read);
                remaining -= read;
            }
            String value = bytes.toString(StandardCharsets.UTF_8.name());
            if (file.length() > maxBytes) value += "\n[truncated]\n";
            return value;
        }
    }

    private static SharedPreferences preferences(Context context) {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    private static String readStoredUri(Context context) {
        return preferences(context).getString(PREF_DOCUMENT_URI, "");
    }

    private static void clearStoredUri(Context context) {
        preferences(context).edit().remove(PREF_DOCUMENT_URI).commit();
    }

    private static final class WriteResult {
        final boolean success;
        final String message;

        private WriteResult(boolean success, String message) {
            this.success = success;
            this.message = message;
        }

        static WriteResult ok(String message) {
            return new WriteResult(true, message);
        }

        static WriteResult fail(String message) {
            return new WriteResult(false, message);
        }
    }
}
