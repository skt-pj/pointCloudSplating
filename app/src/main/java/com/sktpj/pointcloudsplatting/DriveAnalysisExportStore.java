package com.sktpj.pointcloudsplatting;

import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Writes the current complete reconstruction dataset to one user-selected Drive/SAF ZIP.
 *
 * <p>This is an analysis transport, not another reconstruction stage. The ZIP preserves the
 * original Camera2 JPEG bytes, per-RGB metadata, every independent Raw Depth observation, Phase 2
 * geometry/evaluation artifacts, Phase 3 working data/results, and the final splat. The optimizer
 * checkpoint is intentionally omitted because it is large and is not needed to inspect capture,
 * camera geometry, point clouds, training inputs, or the final Gaussian parameters.
 */
public final class DriveAnalysisExportStore {
    public static final String FILE_NAME = "pointCloudSplating-analysis-current.zip";

    private static final String TAG = "DriveAnalysisExport";
    private static final String PREFS_NAME = "drive-analysis-export";
    private static final String PREF_DOCUMENT_URI = "document_uri";
    private static final String CHECKPOINT_NAME = "3dgs_checkpoint.bin";
    private static final int BUFFER_BYTES = 128 * 1024;

    private static final ExecutorService EXECUTOR =
            Executors.newSingleThreadExecutor(runnable -> {
                Thread thread = new Thread(runnable, "DriveAnalysisExport");
                thread.setDaemon(true);
                return thread;
            });
    private static final AtomicBoolean WORKER_QUEUED = new AtomicBoolean(false);
    private static final Object PENDING_LOCK = new Object();
    private static final SimpleDateFormat DATE_TIME =
            new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS Z", Locale.US);

    private static volatile Context appContext;
    private static String pendingDatasetPath = "";

    private DriveAnalysisExportStore() {}

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
        intent.setType("application/zip");
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
            Log.e(TAG, "Could not persist analysis ZIP URI", error);
            return false;
        }
    }

    /** Debounced/coalesced export used after dataset finalization and evaluation/training updates. */
    public static void requestExport(File dataset) {
        Context context = appContext;
        if (context == null || !hasDestination(context) || !isFinalizedDataset(dataset)) return;
        synchronized (PENDING_LOCK) {
            pendingDatasetPath = dataset.getAbsolutePath();
        }
        if (!WORKER_QUEUED.compareAndSet(false, true)) return;
        EXECUTOR.execute(() -> {
            try {
                while (true) {
                    String path;
                    synchronized (PENDING_LOCK) {
                        path = pendingDatasetPath;
                        pendingDatasetPath = "";
                    }
                    if (path.isEmpty()) break;
                    exportInternal(context, new File(path));
                    synchronized (PENDING_LOCK) {
                        if (pendingDatasetPath.isEmpty()) break;
                    }
                }
            } finally {
                WORKER_QUEUED.set(false);
                synchronized (PENDING_LOCK) {
                    if (!pendingDatasetPath.isEmpty()) requestExport(new File(pendingDatasetPath));
                }
            }
        });
    }

    public static void exportLatestNow(Completion completion) {
        Context context = appContext;
        if (context == null) {
            if (completion != null) completion.onComplete(false, "app context unavailable");
            return;
        }
        EXECUTOR.execute(() -> {
            File dataset = findLatestDataset(context);
            WriteResult result = dataset == null
                    ? WriteResult.fail("解析できる保存済み撮影データがありません")
                    : exportInternal(context, dataset);
            if (completion != null) completion.onComplete(result.success, result.message);
        });
    }

    public static void exportNow(File dataset, Completion completion) {
        Context context = appContext;
        if (context == null) {
            if (completion != null) completion.onComplete(false, "app context unavailable");
            return;
        }
        EXECUTOR.execute(() -> {
            WriteResult result = exportInternal(context, dataset);
            if (completion != null) completion.onComplete(result.success, result.message);
        });
    }

    private static WriteResult exportInternal(Context context, File dataset) {
        if (!isFinalizedDataset(dataset)) {
            return WriteResult.fail("保存済み撮影データを確認できませんでした");
        }
        String stored = readStoredUri(context);
        if (stored.isEmpty()) return WriteResult.fail("解析データのDrive保存先が未設定です");

        Uri uri;
        try {
            uri = Uri.parse(stored);
        } catch (RuntimeException error) {
            clearStoredUri(context);
            return WriteResult.fail("解析データのDrive保存先を読み取れませんでした");
        }

        List<File> files = new ArrayList<>();
        List<JSONObject> skipped = new ArrayList<>();
        collectDatasetFiles(dataset, dataset, files, skipped);
        files.sort(Comparator.comparing(file -> relativePath(dataset, file)));

        long sourceBytes = 0L;
        int rgbCount = 0;
        int depthPlyCount = 0;
        for (File file : files) {
            sourceBytes += Math.max(0L, file.length());
            String rel = relativePath(dataset, file);
            if (rel.startsWith("frame_") && rel.endsWith(".jpg")) rgbCount++;
            if (rel.startsWith("depth_obs_") && rel.endsWith(".ply")) depthPlyCount++;
        }

        DiagnosticLog.i(TAG, "Analysis export starting dataset=" + dataset.getName()
                + " files=" + files.size() + " bytes=" + sourceBytes
                + " rgb=" + rgbCount + " depthPly=" + depthPlyCount);

        JSONArray entries = new JSONArray();
        ContentResolver resolver = context.getContentResolver();
        try (OutputStream raw = resolver.openOutputStream(uri, "wt")) {
            if (raw == null) throw new IOException("output stream unavailable");
            try (ZipOutputStream zip = new ZipOutputStream(new BufferedOutputStream(raw, BUFFER_BYTES))) {
                byte[] buffer = new byte[BUFFER_BYTES];
                for (File file : files) {
                    String rel = relativePath(dataset, file);
                    long sizeBefore = file.length();
                    long modifiedBefore = file.lastModified();
                    MessageDigest digest = sha256();
                    ZipEntry entry = new ZipEntry("dataset/" + rel);
                    entry.setTime(modifiedBefore);
                    zip.putNextEntry(entry);
                    long copied = 0L;
                    try (BufferedInputStream in = new BufferedInputStream(
                            new FileInputStream(file), BUFFER_BYTES)) {
                        int read;
                        while ((read = in.read(buffer)) >= 0) {
                            if (read == 0) continue;
                            zip.write(buffer, 0, read);
                            digest.update(buffer, 0, read);
                            copied += read;
                        }
                    }
                    zip.closeEntry();
                    if (file.length() != sizeBefore || file.lastModified() != modifiedBefore
                            || copied != sizeBefore) {
                        throw new IOException("source changed during export: " + rel);
                    }
                    JSONObject item = new JSONObject();
                    item.put("path", rel);
                    item.put("bytes", copied);
                    item.put("sha256", hex(digest.digest()));
                    item.put("category", category(rel));
                    entries.put(item);
                }

                JSONObject manifest = new JSONObject();
                manifest.put("format_version", 1);
                manifest.put("status", "COMPLETE");
                manifest.put("app_version", BuildConfig.VERSION_NAME);
                manifest.put("app_version_code", BuildConfig.VERSION_CODE);
                manifest.put("manufacturer", Build.MANUFACTURER);
                manifest.put("model", Build.MODEL);
                manifest.put("dataset_name", dataset.getName());
                manifest.put("dataset_path", dataset.getAbsolutePath());
                manifest.put("exported_at", DATE_TIME.format(new Date()));
                manifest.put("transport", "android_saf_documents_provider");
                manifest.put("cloud_upload_confirmation", "unavailable_provider_managed_sync");
                manifest.put("original_jpeg_bytes_preserved", true);
                manifest.put("independent_depth_observations_preserved", true);
                manifest.put("phase2_geometry_preserved", new File(dataset, "phase2_geometry_prior.ply").isFile());
                manifest.put("final_splat_preserved", new File(dataset, "splat.ply").isFile());
                manifest.put("optimizer_checkpoint_included", false);
                manifest.put("optimizer_checkpoint_omission_reason",
                        "large optimizer state is not required for capture/geometry/photometric artifact analysis");
                manifest.put("file_count", entries.length());
                manifest.put("source_total_bytes", sourceBytes);
                manifest.put("rgb_original_count", rgbCount);
                manifest.put("depth_observation_ply_count", depthPlyCount);
                manifest.put("entries", entries);
                JSONArray skippedArray = new JSONArray();
                for (JSONObject item : skipped) skippedArray.put(item);
                manifest.put("skipped", skippedArray);

                byte[] manifestBytes = manifest.toString(2).getBytes(StandardCharsets.UTF_8);
                ZipEntry manifestEntry = new ZipEntry("analysis_manifest.json");
                zip.putNextEntry(manifestEntry);
                zip.write(manifestBytes);
                zip.closeEntry();
                zip.finish();
            }
            DiagnosticLog.i(TAG, "Analysis export COMPLETE dataset=" + dataset.getName()
                    + " files=" + files.size() + " sourceBytes=" + sourceBytes);
            DriveDiagnosticLogStore.requestOverwrite();
            return WriteResult.ok("Drive保存先へ解析データを書き込みました（原画像・点群・処理結果を含みます）");
        } catch (SecurityException error) {
            Log.e(TAG, "Analysis export permission lost", error);
            clearStoredUri(context);
            DiagnosticLog.e(TAG, "Analysis export permission lost", error);
            return WriteResult.fail("解析データのDrive保存権限が失われました。保存先をもう一度設定してください");
        } catch (Exception error) {
            Log.e(TAG, "Could not write analysis ZIP", error);
            DiagnosticLog.e(TAG, "Analysis export failed dataset=" + dataset.getName(), error);
            return WriteResult.fail("解析データをDrive保存先へ書き込めませんでした");
        }
    }

    private static void collectDatasetFiles(
            File dataset, File current, List<File> out, List<JSONObject> skipped) {
        File[] children = current.listFiles();
        if (children == null) return;
        Arrays.sort(children, Comparator.comparing(File::getName));
        for (File child : children) {
            if (child.isDirectory()) {
                collectDatasetFiles(dataset, child, out, skipped);
                continue;
            }
            if (!child.isFile()) continue;
            String rel = relativePath(dataset, child);
            if (CHECKPOINT_NAME.equals(child.getName())) {
                try {
                    JSONObject item = new JSONObject();
                    item.put("path", rel);
                    item.put("bytes", child.length());
                    item.put("reason", "optimizer_checkpoint_excluded");
                    skipped.add(item);
                } catch (Exception ignored) {
                }
                continue;
            }
            if (child.getName().endsWith(".tmp") || child.getName().endsWith(".lock")) continue;
            out.add(child);
        }
    }

    private static String category(String rel) {
        String name = new File(rel).getName();
        if (name.startsWith("frame_") && name.endsWith(".jpg")) return "rgb_original";
        if (name.startsWith("frame_") && name.endsWith(".json")) return "rgb_metadata";
        if (name.startsWith("depth_obs_") && name.endsWith(".ply")) return "raw_depth_point_cloud";
        if (name.startsWith("depth_obs_") && name.endsWith(".json")) return "raw_depth_metadata";
        if ("phase2_geometry_prior.ply".equals(name)) return "phase2_fused_geometry";
        if (name.startsWith("phase2_overlay_") && name.endsWith(".jpg")) return "phase2_overlay";
        if ("splat.ply".equals(name)) return "final_3dgs";
        if ("preview_splat.ply".equals(name)) return "preview_3dgs";
        if (rel.startsWith("vksplat_data/")) return "phase3_working_data";
        if (name.endsWith(".json")) return "evaluation_or_processing_metadata";
        return "dataset_artifact";
    }

    private static String relativePath(File root, File file) {
        String rootPath = root.getAbsolutePath();
        String path = file.getAbsolutePath();
        String rel = path.startsWith(rootPath) ? path.substring(rootPath.length()) : file.getName();
        while (rel.startsWith(File.separator)) rel = rel.substring(1);
        return rel.replace(File.separatorChar, '/');
    }

    private static File findLatestDataset(Context context) {
        File pictures = context.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        if (pictures == null || !pictures.isDirectory()) return null;
        File[] datasets = pictures.listFiles(DriveAnalysisExportStore::isFinalizedDataset);
        if (datasets == null || datasets.length == 0) return null;
        Arrays.sort(datasets, Comparator.comparing(File::getName).reversed());
        return datasets[0];
    }

    private static boolean isFinalizedDataset(File file) {
        return file != null && file.isDirectory() && file.getName().startsWith("dataset_")
                && new File(file, ".saved").isFile();
    }

    private static MessageDigest sha256() throws NoSuchAlgorithmException {
        return MessageDigest.getInstance("SHA-256");
    }

    private static String hex(byte[] value) {
        StringBuilder out = new StringBuilder(value.length * 2);
        for (byte b : value) out.append(String.format(Locale.US, "%02x", b & 0xff));
        return out.toString();
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
