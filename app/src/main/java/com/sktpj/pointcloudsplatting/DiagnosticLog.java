package com.sktpj.pointcloudsplatting;

import android.app.ActivityManager;
import android.app.ApplicationExitInfo;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/** Diagnostic log that survives process death so crashes can be inspected from the library. */
public final class DiagnosticLog {
    private static final int MAX_LINES = 500;
    private static final long MAX_FILE_BYTES = 1_000_000L;
    private static final String LOG_FILE_NAME = "diagnostic-history.log";
    private static final String PREFS_NAME = "diagnostic-log";
    private static final String PREF_LAST_EXIT_TIMESTAMP = "last_exit_timestamp";
    private static final String PREF_LAST_NATIVE_TRACE_TIMESTAMP = "last_native_trace_timestamp";
    private static final ArrayDeque<String> LINES = new ArrayDeque<>();
    private static final SimpleDateFormat TIME = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US);
    private static final SimpleDateFormat DATE_TIME =
            new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US);

    private static File persistentFile;
    private static String currentProcessStartMarker;

    private DiagnosticLog() {}

    public static synchronized void initialize(Context context) {
        Context appContext = context.getApplicationContext();
        persistentFile = new File(appContext.getFilesDir(), LOG_FILE_NAME);
        loadRecentPersistentLinesLocked();
        recordHistoricalProcessExitsLocked(appContext);
        currentProcessStartMarker =
                "process start pid=" + android.os.Process.myPid()
                        + " version=" + BuildConfig.VERSION_NAME
                        + " sdk=" + Build.VERSION.SDK_INT;
        addLocked(
                "I",
                "DiagnosticLog",
                currentProcessStartMarker,
                null);
    }

    public static void i(String tag, String message) {
        Log.i(tag, message);
        add("I", tag, message, null);
    }

    public static void w(String tag, String message) {
        Log.w(tag, message);
        add("W", tag, message, null);
    }

    public static void w(String tag, String message, Throwable throwable) {
        Log.w(tag, message, throwable);
        add("W", tag, message, throwable);
    }

    public static void e(String tag, String message) {
        Log.e(tag, message);
        add("E", tag, message, null);
    }

    public static void e(String tag, String message, Throwable throwable) {
        Log.e(tag, message, throwable);
        add("E", tag, message, throwable);
    }

    public static synchronized String snapshot() {
        if (persistentFile != null && persistentFile.isFile()) {
            try {
                return readText(persistentFile);
            } catch (IOException e) {
                Log.w("DiagnosticLog", "Failed to read persistent diagnostics", e);
            }
        }
        StringBuilder out = new StringBuilder();
        for (String line : LINES) {
            out.append(line).append('\n');
        }
        return out.toString();
    }

    /**
     * Returns only the current process section for remote diagnostics. Historical process output is
     * intentionally excluded so the Drive document stays focused on the device run being tested.
     */
    public static synchronized String currentProcessSnapshot() {
        String all = snapshot();
        String marker = currentProcessStartMarker;
        if (marker == null || marker.isEmpty()) return all;
        int markerIndex = all.lastIndexOf(marker);
        if (markerIndex < 0) return all;
        int lineStart = all.lastIndexOf('\n', Math.max(0, markerIndex - 1));
        return all.substring(lineStart < 0 ? 0 : lineStart + 1);
    }

    private static synchronized void add(
            String level, String tag, String message, Throwable throwable) {
        addLocked(level, tag, message, throwable);
    }

    private static void addLocked(
            String level, String tag, String message, Throwable throwable) {
        StringBuilder line = new StringBuilder();
        line.append(TIME.format(new Date()))
                .append(' ')
                .append(level)
                .append('/')
                .append(tag)
                .append(": ")
                .append(message == null ? "" : message);
        if (throwable != null) {
            line.append('\n').append(Log.getStackTraceString(throwable));
        }
        String value = line.toString();
        LINES.addLast(value);
        while (LINES.size() > MAX_LINES) {
            LINES.removeFirst();
        }
        appendPersistentLocked(value);
        DriveDiagnosticLogStore.requestOverwrite();
    }

    private static void appendPersistentLocked(String line) {
        File file = persistentFile;
        if (file == null) return;
        try {
            if (file.length() > MAX_FILE_BYTES) {
                try (FileOutputStream reset = new FileOutputStream(file, false)) {
                    reset.write(("=== diagnostic log rotated "
                            + DATE_TIME.format(new Date())
                            + " ===\n").getBytes(StandardCharsets.UTF_8));
                }
            }
            try (FileOutputStream out = new FileOutputStream(file, true)) {
                out.write(line.getBytes(StandardCharsets.UTF_8));
                out.write('\n');
                out.flush();
            }
        } catch (IOException e) {
            Log.w("DiagnosticLog", "Failed to persist diagnostic line", e);
        }
    }

    private static void loadRecentPersistentLinesLocked() {
        File file = persistentFile;
        if (file == null || !file.isFile()) return;
        ArrayDeque<String> recent = new ArrayDeque<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                recent.addLast(line);
                while (recent.size() > MAX_LINES) recent.removeFirst();
            }
        } catch (IOException e) {
            Log.w("DiagnosticLog", "Failed to load persistent diagnostics", e);
            return;
        }
        LINES.clear();
        LINES.addAll(recent);
    }

    private static void recordHistoricalProcessExitsLocked(Context context) {
        if (Build.VERSION.SDK_INT < 30) return;
        try {
            ActivityManager manager =
                    (ActivityManager) context.getSystemService(Context.ACTIVITY_SERVICE);
            if (manager == null) return;
            List<ApplicationExitInfo> exits = manager.getHistoricalProcessExitReasons(
                    context.getPackageName(), 0, 8);
            if (exits == null || exits.isEmpty()) return;

            SharedPreferences prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
            long lastRecorded = prefs.getLong(PREF_LAST_EXIT_TIMESTAMP, 0L);
            long newestRecorded = lastRecorded;
            List<ApplicationExitInfo> unseen = new ArrayList<>();
            for (ApplicationExitInfo exit : exits) {
                if (exit.getTimestamp() > lastRecorded) unseen.add(exit);
            }
            Collections.reverse(unseen);
            for (ApplicationExitInfo exit : unseen) {
                StringBuilder message = new StringBuilder();
                message.append("previous process exit")
                        .append(" time=").append(DATE_TIME.format(new Date(exit.getTimestamp())))
                        .append(" pid=").append(exit.getPid())
                        .append(" process=").append(exit.getProcessName())
                        .append(" reason=").append(exitReasonName(exit.getReason()))
                        .append('(').append(exit.getReason()).append(')')
                        .append(" status=").append(exit.getStatus())
                        .append(" importance=").append(exit.getImportance())
                        .append(" pssKb=").append(exit.getPss())
                        .append(" rssKb=").append(exit.getRss());
                String description = exit.getDescription();
                if (description != null && !description.isEmpty()) {
                    message.append(" description=").append(description.replace('\n', ' '));
                }
                if (exit.getReason() == ApplicationExitInfo.REASON_CRASH_NATIVE) {
                    message.append(" nativeTraceCapture=enabled");
                }
                addLocked("W", "ProcessExit", message.toString(), null);
                newestRecorded = Math.max(newestRecorded, exit.getTimestamp());
            }
            if (newestRecorded > lastRecorded) {
                prefs.edit().putLong(PREF_LAST_EXIT_TIMESTAMP, newestRecorded).apply();
            }

            recordHistoricalNativeTracesLocked(prefs, exits);
        } catch (RuntimeException e) {
            Log.w("DiagnosticLog", "Failed to read historical process exits", e);
        }
    }

    private static void recordHistoricalNativeTracesLocked(
            SharedPreferences prefs, List<ApplicationExitInfo> exits) {
        if (Build.VERSION.SDK_INT < 31) return;
        long lastTraceRecorded = prefs.getLong(PREF_LAST_NATIVE_TRACE_TIMESTAMP, 0L);
        long newestTraceRecorded = lastTraceRecorded;
        List<ApplicationExitInfo> nativeCrashes = new ArrayList<>();
        for (ApplicationExitInfo exit : exits) {
            if (exit.getReason() == ApplicationExitInfo.REASON_CRASH_NATIVE
                    && exit.getTimestamp() > lastTraceRecorded) {
                nativeCrashes.add(exit);
            }
        }
        Collections.reverse(nativeCrashes);
        for (ApplicationExitInfo exit : nativeCrashes) {
            try (InputStream trace = exit.getTraceInputStream()) {
                if (trace == null) {
                    addLocked("W", "NativeTombstone",
                            "trace unavailable time="
                                    + DATE_TIME.format(new Date(exit.getTimestamp()))
                                    + " pid=" + exit.getPid(), null);
                } else {
                    String parsed = NativeTombstoneParser.parse(trace);
                    addLocked("E", "NativeTombstone",
                            "time=" + DATE_TIME.format(new Date(exit.getTimestamp()))
                                    + " process=" + exit.getProcessName()
                                    + "\n" + parsed,
                            null);
                }
            } catch (IOException | RuntimeException error) {
                addLocked("W", "NativeTombstone",
                        "trace parse failed time="
                                + DATE_TIME.format(new Date(exit.getTimestamp()))
                                + " pid=" + exit.getPid()
                                + " error=" + error,
                        null);
            }
            newestTraceRecorded = Math.max(newestTraceRecorded, exit.getTimestamp());
        }
        if (newestTraceRecorded > lastTraceRecorded) {
            prefs.edit().putLong(PREF_LAST_NATIVE_TRACE_TIMESTAMP, newestTraceRecorded).apply();
        }
    }

    private static String exitReasonName(int reason) {
        switch (reason) {
            case ApplicationExitInfo.REASON_EXIT_SELF:
                return "EXIT_SELF";
            case ApplicationExitInfo.REASON_SIGNALED:
                return "SIGNALED";
            case ApplicationExitInfo.REASON_LOW_MEMORY:
                return "LOW_MEMORY";
            case ApplicationExitInfo.REASON_CRASH:
                return "CRASH";
            case ApplicationExitInfo.REASON_CRASH_NATIVE:
                return "CRASH_NATIVE";
            case ApplicationExitInfo.REASON_ANR:
                return "ANR";
            case ApplicationExitInfo.REASON_INITIALIZATION_FAILURE:
                return "INITIALIZATION_FAILURE";
            case ApplicationExitInfo.REASON_PERMISSION_CHANGE:
                return "PERMISSION_CHANGE";
            case ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE:
                return "EXCESSIVE_RESOURCE_USAGE";
            case ApplicationExitInfo.REASON_USER_REQUESTED:
                return "USER_REQUESTED";
            case ApplicationExitInfo.REASON_USER_STOPPED:
                return "USER_STOPPED";
            case ApplicationExitInfo.REASON_DEPENDENCY_DIED:
                return "DEPENDENCY_DIED";
            case ApplicationExitInfo.REASON_OTHER:
                return "OTHER";
            default:
                return "UNKNOWN";
        }
    }

    private static String readText(File file) throws IOException {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                out.append(line).append('\n');
            }
        }
        return out.toString();
    }
}
