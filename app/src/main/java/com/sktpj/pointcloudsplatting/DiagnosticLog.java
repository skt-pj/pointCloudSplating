package com.sktpj.pointcloudsplatting;

import android.util.Log;

import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.Date;
import java.util.Locale;

/** Small in-process log buffer that can be copied from the app menu. */
public final class DiagnosticLog {
    private static final int MAX_LINES = 500;
    private static final ArrayDeque<String> LINES = new ArrayDeque<>();
    private static final SimpleDateFormat TIME = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US);

    private DiagnosticLog() {}

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
        StringBuilder out = new StringBuilder();
        for (String line : LINES) {
            out.append(line).append('\n');
        }
        return out.toString();
    }

    private static synchronized void add(
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
        LINES.addLast(line.toString());
        while (LINES.size() > MAX_LINES) {
            LINES.removeFirst();
        }
    }
}
