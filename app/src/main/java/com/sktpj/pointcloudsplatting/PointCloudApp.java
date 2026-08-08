package com.sktpj.pointcloudsplatting;

import android.app.Application;
import android.content.Context;

/** Process application used to give background reconstruction jobs a safe application Context. */
public final class PointCloudApp extends Application {
    private static volatile Context appContext;

    @Override
    public void onCreate() {
        super.onCreate();
        appContext = getApplicationContext();
        DiagnosticLog.initialize(appContext);

        Thread.UncaughtExceptionHandler previous = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
            try {
                DiagnosticLog.e(
                        "UncaughtException",
                        "thread=" + (thread == null ? "unknown" : thread.getName()),
                        throwable);
            } catch (Throwable ignored) {
            }
            if (previous != null) {
                previous.uncaughtException(thread, throwable);
            }
        });
    }

    public static Context context() {
        Context value = appContext;
        if (value == null) throw new IllegalStateException("PointCloudApp is not initialized");
        return value;
    }
}
