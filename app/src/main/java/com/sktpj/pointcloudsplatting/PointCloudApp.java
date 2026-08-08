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
    }

    public static Context context() {
        Context value = appContext;
        if (value == null) throw new IllegalStateException("PointCloudApp is not initialized");
        return value;
    }
}
