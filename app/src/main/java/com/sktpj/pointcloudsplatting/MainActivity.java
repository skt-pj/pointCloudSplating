package com.sktpj.pointcloudsplatting;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.display.DisplayManager;
import android.media.Image;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.Surface;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import java.io.IOException;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Minimal Raw Depth point-cloud viewer based on
 * google-ar/arcore-android-sdk samples/raw_depth_java.
 */
public final class MainActivity extends Activity
        implements GLSurfaceView.Renderer, DisplayManager.DisplayListener {

    private static final String TAG = "PointCloudRawDepth";
    private static final int CAMERA_PERMISSION_REQUEST = 1001;

    private final PointCloudRenderer pointCloudRenderer = new PointCloudRenderer();
    private final Object frameInUseLock = new Object();

    private GLSurfaceView surfaceView;
    private TextView statusView;
    private Session session;
    private DisplayManager displayManager;
    private boolean displayListenerRegistered;
    private boolean installRequested;
    private boolean depthReceived;
    private volatile boolean displayGeometryChanged = true;
    private volatile int viewportWidth;
    private volatile int viewportHeight;
    private long depthTimestamp = -1L;
    private String lastStatus = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        surfaceView = new GLSurfaceView(this);
        surfaceView.setPreserveEGLContextOnPause(true);
        surfaceView.setEGLContextClientVersion(2);
        surfaceView.setEGLConfigChooser(8, 8, 8, 0, 16, 0);
        surfaceView.setRenderer(this);
        surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        statusView = new TextView(this);
        statusView.setTextColor(0xFFFFFFFF);
        statusView.setBackgroundColor(0x66000000);
        statusView.setPadding(24, 16, 24, 16);
        statusView.setText("Starting ARCore...");

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(0xFF1A1A1A);
        root.addView(
                surfaceView,
                new FrameLayout.LayoutParams(
                        FrameLayout.LayoutParams.MATCH_PARENT,
                        FrameLayout.LayoutParams.MATCH_PARENT));

        FrameLayout.LayoutParams statusParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT);
        statusParams.gravity = Gravity.TOP | Gravity.START;
        root.addView(statusView, statusParams);
        setContentView(root);

        displayManager = (DisplayManager) getSystemService(Context.DISPLAY_SERVICE);
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!hasCameraPermission()) {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
            return;
        }

        if (session == null && !createSession()) {
            return;
        }

        if (!session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)) {
            setStatus("Raw Depth API is not supported on this device.");
            session.close();
            session = null;
            return;
        }

        try {
            synchronized (frameInUseLock) {
                Config config = session.getConfig();
                config.setDepthMode(Config.DepthMode.RAW_DEPTH_ONLY);
                session.configure(config);
                session.resume();
            }
        } catch (CameraNotAvailableException e) {
            Log.e(TAG, "Camera is not available", e);
            setStatus("Camera is not available. Restart the app.");
            session.close();
            session = null;
            return;
        }

        surfaceView.onResume();
        registerDisplayListener();
        displayGeometryChanged = true;
        setStatus("No depth yet. Move the device.");
    }

    private boolean createSession() {
        try {
            switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                case INSTALL_REQUESTED:
                    installRequested = true;
                    return false;
                case INSTALLED:
                    break;
            }
            session = new Session(this);
            return true;
        } catch (UnavailableArcoreNotInstalledException
                 | UnavailableUserDeclinedInstallationException e) {
            setStatus("Google Play Services for AR is required.");
        } catch (UnavailableApkTooOldException e) {
            setStatus("Update Google Play Services for AR.");
        } catch (UnavailableSdkTooOldException e) {
            setStatus("This app's ARCore SDK is too old for the installed service.");
        } catch (UnavailableDeviceNotCompatibleException e) {
            setStatus("This device is not ARCore compatible.");
        } catch (RuntimeException e) {
            Log.e(TAG, "Failed to create ARCore session", e);
            setStatus("Failed to create the ARCore session.");
        }
        return false;
    }

    @Override
    protected void onPause() {
        super.onPause();
        unregisterDisplayListener();
        if (session != null) {
            surfaceView.onPause();
            session.pause();
        }
    }

    @Override
    protected void onDestroy() {
        if (session != null) {
            session.close();
            session = null;
        }
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode,
            String[] permissions,
            int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != CAMERA_PERMISSION_REQUEST) {
            return;
        }

        if (hasCameraPermission()) {
            recreate();
        } else {
            Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show();
            finish();
        }
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        try {
            pointCloudRenderer.createOnGlThread(this);
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize OpenGL resources", e);
            setStatus("Failed to initialize OpenGL resources.");
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        viewportWidth = width;
        viewportHeight = height;
        displayGeometryChanged = true;
        GLES20.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
        if (session == null) {
            return;
        }

        synchronized (frameInUseLock) {
            try {
                updateDisplayGeometryIfNeeded();

                // Raw Depth sample does not render the camera image, but ARCore still requires
                // a camera texture name before Session.update().
                session.setCameraTextureNames(new int[] {0});

                Frame frame = session.update();
                Camera camera = frame.getCamera();

                if (camera.getTrackingState() != TrackingState.TRACKING) {
                    if (depthReceived) {
                        setStatus("AR tracking is paused. Move the device slowly.");
                    }
                    return;
                }

                boolean containsNewDepthData = false;
                try (Image depthImage = frame.acquireRawDepthImage16Bits()) {
                    long currentTimestamp = depthImage.getTimestamp();
                    containsNewDepthData = currentTimestamp != depthTimestamp;
                    depthTimestamp = currentTimestamp;
                } catch (NotYetAvailableException e) {
                    // Normal while Raw Depth is initializing.
                }

                if (containsNewDepthData) {
                    DepthData depth = DepthData.create(session, frame);
                    if (depth != null) {
                        depthReceived = true;
                        pointCloudRenderer.update(depth);
                    }
                }

                float[] projectionMatrix = new float[16];
                camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100.0f);
                float[] viewMatrix = new float[16];
                camera.getViewMatrix(viewMatrix, 0);

                pointCloudRenderer.draw(viewMatrix, projectionMatrix);

                if (depthReceived) {
                    setStatus(
                            "Accumulated raw depth frames: "
                                    + pointCloudRenderer.getStoredFrameCount());
                } else {
                    setStatus("No depth yet. Move the device.");
                }
            } catch (Throwable t) {
                Log.e(TAG, "OpenGL/ARCore frame failed", t);
                setStatus("Frame processing failed. See logcat: "
                        + t.getClass().getSimpleName());
            }
        }
    }

    private void updateDisplayGeometryIfNeeded() {
        if (!displayGeometryChanged || viewportWidth == 0 || viewportHeight == 0) {
            return;
        }

        int rotation = Surface.ROTATION_0;
        if (getWindowManager().getDefaultDisplay() != null) {
            rotation = getWindowManager().getDefaultDisplay().getRotation();
        }
        session.setDisplayGeometry(rotation, viewportWidth, viewportHeight);
        displayGeometryChanged = false;
    }

    private boolean hasCameraPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void registerDisplayListener() {
        if (!displayListenerRegistered && displayManager != null) {
            displayManager.registerDisplayListener(this, null);
            displayListenerRegistered = true;
        }
    }

    private void unregisterDisplayListener() {
        if (displayListenerRegistered && displayManager != null) {
            displayManager.unregisterDisplayListener(this);
            displayListenerRegistered = false;
        }
    }

    private void setStatus(String text) {
        if (text.equals(lastStatus)) {
            return;
        }
        lastStatus = text;
        runOnUiThread(() -> statusView.setText(text));
    }

    @Override
    public void onDisplayAdded(int displayId) {}

    @Override
    public void onDisplayRemoved(int displayId) {}

    @Override
    public void onDisplayChanged(int displayId) {
        displayGeometryChanged = true;
    }
}
