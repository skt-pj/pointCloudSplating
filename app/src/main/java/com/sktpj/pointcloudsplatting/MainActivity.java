package com.sktpj.pointcloudsplatting;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.display.DisplayManager;
import android.media.Image;
import android.media.ImageReader;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.CameraConfig;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.Session;
import com.google.ar.core.SharedCamera;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;
import com.google.ar.core.exceptions.UnavailableApkTooOldException;
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException;
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException;
import com.google.ar.core.exceptions.UnavailableSdkTooOldException;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Raw Depth point-cloud scanner with SharedCamera high-resolution texture capture.
 *
 * <p>ARCore owns tracking and Raw Depth. Camera2 shares the same ARCore camera and adds an
 * occasional high-resolution JPEG stream. Each JPEG is synchronized to ARCore pose by sensor
 * timestamp in {@link HighResPhotoCaptureManager}.
 */
public final class MainActivity extends Activity
        implements GLSurfaceView.Renderer, DisplayManager.DisplayListener {

    private static final String TAG = "PointCloudRawDepth";
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final String HIGH_RES_CAPTURE_TAG = "pointcloud_high_res_texture";

    private final PointCloudRenderer pointCloudRenderer = new PointCloudRenderer();
    private final Object frameInUseLock = new Object();

    private GLSurfaceView surfaceView;
    private TextView statusView;
    private DisplayManager displayManager;

    private Session session;
    private SharedCamera sharedCamera;
    private CameraManager cameraManager;
    private CameraCharacteristics cameraCharacteristics;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder previewCaptureRequestBuilder;
    private ImageReader jpegReader;
    private HighResPhotoCaptureManager photoCaptureManager;

    private HandlerThread cameraThread;
    private Handler cameraHandler;

    private String cameraId;
    private Size jpegSize;
    private int cameraTextureId = -1;

    private boolean installRequested;
    private boolean activityResumed;
    private boolean surfaceCreated;
    private boolean cameraOpening;
    private volatile boolean arcoreActive;
    private boolean depthReceived;
    private boolean displayListenerRegistered;

    private volatile boolean displayGeometryChanged = true;
    private volatile int viewportWidth;
    private volatile int viewportHeight;
    private long depthTimestamp = -1L;
    private String lastStatus = "";
    private String cameraConfigSummary = "camera=?";

    private final CameraDevice.StateCallback cameraDeviceCallback =
            new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice device) {
                    Log.i(TAG, "Shared camera opened: " + device.getId());
                    cameraOpening = false;
                    cameraDevice = device;
                    createCameraCaptureSession();
                }

                @Override
                public void onDisconnected(CameraDevice device) {
                    Log.w(TAG, "Shared camera disconnected");
                    device.close();
                    cameraDevice = null;
                    cameraOpening = false;
                    setStatus("Camera disconnected.");
                }

                @Override
                public void onError(CameraDevice device, int error) {
                    Log.e(TAG, "Shared camera error=" + error);
                    device.close();
                    cameraDevice = null;
                    cameraOpening = false;
                    setStatus("Camera2/SharedCamera error: " + error);
                }

                @Override
                public void onClosed(CameraDevice device) {
                    if (cameraDevice == device) {
                        cameraDevice = null;
                    }
                    cameraOpening = false;
                }
            };

    private final CameraCaptureSession.StateCallback cameraSessionStateCallback =
            new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession configuredSession) {
                    captureSession = configuredSession;
                    try {
                        // This repeating request is submitted before ARCore is resumed. Once ARCore
                        // becomes active it owns the repeating request; the app only submits
                        // occasional one-shot high-resolution captures.
                        captureSession.setRepeatingRequest(
                                previewCaptureRequestBuilder.build(),
                                cameraCaptureCallback,
                                cameraHandler);
                    } catch (CameraAccessException | IllegalStateException e) {
                        Log.e(TAG, "Failed to start SharedCamera repeating request", e);
                        setStatus("Failed to start shared camera stream.");
                        return;
                    }
                }

                @Override
                public void onActive(CameraCaptureSession activeSession) {
                    if (activityResumed && !arcoreActive) {
                        resumeArCore();
                    }
                }

                @Override
                public void onConfigureFailed(CameraCaptureSession failedSession) {
                    Log.e(TAG, "SharedCamera capture session configuration failed");
                    setStatus(
                            "Pixel 10a high-res camera stream could not be configured with ARCore.");
                }

                @Override
                public void onClosed(CameraCaptureSession closedSession) {
                    if (captureSession == closedSession) {
                        captureSession = null;
                    }
                }
            };

    private final CameraCaptureSession.CaptureCallback cameraCaptureCallback =
            new CameraCaptureSession.CaptureCallback() {
                @Override
                public void onCaptureCompleted(
                        CameraCaptureSession captureSession,
                        CaptureRequest request,
                        TotalCaptureResult result) {
                    if (HIGH_RES_CAPTURE_TAG.equals(request.getTag())
                            && photoCaptureManager != null) {
                        photoCaptureManager.onCaptureCompleted(result);
                    }
                }

                @Override
                public void onCaptureFailed(
                        CameraCaptureSession captureSession,
                        CaptureRequest request,
                        CaptureFailure failure) {
                    if (HIGH_RES_CAPTURE_TAG.equals(request.getTag())
                            && photoCaptureManager != null) {
                        Log.w(TAG, "High-res JPEG capture failed: " + failure.getReason());
                        photoCaptureManager.onCaptureRequestFailed();
                    }
                }
            };

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
        statusView.setText("Starting ARCore SharedCamera...");

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
        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        startCameraThread();

        try {
            photoCaptureManager = new HighResPhotoCaptureManager(
                    HighResPhotoCaptureManager.getPicturesDirectory(this));
            Log.i(TAG, "Texture captures: " + photoCaptureManager.getCaptureDirectoryPath());
        } catch (RuntimeException e) {
            Log.e(TAG, "Failed to initialize texture capture directory", e);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        activityResumed = true;

        if (!hasCameraPermission()) {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
            return;
        }

        if (session == null && !createSharedArSession()) {
            return;
        }

        surfaceView.onResume();
        registerDisplayListener();
        displayGeometryChanged = true;

        if (surfaceCreated) {
            openCameraForSharing();
        }

        setStatus("Starting Pixel 10a SharedCamera / Raw Depth...");
    }

    private boolean createSharedArSession() {
        try {
            switch (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                case INSTALL_REQUESTED:
                    installRequested = true;
                    return false;
                case INSTALLED:
                    break;
            }

            session = new Session(this, EnumSet.of(Session.Feature.SHARED_CAMERA));
            CameraConfig cameraConfig = CameraConfigSelector.selectHighestResolution30Fps(session);
            session.setCameraConfig(cameraConfig);

            if (!session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)) {
                setStatus(
                        "Raw Depth is not available with SharedCamera on this Pixel 10a camera config.");
                session.close();
                session = null;
                return false;
            }

            Config config = session.getConfig();
            config.setDepthMode(Config.DepthMode.RAW_DEPTH_ONLY);
            config.setFocusMode(Config.FocusMode.AUTO);
            config.setImageStabilizationMode(Config.ImageStabilizationMode.OFF);
            config.setPlaneFindingMode(Config.PlaneFindingMode.DISABLED);
            config.setLightEstimationMode(Config.LightEstimationMode.DISABLED);
            session.configure(config);

            sharedCamera = session.getSharedCamera();
            cameraId = session.getCameraConfig().getCameraId();
            cameraCharacteristics = cameraManager.getCameraCharacteristics(cameraId);

            jpegSize = HighResPhotoCaptureManager.chooseJpegSize(cameraCharacteristics);
            jpegReader = ImageReader.newInstance(
                    jpegSize.getWidth(), jpegSize.getHeight(), ImageFormat.JPEG, 2);
            if (photoCaptureManager != null) {
                jpegReader.setOnImageAvailableListener(
                        photoCaptureManager::onJpegAvailable, cameraHandler);
                photoCaptureManager.configureCamera(cameraId, jpegSize, cameraCharacteristics);
            }

            // Tell ARCore about the custom high-resolution still surface before opening the camera.
            sharedCamera.setAppSurfaces(
                    cameraId, Collections.singletonList(jpegReader.getSurface()));

            Size cpu = session.getCameraConfig().getImageSize();
            Size gpu = session.getCameraConfig().getTextureSize();
            cameraConfigSummary =
                    "AR CPU " + cpu.getWidth() + "x" + cpu.getHeight()
                            + " / GPU " + gpu.getWidth() + "x" + gpu.getHeight()
                            + " / JPEG " + jpegSize.getWidth() + "x" + jpegSize.getHeight();
            Log.i(TAG, cameraConfigSummary
                    + " cameraId=" + cameraId
                    + " depthSensorUsage=" + session.getCameraConfig().getDepthSensorUsage());
            return true;
        } catch (UnavailableArcoreNotInstalledException
                 | UnavailableUserDeclinedInstallationException e) {
            setStatus("Google Play Services for AR is required.");
        } catch (UnavailableApkTooOldException e) {
            setStatus("Update Google Play Services for AR.");
        } catch (UnavailableSdkTooOldException e) {
            setStatus("Update this app / ARCore SDK.");
        } catch (UnavailableDeviceNotCompatibleException e) {
            setStatus("This device is not ARCore compatible.");
        } catch (CameraAccessException | RuntimeException e) {
            Log.e(TAG, "Failed to create SharedCamera ARCore session", e);
            setStatus("Failed to configure ARCore SharedCamera: " + e.getClass().getSimpleName());
        }

        if (session != null) {
            session.close();
            session = null;
        }
        return false;
    }

    private void openCameraForSharing() {
        if (!activityResumed
                || !surfaceCreated
                || session == null
                || sharedCamera == null
                || cameraId == null
                || cameraHandler == null
                || cameraDevice != null
                || cameraOpening) {
            return;
        }

        try {
            session.setCameraTextureName(cameraTextureId);
            CameraDevice.StateCallback wrapped =
                    sharedCamera.createARDeviceStateCallback(cameraDeviceCallback, cameraHandler);
            cameraOpening = true;
            cameraManager.openCamera(cameraId, wrapped, cameraHandler);
        } catch (CameraAccessException | SecurityException | IllegalArgumentException e) {
            cameraOpening = false;
            Log.e(TAG, "Failed to open ARCore shared camera", e);
            setStatus("Failed to open SharedCamera.");
        }
    }

    private void createCameraCaptureSession() {
        if (cameraDevice == null || sharedCamera == null || jpegReader == null) {
            return;
        }

        try {
            List<Surface> arCoreSurfaces = new ArrayList<>(sharedCamera.getArCoreSurfaces());
            List<Surface> sessionSurfaces = new ArrayList<>(arCoreSurfaces);
            sessionSurfaces.add(jpegReader.getSurface());

            previewCaptureRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            for (Surface surface : arCoreSurfaces) {
                previewCaptureRequestBuilder.addTarget(surface);
            }
            previewCaptureRequestBuilder.set(
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);

            CameraCaptureSession.StateCallback wrapped =
                    sharedCamera.createARSessionStateCallback(
                            cameraSessionStateCallback, cameraHandler);
            cameraDevice.createCaptureSession(sessionSurfaces, wrapped, cameraHandler);
        } catch (CameraAccessException | IllegalStateException e) {
            Log.e(TAG, "Failed to create SharedCamera capture session", e);
            setStatus("Failed to create SharedCamera capture session.");
        }
    }

    private void resumeArCore() {
        if (session == null || sharedCamera == null || arcoreActive || !activityResumed) {
            return;
        }
        try {
            session.resume();
            arcoreActive = true;
            sharedCamera.setCaptureCallback(cameraCaptureCallback, cameraHandler);
            setStatus("SharedCamera active. Move slowly around the subject.\n" + cameraConfigSummary);
        } catch (CameraNotAvailableException e) {
            Log.e(TAG, "Camera unavailable while resuming ARCore", e);
            setStatus("Camera unavailable. Restart the app.");
        }
    }

    private void requestHighResolutionStill() {
        if (cameraHandler == null) {
            if (photoCaptureManager != null) {
                photoCaptureManager.onCaptureRequestFailed();
            }
            return;
        }

        cameraHandler.post(() -> {
            if (!arcoreActive
                    || cameraDevice == null
                    || captureSession == null
                    || jpegReader == null
                    || sharedCamera == null) {
                if (photoCaptureManager != null) {
                    photoCaptureManager.onCaptureRequestFailed();
                }
                return;
            }

            try {
                CaptureRequest.Builder still =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);

                // Feed the same sensor capture to ARCore surfaces as well as the JPEG reader. This
                // gives ARCore an opportunity to expose a pose close to the JPEG sensor timestamp.
                for (Surface surface : sharedCamera.getArCoreSurfaces()) {
                    still.addTarget(surface);
                }
                still.addTarget(jpegReader.getSurface());
                still.setTag(HIGH_RES_CAPTURE_TAG);

                still.set(
                        CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                still.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON);
                still.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO);
                still.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF);
                still.set(
                        CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                        CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);
                still.set(CaptureRequest.JPEG_QUALITY, (byte) 95);
                still.set(CaptureRequest.JPEG_ORIENTATION, computeJpegOrientation());

                // Lock the already-good auto exposure / white balance for the one-shot still so
                // the camera does not suddenly lengthen shutter time after the blur gate passed.
                Boolean aeLockAvailable =
                        cameraCharacteristics.get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE);
                if (Boolean.TRUE.equals(aeLockAvailable)) {
                    still.set(CaptureRequest.CONTROL_AE_LOCK, true);
                }
                Boolean awbLockAvailable =
                        cameraCharacteristics.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE);
                if (Boolean.TRUE.equals(awbLockAvailable)) {
                    still.set(CaptureRequest.CONTROL_AWB_LOCK, true);
                }

                // At <=1/100 s and very low phone motion we prefer stable optics/calibration over
                // moving optical stabilization elements. Disable OIS only for the high-res still
                // when Camera2 reports that OIS control exists.
                int[] oisModes = cameraCharacteristics.get(
                        CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                if (contains(oisModes, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)) {
                    still.set(
                            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF);
                }

                captureSession.capture(still.build(), cameraCaptureCallback, cameraHandler);
            } catch (CameraAccessException | IllegalStateException | IllegalArgumentException e) {
                Log.e(TAG, "Failed to submit high-resolution JPEG", e);
                if (photoCaptureManager != null) {
                    photoCaptureManager.onCaptureRequestFailed();
                }
            }
        });
    }

    @Override
    protected void onPause() {
        activityResumed = false;
        unregisterDisplayListener();
        surfaceView.onPause();

        synchronized (frameInUseLock) {
            if (session != null && arcoreActive) {
                session.pause();
                arcoreActive = false;
            }
        }
        closeCamera();
        super.onPause();
    }

    private void closeCamera() {
        if (captureSession != null) {
            try {
                captureSession.close();
            } catch (RuntimeException e) {
                Log.w(TAG, "Error closing capture session", e);
            }
            captureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
        cameraOpening = false;
    }

    @Override
    protected void onDestroy() {
        activityResumed = false;
        closeCamera();

        if (jpegReader != null) {
            jpegReader.close();
            jpegReader = null;
        }
        if (photoCaptureManager != null) {
            photoCaptureManager.shutdown();
            photoCaptureManager = null;
        }
        if (session != null) {
            session.close();
            session = null;
        }
        stopCameraThread();
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
            cameraTextureId = createExternalCameraTexture();
            surfaceCreated = true;
            runOnUiThread(this::openCameraForSharing);
        } catch (IOException | RuntimeException e) {
            Log.e(TAG, "Failed to initialize OpenGL resources", e);
            setStatus("Failed to initialize OpenGL resources.");
        }
    }

    private static int createExternalCameraTexture() {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0]);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MIN_FILTER,
                GLES20.GL_LINEAR);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MAG_FILTER,
                GLES20.GL_LINEAR);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_S,
                GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_T,
                GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);
        return textures[0];
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
        if (session == null || !arcoreActive) {
            return;
        }

        synchronized (frameInUseLock) {
            try {
                updateDisplayGeometryIfNeeded();

                Frame frame = session.update();
                Camera camera = frame.getCamera();
                if (camera.getTrackingState() != TrackingState.TRACKING) {
                    if (depthReceived) {
                        setStatus("AR tracking paused. Move slowly and keep scene detail visible.");
                    }
                    return;
                }

                if (photoCaptureManager != null
                        && photoCaptureManager.onArFrame(frame, camera)) {
                    requestHighResolutionStill();
                }

                boolean containsNewDepthData = false;
                try (Image depthImage = frame.acquireRawDepthImage16Bits()) {
                    long currentTimestamp = depthImage.getTimestamp();
                    containsNewDepthData = currentTimestamp != depthTimestamp;
                    depthTimestamp = currentTimestamp;
                } catch (NotYetAvailableException e) {
                    // Normal while motion depth initializes.
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

                int photos = photoCaptureManager == null ? 0 : photoCaptureManager.getSavedCount();
                String profile = photoCaptureManager == null
                        ? "none" : photoCaptureManager.getProfileName();
                setStatus(
                        "Raw depth frames: " + pointCloudRenderer.getStoredFrameCount()
                                + " / high-res photos: " + photos
                                + " / " + profile
                                + "\n" + cameraConfigSummary);
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
        session.setDisplayGeometry(
                getWindowManager().getDefaultDisplay().getRotation(),
                viewportWidth,
                viewportHeight);
        displayGeometryChanged = false;
    }

    private int computeJpegOrientation() {
        int rotation = getWindowManager().getDefaultDisplay().getRotation();
        int displayDegrees;
        switch (rotation) {
            case Surface.ROTATION_90:
                displayDegrees = 90;
                break;
            case Surface.ROTATION_180:
                displayDegrees = 180;
                break;
            case Surface.ROTATION_270:
                displayDegrees = 270;
                break;
            case Surface.ROTATION_0:
            default:
                displayDegrees = 0;
                break;
        }
        Integer sensorOrientation =
                cameraCharacteristics == null
                        ? null
                        : cameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        int sensor = sensorOrientation == null ? 90 : sensorOrientation;
        return (sensor - displayDegrees + 360) % 360;
    }

    private static boolean contains(int[] values, int target) {
        if (values == null) {
            return false;
        }
        for (int value : values) {
            if (value == target) {
                return true;
            }
        }
        return false;
    }

    private boolean hasCameraPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void startCameraThread() {
        if (cameraThread != null) {
            return;
        }
        cameraThread = new HandlerThread("PointCloudCamera2");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
    }

    private void stopCameraThread() {
        if (cameraThread == null) {
            return;
        }
        cameraThread.quitSafely();
        try {
            cameraThread.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        cameraThread = null;
        cameraHandler = null;
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
