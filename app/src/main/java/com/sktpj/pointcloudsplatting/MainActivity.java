package com.sktpj.pointcloudsplatting;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
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
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.display.DisplayManager;
import android.media.Image;
import android.media.ImageReader;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Range;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.PopupMenu;
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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/** Raw Depth scanner + Camera2 SharedCamera texture capture for photogrammetry / 3DGS. */
public final class MainActivity extends Activity
        implements GLSurfaceView.Renderer, DisplayManager.DisplayListener {

    private static final String TAG = "PointCloudRawDepth";
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final int MENU_COPY_LOG = 1;
    private static final String HIGH_RES_CAPTURE_TAG = "pointcloud_high_res_texture";

    // Pixel 10a photogrammetry policy.
    private static final long TARGET_STILL_EXPOSURE_NS = 2_000_000L; // 1/500 s
    private static final long MAX_STILL_EXPOSURE_NS = 3_333_333L;    // 1/300 s
    private static final int MAX_PHOTOGRAMMETRY_ISO = 1600;

    private final PointCloudRenderer pointCloudRenderer = new PointCloudRenderer();
    private final CameraBackgroundRenderer cameraBackgroundRenderer = new CameraBackgroundRenderer();
    private final Object frameInUseLock = new Object();

    private GLSurfaceView surfaceView;
    private TextView statusView;
    private Button menuButton;
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

    private boolean installRequested;
    private boolean activityResumed;
    private boolean surfaceCreated;
    private boolean cameraOpening;
    private volatile boolean arcoreActive;
    private boolean depthReceived;
    private boolean displayListenerRegistered;
    private boolean manualSensorSupported;

    private volatile boolean displayGeometryChanged = true;
    private volatile int viewportWidth;
    private volatile int viewportHeight;
    private long depthTimestamp = -1L;
    private String lastStatus = "";
    private String cameraConfigSummary = "camera=?";

    // Latest ARCore/Camera2 repeating metadata used to create one fast, focus-locked still.
    private volatile Long latestExposureTimeNs;
    private volatile Integer latestIso;
    private volatile Float latestFocusDistanceDiopters;
    private volatile Integer latestAfState;
    private volatile Integer latestLensState;

    private final CameraDevice.StateCallback cameraDeviceCallback =
            new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice device) {
                    DiagnosticLog.i(TAG, "Shared camera opened id=" + device.getId());
                    cameraOpening = false;
                    cameraDevice = device;
                    createCameraCaptureSession();
                }

                @Override
                public void onDisconnected(CameraDevice device) {
                    DiagnosticLog.w(TAG, "Shared camera disconnected id=" + device.getId());
                    device.close();
                    cameraDevice = null;
                    cameraOpening = false;
                    setStatus("Camera disconnected. メニュー → ログをコピー");
                }

                @Override
                public void onError(CameraDevice device, int error) {
                    String errorName = cameraErrorName(error);
                    DiagnosticLog.e(TAG,
                            "Shared camera fatal error=" + error + " (" + errorName + ")"
                                    + " config=" + cameraConfigSummary);
                    device.close();
                    cameraDevice = null;
                    cameraOpening = false;
                    setStatus(
                            "Camera2/SharedCamera error: " + error + " (" + errorName + ")\n"
                                    + "メニュー → ログをコピー");
                }

                @Override
                public void onClosed(CameraDevice device) {
                    DiagnosticLog.i(TAG, "Shared camera closed id=" + device.getId());
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
                    DiagnosticLog.i(TAG, "Camera capture session configured");
                    captureSession = configuredSession;
                    try {
                        // Match Google's SharedCamera startup sequence: start a Camera2 repeating
                        // request first, then resume ARCore from onActive(). The repeating request
                        // targets only ARCore's surfaces. The JPEG surface is intentionally one-shot
                        // and is not registered as an ARCore repeating app surface.
                        captureSession.setRepeatingRequest(
                                previewCaptureRequestBuilder.build(),
                                cameraCaptureCallback,
                                cameraHandler);
                    } catch (CameraAccessException | IllegalStateException e) {
                        DiagnosticLog.e(TAG, "Failed to start SharedCamera repeating request", e);
                        setStatus("Failed to start shared camera stream.\nメニュー → ログをコピー");
                    }
                }

                @Override
                public void onActive(CameraCaptureSession activeSession) {
                    DiagnosticLog.i(TAG, "Camera capture session active");
                    if (activityResumed && !arcoreActive) {
                        resumeArCore();
                    }
                }

                @Override
                public void onReady(CameraCaptureSession readySession) {
                    DiagnosticLog.i(TAG, "Camera capture session ready");
                }

                @Override
                public void onConfigureFailed(CameraCaptureSession failedSession) {
                    DiagnosticLog.e(TAG,
                            "SharedCamera capture session configuration failed: "
                                    + cameraConfigSummary);
                    setStatus("Camera stream configuration failed.\nメニュー → ログをコピー");
                }

                @Override
                public void onClosed(CameraCaptureSession closedSession) {
                    DiagnosticLog.i(TAG, "Camera capture session closed");
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
                    recordLatestCameraState(result);
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
                    DiagnosticLog.w(TAG,
                            "Camera capture failed reason=" + failure.getReason()
                                    + " frame=" + failure.getFrameNumber());
                    if (HIGH_RES_CAPTURE_TAG.equals(request.getTag())
                            && photoCaptureManager != null) {
                        photoCaptureManager.onCaptureRequestFailed();
                    }
                }
            };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        DiagnosticLog.i(TAG,
                "App create model=" + Build.MANUFACTURER + " " + Build.MODEL
                        + " sdk=" + Build.VERSION.SDK_INT);

        surfaceView = new GLSurfaceView(this);
        surfaceView.setPreserveEGLContextOnPause(true);
        surfaceView.setEGLContextClientVersion(2);
        surfaceView.setEGLConfigChooser(8, 8, 8, 0, 16, 0);
        surfaceView.setRenderer(this);
        surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        statusView = new TextView(this);
        statusView.setTextColor(0xFFFFFFFF);
        statusView.setBackgroundColor(0x77000000);
        statusView.setPadding(dp(12), dp(8), dp(12), dp(8));
        statusView.setText("Starting ARCore SharedCamera...");

        menuButton = new Button(this);
        menuButton.setText("⋮");
        menuButton.setTextSize(24f);
        menuButton.setTextColor(0xFFFFFFFF);
        menuButton.setBackgroundColor(0x77000000);
        menuButton.setMinWidth(0);
        menuButton.setMinHeight(0);
        menuButton.setPadding(0, 0, 0, 0);
        menuButton.setOnClickListener(v -> showMenu());

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
        statusParams.leftMargin = dp(8);
        statusParams.topMargin = dp(8);
        root.addView(statusView, statusParams);

        FrameLayout.LayoutParams menuParams = new FrameLayout.LayoutParams(dp(48), dp(48));
        menuParams.gravity = Gravity.TOP | Gravity.END;
        menuParams.topMargin = dp(6);
        menuParams.rightMargin = dp(6);
        root.addView(menuButton, menuParams);
        setContentView(root);

        displayManager = (DisplayManager) getSystemService(Context.DISPLAY_SERVICE);
        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        startCameraThread();

        try {
            photoCaptureManager = new HighResPhotoCaptureManager(
                    HighResPhotoCaptureManager.getPicturesDirectory(this));
            DiagnosticLog.i(TAG,
                    "Texture capture directory=" + photoCaptureManager.getCaptureDirectoryPath());
        } catch (RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to initialize texture capture directory", e);
        }
    }

    private void showMenu() {
        PopupMenu popup = new PopupMenu(this, menuButton);
        popup.getMenu().add(0, MENU_COPY_LOG, 0, "ログをコピー");
        popup.setOnMenuItemClickListener(item -> {
            if (item.getItemId() == MENU_COPY_LOG) {
                copyLogsToClipboard();
                return true;
            }
            return false;
        });
        popup.show();
    }

    private void copyLogsToClipboard() {
        Toast.makeText(this, "ログを収集中…", Toast.LENGTH_SHORT).show();
        new Thread(() -> {
            String report = buildDiagnosticReport();
            runOnUiThread(() -> {
                ClipboardManager clipboard =
                        (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                clipboard.setPrimaryClip(ClipData.newPlainText("pointCloudSplating log", report));
                Toast.makeText(this, "ログをコピーしました", Toast.LENGTH_SHORT).show();
            });
        }, "CopyDiagnostics").start();
    }

    private String buildDiagnosticReport() {
        StringBuilder report = new StringBuilder();
        report.append("pointCloudSplating diagnostics\n")
                .append("manufacturer=").append(Build.MANUFACTURER).append('\n')
                .append("model=").append(Build.MODEL).append('\n')
                .append("device=").append(Build.DEVICE).append('\n')
                .append("sdk=").append(Build.VERSION.SDK_INT).append('\n')
                .append("cameraId=").append(cameraId).append('\n')
                .append("cameraConfig=").append(cameraConfigSummary).append('\n')
                .append("arcoreActive=").append(arcoreActive).append('\n')
                .append("surfaceCreated=").append(surfaceCreated).append('\n')
                .append("manualSensorSupported=").append(manualSensorSupported).append('\n')
                .append("status=").append(lastStatus).append("\n\n")
                .append("=== in-app log ===\n")
                .append(DiagnosticLog.snapshot())
                .append("\n=== process logcat ===\n")
                .append(readOwnLogcat());
        return report.toString();
    }

    private static String readOwnLogcat() {
        StringBuilder out = new StringBuilder();
        java.lang.Process process = null;
        try {
            process = Runtime.getRuntime().exec(
                    new String[] {
                            "logcat", "-d", "-v", "threadtime", "--pid=" + android.os.Process.myPid()
                    });
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    out.append(line).append('\n');
                    if (out.length() > 200_000) {
                        out.append("[logcat truncated]\n");
                        break;
                    }
                }
            }
        } catch (IOException | RuntimeException e) {
            out.append("logcat unavailable: ").append(e).append('\n');
        } finally {
            if (process != null) {
                process.destroy();
            }
        }
        return out.toString();
    }

    @Override
    protected void onResume() {
        super.onResume();
        DiagnosticLog.i(TAG, "onResume");
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
            CameraConfig cameraConfig = CameraConfigSelector.selectPhotogrammetry30Fps(session);
            session.setCameraConfig(cameraConfig);

            if (!session.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY)) {
                DiagnosticLog.e(TAG, "RAW_DEPTH_ONLY unsupported with selected SharedCamera config");
                setStatus("Raw Depth is not available with this SharedCamera config.");
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
            manualSensorSupported = supportsManualSensor(cameraCharacteristics);

            jpegSize = HighResPhotoCaptureManager.chooseJpegSize(cameraCharacteristics);
            jpegReader = ImageReader.newInstance(
                    jpegSize.getWidth(), jpegSize.getHeight(), ImageFormat.JPEG, 2);
            if (photoCaptureManager != null) {
                jpegReader.setOnImageAvailableListener(
                        photoCaptureManager::onJpegAvailable, cameraHandler);
                photoCaptureManager.configureCamera(cameraId, jpegSize, cameraCharacteristics);
            }

            // Do NOT call SharedCamera.setAppSurfaces() for the high-resolution JPEG. That API is
            // intended for surfaces ARCore must include in its repeating request. Our JPEG is an
            // occasional one-shot Camera2 target. Registering a ~12 MP JPEG as an ARCore repeating
            // app surface increases stream pressure and was part of the Pixel 10a error-4 startup.
            Size cpu = session.getCameraConfig().getImageSize();
            Size gpu = session.getCameraConfig().getTextureSize();
            cameraConfigSummary =
                    "AR CPU " + cpu.getWidth() + "x" + cpu.getHeight()
                            + " / GPU " + gpu.getWidth() + "x" + gpu.getHeight()
                            + " / JPEG " + jpegSize.getWidth() + "x" + jpegSize.getHeight()
                            + " / manual=" + manualSensorSupported;
            DiagnosticLog.i(TAG,
                    "SharedCamera configured " + cameraConfigSummary
                            + " cameraId=" + cameraId
                            + " depthSensorUsage=" + session.getCameraConfig().getDepthSensorUsage());
            return true;
        } catch (UnavailableArcoreNotInstalledException
                 | UnavailableUserDeclinedInstallationException e) {
            DiagnosticLog.e(TAG, "ARCore is required", e);
            setStatus("Google Play Services for AR is required.");
        } catch (UnavailableApkTooOldException e) {
            DiagnosticLog.e(TAG, "ARCore APK too old", e);
            setStatus("Update Google Play Services for AR.");
        } catch (UnavailableSdkTooOldException e) {
            DiagnosticLog.e(TAG, "App ARCore SDK too old", e);
            setStatus("Update this app / ARCore SDK.");
        } catch (UnavailableDeviceNotCompatibleException e) {
            DiagnosticLog.e(TAG, "Device is not ARCore compatible", e);
            setStatus("This device is not ARCore compatible.");
        } catch (CameraAccessException | RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to configure SharedCamera", e);
            setStatus("Failed to configure SharedCamera: " + e.getClass().getSimpleName()
                    + "\nメニュー → ログをコピー");
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
            int textureId = cameraBackgroundRenderer.getTextureId();
            session.setCameraTextureName(textureId);
            CameraDevice.StateCallback wrapped =
                    sharedCamera.createARDeviceStateCallback(cameraDeviceCallback, cameraHandler);
            cameraOpening = true;
            DiagnosticLog.i(TAG, "Opening SharedCamera id=" + cameraId + " texture=" + textureId);
            cameraManager.openCamera(cameraId, wrapped, cameraHandler);
        } catch (CameraAccessException | SecurityException | IllegalArgumentException e) {
            cameraOpening = false;
            DiagnosticLog.e(TAG, "Failed to open ARCore shared camera", e);
            setStatus("Failed to open SharedCamera.\nメニュー → ログをコピー");
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

            DiagnosticLog.i(TAG,
                    "Creating capture session arCoreSurfaces=" + arCoreSurfaces.size()
                            + " totalSurfaces=" + sessionSurfaces.size()
                            + " JPEG=" + jpegSize);

            CameraCaptureSession.StateCallback wrapped =
                    sharedCamera.createARSessionStateCallback(
                            cameraSessionStateCallback, cameraHandler);
            cameraDevice.createCaptureSession(sessionSurfaces, wrapped, cameraHandler);
        } catch (CameraAccessException | IllegalStateException | IllegalArgumentException e) {
            DiagnosticLog.e(TAG, "Failed to create SharedCamera capture session", e);
            setStatus("Failed to create SharedCamera capture session.\nメニュー → ログをコピー");
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
            DiagnosticLog.i(TAG, "ARCore resumed with SharedCamera");
            setStatus("SharedCamera active. Move steadily around the subject.\n" + cameraConfigSummary);
        } catch (CameraNotAvailableException e) {
            DiagnosticLog.e(TAG, "Camera unavailable while resuming ARCore", e);
            setStatus("Camera unavailable.\nメニュー → ログをコピー");
        }
    }

    private void recordLatestCameraState(TotalCaptureResult result) {
        Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
        Integer iso = result.get(CaptureResult.SENSOR_SENSITIVITY);
        Float focus = result.get(CaptureResult.LENS_FOCUS_DISTANCE);
        Integer af = result.get(CaptureResult.CONTROL_AF_STATE);
        Integer lens = result.get(CaptureResult.LENS_STATE);
        if (exposure != null) {
            latestExposureTimeNs = exposure;
        }
        if (iso != null) {
            latestIso = iso;
        }
        if (focus != null) {
            latestFocusDistanceDiopters = focus;
        }
        if (af != null) {
            latestAfState = af;
        }
        if (lens != null) {
            latestLensState = lens;
        }
    }

    private void requestHighResolutionStill() {
        if (cameraHandler == null) {
            failPendingPhoto();
            return;
        }

        cameraHandler.post(() -> {
            if (!arcoreActive
                    || cameraDevice == null
                    || captureSession == null
                    || jpegReader == null
                    || sharedCamera == null
                    || cameraCharacteristics == null) {
                failPendingPhoto();
                return;
            }

            try {
                CaptureRequest.Builder still =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
                for (Surface surface : sharedCamera.getArCoreSurfaces()) {
                    still.addTarget(surface);
                }
                still.addTarget(jpegReader.getSurface());
                still.setTag(HIGH_RES_CAPTURE_TAG);
                still.set(CaptureRequest.JPEG_QUALITY, (byte) 95);
                still.set(CaptureRequest.JPEG_ORIENTATION, computeJpegOrientation());
                still.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF);
                still.set(
                        CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                        CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);

                if (!applyPhotogrammetryExposureAndFocus(still)) {
                    failPendingPhoto();
                    return;
                }

                still.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO);
                Boolean awbLockAvailable =
                        cameraCharacteristics.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE);
                if (Boolean.TRUE.equals(awbLockAvailable)) {
                    still.set(CaptureRequest.CONTROL_AWB_LOCK, true);
                }

                int[] oisModes = cameraCharacteristics.get(
                        CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                if (contains(oisModes, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)) {
                    still.set(
                            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF);
                }

                captureSession.capture(still.build(), cameraCaptureCallback, cameraHandler);
            } catch (CameraAccessException | IllegalStateException | IllegalArgumentException e) {
                DiagnosticLog.e(TAG, "Failed to submit photogrammetry JPEG", e);
                failPendingPhoto();
            }
        });
    }

    private boolean applyPhotogrammetryExposureAndFocus(CaptureRequest.Builder still) {
        if (!manualSensorSupported) {
            return false;
        }

        Long autoExposure = latestExposureTimeNs;
        Integer autoIso = latestIso;
        Float focusDistance = latestFocusDistanceDiopters;
        Integer lensState = latestLensState;
        Integer afState = latestAfState;
        if (autoExposure == null || autoIso == null || focusDistance == null) {
            return false;
        }
        if (lensState != null && lensState != CaptureResult.LENS_STATE_STATIONARY) {
            return false;
        }
        if (!isFocusedState(afState)) {
            return false;
        }

        Range<Long> exposureRange = cameraCharacteristics.get(
                CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE);
        Range<Integer> isoRange = cameraCharacteristics.get(
                CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);
        if (exposureRange == null || isoRange == null) {
            return false;
        }

        long minExposure = exposureRange.getLower();
        long maxExposure = Math.min(exposureRange.getUpper(), MAX_STILL_EXPOSURE_NS);
        int minIso = isoRange.getLower();
        int maxIso = Math.min(isoRange.getUpper(), MAX_PHOTOGRAMMETRY_ISO);
        if (minExposure > maxExposure || minIso > maxIso) {
            return false;
        }

        double exposureProduct = (double) autoExposure * (double) autoIso;
        long exposure = clampLong(TARGET_STILL_EXPOSURE_NS, minExposure, maxExposure);
        int iso = (int) Math.ceil(exposureProduct / exposure);

        if (iso < minIso) {
            iso = minIso;
            exposure = clampLong(Math.round(exposureProduct / iso), minExposure, maxExposure);
        } else if (iso > maxIso) {
            iso = maxIso;
            long requiredExposure = (long) Math.ceil(exposureProduct / iso);
            if (requiredExposure > maxExposure) {
                return false;
            }
            exposure = clampLong(requiredExposure, minExposure, maxExposure);
        }

        iso = clampInt((int) Math.round(exposureProduct / exposure), minIso, maxIso);
        if (exposure > MAX_STILL_EXPOSURE_NS || iso > MAX_PHOTOGRAMMETRY_ISO) {
            return false;
        }

        still.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
        still.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposure);
        still.set(CaptureRequest.SENSOR_SENSITIVITY, iso);

        float lockedFocusDistance = Math.max(0f, focusDistance);
        Float minimumFocusDistance = cameraCharacteristics.get(
                CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);
        if (minimumFocusDistance != null) {
            lockedFocusDistance = Math.min(lockedFocusDistance, minimumFocusDistance);
        }
        still.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF);
        still.set(CaptureRequest.LENS_FOCUS_DISTANCE, lockedFocusDistance);
        DiagnosticLog.i(TAG,
                "Texture still exposure=" + exposure + "ns ISO=" + iso
                        + " focus=" + lockedFocusDistance + "D");
        return true;
    }

    private void failPendingPhoto() {
        if (photoCaptureManager != null) {
            photoCaptureManager.onCaptureRequestFailed();
        }
    }

    @Override
    protected void onPause() {
        DiagnosticLog.i(TAG, "onPause");
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
                DiagnosticLog.w(TAG, "Error closing capture session", e);
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
        DiagnosticLog.i(TAG, "onDestroy");
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
            int requestCode, String[] permissions, int[] grantResults) {
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
            cameraBackgroundRenderer.createOnGlThread(this);
            pointCloudRenderer.createOnGlThread(this);
            surfaceCreated = true;
            DiagnosticLog.i(TAG,
                    "GL resources ready cameraTexture=" + cameraBackgroundRenderer.getTextureId());
            runOnUiThread(this::openCameraForSharing);
        } catch (IOException | RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to initialize OpenGL resources", e);
            setStatus("Failed to initialize OpenGL resources.\nメニュー → ログをコピー");
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
        if (session == null || !arcoreActive) {
            return;
        }

        synchronized (frameInUseLock) {
            try {
                updateDisplayGeometryIfNeeded();
                Frame frame = session.update();

                // The previous build created the OES camera texture but never drew it. Always draw
                // the camera frame first, even when AR tracking is temporarily paused.
                cameraBackgroundRenderer.draw(frame);

                Camera camera = frame.getCamera();
                if (camera.getTrackingState() != TrackingState.TRACKING) {
                    setStatus("AR tracking paused. Move slowly and keep scene detail visible.");
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
                setStatus(
                        "Raw depth: " + pointCloudRenderer.getStoredFrameCount()
                                + " / texture photos: " + photos
                                + "\n1/500 target / <=1/300 / ISO<=1600"
                                + "\n" + cameraConfigSummary);
            } catch (Throwable t) {
                DiagnosticLog.e(TAG, "OpenGL/ARCore frame failed", t);
                setStatus("Frame processing failed: " + t.getClass().getSimpleName()
                        + "\nメニュー → ログをコピー");
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
        Integer sensorOrientation = cameraCharacteristics == null
                ? null
                : cameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        int sensor = sensorOrientation == null ? 90 : sensorOrientation;
        return (sensor - displayDegrees + 360) % 360;
    }

    private static boolean supportsManualSensor(CameraCharacteristics characteristics) {
        int[] capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
        return contains(
                capabilities,
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR);
    }

    private static boolean isFocusedState(Integer afState) {
        if (afState == null) {
            return false;
        }
        return afState == CaptureResult.CONTROL_AF_STATE_INACTIVE
                || afState == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED
                || afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED;
    }

    private static long clampLong(long value, long min, long max) {
        return Math.max(min, Math.min(max, value));
    }

    private static int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
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

    private static String cameraErrorName(int error) {
        switch (error) {
            case CameraDevice.StateCallback.ERROR_CAMERA_IN_USE:
                return "CAMERA_IN_USE";
            case CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE:
                return "MAX_CAMERAS_IN_USE";
            case CameraDevice.StateCallback.ERROR_CAMERA_DISABLED:
                return "CAMERA_DISABLED";
            case CameraDevice.StateCallback.ERROR_CAMERA_DEVICE:
                return "CAMERA_DEVICE";
            case CameraDevice.StateCallback.ERROR_CAMERA_SERVICE:
                return "CAMERA_SERVICE";
            default:
                return "UNKNOWN";
        }
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

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
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
