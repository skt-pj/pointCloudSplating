package com.sktpj.pointcloudsplatting;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.ImageFormat;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
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
import android.os.Looper;
import android.util.Range;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.google.ar.core.Anchor;
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
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/** Pixel 10a Raw Depth scanner with synchronized Camera2 reconstruction-image recording. */
public final class ScannerActivity extends Activity
        implements GLSurfaceView.Renderer, DisplayManager.DisplayListener {

    private static final String TAG = "PointCloudScanner";
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final int MENU_LIBRARY = 1;
    private static final int MENU_COPY_LOG = 2;
    private static final int MENU_COPY_DATASET_PATH = 3;
    private static final int MENU_TOGGLE_POINTS = 4;
    private static final String HIGH_RES_CAPTURE_TAG = "dataset_texture_still";

    private static final long TARGET_STILL_EXPOSURE_NS = 2_000_000L; // 1/500 s
    private static final long INDOOR_MAX_STILL_EXPOSURE_NS = 8_000_000L; // 1/125 s
    private static final int INDOOR_MAX_ISO = 6400;
    private static final long OUTDOOR_MAX_STILL_EXPOSURE_NS = 4_000_000L; // 1/250 s
    private static final int OUTDOOR_MAX_ISO = 3200;

    private enum CaptureEnvironment {
        INDOOR,
        OUTDOOR
    }

    private final PointCloudRenderer pointCloudRenderer = new PointCloudRenderer();
    private final CameraBackgroundRenderer cameraBackgroundRenderer = new CameraBackgroundRenderer();
    private final Object frameInUseLock = new Object();

    private GLSurfaceView surfaceView;
    private TextView statusTitleView;
    private TextView statusView;
    private TextView captureCountView;
    private TextView feedbackView;
    private ProgressBar progressBar;
    private Button menuButton;
    private Button modeButton;
    private Button saveButton;
    private Button gaussianButton;
    private DisplayManager displayManager;
    private Handler uiHandler;
    private Runnable hideFeedbackRunnable;

    private Session session;
    private SharedCamera sharedCamera;
    private CameraManager cameraManager;
    private CameraCharacteristics cameraCharacteristics;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder previewCaptureRequestBuilder;
    private ImageReader jpegReader;
    private DatasetCaptureManager datasetCaptureManager;

    private HandlerThread cameraThread;
    private Handler cameraHandler;

    private String cameraId;
    private Size jpegSize;
    private boolean installRequested;
    private boolean activityResumed;
    private boolean surfaceCreated;
    private boolean cameraOpening;
    private volatile boolean arcoreActive;
    private boolean displayListenerRegistered;
    private boolean manualSensorSupported;
    private volatile boolean displayGeometryChanged = true;
    private volatile int viewportWidth;
    private volatile int viewportHeight;
    private long depthTimestamp = -1L;
    private volatile String lastStatus = "";
    private volatile String lastUiSignature = "";
    private String cameraConfigSummary = "camera=?";
    private volatile boolean captureFinalized;
    private volatile boolean saveInProgress;
    private volatile boolean processingModel;
    private volatile boolean runGaussianAfterSave;
    private volatile boolean showPointCloud;
    private volatile String finalizedDatasetPath;
    private volatile CaptureEnvironment captureEnvironment = CaptureEnvironment.INDOOR;
    private Anchor datasetRootAnchor;

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
                    showState("カメラが切断されました",
                            "アプリを開き直してください。直らない場合はメニューから診断情報をコピーしてください。");
                }

                @Override
                public void onError(CameraDevice device, int error) {
                    DiagnosticLog.e(TAG,
                            "Shared camera error=" + error + " (" + cameraErrorName(error) + ")"
                                    + " config=" + cameraConfigSummary);
                    device.close();
                    cameraDevice = null;
                    cameraOpening = false;
                    showState("カメラを開始できませんでした",
                            "アプリを開き直してください。直らない場合はメニューから診断情報をコピーしてください。");
                }

                @Override
                public void onClosed(CameraDevice device) {
                    DiagnosticLog.i(TAG, "Shared camera closed id=" + device.getId());
                    if (cameraDevice == device) cameraDevice = null;
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
                        captureSession.setRepeatingRequest(
                                previewCaptureRequestBuilder.build(),
                                cameraCaptureCallback,
                                cameraHandler);
                    } catch (CameraAccessException | IllegalStateException e) {
                        DiagnosticLog.e(TAG, "Failed to start SharedCamera repeating request", e);
                        showState("カメラを開始できませんでした",
                                "アプリを開き直してください。直らない場合は診断情報をコピーしてください。");
                    }
                }

                @Override
                public void onActive(CameraCaptureSession activeSession) {
                    DiagnosticLog.i(TAG, "Camera capture session active");
                    if (activityResumed && !arcoreActive) resumeArCore();
                }

                @Override
                public void onReady(CameraCaptureSession readySession) {
                    DiagnosticLog.i(TAG, "Camera capture session ready");
                }

                @Override
                public void onConfigureFailed(CameraCaptureSession failedSession) {
                    DiagnosticLog.e(TAG, "Capture session configuration failed: " + cameraConfigSummary);
                    showState("カメラを準備できませんでした",
                            "アプリを開き直してください。直らない場合は診断情報をコピーしてください。");
                }

                @Override
                public void onClosed(CameraCaptureSession closedSession) {
                    DiagnosticLog.i(TAG, "Camera capture session closed");
                    if (captureSession == closedSession) captureSession = null;
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
                            && datasetCaptureManager != null) {
                        datasetCaptureManager.onCaptureCompleted(result);
                    }
                }

                @Override
                public void onCaptureFailed(
                        CameraCaptureSession captureSession,
                        CaptureRequest request,
                        CaptureFailure failure) {
                    DiagnosticLog.w(TAG,
                            "Capture failed reason=" + failure.getReason()
                                    + " frame=" + failure.getFrameNumber());
                    if (HIGH_RES_CAPTURE_TAG.equals(request.getTag())
                            && datasetCaptureManager != null) {
                        datasetCaptureManager.onCaptureRequestFailed(
                                "Camera2 capture failed reason=" + failure.getReason());
                    }
                }
            };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        DiagnosticLog.i(TAG,
                "App create version=" + BuildConfig.VERSION_NAME
                        + " model=" + Build.MANUFACTURER + " " + Build.MODEL
                        + " sdk=" + Build.VERSION.SDK_INT);
        uiHandler = new Handler(Looper.getMainLooper());

        surfaceView = new GLSurfaceView(this);
        // Do not retain the camera GL context while another Activity is doing Vulkan 3DGS.
        // Releasing it on pause returns camera/point-cloud GPU allocations before training.
        surfaceView.setPreserveEGLContextOnPause(false);
        surfaceView.setEGLContextClientVersion(2);
        surfaceView.setEGLConfigChooser(8, 8, 8, 0, 16, 0);
        surfaceView.setRenderer(this);
        surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(0xFF101010);
        root.addView(surfaceView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));
        buildScannerUi(root);
        setContentView(root);

        displayManager = (DisplayManager) getSystemService(Context.DISPLAY_SERVICE);
        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        startCameraThread();

        try {
            datasetCaptureManager = new DatasetCaptureManager(
                    DatasetCaptureManager.getPicturesDirectory(this));
            DiagnosticLog.i(TAG,
                    "Dataset directory=" + datasetCaptureManager.getCaptureDirectoryPath()
                            + " captureMode=" + captureEnvironment.name());
        } catch (RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to initialize dataset directory", e);
            showState("撮影データを準備できませんでした",
                    "空き容量を確認して、アプリを開き直してください。");
        }
    }

    private void buildScannerUi(FrameLayout root) {
        LinearLayout statusCard = new LinearLayout(this);
        statusCard.setOrientation(LinearLayout.VERTICAL);
        statusCard.setPadding(dp(16), dp(12), dp(16), dp(12));
        statusCard.setBackground(roundedBackground(0xD91A1A1A, 16));

        statusTitleView = new TextView(this);
        statusTitleView.setText("準備中");
        statusTitleView.setTextColor(0xFFFFFFFF);
        statusTitleView.setTextSize(19f);
        statusTitleView.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        statusCard.addView(statusTitleView);

        statusView = new TextView(this);
        statusView.setText("カメラを準備しています…");
        statusView.setTextColor(0xFFF1F3F4);
        statusView.setTextSize(15f);
        statusView.setPadding(0, dp(4), 0, 0);
        statusCard.addView(statusView);

        captureCountView = new TextView(this);
        captureCountView.setText("保存できた写真 0枚");
        captureCountView.setTextColor(0xFFDADCE0);
        captureCountView.setTextSize(14f);
        captureCountView.setPadding(0, dp(8), 0, 0);
        statusCard.addView(captureCountView);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(6));
        progressParams.topMargin = dp(10);
        statusCard.addView(progressBar, progressParams);

        FrameLayout.LayoutParams statusParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT);
        statusParams.gravity = Gravity.TOP;
        statusParams.leftMargin = dp(12);
        statusParams.rightMargin = dp(72);
        statusParams.topMargin = dp(12);
        root.addView(statusCard, statusParams);

        menuButton = new Button(this);
        menuButton.setText("⋮");
        menuButton.setTextSize(24f);
        menuButton.setTextColor(0xFFFFFFFF);
        menuButton.setBackground(roundedBackground(0xD91A1A1A, 16));
        menuButton.setMinWidth(dp(52));
        menuButton.setMinHeight(dp(52));
        menuButton.setPadding(0, 0, 0, 0);
        menuButton.setContentDescription("メニュー");
        menuButton.setOnClickListener(v -> showMenu());
        FrameLayout.LayoutParams menuParams = new FrameLayout.LayoutParams(dp(52), dp(52));
        menuParams.gravity = Gravity.TOP | Gravity.END;
        menuParams.topMargin = dp(12);
        menuParams.rightMargin = dp(12);
        root.addView(menuButton, menuParams);

        feedbackView = new TextView(this);
        feedbackView.setTextColor(0xFFFFFFFF);
        feedbackView.setTextSize(14f);
        feedbackView.setGravity(Gravity.CENTER);
        feedbackView.setPadding(dp(16), dp(12), dp(16), dp(12));
        feedbackView.setBackground(roundedBackground(0xEE202124, 16));
        feedbackView.setVisibility(View.GONE);
        FrameLayout.LayoutParams feedbackParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT);
        feedbackParams.gravity = Gravity.BOTTOM;
        feedbackParams.leftMargin = dp(24);
        feedbackParams.rightMargin = dp(24);
        feedbackParams.bottomMargin = dp(162);
        root.addView(feedbackView, feedbackParams);

        modeButton = new Button(this);
        modeButton.setText(captureEnvironmentLabel());
        styleButton(modeButton, 0xDD303134, 14f);
        modeButton.setContentDescription("撮影モードを切り替える");
        modeButton.setOnClickListener(v -> toggleCaptureEnvironment());
        FrameLayout.LayoutParams modeParams = new FrameLayout.LayoutParams(dp(104), dp(48));
        modeParams.gravity = Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL;
        modeParams.bottomMargin = dp(102);
        root.addView(modeButton, modeParams);

        LinearLayout actionBar = new LinearLayout(this);
        actionBar.setOrientation(LinearLayout.HORIZONTAL);
        actionBar.setGravity(Gravity.CENTER_VERTICAL);
        actionBar.setPadding(dp(8), dp(8), dp(8), dp(8));
        actionBar.setBackground(roundedBackground(0xE6151515, 20));

        saveButton = new Button(this);
        saveButton.setText("撮影を保存");
        styleButton(saveButton, 0xFF3C4043, 15f);
        saveButton.setContentDescription("撮影した写真を保存する");
        saveButton.setOnClickListener(v -> saveCurrentDataset());
        LinearLayout.LayoutParams saveParams = new LinearLayout.LayoutParams(0, dp(56), 0.42f);
        saveParams.rightMargin = dp(6);
        actionBar.addView(saveButton, saveParams);

        gaussianButton = new Button(this);
        gaussianButton.setText("3Dプレビューを作成");
        styleButton(gaussianButton, 0xFF0B57D0, 15f);
        gaussianButton.setContentDescription("撮影した写真から3Dプレビューを作成する");
        gaussianButton.setOnClickListener(v -> startGaussianSplatting());
        LinearLayout.LayoutParams modelParams = new LinearLayout.LayoutParams(0, dp(56), 0.58f);
        modelParams.leftMargin = dp(6);
        actionBar.addView(gaussianButton, modelParams);

        FrameLayout.LayoutParams actionParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(72));
        actionParams.gravity = Gravity.BOTTOM;
        actionParams.leftMargin = dp(12);
        actionParams.rightMargin = dp(12);
        actionParams.bottomMargin = dp(16);
        root.addView(actionBar, actionParams);
    }

    private void styleButton(Button button, int backgroundColor, float textSizeSp) {
        button.setAllCaps(false);
        button.setTextColor(0xFFFFFFFF);
        button.setTextSize(textSizeSp);
        button.setMinHeight(dp(48));
        button.setBackgroundTintList(ColorStateList.valueOf(backgroundColor));
    }

    private GradientDrawable roundedBackground(int color, int radiusDp) {
        GradientDrawable background = new GradientDrawable();
        background.setColor(color);
        background.setCornerRadius(dp(radiusDp));
        return background;
    }

    private void showMenu() {
        PopupMenu popup = new PopupMenu(this, menuButton);
        popup.getMenu().add(0, MENU_LIBRARY, 0, "保存したスキャン");
        popup.getMenu().add(0, MENU_COPY_LOG, 1, "診断情報をコピー");
        popup.getMenu().add(0, MENU_COPY_DATASET_PATH, 2, "保存場所をコピー");
        popup.getMenu().add(0, MENU_TOGGLE_POINTS, 3,
                showPointCloud ? "計測点を隠す" : "計測点を表示（診断用）");
        popup.setOnMenuItemClickListener(item -> {
            if (item.getItemId() == MENU_LIBRARY) {
                startActivity(new Intent(this, LibraryActivity.class));
                return true;
            }
            if (item.getItemId() == MENU_COPY_LOG) {
                copyLogsToClipboard();
                return true;
            }
            if (item.getItemId() == MENU_COPY_DATASET_PATH) {
                copyText("pointCloudSplating dataset", getCurrentDatasetPath());
                showFeedback("保存場所をコピーしました");
                return true;
            }
            if (item.getItemId() == MENU_TOGGLE_POINTS) {
                showPointCloud = !showPointCloud;
                DiagnosticLog.i(TAG, "Diagnostic point overlay=" + showPointCloud);
                showFeedback(showPointCloud
                        ? "診断用の計測点を表示します"
                        : "計測点を隠しました。撮影はそのまま続きます");
                return true;
            }
            return false;
        });
        popup.show();
    }

    private void toggleCaptureEnvironment() {
        if (captureFinalized || saveInProgress || processingModel) {
            showFeedback("撮影を保存した後は撮影モードを変更できません");
            return;
        }
        captureEnvironment = captureEnvironment == CaptureEnvironment.INDOOR
                ? CaptureEnvironment.OUTDOOR
                : CaptureEnvironment.INDOOR;
        modeButton.setText(captureEnvironmentLabel());
        DiagnosticLog.i(TAG,
                "Capture environment changed to " + captureEnvironment.name()
                        + " policy=" + capturePolicySummary());
        showFeedback(captureEnvironment == CaptureEnvironment.INDOOR
                ? "室内向けの撮影に切り替えました"
                : "屋外向けの撮影に切り替えました");
    }

    private String captureEnvironmentLabel() {
        return captureEnvironment == CaptureEnvironment.INDOOR ? "室内" : "屋外";
    }

    private long maxStillExposureNs() {
        return captureEnvironment == CaptureEnvironment.INDOOR
                ? INDOOR_MAX_STILL_EXPOSURE_NS
                : OUTDOOR_MAX_STILL_EXPOSURE_NS;
    }

    private int maxPhotogrammetryIso() {
        return captureEnvironment == CaptureEnvironment.INDOOR
                ? INDOOR_MAX_ISO
                : OUTDOOR_MAX_ISO;
    }

    private String capturePolicySummary() {
        if (captureEnvironment == CaptureEnvironment.INDOOR) {
            return "室内 1/500 target / <=1/125 / ISO<=6400";
        }
        return "屋外 1/500 target / <=1/250 / ISO<=3200";
    }

    private String getCurrentDatasetPath() {
        if (finalizedDatasetPath != null) return finalizedDatasetPath;
        return datasetCaptureManager == null
                ? "dataset unavailable"
                : datasetCaptureManager.getCaptureDirectoryPath();
    }

    private void saveCurrentDataset() {
        DatasetCaptureManager manager = datasetCaptureManager;
        if (manager == null) {
            runGaussianAfterSave = false;
            showFeedback("撮影データを保存できる状態ではありません");
            return;
        }
        if (captureFinalized) {
            if (runGaussianAfterSave) {
                runGaussianAfterSave = false;
                startGaussianSplatting();
            } else {
                showFeedback("この撮影はすでに保存されています");
            }
            return;
        }
        if (saveInProgress) {
            showFeedback("いま撮影を保存しています。少しお待ちください");
            return;
        }

        saveInProgress = true;
        saveButton.setEnabled(false);
        gaussianButton.setEnabled(false);
        modeButton.setEnabled(false);
        showOperation("撮影を保存しています", "最後の写真を保存しています…", -1);

        new Thread(() -> {
            boolean flushed = manager.stopCaptureAndFlush(5_000L);
            if (!flushed) {
                manager.resumeCapture();
                runOnUiThread(() -> {
                    saveInProgress = false;
                    runGaussianAfterSave = false;
                    saveButton.setEnabled(true);
                    gaussianButton.setEnabled(true);
                    modeButton.setEnabled(true);
                    showState("保存できませんでした",
                            "写真の保存が終わるまで少し待って、もう一度「撮影を保存」を押してください。");
                });
                return;
            }

            DatasetFinalizer.Result result = DatasetFinalizer.finalizeDataset(
                    new File(manager.getCaptureDirectoryPath()));
            runOnUiThread(() -> {
                saveInProgress = false;
                if (result.success) {
                    captureFinalized = true;
                    finalizedDatasetPath = result.directory.getAbsolutePath();
                    saveButton.setText("撮影済み");
                    saveButton.setEnabled(false);
                    gaussianButton.setEnabled(true);
                    modeButton.setEnabled(false);
                    captureCountView.setText("保存した写真 " + result.frameCount + "枚");
                    showState("撮影を保存しました",
                            result.frameCount + "枚の写真を保存しました。3Dプレビューを作成できます。");
                    if (runGaussianAfterSave) {
                        runGaussianAfterSave = false;
                        startGaussianSplatting();
                    }
                } else {
                    runGaussianAfterSave = false;
                    manager.resumeCapture();
                    saveButton.setEnabled(true);
                    gaussianButton.setEnabled(true);
                    modeButton.setEnabled(true);
                    showState("保存できませんでした",
                            "まだ保存できた写真がありません。対象の周りをゆっくり撮影してから、もう一度お試しください。");
                }
            });
        }, "FinalizeDataset").start();
    }

    private void startGaussianSplatting() {
        if (saveInProgress) {
            showFeedback("撮影の保存が終わるまで少しお待ちください");
            return;
        }
        if (processingModel) {
            showFeedback("3Dプレビューを準備しています。このままお待ちください");
            return;
        }

        if (!captureFinalized || finalizedDatasetPath == null) {
            DatasetCaptureManager manager = datasetCaptureManager;
            int count = manager == null ? 0 : manager.getSavedCount();
            String decision = manager == null ? "" : manager.getLastDecision();
            if (count == 0 && !decisionIndicatesPendingPhoto(decision)) {
                showFeedback("まだ写真を保存できていません。上の「保存できた写真」が増えるまで、対象の周りをゆっくり撮影してください。");
                updateCaptureUi(count, decision);
                return;
            }
            runGaussianAfterSave = true;
            showOperation("3Dプレビューを準備します", "最後の写真を保存しています…", -1);
            saveCurrentDataset();
            return;
        }

        processingModel = true;
        gaussianButton.setEnabled(false);
        saveButton.setEnabled(false);
        modeButton.setEnabled(false);
        showOperation("3Dプレビューを準備中", "撮影データを確認しています…", 0);
        File datasetDirectory = new File(finalizedDatasetPath);
        new Thread(() -> {
            GaussianSplatJob.Result result = GaussianSplatJob.prepare(
                    datasetDirectory,
                    (percent, message) -> runOnUiThread(() ->
                            showOperation("3Dプレビューを準備中", message, percent)));
            runOnUiThread(() -> {
                processingModel = false;
                gaussianButton.setEnabled(true);
                if (result.success) {
                    showOperation("高品質3Dモデルを作成しました",
                            "完成した3Dモデルを表示します。", 100);
                    openViewer(datasetDirectory);
                } else if (result.hqReady) {
                    showOperation("3Dプレビューを準備しました",
                            "撮影した高画質写真を反映したプレビューを表示します。", 100);
                    openViewer(datasetDirectory);
                } else if (result.priorReady) {
                    showState("3Dプレビューを仕上げられませんでした",
                            "撮影データは保存されています。「3Dプレビューを作成」をもう一度押してください。");
                    showFeedback("撮影データは失われていません。もう一度準備できます");
                } else {
                    showState("3Dプレビューを準備できませんでした",
                            "保存したスキャンを確認して、もう一度お試しください。直らない場合は診断情報をコピーしてください。");
                    DiagnosticLog.w(TAG, "3D preview preparation failed userMessage=" + result.message);
                }
            });
        }, "GenerateHighQualityGaussianPreview").start();
    }

    private void openViewer(File datasetDirectory) {
        Intent intent = new Intent(this, GaussianViewerActivity.class);
        intent.putExtra(GaussianViewerActivity.EXTRA_DATASET_PATH,
                datasetDirectory.getAbsolutePath());
        startActivity(intent);
    }

    private static boolean decisionIndicatesPendingPhoto(String decision) {
        if (decision == null) return false;
        return decision.contains("photo capture in flight")
                || decision.contains("requesting texture photo")
                || decision.contains("waiting briefly for Raw Depth prior");
    }

    private void copyLogsToClipboard() {
        showFeedback("診断情報を準備しています…");
        new Thread(() -> {
            String report = buildDiagnosticReport();
            runOnUiThread(() -> {
                copyText("pointCloudSplating diagnostics", report);
                showFeedback("診断情報をコピーしました");
            });
        }, "CopyDiagnostics").start();
    }

    private void copyText(String label, String text) {
        ClipboardManager clipboard =
                (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (clipboard != null) clipboard.setPrimaryClip(ClipData.newPlainText(label, text));
    }

    private String buildDiagnosticReport() {
        int photos = datasetCaptureManager == null ? 0 : datasetCaptureManager.getSavedCount();
        String dataset = getCurrentDatasetPath();
        String decision = datasetCaptureManager == null
                ? "unavailable" : datasetCaptureManager.getLastDecision();
        return new StringBuilder()
                .append("pointCloudSplating diagnostics\n")
                .append("version=").append(BuildConfig.VERSION_NAME).append('\n')
                .append("manufacturer=").append(Build.MANUFACTURER).append('\n')
                .append("model=").append(Build.MODEL).append('\n')
                .append("device=").append(Build.DEVICE).append('\n')
                .append("sdk=").append(Build.VERSION.SDK_INT).append('\n')
                .append("cameraId=").append(cameraId).append('\n')
                .append("cameraConfig=").append(cameraConfigSummary).append('\n')
                .append("captureMode=").append(captureEnvironment.name()).append('\n')
                .append("capturePolicy=").append(capturePolicySummary()).append('\n')
                .append("arcoreActive=").append(arcoreActive).append('\n')
                .append("surfaceCreated=").append(surfaceCreated).append('\n')
                .append("manualSensorSupported=").append(manualSensorSupported).append('\n')
                .append("depthPreviewFrames=").append(pointCloudRenderer.getStoredFrameCount()).append('\n')
                .append("savedDatasetFrames=").append(photos).append('\n')
                .append("captureFinalized=").append(captureFinalized).append('\n')
                .append("processingModel=").append(processingModel).append('\n')
                .append("photoDecision=").append(decision).append('\n')
                .append("dataset=").append(dataset).append('\n')
                .append("status=").append(lastStatus).append("\n\n")
                .append("=== in-app log ===\n")
                .append(DiagnosticLog.snapshot())
                .append("\n=== process logcat ===\n")
                .append(readOwnLogcat())
                .toString();
    }

    private static String readOwnLogcat() {
        StringBuilder out = new StringBuilder();
        java.lang.Process process = null;
        try {
            process = Runtime.getRuntime().exec(new String[] {
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
            if (process != null) process.destroy();
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
        if (session == null && !createSharedArSession()) return;

        surfaceView.onResume();
        registerDisplayListener();
        displayGeometryChanged = true;
        if (surfaceCreated) openCameraForSharing();
        if (!captureFinalized && !processingModel) showState("準備中", "カメラを準備しています…");
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
                showState("この端末では計測を開始できません",
                        "必要な深度計測を利用できません。診断情報をコピーして確認してください。");
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

            jpegSize = DatasetCaptureManager.chooseJpegSize(cameraCharacteristics);
            jpegReader = ImageReader.newInstance(
                    jpegSize.getWidth(), jpegSize.getHeight(), ImageFormat.JPEG, 2);
            if (datasetCaptureManager != null) {
                jpegReader.setOnImageAvailableListener(
                        datasetCaptureManager::onJpegAvailable, cameraHandler);
                datasetCaptureManager.configureCamera(cameraId, jpegSize, cameraCharacteristics);
            }

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
            showState("AR機能を利用できません",
                    "Google Play 開発者サービス（AR）をインストールしてから、もう一度開いてください。");
        } catch (UnavailableApkTooOldException e) {
            DiagnosticLog.e(TAG, "ARCore APK too old", e);
            showState("AR機能の更新が必要です", "Google Play 開発者サービス（AR）を更新してください。");
        } catch (UnavailableSdkTooOldException e) {
            DiagnosticLog.e(TAG, "ARCore SDK too old", e);
            showState("アプリの更新が必要です", "最新のpointCloudSplatingをインストールしてください。");
        } catch (UnavailableDeviceNotCompatibleException e) {
            DiagnosticLog.e(TAG, "Device is not ARCore compatible", e);
            showState("この端末では利用できません", "この端末は必要なAR機能に対応していません。");
        } catch (CameraAccessException | RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to configure SharedCamera", e);
            showState("カメラを準備できませんでした",
                    "アプリを開き直してください。直らない場合は診断情報をコピーしてください。");
        }

        if (datasetRootAnchor != null) {
            datasetRootAnchor.detach();
            datasetRootAnchor = null;
        }
        if (session != null) {
            session.close();
            session = null;
        }
        return false;
    }

    private void openCameraForSharing() {
        if (!activityResumed || !surfaceCreated || session == null || sharedCamera == null
                || cameraId == null || cameraHandler == null || cameraDevice != null || cameraOpening) return;
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
            DiagnosticLog.e(TAG, "Failed to open SharedCamera", e);
            showState("カメラを開けませんでした", "カメラの許可を確認して、アプリを開き直してください。");
        }
    }

    private void createCameraCaptureSession() {
        if (cameraDevice == null || sharedCamera == null || jpegReader == null) return;
        try {
            List<Surface> arCoreSurfaces = new ArrayList<>(sharedCamera.getArCoreSurfaces());
            List<Surface> sessionSurfaces = new ArrayList<>(arCoreSurfaces);
            sessionSurfaces.add(jpegReader.getSurface());

            previewCaptureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            for (Surface surface : arCoreSurfaces) previewCaptureRequestBuilder.addTarget(surface);
            previewCaptureRequestBuilder.set(
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);

            DiagnosticLog.i(TAG,
                    "Creating capture session arCoreSurfaces=" + arCoreSurfaces.size()
                            + " totalSurfaces=" + sessionSurfaces.size()
                            + " JPEG=" + jpegSize);
            CameraCaptureSession.StateCallback wrapped =
                    sharedCamera.createARSessionStateCallback(cameraSessionStateCallback, cameraHandler);
            cameraDevice.createCaptureSession(sessionSurfaces, wrapped, cameraHandler);
        } catch (CameraAccessException | IllegalStateException | IllegalArgumentException e) {
            DiagnosticLog.e(TAG, "Failed to create SharedCamera capture session", e);
            showState("カメラを準備できませんでした",
                    "アプリを開き直してください。直らない場合は診断情報をコピーしてください。");
        }
    }

    private void resumeArCore() {
        if (session == null || sharedCamera == null || arcoreActive || !activityResumed) return;
        try {
            session.resume();
            if (datasetCaptureManager != null) datasetCaptureManager.onCameraResumed();
            arcoreActive = true;
            sharedCamera.setCaptureCallback(cameraCaptureCallback, cameraHandler);
            DiagnosticLog.i(TAG, "ARCore resumed with SharedCamera");
        } catch (CameraNotAvailableException e) {
            DiagnosticLog.e(TAG, "Camera unavailable while resuming ARCore", e);
            showState("カメラを利用できません",
                    "ほかのアプリがカメラを使用していないか確認して、もう一度お試しください。");
        }
    }

    private void recordLatestCameraState(TotalCaptureResult result) {
        Long exposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME);
        Integer iso = result.get(CaptureResult.SENSOR_SENSITIVITY);
        Float focus = result.get(CaptureResult.LENS_FOCUS_DISTANCE);
        Integer af = result.get(CaptureResult.CONTROL_AF_STATE);
        Integer lens = result.get(CaptureResult.LENS_STATE);
        if (exposure != null) latestExposureTimeNs = exposure;
        if (iso != null) latestIso = iso;
        if (focus != null) latestFocusDistanceDiopters = focus;
        if (af != null) latestAfState = af;
        if (lens != null) latestLensState = lens;
    }

    private void requestHighResolutionStill() {
        if (cameraHandler == null) {
            failPendingPhoto("camera handler unavailable");
            return;
        }
        cameraHandler.post(() -> {
            if (!arcoreActive || cameraDevice == null || captureSession == null || jpegReader == null
                    || sharedCamera == null || cameraCharacteristics == null) {
                failPendingPhoto("camera not ready for still");
                return;
            }
            try {
                CaptureRequest.Builder still = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE);
                for (Surface surface : sharedCamera.getArCoreSurfaces()) still.addTarget(surface);
                still.addTarget(jpegReader.getSurface());
                still.setTag(HIGH_RES_CAPTURE_TAG);
                still.set(CaptureRequest.JPEG_QUALITY, (byte) 95);
                still.set(CaptureRequest.JPEG_ORIENTATION, 0);
                still.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF);
                still.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                        CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);

                String rejection = applyPhotogrammetryExposureAndFocus(still);
                if (rejection != null) {
                    DiagnosticLog.i(TAG, "Texture still skipped: " + rejection);
                    failPendingPhoto(rejection);
                    return;
                }

                still.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO);
                Boolean awbLockAvailable = cameraCharacteristics.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE);
                if (Boolean.TRUE.equals(awbLockAvailable)) still.set(CaptureRequest.CONTROL_AWB_LOCK, true);

                int[] oisModes = cameraCharacteristics.get(
                        CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                if (contains(oisModes, CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)) {
                    still.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF);
                }

                captureSession.capture(still.build(), cameraCaptureCallback, cameraHandler);
            } catch (CameraAccessException | IllegalStateException | IllegalArgumentException e) {
                DiagnosticLog.e(TAG, "Failed to submit texture still", e);
                failPendingPhoto("Camera2 still submit failed: " + e.getClass().getSimpleName());
            }
        });
    }

    private String applyPhotogrammetryExposureAndFocus(CaptureRequest.Builder still) {
        if (!manualSensorSupported) return "manual sensor unsupported";
        Long autoExposure = latestExposureTimeNs;
        Integer autoIso = latestIso;
        Float focusDistance = latestFocusDistanceDiopters;
        Integer lensState = latestLensState;
        Integer afState = latestAfState;
        if (autoExposure == null || autoIso == null || focusDistance == null) return "waiting for Camera2 exposure/focus metadata";
        if (lensState != null && lensState != CaptureResult.LENS_STATE_STATIONARY) return "lens moving";
        if (!isFocusedState(afState)) return "autofocus not converged";

        Range<Long> exposureRange = cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE);
        Range<Integer> isoRange = cameraCharacteristics.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);
        if (exposureRange == null || isoRange == null) return "manual exposure range unavailable";

        long policyMaxExposure = maxStillExposureNs();
        int policyMaxIso = maxPhotogrammetryIso();
        long minExposure = exposureRange.getLower();
        long maxExposure = Math.min(exposureRange.getUpper(), policyMaxExposure);
        int minIso = isoRange.getLower();
        int maxIso = Math.min(isoRange.getUpper(), policyMaxIso);
        if (minExposure > maxExposure || minIso > maxIso) return "invalid sensor exposure range";

        double exposureProduct = (double) autoExposure * (double) autoIso;
        long exposure = clampLong(TARGET_STILL_EXPOSURE_NS, minExposure, maxExposure);
        int iso = (int) Math.ceil(exposureProduct / exposure);
        if (iso < minIso) {
            iso = minIso;
            exposure = clampLong(Math.round(exposureProduct / iso), minExposure, maxExposure);
        } else if (iso > maxIso) {
            iso = maxIso;
            long requiredExposure = (long) Math.ceil(exposureProduct / iso);
            if (requiredExposure > maxExposure) return "scene too dark for " + capturePolicySummary();
            exposure = clampLong(requiredExposure, minExposure, maxExposure);
        }
        iso = clampInt((int) Math.round(exposureProduct / exposure), minIso, maxIso);
        if (exposure > policyMaxExposure || iso > policyMaxIso) return "exposure quality limit exceeded for " + captureEnvironmentLabel();

        still.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF);
        still.set(CaptureRequest.SENSOR_EXPOSURE_TIME, exposure);
        still.set(CaptureRequest.SENSOR_SENSITIVITY, iso);

        float lockedFocusDistance = Math.max(0f, focusDistance);
        Float minimumFocusDistance = cameraCharacteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);
        if (minimumFocusDistance != null) lockedFocusDistance = Math.min(lockedFocusDistance, minimumFocusDistance);
        still.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF);
        still.set(CaptureRequest.LENS_FOCUS_DISTANCE, lockedFocusDistance);
        DiagnosticLog.i(TAG,
                "Texture still mode=" + captureEnvironment.name()
                        + " exp=" + exposure + "ns ISO=" + iso
                        + " focus=" + lockedFocusDistance + "D");
        return null;
    }

    private void failPendingPhoto(String reason) {
        if (datasetCaptureManager != null) datasetCaptureManager.onCaptureRequestFailed(reason);
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
        if (datasetCaptureManager != null) datasetCaptureManager.onCameraPaused();
        super.onPause();
    }

    private void closeCamera() {
        if (captureSession != null) {
            try { captureSession.close(); }
            catch (RuntimeException e) { DiagnosticLog.w(TAG, "Error closing capture session: " + e.getMessage()); }
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
        if (jpegReader != null) { jpegReader.close(); jpegReader = null; }
        if (datasetCaptureManager != null) { datasetCaptureManager.shutdown(); datasetCaptureManager = null; }
        if (session != null) { session.close(); session = null; }
        if (uiHandler != null && hideFeedbackRunnable != null) uiHandler.removeCallbacks(hideFeedbackRunnable);
        stopCameraThread();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode != CAMERA_PERMISSION_REQUEST) return;
        if (hasCameraPermission()) recreate();
        else {
            Toast.makeText(this, "3Dスキャンにはカメラの許可が必要です。", Toast.LENGTH_LONG).show();
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
            DiagnosticLog.i(TAG, "GL resources ready cameraTexture=" + cameraBackgroundRenderer.getTextureId());
            runOnUiThread(this::openCameraForSharing);
        } catch (IOException | RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to initialize OpenGL resources", e);
            showState("画面を準備できませんでした",
                    "アプリを開き直してください。直らない場合は診断情報をコピーしてください。");
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
        if (session == null || !arcoreActive) return;

        synchronized (frameInUseLock) {
            try {
                updateDisplayGeometryIfNeeded();
                Frame frame = session.update();
                cameraBackgroundRenderer.draw(frame);

                Camera camera = frame.getCamera();
                if (camera.getTrackingState() != TrackingState.TRACKING) {
                    if (!saveInProgress && !processingModel && !captureFinalized) {
                        showState("位置を確認しています",
                                "端末をゆっくり動かし、模様のある場所へカメラを向けてください。");
                    }
                    return;
                }

                if (datasetRootAnchor == null && !captureFinalized && !saveInProgress) {
                    datasetRootAnchor = session.createAnchor(camera.getPose());
                    DiagnosticLog.i(TAG, "Dataset root anchor created");
                }
                com.google.ar.core.Pose datasetRootPose = datasetRootAnchor == null
                        ? camera.getPose() : datasetRootAnchor.getPose();

                if (datasetCaptureManager != null
                        && datasetCaptureManager.onArFrame(frame, camera, datasetRootPose)) {
                    requestHighResolutionStill();
                }

                boolean containsNewDepthData = false;
                try (Image depthImage = frame.acquireRawDepthImage16Bits()) {
                    long currentTimestamp = depthImage.getTimestamp();
                    containsNewDepthData = currentTimestamp != depthTimestamp;
                    depthTimestamp = currentTimestamp;
                } catch (NotYetAvailableException ignored) {}

                if (containsNewDepthData) {
                    DepthData depth = DepthData.create(session, frame);
                    if (depth != null) {
                        pointCloudRenderer.update(depth);
                        if (datasetCaptureManager != null) datasetCaptureManager.onDepthFrame(depth, datasetRootPose);
                    }
                }

                if (showPointCloud) {
                    float[] projectionMatrix = new float[16];
                    camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100.0f);
                    float[] viewMatrix = new float[16];
                    camera.getViewMatrix(viewMatrix, 0);
                    pointCloudRenderer.draw(viewMatrix, projectionMatrix);
                }

                int photos = datasetCaptureManager == null ? 0 : datasetCaptureManager.getSavedCount();
                String decision = datasetCaptureManager == null ? "dataset unavailable" : datasetCaptureManager.getLastDecision();
                updateCaptureUi(photos, decision);
            } catch (Throwable t) {
                DiagnosticLog.e(TAG, "OpenGL/ARCore frame failed", t);
                showState("撮影を続けられませんでした",
                        "アプリを開き直してください。直らない場合は診断情報をコピーしてください。");
            }
        }
    }

    private void updateCaptureUi(int photos, String decision) {
        if (captureFinalized || saveInProgress || processingModel) return;
        String title = photos > 0 ? "撮影できています" : "撮影中";
        String instruction = userInstructionForDecision(decision, photos);
        String signature = "capture|" + photos + "|" + instruction + "|" + captureEnvironment.name();
        if (signature.equals(lastUiSignature)) return;
        lastUiSignature = signature;
        lastStatus = title + " / " + instruction;
        runOnUiThread(() -> {
            statusTitleView.setText(title);
            statusView.setText(instruction);
            captureCountView.setText("保存できた写真 " + photos + "枚");
            progressBar.setVisibility(View.GONE);
        });
    }

    private String userInstructionForDecision(String decision, int photos) {
        if (decision == null) return photos == 0
                ? "対象の周りをゆっくり動かしてください。"
                : "少しずつ角度を変えて、対象の周りを撮影してください。";
        String lower = decision.toLowerCase(java.util.Locale.US);
        if (lower.contains("too dark") || lower.contains("exposure quality")) return "少し明るい場所へ移動するか、照明をつけてください。";
        if (lower.contains("lower blur")) return "端末をもう少しゆっくり動かしてください。";
        if (lower.contains("focus") || lower.contains("lens")) return "ピントを合わせています。端末を少し止めてください。";
        if (lower.contains("capture in flight")
                || lower.contains("requesting texture photo")
                || lower.contains("waiting briefly for raw depth prior")) {
            return "写真を保存しています。端末を少し止めてください。";
        }
        if (lower.contains("discarded photo")) return "この角度の写真を保存できませんでした。少しゆっくり動かして、もう一度撮影してください。";
        if (lower.contains("next viewpoint") || lower.contains("new viewpoint")) return photos == 0
                ? "対象にカメラを向け、ゆっくり動かしてください。"
                : "少し角度を変えて、対象の周りをゆっくり撮影してください。";
        if (lower.contains("saved frame")) return "写真を保存できました。別の角度からも続けてください。";
        if (lower.contains("camera resumed") || lower.contains("capture resumed")) return "対象の周りをゆっくり動かして撮影を続けてください。";
        if (lower.contains("camera not ready") || lower.contains("metadata")) return "カメラを準備しています。端末を少し止めてください。";
        return photos == 0 ? "対象の周りをゆっくり動かしてください。" : "別の角度からもゆっくり撮影してください。";
    }

    private void showState(String title, String message) {
        String signature = "state|" + title + "|" + message;
        if (signature.equals(lastUiSignature)) return;
        lastUiSignature = signature;
        lastStatus = title + " / " + message;
        int count = datasetCaptureManager == null ? 0 : datasetCaptureManager.getSavedCount();
        runOnUiThread(() -> {
            statusTitleView.setText(title);
            statusView.setText(message);
            captureCountView.setText(captureFinalized
                    ? "保存した写真 " + count + "枚"
                    : "保存できた写真 " + count + "枚");
            progressBar.setVisibility(View.GONE);
        });
    }

    private void showOperation(String title, String message, int percent) {
        String signature = "op|" + title + "|" + message + "|" + percent;
        if (signature.equals(lastUiSignature)) return;
        lastUiSignature = signature;
        lastStatus = title + " / " + message + " / " + percent;
        int count = datasetCaptureManager == null ? 0 : datasetCaptureManager.getSavedCount();
        runOnUiThread(() -> {
            statusTitleView.setText(title);
            statusView.setText(message);
            captureCountView.setText((captureFinalized ? "保存した写真 " : "保存できた写真 ")
                    + count + "枚");
            progressBar.setVisibility(View.VISIBLE);
            if (percent < 0) progressBar.setIndeterminate(true);
            else {
                progressBar.setIndeterminate(false);
                progressBar.setProgress(Math.max(0, Math.min(100, percent)));
            }
        });
    }

    private void showFeedback(String message) {
        if (uiHandler == null || feedbackView == null) return;
        runOnUiThread(() -> {
            if (hideFeedbackRunnable != null) uiHandler.removeCallbacks(hideFeedbackRunnable);
            feedbackView.setText(message);
            feedbackView.setVisibility(View.VISIBLE);
            hideFeedbackRunnable = () -> feedbackView.setVisibility(View.GONE);
            uiHandler.postDelayed(hideFeedbackRunnable, 2800L);
        });
    }

    private void updateDisplayGeometryIfNeeded() {
        if (!displayGeometryChanged || viewportWidth == 0 || viewportHeight == 0) return;
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
            case Surface.ROTATION_90: displayDegrees = 90; break;
            case Surface.ROTATION_180: displayDegrees = 180; break;
            case Surface.ROTATION_270: displayDegrees = 270; break;
            case Surface.ROTATION_0:
            default: displayDegrees = 0; break;
        }
        Integer sensorOrientation = cameraCharacteristics == null
                ? null : cameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        int sensor = sensorOrientation == null ? 90 : sensorOrientation;
        return (sensor - displayDegrees + 360) % 360;
    }

    private static boolean supportsManualSensor(CameraCharacteristics characteristics) {
        int[] capabilities = characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
        return contains(capabilities, CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR);
    }

    private static boolean isFocusedState(Integer afState) {
        if (afState == null) return true;
        return afState == CaptureResult.CONTROL_AF_STATE_INACTIVE
                || afState == CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED
                || afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED;
    }

    private static long clampLong(long value, long min, long max) { return Math.max(min, Math.min(max, value)); }
    private static int clampInt(int value, int min, int max) { return Math.max(min, Math.min(max, value)); }

    private static boolean contains(int[] values, int target) {
        if (values == null) return false;
        for (int value : values) if (value == target) return true;
        return false;
    }

    private static String cameraErrorName(int error) {
        switch (error) {
            case CameraDevice.StateCallback.ERROR_CAMERA_IN_USE: return "CAMERA_IN_USE";
            case CameraDevice.StateCallback.ERROR_MAX_CAMERAS_IN_USE: return "MAX_CAMERAS_IN_USE";
            case CameraDevice.StateCallback.ERROR_CAMERA_DISABLED: return "CAMERA_DISABLED";
            case CameraDevice.StateCallback.ERROR_CAMERA_DEVICE: return "CAMERA_DEVICE";
            case CameraDevice.StateCallback.ERROR_CAMERA_SERVICE: return "CAMERA_SERVICE";
            default: return "UNKNOWN";
        }
    }

    private boolean hasCameraPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void startCameraThread() {
        if (cameraThread != null) return;
        cameraThread = new HandlerThread("PointCloudCamera2");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
    }

    private void stopCameraThread() {
        if (cameraThread == null) return;
        cameraThread.quitSafely();
        try { cameraThread.join(); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); }
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

    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }

    @Override public void onDisplayAdded(int displayId) {}
    @Override public void onDisplayRemoved(int displayId) {}
    @Override public void onDisplayChanged(int displayId) { displayGeometryChanged = true; }
}
