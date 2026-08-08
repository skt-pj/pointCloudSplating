package com.sktpj.pointcloudsplatting;

import android.app.Activity;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;

/**
 * OpenGL ES 3 viewer for splat.ply.
 *
 * <p>The viewer renders the actual anisotropic Gaussian parameters: 3D scale, quaternion rotation,
 * opacity and SH1 appearance. Gaussians are sorted back-to-front whenever the view direction
 * changes and alpha composited without depth writes. The vertex shader projects the full 3D
 * covariance into screen space and draws a 3-sigma ellipse rather than an isotropic point sprite.
 */
public final class GaussianViewerActivity extends Activity {
    public static final String EXTRA_DATASET_PATH = "dataset_path";

    private GLSurfaceView glView;
    private GaussianEs3Renderer renderer;
    private TextView statusView;
    private TextView sizeLabel;
    private ScaleGestureDetector scaleDetector;
    private float lastX;
    private float lastY;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        String datasetPath = getIntent().getStringExtra(EXTRA_DATASET_PATH);
        if (datasetPath == null) {
            Toast.makeText(this, "3Dモデルを開けませんでした。", Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        File splat = new File(new File(datasetPath), "splat.ply");
        if (!splat.isFile()) {
            Toast.makeText(this, "3Dモデルを開けませんでした。", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        renderer = new GaussianEs3Renderer(message -> runOnUiThread(() -> {
            if (statusView != null) {
                statusView.setText(message);
            }
        }));

        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(3);
        glView.setEGLConfigChooser(8, 8, 8, 8, 24, 0);
        glView.setPreserveEGLContextOnPause(true);
        glView.setRenderer(renderer);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        scaleDetector = new ScaleGestureDetector(this,
                new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                    @Override
                    public boolean onScale(ScaleGestureDetector detector) {
                        renderer.zoom(detector.getScaleFactor());
                        return true;
                    }
                });
        glView.setOnTouchListener(this::handleTouch);
        glView.setContentDescription("3Dモデル表示。指一本で回転、二本指で拡大縮小できます");

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(0xFF101010);
        root.addView(glView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        top.setBackgroundColor(0x77000000);
        top.addView(makeButton("戻る", "保存したスキャンに戻る", v -> finish(), 86));

        statusView = new TextView(this);
        statusView.setTextColor(0xFFFFFFFF);
        statusView.setTextSize(13f);
        statusView.setGravity(Gravity.CENTER);
        statusView.setText("3Dモデルを読み込んでいます…\n少しお待ちください");
        top.addView(statusView, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.MATCH_PARENT, 1f));

        top.addView(makeButton(
                "正面に戻す",
                "3Dモデルの向きと大きさを元に戻す",
                v -> renderer.resetCamera(),
                128));

        FrameLayout.LayoutParams topParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(64));
        topParams.gravity = Gravity.TOP;
        topParams.leftMargin = dp(6);
        topParams.rightMargin = dp(6);
        topParams.topMargin = dp(6);
        root.addView(top, topParams);

        LinearLayout sizePanel = new LinearLayout(this);
        sizePanel.setOrientation(LinearLayout.VERTICAL);
        sizePanel.setGravity(Gravity.CENTER_VERTICAL);
        sizePanel.setPadding(dp(16), dp(4), dp(16), dp(4));
        sizePanel.setBackgroundColor(0x99000000);

        sizeLabel = new TextView(this);
        sizeLabel.setTextColor(0xFFFFFFFF);
        sizeLabel.setTextSize(14f);
        sizeLabel.setText("表示サイズ: 最小");
        sizePanel.addView(sizeLabel, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(24)));

        SeekBar sizeSlider = new SeekBar(this);
        sizeSlider.setMax(100);
        sizeSlider.setProgress(0);
        sizeSlider.setMinHeight(dp(48));
        sizeSlider.setContentDescription("3Dモデルの表示サイズ");
        sizeSlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                renderer.setDisplaySizeProgress(progress);
                if (progress == 0) {
                    sizeLabel.setText("表示サイズ: 最小");
                } else if (progress < 55) {
                    sizeLabel.setText("表示サイズ: 小さめ " + progress + "%");
                } else if (progress < 80) {
                    sizeLabel.setText("表示サイズ: 標準 " + progress + "%");
                } else if (progress == 100) {
                    sizeLabel.setText("表示サイズ: 最大");
                } else {
                    sizeLabel.setText("表示サイズ: 大きめ " + progress + "%");
                }
            }

            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        sizePanel.addView(sizeSlider, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));

        FrameLayout.LayoutParams sizeParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(80));
        sizeParams.gravity = Gravity.BOTTOM;
        sizeParams.leftMargin = dp(8);
        sizeParams.rightMargin = dp(8);
        sizeParams.bottomMargin = dp(8);
        root.addView(sizePanel, sizeParams);

        setContentView(root);
        loadModelAsync(splat);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (glView != null) {
            glView.onResume();
        }
    }

    @Override
    protected void onPause() {
        if (glView != null) {
            glView.onPause();
        }
        super.onPause();
    }

    private Button makeButton(
            String text,
            String description,
            View.OnClickListener listener,
            int widthDp) {
        Button button = new Button(this);
        button.setText(text);
        button.setAllCaps(false);
        button.setContentDescription(description);
        button.setMinHeight(dp(48));
        button.setOnClickListener(listener);
        button.setLayoutParams(new LinearLayout.LayoutParams(
                dp(widthDp), LinearLayout.LayoutParams.MATCH_PARENT));
        return button;
    }

    private void loadModelAsync(File splat) {
        new Thread(() -> {
            try {
                GaussianModel model = GaussianPlyModelReader.read(splat);
                renderer.setModel(model);
                runOnUiThread(() -> statusView.setText(
                        "3Dモデルを画面に準備しています…\n少しお待ちください"));
            } catch (Exception e) {
                DiagnosticLog.e("GaussianViewer", "Failed to load splat.ply", e);
                runOnUiThread(() -> {
                    statusView.setText("3Dモデルを表示できませんでした");
                    Toast.makeText(
                            this,
                            "3Dモデルを表示できませんでした。",
                            Toast.LENGTH_LONG).show();
                });
            }
        }, "LoadGaussianPly").start();
    }

    private boolean handleTouch(View view, MotionEvent event) {
        scaleDetector.onTouchEvent(event);
        if (event.getPointerCount() == 1 && !scaleDetector.isInProgress()) {
            if (event.getActionMasked() == MotionEvent.ACTION_DOWN) {
                lastX = event.getX();
                lastY = event.getY();
            } else if (event.getActionMasked() == MotionEvent.ACTION_MOVE) {
                float x = event.getX();
                float y = event.getY();
                renderer.rotate((x - lastX) * 0.35f, (y - lastY) * 0.35f);
                lastX = x;
                lastY = y;
            }
        }
        return true;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
