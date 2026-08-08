package com.sktpj.pointcloudsplatting;

import android.app.Activity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

/** Lightweight foreground screen used while the scanner Activity is fully paused for 3DGS. */
public final class ModelProcessingActivity extends Activity {
    private TextView titleView;
    private TextView messageView;
    private ProgressBar progressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(0xFF101010);

        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setGravity(Gravity.CENTER_HORIZONTAL);
        card.setPadding(dp(28), dp(28), dp(28), dp(28));

        titleView = new TextView(this);
        titleView.setText("3Dモデルを変換中");
        titleView.setTextColor(0xFFFFFFFF);
        titleView.setTextSize(22f);
        titleView.setGravity(Gravity.CENTER);
        card.addView(titleView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        messageView = new TextView(this);
        messageView.setText("カメラを停止しています…");
        messageView.setTextColor(0xFFDADCE0);
        messageView.setTextSize(15f);
        messageView.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams messageParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        messageParams.topMargin = dp(16);
        card.addView(messageView, messageParams);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setProgress(0);
        LinearLayout.LayoutParams progressParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(8));
        progressParams.topMargin = dp(22);
        card.addView(progressBar, progressParams);

        TextView note = new TextView(this);
        note.setText("変換中はカメラ・AR表示を停止して、端末のメモリとGPUを3D処理へ優先します。");
        note.setTextColor(0xFF9AA0A6);
        note.setTextSize(13f);
        note.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams noteParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        noteParams.topMargin = dp(18);
        card.addView(note, noteParams);

        FrameLayout.LayoutParams cardParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        cardParams.gravity = Gravity.CENTER;
        cardParams.leftMargin = dp(20);
        cardParams.rightMargin = dp(20);
        root.addView(card, cardParams);
        setContentView(root);
    }

    @Override
    protected void onResume() {
        super.onResume();
        ModelProcessingCoordinator.attach(this);
    }

    @Override
    protected void onDestroy() {
        ModelProcessingCoordinator.detach(this);
        super.onDestroy();
    }

    void updateProgress(int progress, String message, boolean active) {
        if (isFinishing()) return;
        runOnUiThread(() -> {
            if (isFinishing()) return;
            progressBar.setProgress(Math.max(0, Math.min(100, progress)));
            if (message != null && !message.isEmpty()) messageView.setText(message);
            if (!active && progress >= 100) titleView.setText("3Dモデルを作成しました");
        });
    }

    @Override
    public void onBackPressed() {
        if (ModelProcessingCoordinator.isActive()) {
            Toast.makeText(this, "変換中です。完了するまでお待ちください。", Toast.LENGTH_SHORT).show();
            return;
        }
        super.onBackPressed();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
