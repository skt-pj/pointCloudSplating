package com.sktpj.pointcloudsplatting;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Environment;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/** Thumbnail library for saved reconstruction datasets and their real training state. */
public final class LibraryActivity extends Activity {
    private static final int CARD_THUMBNAIL_HEIGHT_DP = 150;
    private static final String FINAL_SPLAT = "splat.ply";
    private static final String DEPTH_PRIOR = "depth_prior.ply";

    private GridLayout grid;
    private TextView emptyView;
    private volatile boolean generationInProgress;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildContentView());
    }

    @Override
    protected void onResume() {
        super.onResume();
        reloadLibrary();
    }

    private View buildContentView() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFF101010);

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(dp(12), dp(8), dp(12), dp(8));
        header.setBackgroundColor(0xFF202020);

        Button back = new Button(this);
        back.setText("戻る");
        back.setAllCaps(false);
        back.setContentDescription("撮影画面に戻る");
        back.setMinHeight(dp(48));
        back.setOnClickListener(v -> finish());
        header.addView(back, new LinearLayout.LayoutParams(dp(84), dp(52)));

        TextView title = new TextView(this);
        title.setText("保存したスキャン");
        title.setTextColor(0xFFFFFFFF);
        title.setTextSize(20f);
        title.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(0, dp(52), 1f);
        header.addView(title, titleParams);

        Button reload = new Button(this);
        reload.setText("更新");
        reload.setAllCaps(false);
        reload.setContentDescription("保存したスキャンの一覧を更新する");
        reload.setMinHeight(dp(48));
        reload.setOnClickListener(v -> reloadLibrary());
        header.addView(reload, new LinearLayout.LayoutParams(dp(84), dp(52)));
        root.addView(header);

        FrameLayout content = new FrameLayout(this);
        LinearLayout.LayoutParams contentParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f);
        root.addView(content, contentParams);

        ScrollView scroll = new ScrollView(this);
        grid = new GridLayout(this);
        grid.setColumnCount(2);
        grid.setUseDefaultMargins(false);
        grid.setPadding(dp(6), dp(6), dp(6), dp(24));
        scroll.addView(grid, new ScrollView.LayoutParams(
                ScrollView.LayoutParams.MATCH_PARENT,
                ScrollView.LayoutParams.WRAP_CONTENT));
        content.addView(scroll, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        emptyView = new TextView(this);
        emptyView.setText("保存したスキャンはまだありません\n\n撮影画面で対象の周りを撮影し、\n「撮影を保存」を押してください。");
        emptyView.setTextColor(0xFFCCCCCC);
        emptyView.setTextSize(17f);
        emptyView.setGravity(Gravity.CENTER);
        emptyView.setPadding(dp(24), dp(24), dp(24), dp(24));
        content.addView(emptyView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        return root;
    }

    private void reloadLibrary() {
        File pictures = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        List<File> datasets = findSavedDatasets(pictures);
        grid.removeAllViews();
        emptyView.setVisibility(datasets.isEmpty() ? View.VISIBLE : View.GONE);
        for (File dataset : datasets) {
            migrateLegacyDepthPrior(dataset);
            grid.addView(createDatasetCard(dataset));
        }
    }

    private View createDatasetCard(File dataset) {
        LinearLayout card = new LinearLayout(this);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setPadding(dp(8), dp(8), dp(8), dp(10));
        card.setMinimumHeight(dp(220));
        GradientDrawable background = new GradientDrawable();
        background.setColor(0xFF282828);
        background.setCornerRadius(dp(12));
        card.setBackground(background);

        GridLayout.LayoutParams params = new GridLayout.LayoutParams();
        params.width = 0;
        params.height = GridLayout.LayoutParams.WRAP_CONTENT;
        params.columnSpec = GridLayout.spec(GridLayout.UNDEFINED, 1f);
        params.setMargins(dp(5), dp(5), dp(5), dp(5));
        card.setLayoutParams(params);

        ImageView thumbnail = new ImageView(this);
        thumbnail.setScaleType(ImageView.ScaleType.CENTER_CROP);
        thumbnail.setBackgroundColor(0xFF151515);
        thumbnail.setContentDescription("保存したスキャンの写真");
        Bitmap bitmap = decodeThumbnail(findFirstJpeg(dataset), 480, 360);
        if (bitmap != null) thumbnail.setImageBitmap(bitmap);
        card.addView(thumbnail, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(CARD_THUMBNAIL_HEIGHT_DP)));

        TextView name = new TextView(this);
        name.setText(formatDatasetName(dataset.getName()));
        name.setTextColor(0xFFFFFFFF);
        name.setTextSize(15f);
        name.setMaxLines(1);
        name.setPadding(dp(4), dp(8), dp(4), 0);
        card.addView(name);

        TextView status = new TextView(this);
        status.setText(buildDatasetStatus(dataset));
        status.setTextColor(0xFFDDDDDD);
        status.setTextSize(14f);
        status.setPadding(dp(4), dp(4), dp(4), dp(4));
        card.addView(status);

        card.setContentDescription(buildCardDescription(dataset));
        card.setClickable(true);
        card.setFocusable(true);
        card.setOnClickListener(v -> openOrPrepare(dataset, status));
        return card;
    }

    private String buildCardDescription(File dataset) {
        return formatDatasetName(dataset.getName()) + "。" + buildDatasetStatus(dataset)
                .replace('\n', ' ') + "。タップして開く、または3Dプレビューを準備する。";
    }

    private void openOrPrepare(File dataset, TextView status) {
        migrateLegacyDepthPrior(dataset);
        if (isViewableGaussian(dataset)) {
            openViewer(dataset);
            return;
        }
        if (generationInProgress) {
            Toast.makeText(this, "3Dプレビューを準備しています。少しお待ちください。",
                    Toast.LENGTH_SHORT).show();
            return;
        }
        generationInProgress = true;
        status.setText("3Dプレビューを準備中\n撮影データを確認しています…");
        new Thread(() -> {
            GaussianSplatJob.Result result = GaussianSplatJob.prepare(
                    dataset,
                    (percent, message) -> runOnUiThread(() -> status.setText(
                            "3Dプレビューを準備中 " + percent + "%\n" + message)));
            runOnUiThread(() -> {
                generationInProgress = false;
                status.setText(buildDatasetStatus(dataset));
                if (result.success || result.hqReady) {
                    openViewer(dataset);
                } else {
                    status.setText("準備できませんでした\nタップしてもう一度試す");
                    Toast.makeText(this,
                            "3Dプレビューを準備できませんでした。撮影データは残っています。",
                            Toast.LENGTH_LONG).show();
                }
            });
        }, "LibraryPreparePreview").start();
    }

    private void openViewer(File dataset) {
        if (!isViewableGaussian(dataset)) {
            Toast.makeText(this, "表示できる3Dデータがまだありません。", Toast.LENGTH_LONG).show();
            return;
        }
        Intent intent = new Intent(this, GaussianViewerActivity.class);
        intent.putExtra(GaussianViewerActivity.EXTRA_DATASET_PATH, dataset.getAbsolutePath());
        startActivity(intent);
    }

    private String buildDatasetStatus(File dataset) {
        int frames = readFrameCount(dataset);
        String photos = frames + "枚の写真";
        if (isPhotometricComplete(dataset)) return photos + "\n高品質3Dモデル完成";
        if (isHqPreview(dataset)) return photos + "\n3Dプレビュー作成済み";
        if (new File(dataset, DEPTH_PRIOR).isFile()) return photos + "\nタップしてプレビュー作成を再開";
        return photos + "\nタップして3Dプレビューを作成";
    }

    private static boolean isPhotometricComplete(File dataset) {
        File splat = new File(dataset, FINAL_SPLAT);
        if (!splat.isFile()) return false;
        JSONObject result = readResult(dataset);
        return result != null
                && result.optBoolean("photometric_optimization", false)
                && result.optBoolean("rasterized_image_loss", false)
                && result.optBoolean("l1_ssim_backward", false)
                && result.optBoolean("final_3dgs", false)
                && "COMPLETE".equals(result.optString("status", ""));
    }

    private static boolean isHqPreview(File dataset) {
        File splat = new File(dataset, FINAL_SPLAT);
        if (!splat.isFile()) return false;
        JSONObject result = readResult(dataset);
        return result != null
                && "HQ_RGB_REFINED".equals(result.optString("status", ""))
                && result.optBoolean("appearance_refinement", false)
                && !result.optBoolean("final_3dgs", false);
    }

    private static boolean isViewableGaussian(File dataset) {
        return isPhotometricComplete(dataset) || isHqPreview(dataset);
    }

    /** Move old depth-only experimental splat.ply away from the final/viewable artifact name. */
    private static void migrateLegacyDepthPrior(File dataset) {
        File finalSplat = new File(dataset, FINAL_SPLAT);
        File prior = new File(dataset, DEPTH_PRIOR);
        if (!finalSplat.isFile() || prior.isFile()) return;
        JSONObject result = readResult(dataset);
        if (result == null || result.optBoolean("photometric_optimization", false)) return;
        if (!finalSplat.renameTo(prior)) return;
        try {
            result.put("status", "DEPTH_PRIOR_READY");
            result.put("output", DEPTH_PRIOR);
            result.put("appearance_refinement", false);
            result.put("photometric_optimization", false);
            result.put("final_3dgs", false);
            result.put("note", "Migrated legacy depth-only Gaussian output; not a completed 3DGS model.");
            writeText(new File(dataset, "3dgs_result.json"), result.toString(2));
        } catch (Exception ignored) {}
    }

    private static List<File> findSavedDatasets(File pictures) {
        List<File> out = new ArrayList<>();
        if (pictures == null || !pictures.isDirectory()) return out;
        File[] dirs = pictures.listFiles(File::isDirectory);
        if (dirs == null) return out;
        for (File dir : dirs) {
            if (new File(dir, ".saved").isFile()
                    || (dir.getName().startsWith("dataset_") && new File(dir, "transforms.json").isFile())) {
                out.add(dir);
            }
        }
        out.sort(Comparator.comparingLong(File::lastModified).reversed());
        return out;
    }

    private static File findFirstJpeg(File dataset) {
        File[] images = dataset.listFiles((dir, name) ->
                name.startsWith("frame_") && name.toLowerCase(Locale.US).endsWith(".jpg"));
        if (images == null || images.length == 0) return null;
        Arrays.sort(images, Comparator.comparing(File::getName));
        return images[0];
    }

    private static Bitmap decodeThumbnail(File file, int targetWidth, int targetHeight) {
        if (file == null || !file.isFile()) return null;
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(file.getAbsolutePath(), bounds);
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null;
        int sample = 1;
        while (bounds.outWidth / (sample * 2) >= targetWidth
                && bounds.outHeight / (sample * 2) >= targetHeight) sample *= 2;
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = Math.max(1, sample);
        options.inPreferredConfig = Bitmap.Config.RGB_565;
        return BitmapFactory.decodeFile(file.getAbsolutePath(), options);
    }

    private static int readFrameCount(File dataset) {
        File manifest = new File(dataset, "dataset_manifest.json");
        try {
            if (manifest.isFile()) {
                JSONObject json = new JSONObject(readText(manifest));
                return json.optInt("frame_count", countJpegs(dataset));
            }
        } catch (Exception ignored) {}
        return countJpegs(dataset);
    }

    private static JSONObject readResult(File dataset) {
        File result = new File(dataset, "3dgs_result.json");
        try { if (result.isFile()) return new JSONObject(readText(result)); } catch (Exception ignored) {}
        return null;
    }

    private static int countJpegs(File dataset) {
        File[] files = dataset.listFiles((dir, name) ->
                name.startsWith("frame_") && name.toLowerCase(Locale.US).endsWith(".jpg"));
        return files == null ? 0 : files.length;
    }

    private static String readText(File file) throws Exception {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) out.append(line).append('\n');
        }
        return out.toString();
    }

    private static void writeText(File file, String text) throws Exception {
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(text.getBytes(StandardCharsets.UTF_8));
        }
    }

    private static String formatDatasetName(String name) {
        String value = name;
        if (value.startsWith("dataset_")) value = value.substring("dataset_".length());
        try {
            Date date = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).parse(value);
            if (date != null) return new SimpleDateFormat("yyyy/MM/dd HH:mm:ss", Locale.getDefault()).format(date);
        } catch (ParseException ignored) {}
        return name;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
