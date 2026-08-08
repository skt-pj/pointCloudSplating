package com.sktpj.pointcloudsplatting;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/** Runs the final on-device differentiable 3D Gaussian Splatting training job. */
public final class GaussianSplatJob {
    private static final String TAG = "GaussianSplatJob";

    private GaussianSplatJob() {}

    public interface ProgressListener {
        void onProgress(int percent, String message);
    }

    public static final class Result {
        /** True only when rasterized L1+SSIM backward training produced final splat.ply. */
        public final boolean success;
        /** Retained for UI/source compatibility. A final run never reports prior-only success. */
        public final boolean priorReady;
        /** Retained for UI/source compatibility. A final run never reports preview as completion. */
        public final boolean hqReady;
        public final String message;
        public final int frameCount;
        public final int gaussianCount;
        public final File outputFile;

        private Result(boolean success, String message, int frameCount, int gaussianCount,
                File outputFile) {
            this.success = success;
            this.priorReady = false;
            this.hqReady = false;
            this.message = message;
            this.frameCount = frameCount;
            this.gaussianCount = gaussianCount;
            this.outputFile = outputFile;
        }

        private static Result fail(String message, int frames) {
            return new Result(false, message, frames, 0, null);
        }

        private static Result complete(String message, int frames, int gaussians, File output) {
            return new Result(true, message, frames, gaussians, output);
        }
    }

    public static Result prepare(Context context, File datasetDirectory, ProgressListener listener) {
        notifyProgress(listener, 1, "撮影した写真とカメラ位置を確認しています…");
        if (context == null || datasetDirectory == null || !datasetDirectory.isDirectory()) {
            return Result.fail("撮影データが見つかりませんでした。", 0);
        }
        File transformsFile = new File(datasetDirectory, "transforms.json");
        if (!transformsFile.isFile()) {
            return Result.fail("撮影データの保存が完了していません。", 0);
        }

        try {
            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frames = transforms.getJSONArray("frames");
            int count = frames.length();
            if (count < 2) {
                return Result.fail("異なる位置から撮影した写真がもう少し必要です。", count);
            }
            DatasetStats stats = validatePhotometricDataset(datasetDirectory, frames);
            DiagnosticLog.i(TAG,
                    "Full 3DGS dataset ready frames=" + count
                            + " maxRgb=" + stats.maxWidth + "x" + stats.maxHeight
                            + " pose+intrinsics=" + stats.validCameraFrames + "/" + count);

            // If a verified final artifact already exists, do not replace it with a preview path.
            File existing = new File(datasetDirectory, "splat.ply");
            JSONObject previous = readResult(datasetDirectory);
            if (existing.isFile() && previous != null
                    && "COMPLETE".equals(previous.optString("status", ""))
                    && previous.optBoolean("photometric_optimization", false)
                    && previous.optBoolean("rasterized_image_loss", false)
                    && previous.optBoolean("l1_ssim_backward", false)
                    && previous.optBoolean("density_control", false)
                    && previous.optBoolean("final_3dgs", false)) {
                int gaussians = previous.optInt("gaussian_count", 0);
                notifyProgress(listener, 100, "3Dモデルは作成済みです");
                return Result.complete("3DGS学習済み", count, gaussians, existing);
            }

            // Remove only stale/legacy files carrying the reserved final name. Preview artifacts
            // remain separate as preview_splat.ply and are never promoted to completion.
            if (existing.isFile() && !existing.delete()) {
                return Result.fail("以前の未完成データを更新できませんでした。", count);
            }

            notifyProgress(listener, 4, "端末内で3Dモデルの学習を始めます…");
            NativeGaussianTrainer.Result trained = NativeGaussianTrainer.train(
                    context.getApplicationContext(),
                    datasetDirectory,
                    (percent, message) -> notifyProgress(listener, percent, message));
            if (!trained.success) {
                DiagnosticLog.w(TAG, "Full 3DGS failed: " + trained.message);
                return Result.fail(trained.message, count);
            }

            JSONObject result = readResult(datasetDirectory);
            boolean verified = trained.outputFile != null && trained.outputFile.isFile()
                    && trained.outputFile.length() > 0L
                    && result != null
                    && "COMPLETE".equals(result.optString("status", ""))
                    && result.optBoolean("photometric_optimization", false)
                    && result.optBoolean("rasterized_image_loss", false)
                    && result.optBoolean("l1_ssim_backward", false)
                    && result.optBoolean("density_control", false)
                    && result.optBoolean("final_3dgs", false);
            if (!verified) {
                DiagnosticLog.e(TAG, "Native trainer returned without a verified final 3DGS artifact");
                return Result.fail("3Dモデルの学習結果を確認できませんでした。", count);
            }

            DiagnosticLog.i(TAG,
                    "Full 3DGS COMPLETE frames=" + count
                            + " gaussians=" + trained.gaussianCount
                            + " steps=" + trained.steps
                            + " validationPsnr=" + trained.validationPsnr
                            + " device=" + trained.device);
            notifyProgress(listener, 100, "3Dモデルを作成しました");
            return Result.complete("3DGS学習完了", count, trained.gaussianCount, trained.outputFile);
        } catch (Exception e) {
            DiagnosticLog.e(TAG, "Full 3DGS job failed", e);
            return Result.fail("3Dモデルの学習に失敗しました。", 0);
        }
    }

    private static DatasetStats validatePhotometricDataset(File dataset, JSONArray frames)
            throws Exception {
        int maxWidth = 0;
        int maxHeight = 0;
        int valid = 0;
        for (int i = 0; i < frames.length(); i++) {
            JSONObject frame = frames.getJSONObject(i);
            File image = new File(dataset, frame.getString("file_path"));
            if (!image.isFile() || image.length() == 0L) {
                throw new IOException("missing RGB image " + image.getName());
            }
            int w = frame.getInt("w");
            int h = frame.getInt("h");
            requirePositive(frame.getDouble("fl_x"));
            requirePositive(frame.getDouble("fl_y"));
            requireFinite(frame.getDouble("cx"));
            requireFinite(frame.getDouble("cy"));
            JSONArray matrix = frame.getJSONArray("transform_matrix");
            if (matrix.length() != 4) throw new IOException("camera matrix must be 4x4");
            for (int r = 0; r < 4; r++) {
                JSONArray row = matrix.getJSONArray(r);
                if (row.length() != 4) throw new IOException("camera matrix must be 4x4");
                for (int c = 0; c < 4; c++) requireFinite(row.getDouble(c));
            }
            maxWidth = Math.max(maxWidth, w);
            maxHeight = Math.max(maxHeight, h);
            valid++;
        }
        return new DatasetStats(maxWidth, maxHeight, valid);
    }

    private static void requirePositive(double value) throws IOException {
        if (!Double.isFinite(value) || value <= 0.0) throw new IOException("invalid camera intrinsics");
    }

    private static void requireFinite(double value) throws IOException {
        if (!Double.isFinite(value)) throw new IOException("invalid camera data");
    }

    private static JSONObject readResult(File dataset) {
        File file = new File(dataset, "3dgs_result.json");
        if (!file.isFile()) return null;
        try {
            return new JSONObject(readText(file));
        } catch (Exception ignored) {
            return null;
        }
    }

    private static String readText(File file) throws IOException {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) out.append(line).append('\n');
        }
        return out.toString();
    }

    private static void notifyProgress(ProgressListener listener, int percent, String message) {
        if (listener != null) listener.onProgress(Math.max(0, Math.min(100, percent)), message);
    }

    private static final class DatasetStats {
        final int maxWidth;
        final int maxHeight;
        final int validCameraFrames;
        DatasetStats(int maxWidth, int maxHeight, int validCameraFrames) {
            this.maxWidth = maxWidth;
            this.maxHeight = maxHeight;
            this.validCameraFrames = validCameraFrames;
        }
    }
}
