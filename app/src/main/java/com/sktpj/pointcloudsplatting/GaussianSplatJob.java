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
    private static final String FINAL_SPLAT = "splat.ply";
    private static final String CHECKPOINT_RELATIVE = "vksplat_data/3dgs_checkpoint.bin";

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

    /** Existing scanner/library entry point; chooses the normal first-run mobile step default. */
    public static Result prepare(File datasetDirectory, ProgressListener listener) {
        return prepare(PointCloudApp.context(), datasetDirectory, listener);
    }

    public static Result prepare(Context context, File datasetDirectory, ProgressListener listener) {
        return runInitial(context, datasetDirectory, 0, listener);
    }

    /** Optional explicit first-run step count. No quality ceiling is imposed by the app. */
    public static Result prepare(Context context, File datasetDirectory, int requestedSteps,
            ProgressListener listener) {
        if (requestedSteps <= 0) return Result.fail("学習回数は1以上を指定してください。", 0);
        return runInitial(context, datasetDirectory, requestedSteps, listener);
    }

    /** True only for new models that persisted Gaussian + Adam + RNG + cumulative step state. */
    public static boolean canContinueTraining(File datasetDirectory) {
        return isVerifiedFinal(datasetDirectory) && checkpointFile(datasetDirectory).isFile()
                && checkpointFile(datasetDirectory).length() > 0L;
    }

    /**
     * Continues a completed model for exactly additionalSteps more optimizer iterations.
     * This never treats a PLY-only legacy model as a true resume because its Adam state is gone.
     */
    public static Result continueTraining(File datasetDirectory, int additionalSteps,
            ProgressListener listener) {
        return continueTraining(PointCloudApp.context(), datasetDirectory, additionalSteps, listener);
    }

    public static Result continueTraining(Context context, File datasetDirectory, int additionalSteps,
            ProgressListener listener) {
        if (additionalSteps <= 0) return Result.fail("追加学習回数は1以上を指定してください。", 0);
        if (context == null || datasetDirectory == null || !datasetDirectory.isDirectory()) {
            return Result.fail("撮影データが見つかりませんでした。", 0);
        }
        if (!isVerifiedFinal(datasetDirectory)) {
            return Result.fail("追加学習できる完成済み3DGSモデルがありません。", 0);
        }
        File checkpoint = checkpointFile(datasetDirectory);
        if (!checkpoint.isFile() || checkpoint.length() == 0L) {
            return Result.fail(
                    "このモデルは旧版で作成され、optimizerの学習状態が保存されていません。"
                            + "正確な追加学習には新版で一度モデルを作成し直してください。",
                    readFrameCount(datasetDirectory));
        }

        JSONObject before = readResult(datasetDirectory);
        int previousSteps = before == null ? 0 : before.optInt("training_steps", 0);
        int frameCount = readFrameCount(datasetDirectory);
        notifyProgress(listener, 1,
                "保存済み3DGSの " + previousSteps + " step から追加学習を準備しています…");

        Context appContext = context.getApplicationContext();
        if (!ModelProcessingCoordinator.enter(appContext)) {
            return Result.fail("変換用画面を開始できなかったため、安全のため追加学習を開始しませんでした。",
                    frameCount);
        }

        NativeGaussianTrainer.Result trained;
        try {
            notifyProgress(listener, 4, "保存したoptimizer状態から3DGS追加学習を始めます…");
            trained = NativeGaussianTrainer.train(
                    appContext,
                    datasetDirectory,
                    additionalSteps,
                    (percent, message) -> notifyProgress(listener, percent, message));
        } finally {
            ModelProcessingCoordinator.exit();
        }

        if (!trained.success) {
            DiagnosticLog.w(TAG, "3DGS continuation failed: " + trained.message);
            return Result.fail(trained.message, frameCount);
        }

        JSONObject after = readResult(datasetDirectory);
        int completedSteps = after == null ? 0 : after.optInt("training_steps", 0);
        boolean verified = isVerifiedFinal(datasetDirectory)
                && checkpoint.isFile() && checkpoint.length() > 0L
                && after != null
                && after.optBoolean("resumable_training", false)
                && after.optBoolean("checkpoint_available", false)
                && completedSteps == previousSteps + additionalSteps
                && after.optInt("last_run_added_steps", -1) == additionalSteps
                && after.optBoolean("last_run_resumed", false);
        if (!verified) {
            DiagnosticLog.e(TAG, "Continuation returned without cumulative checkpoint verification"
                    + " previous=" + previousSteps + " requested=" + additionalSteps
                    + " completed=" + completedSteps);
            return Result.fail("追加学習後のcheckpointを確認できませんでした。", frameCount);
        }

        DiagnosticLog.i(TAG, "3DGS CONTINUED previousSteps=" + previousSteps
                + " addedSteps=" + additionalSteps + " totalSteps=" + completedSteps
                + " gaussians=" + trained.gaussianCount
                + " validationPsnr=" + trained.validationPsnr
                + " device=" + trained.device);
        notifyProgress(listener, 100, "追加学習が完了しました（累計 " + completedSteps + " step）");
        return Result.complete("3DGS追加学習完了", frameCount, trained.gaussianCount,
                trained.outputFile);
    }

    private static Result runInitial(Context context, File datasetDirectory, int requestedSteps,
            ProgressListener listener) {
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

            // A completed model is opened as-is. Explicit additional training goes through
            // continueTraining so an existing optimizer checkpoint is never accidentally reset.
            File existing = new File(datasetDirectory, FINAL_SPLAT);
            JSONObject previous = readResult(datasetDirectory);
            if (existing.isFile() && isCompleteResult(previous)) {
                int gaussians = previous.optInt("gaussian_count", 0);
                notifyProgress(listener, 100, "3Dモデルは作成済みです");
                return Result.complete("3DGS学習済み", count, gaussians, existing);
            }

            // Remove only stale/legacy files carrying the reserved final name. Preview artifacts
            // remain separate as preview_splat.ply and are never promoted to completion.
            if (existing.isFile() && !existing.delete()) {
                return Result.fail("以前の未完成データを更新できませんでした。", count);
            }

            notifyProgress(listener, 3, "変換中のカメラとAR表示を停止しています…");
            Context appContext = context.getApplicationContext();
            if (!ModelProcessingCoordinator.enter(appContext)) {
                return Result.fail("変換用画面を開始できなかったため、安全のため3D処理を開始しませんでした。", count);
            }

            NativeGaussianTrainer.Result trained;
            try {
                notifyProgress(listener, 4, "端末内で3Dモデルの学習を始めます…");
                trained = requestedSteps > 0
                        ? NativeGaussianTrainer.train(appContext, datasetDirectory, requestedSteps,
                                (percent, message) -> notifyProgress(listener, percent, message))
                        : NativeGaussianTrainer.train(appContext, datasetDirectory,
                                (percent, message) -> notifyProgress(listener, percent, message));
            } finally {
                ModelProcessingCoordinator.exit();
            }

            if (!trained.success) {
                DiagnosticLog.w(TAG, "Full 3DGS failed: " + trained.message);
                return Result.fail(trained.message, count);
            }

            JSONObject result = readResult(datasetDirectory);
            boolean verified = trained.outputFile != null && trained.outputFile.isFile()
                    && trained.outputFile.length() > 0L
                    && isCompleteResult(result)
                    && result.optBoolean("resumable_training", false)
                    && result.optBoolean("checkpoint_available", false)
                    && checkpointFile(datasetDirectory).isFile()
                    && checkpointFile(datasetDirectory).length() > 0L;
            if (!verified) {
                DiagnosticLog.e(TAG, "Native trainer returned without a verified resumable 3DGS artifact");
                return Result.fail("3Dモデルの学習結果を確認できませんでした。", count);
            }

            DiagnosticLog.i(TAG,
                    "Full 3DGS COMPLETE frames=" + count
                            + " gaussians=" + trained.gaussianCount
                            + " steps=" + trained.steps
                            + " checkpoint=true"
                            + " validationPsnr=" + trained.validationPsnr
                            + " device=" + trained.device);
            notifyProgress(listener, 100, "3Dモデルを作成しました");
            return Result.complete("3DGS学習完了", count, trained.gaussianCount, trained.outputFile);
        } catch (Exception e) {
            ModelProcessingCoordinator.exit();
            DiagnosticLog.e(TAG, "Full 3DGS job failed", e);
            return Result.fail("3Dモデルの学習に失敗しました。", 0);
        }
    }

    private static boolean isVerifiedFinal(File datasetDirectory) {
        if (datasetDirectory == null) return false;
        File output = new File(datasetDirectory, FINAL_SPLAT);
        return output.isFile() && output.length() > 0L && isCompleteResult(readResult(datasetDirectory));
    }

    private static boolean isCompleteResult(JSONObject result) {
        return result != null
                && "COMPLETE".equals(result.optString("status", ""))
                && result.optBoolean("photometric_optimization", false)
                && result.optBoolean("rasterized_image_loss", false)
                && result.optBoolean("l1_ssim_backward", false)
                && result.optBoolean("density_control", false)
                && result.optBoolean("final_3dgs", false);
    }

    private static File checkpointFile(File datasetDirectory) {
        return new File(datasetDirectory, CHECKPOINT_RELATIVE);
    }

    private static int readFrameCount(File datasetDirectory) {
        try {
            File transformsFile = new File(datasetDirectory, "transforms.json");
            if (!transformsFile.isFile()) return 0;
            return new JSONObject(readText(transformsFile)).getJSONArray("frames").length();
        } catch (Exception ignored) {
            return 0;
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
        int clamped = Math.max(0, Math.min(100, percent));
        ModelProcessingCoordinator.publishProgress(clamped, message);
        if (listener != null) listener.onProgress(clamped, message);
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
