package com.sktpj.pointcloudsplatting;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Validates a saved RGB/Pose dataset, prepares an optional Raw Depth geometry prior, and runs the
 * phone-side high-resolution multi-view appearance refinement. Full final 3DGS is deliberately
 * reserved for a differentiable rasterized L1+SSIM optimizer with density control.
 */
public final class GaussianSplatJob {
    private static final String TAG = "GaussianSplatJob";
    private static final String DEPTH_PRIOR_NAME = "depth_prior.ply";
    private static final String PREVIEW_SPLAT_NAME = "preview_splat.ply";
    private static final String FINAL_SPLAT_NAME = "splat.ply";

    private GaussianSplatJob() {}

    public interface ProgressListener { void onProgress(int percent, String message); }

    public static final class Result {
        public final boolean success;
        public final boolean priorReady;
        public final boolean hqReady;
        public final String message;
        public final int frameCount;
        public final int gaussianCount;
        public final File outputFile;

        private Result(boolean success, boolean priorReady, boolean hqReady, String message,
                int frameCount, int gaussianCount, File outputFile) {
            this.success = success;
            this.priorReady = priorReady;
            this.hqReady = hqReady;
            this.message = message;
            this.frameCount = frameCount;
            this.gaussianCount = gaussianCount;
            this.outputFile = outputFile;
        }
        private static Result fail(String m, int f) { return new Result(false,false,false,m,f,0,null); }
        private static Result priorReady(String m,int f,int g,File o) { return new Result(false,true,false,m,f,g,o); }
        private static Result hqReady(String m,int f,int g,File o) { return new Result(false,true,true,m,f,g,o); }
    }

    public static Result prepare(File datasetDirectory) { return prepare(datasetDirectory, null); }

    public static Result prepare(File datasetDirectory, ProgressListener listener) {
        notifyProgress(listener, 2, "撮影データを確認しています…");
        if (datasetDirectory == null || !datasetDirectory.isDirectory()) return Result.fail("撮影データが見つかりませんでした。", 0);
        File transformsFile = new File(datasetDirectory, "transforms.json");
        if (!transformsFile.isFile()) return Result.fail("撮影データの保存が完了していません。", 0);
        try {
            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frames = transforms.getJSONArray("frames");
            int count = frames.length();
            if (count == 0) return Result.fail("保存できた写真がありません。", 0);
            DatasetStats stats = validatePhotometricDataset(datasetDirectory, frames);
            DiagnosticLog.i(TAG, "RGB dataset ready frames=" + count + " maxRgb=" + stats.maxWidth + "x" + stats.maxHeight
                    + " pose+intrinsics=" + stats.validCameraFrames + "/" + count);

            File jobFile = new File(datasetDirectory, "3dgs_job.json");
            JSONObject job = new JSONObject();
            job.put("format_version", 6);
            job.put("status", "PREPARING_GEOMETRY");
            job.put("requested_at_unix_ms", System.currentTimeMillis());
            job.put("frame_count", count);
            job.put("transforms", "transforms.json");
            job.put("rgb_pattern", "frame_*.jpg");
            job.put("rgb_role", "high_resolution_primary_observation");
            job.put("max_rgb_width", stats.maxWidth);
            job.put("max_rgb_height", stats.maxHeight);
            job.put("camera_source", "saved_ARCore_pose_and_intrinsics");
            job.put("depth_prior_pattern", "frame_*.ply_optional");
            job.put("camera_convention", "OpenGL camera-to-world, root-anchor local");
            job.put("target_backend", "android_ndk_vulkan_differentiable_3dgs");
            job.put("initializer_backend", "android_dense_depth_prior_v3");
            job.put("preview_backend", "android_highres_multiview_gaussian_v1");
            job.put("preview_output", PREVIEW_SPLAT_NAME);
            job.put("final_output", FINAL_SPLAT_NAME);
            job.put("appearance_refinement", false);
            job.put("photometric_optimization", false);
            job.put("rasterized_image_loss", false);
            job.put("l1_ssim_backward", false);
            job.put("density_control", false);
            job.put("final_3dgs", false);
            job.put("note", "preview_splat.ply is never a COMPLETE artifact. splat.ply is reserved for final differentiable training.");
            writeJson(jobFile, job);

            File priorFile = new File(datasetDirectory, DEPTH_PRIOR_NAME);
            int priorCount = 0;
            if (!priorFile.isFile()) {
                notifyProgress(listener, 8, "3Dの形を準備しています…");
                job.put("status", "INITIALIZING_DEPTH_PRIOR");
                writeJson(jobFile, job);
                GaussianSplatTrainer.Result initialization = GaussianSplatTrainer.train(datasetDirectory,
                        (percent, message) -> {
                            DiagnosticLog.i(TAG, "depth-prior " + percent + "% " + message);
                            notifyProgress(listener, 8 + Math.round(percent * 0.27f), "3Dの形を準備しています…");
                        });
                if (!initialization.success) {
                    job.put("status", "FAILED_DEPTH_PRIOR");
                    job.put("error", initialization.message);
                    writeJson(jobFile, job);
                    return Result.fail("3Dの形を作るための情報を準備できませんでした。", count);
                }
                priorFile = moveInitializerOutput(datasetDirectory, initialization.outputFile);
                priorCount = initialization.gaussianCount;
                rewriteInitializerResult(datasetDirectory, priorFile, priorCount);
            } else {
                JSONObject existing = readResult(datasetDirectory);
                priorCount = existing == null ? 0 : existing.optInt("gaussian_count", 0);
                DiagnosticLog.i(TAG, "Reusing depth prior " + priorFile.getAbsolutePath());
            }

            notifyProgress(listener, 36, "写真から色と質感を読み取っています…");
            job.put("status", "REFINING_HIGH_RES_RGB");
            job.put("depth_prior_output", priorFile.getName());
            job.put("depth_prior_gaussian_count", priorCount);
            writeJson(jobFile, job);

            HighQualityGaussianTrainer.Result hq = HighQualityGaussianTrainer.train(datasetDirectory,
                    (percent, message) -> {
                        DiagnosticLog.i(TAG, "hq " + percent + "% " + message);
                        notifyProgress(listener, 36 + Math.round(percent * 0.62f), userStageMessage(percent));
                    });
            if (!hq.success) {
                job.put("status", "DEPTH_PRIOR_READY");
                job.put("hq_error", hq.message);
                job.put("appearance_refinement", false);
                job.put("photometric_optimization", false);
                job.put("final_3dgs", false);
                writeJson(jobFile, job);
                DiagnosticLog.w(TAG, "RGB appearance refinement failed after depth prior: " + hq.message);
                return Result.priorReady("3Dプレビューを仕上げられませんでした。もう一度お試しください。",
                        count, priorCount, priorFile);
            }

            File previewFile = movePreviewOutput(datasetDirectory, hq.outputFile);
            normalizeAppearanceResult(datasetDirectory, hq, previewFile);
            job.put("status", "HQ_RGB_REFINED");
            job.put("hq_completed_at_unix_ms", System.currentTimeMillis());
            job.put("gaussian_count", hq.gaussianCount);
            job.put("textured_gaussian_count", hq.texturedGaussianCount);
            job.put("preview_output", previewFile.getName());
            job.put("appearance_refinement", true);
            job.put("appearance_fit", "weighted_multiview_SH1_least_squares");
            job.put("photometric_optimization", false);
            job.put("rasterized_image_loss", false);
            job.put("l1_ssim_backward", false);
            job.put("density_control", false);
            job.put("final_3dgs", false);
            if (Double.isFinite(hq.photometricRmse)) job.put("appearance_rmse", hq.photometricRmse);
            writeJson(jobFile, job);

            DiagnosticLog.i(TAG, "RGB appearance preview ready: " + hq.gaussianCount + " gaussians / "
                    + hq.texturedGaussianCount + " JPEG-observed / appearanceRmse=" + hq.photometricRmse
                    + " full3dgs=false output=" + previewFile.getAbsolutePath());
            notifyProgress(listener, 100, "3Dプレビューを準備しました");
            return Result.hqReady("3Dプレビューを準備しました。", count, hq.gaussianCount, previewFile);
        } catch (IOException | JSONException | RuntimeException e) {
            DiagnosticLog.e(TAG, "Failed to prepare/refine 3DGS dataset", e);
            return Result.fail("3Dプレビューを準備できませんでした。", 0);
        }
    }

    private static File movePreviewOutput(File dataset, File generated) throws IOException {
        if (generated == null || !generated.isFile()) throw new IOException("appearance refiner produced no PLY");
        File preview = new File(dataset, PREVIEW_SPLAT_NAME);
        if (preview.exists() && !preview.delete()) throw new IOException("cannot replace " + PREVIEW_SPLAT_NAME);
        if (!generated.renameTo(preview)) {
            copyFile(generated, preview);
            if (!generated.delete()) DiagnosticLog.w(TAG, "Could not delete temporary preview output " + generated.getAbsolutePath());
        }
        // Never leave the Java preview under the reserved final name.
        File accidentalFinal = new File(dataset, FINAL_SPLAT_NAME);
        if (accidentalFinal.isFile() && accidentalFinal.length() == preview.length()) accidentalFinal.delete();
        return preview;
    }

    private static void normalizeAppearanceResult(File dataset, HighQualityGaussianTrainer.Result hq, File preview)
            throws IOException, JSONException {
        File resultFile = new File(dataset, "3dgs_result.json");
        JSONObject result = resultFile.isFile() ? new JSONObject(readText(resultFile)) : new JSONObject();
        result.put("format_version", 5);
        result.put("status", "HQ_RGB_REFINED");
        result.put("backend", "android_highres_multiview_gaussian_v1");
        result.put("output", preview.getName());
        result.put("preview_output", preview.getName());
        result.put("final_output", FINAL_SPLAT_NAME);
        result.put("gaussian_count", hq.gaussianCount);
        result.put("textured_gaussian_count", hq.texturedGaussianCount);
        result.put("appearance_refinement", true);
        result.put("appearance_fit", "weighted_multiview_SH1_least_squares");
        if (Double.isFinite(hq.photometricRmse)) result.put("appearance_rmse", hq.photometricRmse);
        result.remove("photometric_rmse");
        result.put("photometric_optimization", false);
        result.put("rasterized_image_loss", false);
        result.put("l1_ssim_backward", false);
        result.put("density_control", false);
        result.put("final_3dgs", false);
        result.put("note", "High-resolution JPEGs were sampled and fitted to SH1 appearance. This file is a preview, not differentiable 3DGS training completion.");
        writeJson(resultFile, result);
    }

    private static String userStageMessage(int p) {
        if (p < 15) return "3Dの形を整えています…";
        if (p < 70) return "写真の色を3Dの形に合わせています…";
        if (p < 90) return "複数の角度の写真をまとめています…";
        return "3Dプレビューを仕上げています…";
    }

    private static void notifyProgress(ProgressListener l, int p, String m) {
        if (l != null) l.onProgress(Math.max(0, Math.min(100, p)), m);
    }

    private static DatasetStats validatePhotometricDataset(File dataset, JSONArray frames)
            throws JSONException, IOException {
        int maxWidth=0,maxHeight=0,valid=0;
        for (int i=0;i<frames.length();i++) {
            JSONObject frame=frames.getJSONObject(i);
            String path=frame.getString("file_path");
            File image=new File(dataset,path);
            if (!image.isFile() || image.length()==0L) throw new IOException("missing RGB observation: "+path);
            int w=frame.getInt("w"),h=frame.getInt("h");
            if (w<=0||h<=0) throw new IOException("invalid RGB dimensions: "+path);
            requireFinitePositive(frame.getDouble("fl_x"),"fl_x");
            requireFinitePositive(frame.getDouble("fl_y"),"fl_y");
            requireFinite(frame.getDouble("cx"),"cx");
            requireFinite(frame.getDouble("cy"),"cy");
            validateTransform(frame.getJSONArray("transform_matrix"));
            maxWidth=Math.max(maxWidth,w); maxHeight=Math.max(maxHeight,h); valid++;
        }
        return new DatasetStats(maxWidth,maxHeight,valid);
    }

    private static void validateTransform(JSONArray m) throws JSONException,IOException {
        if (m.length()!=4) throw new IOException("camera transform must be 4x4");
        for(int r=0;r<4;r++){JSONArray row=m.getJSONArray(r);if(row.length()!=4)throw new IOException("camera transform must be 4x4");for(int c=0;c<4;c++)requireFinite(row.getDouble(c),"transform_matrix");}
    }
    private static void requireFinitePositive(double v,String n)throws IOException{if(!Double.isFinite(v)||v<=0)throw new IOException("invalid camera "+n);}
    private static void requireFinite(double v,String n)throws IOException{if(!Double.isFinite(v))throw new IOException("invalid camera "+n);}

    private static File moveInitializerOutput(File dataset, File generated)throws IOException{
        if(generated==null||!generated.isFile())throw new IOException("depth prior initializer produced no PLY");
        File prior=new File(dataset,DEPTH_PRIOR_NAME);if(prior.exists()&&!prior.delete())throw new IOException("cannot replace "+DEPTH_PRIOR_NAME);
        if(!generated.renameTo(prior)){copyFile(generated,prior);if(!generated.delete())DiagnosticLog.w(TAG,"Could not delete temporary initializer output "+generated.getAbsolutePath());}
        return prior;
    }

    private static void rewriteInitializerResult(File dataset,File prior,int count)throws IOException,JSONException{
        File f=new File(dataset,"3dgs_result.json");JSONObject r=f.isFile()?new JSONObject(readText(f)):new JSONObject();
        r.put("format_version",5);r.put("status","DEPTH_PRIOR_READY");r.put("backend","android_dense_depth_prior_v3");r.put("output",prior.getName());r.put("gaussian_count",count);
        r.put("appearance_refinement",false);r.put("photometric_optimization",false);r.put("rasterized_image_loss",false);r.put("l1_ssim_backward",false);r.put("density_control",false);r.put("final_3dgs",false);
        r.put("note","Geometry initialization only. It is not a completed 3DGS model.");writeJson(f,r);
    }
    private static JSONObject readResult(File dataset){File f=new File(dataset,"3dgs_result.json");if(!f.isFile())return null;try{return new JSONObject(readText(f));}catch(Exception e){return null;}}
    private static void copyFile(File s,File d)throws IOException{try(FileInputStream in=new FileInputStream(s);FileOutputStream out=new FileOutputStream(d)){byte[]b=new byte[65536];int n;while((n=in.read(b))!=-1)out.write(b,0,n);}}
    private static void writeJson(File f,JSONObject j)throws IOException,JSONException{try(FileOutputStream out=new FileOutputStream(f)){out.write(j.toString(2).getBytes(StandardCharsets.UTF_8));}}
    private static String readText(File f)throws IOException{StringBuilder out=new StringBuilder();try(BufferedReader r=new BufferedReader(new InputStreamReader(new FileInputStream(f),StandardCharsets.UTF_8))){String l;while((l=r.readLine())!=null)out.append(l).append('\n');}return out.toString();}
    private static final class DatasetStats{final int maxWidth,maxHeight,validCameraFrames;DatasetStats(int w,int h,int v){maxWidth=w;maxHeight=h;validCameraFrames=v;}}
}
