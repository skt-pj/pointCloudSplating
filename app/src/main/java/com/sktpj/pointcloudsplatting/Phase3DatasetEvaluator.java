package com.sktpj.pointcloudsplatting;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/** Machine-side Phase 3 gate. Final PASS still requires user visual review in the viewer. */
public final class Phase3DatasetEvaluator {
    private static final String TAG = "Phase3Eval";
    private static final String REPORT = "phase3_evaluation.json";
    private static final String FINAL_SPLAT = "splat.ply";
    private static final String RESULT = "3dgs_result.json";
    private static final String CHECKPOINT = "vksplat_data/3dgs_checkpoint.bin";

    private Phase3DatasetEvaluator() {}

    public static final class Result {
        public final boolean machineGatePassed;
        public final boolean reviewRequired;
        public final int failureCount;
        public final int warningCount;
        public final File reportFile;
        public final JSONObject report;

        Result(boolean machineGatePassed, boolean reviewRequired, int failureCount,
                int warningCount, File reportFile, JSONObject report) {
            this.machineGatePassed = machineGatePassed;
            this.reviewRequired = reviewRequired;
            this.failureCount = failureCount;
            this.warningCount = warningCount;
            this.reportFile = reportFile;
            this.report = report;
        }
    }

    public static Result evaluate(File dataset) {
        JSONArray failures = new JSONArray();
        JSONArray warnings = new JSONArray();
        JSONObject training = new JSONObject();
        JSONObject holdout = new JSONObject();
        File reportFile = dataset == null ? null : new File(dataset, REPORT);

        try {
            if (dataset == null || !dataset.isDirectory()) {
                fail(failures, "dataset_unavailable", "dataset directory unavailable");
            } else {
                if (!Phase1DatasetEvaluator.hasStoredPass(dataset)) {
                    fail(failures, "phase1_not_pass", "Phase 1 evaluation is not PASS");
                }
                if (!Phase2DatasetEvaluator.hasStoredPass(dataset)) {
                    fail(failures, "phase2_not_pass", "Phase 2 evaluation is not PASS");
                }

                File geometry = new File(dataset, "phase2_geometry_prior.ply");
                if (!geometry.isFile() || geometry.length() <= 0L) {
                    fail(failures, "phase2_geometry_missing", "phase2_geometry_prior.ply missing");
                }

                File splat = new File(dataset, FINAL_SPLAT);
                if (!splat.isFile() || splat.length() <= 0L) {
                    fail(failures, "final_splat_missing", "splat.ply missing or empty");
                }
                File checkpoint = new File(dataset, CHECKPOINT);
                if (!checkpoint.isFile() || checkpoint.length() <= 0L) {
                    fail(failures, "checkpoint_missing", "resumable 3DGS checkpoint missing");
                }

                JSONObject result = readJson(new File(dataset, RESULT));
                if (result == null) {
                    fail(failures, "result_missing", "3dgs_result.json missing or invalid");
                } else {
                    requireTrue(result, "photometric_optimization", failures);
                    requireTrue(result, "rasterized_image_loss", failures);
                    requireTrue(result, "l1_ssim_backward", failures);
                    requireTrue(result, "density_control", failures);
                    requireTrue(result, "final_3dgs", failures);
                    requireTrue(result, "checkpoint_available", failures);
                    if (!"COMPLETE".equals(result.optString("status", ""))) {
                        fail(failures, "trainer_not_complete", "3dgs_result status is not COMPLETE");
                    }
                    if (result.optInt("phase3_pipeline_version", 0) < 1) {
                        fail(failures, "phase3_pipeline_old", "result predates Phase 3 progressive pipeline");
                    }
                    String profile = result.optString("training_profile", "");
                    if (!"progressive_low_mid_high_patch".equals(profile)) {
                        fail(failures, "training_profile", "progressive low/mid/high-patch profile missing");
                    }
                    if (!"phase2_geometry_prior.ply".equals(result.optString("geometry_source", ""))) {
                        fail(failures, "geometry_source", "Phase 3 did not initialize from Phase 2 geometry prior");
                    }
                    int steps = result.optInt("training_steps", 0);
                    int gaussians = result.optInt("gaussian_count", 0);
                    if (steps <= 0) fail(failures, "training_steps", "no optimizer steps recorded");
                    if (gaussians < 64) fail(failures, "gaussian_count", "final Gaussian count is too small");

                    training.put("training_steps", steps);
                    training.put("gaussian_count", gaussians);
                    training.put("gaussian_budget", result.optInt("gaussian_budget", 0));
                    training.put("density_strategy", result.optString("density_strategy", ""));
                    training.put("optimized_parameters", result.optString("optimized_parameters", ""));
                    training.put("training_profile", profile);
                    training.put("geometry_source", result.optString("geometry_source", ""));
                    training.put("vulkan_device", result.optString("vulkan_device", ""));
                    training.put("peak_vram_mb", finiteOrNull(result.optDouble("peak_vram_mb", Double.NaN)));

                    int holdoutViews = result.optInt("holdout_view_count", 0);
                    double psnr = result.optDouble("validation_psnr", Double.NaN);
                    double ssim = result.optDouble("validation_ssim", Double.NaN);
                    holdout.put("view_count", holdoutViews);
                    holdout.put("psnr", finiteOrNull(psnr));
                    holdout.put("ssim", finiteOrNull(ssim));
                    holdout.put("role", "auxiliary_metric_not_final_quality_pass");
                    if (holdoutViews <= 0) {
                        fail(failures, "holdout_missing", "no hold-out RGB views were evaluated");
                    }
                    if (!Double.isFinite(psnr)) {
                        fail(failures, "holdout_psnr_missing", "hold-out PSNR is unavailable");
                    }
                    if (!Double.isFinite(ssim)) {
                        fail(failures, "holdout_ssim_missing", "hold-out SSIM is unavailable");
                    }

                    int renderCount = countHoldoutRenders(dataset);
                    holdout.put("render_count", renderCount);
                    holdout.put("render_format", "PPM P6");
                    if (renderCount < holdoutViews) {
                        warn(warnings, "holdout_render_incomplete",
                                "saved hold-out renders=" + renderCount + " expected=" + holdoutViews);
                    }
                }

                // v1.0.9 intentionally does not auto-PASS final visual quality. The viewer is the
                // final gate for double images, floaters, holes, smearing and novel-view continuity.
                warn(warnings, "visual_review_required",
                        "Final quality must be judged in the device viewer; numeric metrics cannot auto-PASS Phase 3");
                warn(warnings, "ab_fixed_view_pending",
                        "Automated Phase2-vs-final fixed-view A/B images are not yet generated in v1.0.9");
            }
        } catch (Throwable error) {
            fail(failures, "phase3_evaluator_exception",
                    error.getClass().getSimpleName() + ": " + String.valueOf(error.getMessage()));
        }

        boolean machinePass = failures.length() == 0;
        boolean reviewRequired = machinePass;
        String status = machinePass ? "REVIEW_REQUIRED" : "FAIL";
        JSONObject report = new JSONObject();
        try {
            report.put("format_version", 1);
            report.put("phase", "phase3");
            report.put("evaluator_build", BuildConfig.VERSION_NAME);
            report.put("status", status);
            report.put("pass", false);
            report.put("machine_gate_pass", machinePass);
            report.put("review_required", reviewRequired);
            report.put("next_phase_allowed", false);
            report.put("final_visual_pass_requires_user", true);
            report.put("phase1_pass", dataset != null && Phase1DatasetEvaluator.hasStoredPass(dataset));
            report.put("phase2_pass", dataset != null && Phase2DatasetEvaluator.hasStoredPass(dataset));
            report.put("training", training);
            report.put("holdout", holdout);
            report.put("failure_count", failures.length());
            report.put("warning_count", warnings.length());
            report.put("failures", failures);
            report.put("warnings", warnings);
            report.put("evaluated_at_unix_ms", System.currentTimeMillis());
        } catch (Exception ignored) {
        }

        if (reportFile != null) {
            try (FileOutputStream out = new FileOutputStream(reportFile, false)) {
                out.write(report.toString(2).getBytes(StandardCharsets.UTF_8));
                out.getFD().sync();
            } catch (Exception error) {
                DiagnosticLog.e(TAG, "Could not write phase3_evaluation.json", error);
                machinePass = false;
                reviewRequired = false;
            }
        }

        DiagnosticLog.i(TAG, String.format(Locale.US,
                "PHASE3_EVAL %s machineGate=%s failures=%d warnings=%d holdoutViews=%d psnr=%s ssim=%s",
                status,
                Boolean.toString(machinePass),
                failures.length(),
                warnings.length(),
                holdout.optInt("view_count", 0),
                String.valueOf(holdout.opt("psnr")),
                String.valueOf(holdout.opt("ssim"))));
        DriveDiagnosticLogStore.setPrimaryDataset(dataset);
        DriveDiagnosticLogStore.requestOverwrite();
        return new Result(machinePass, reviewRequired, failures.length(), warnings.length(), reportFile, report);
    }

    public static boolean needsEvaluation(File dataset) {
        JSONObject report = readJson(dataset == null ? null : new File(dataset, REPORT));
        return report == null || !BuildConfig.VERSION_NAME.equals(report.optString("evaluator_build", ""));
    }

    public static boolean hasMachineGatePass(File dataset) {
        JSONObject report = readJson(dataset == null ? null : new File(dataset, REPORT));
        return report != null && report.optBoolean("machine_gate_pass", false);
    }

    private static int countHoldoutRenders(File dataset) {
        File root = new File(dataset, "vksplat_data");
        File[] files = root.listFiles((dir, name) -> name.startsWith("phase3_holdout_render_") && name.endsWith(".ppm"));
        return files == null ? 0 : files.length;
    }

    private static void requireTrue(JSONObject result, String key, JSONArray failures) {
        if (!result.optBoolean(key, false)) {
            fail(failures, "trainer_contract_" + key, key + " is not true");
        }
    }

    private static Object finiteOrNull(double value) {
        return Double.isFinite(value) ? value : JSONObject.NULL;
    }

    private static void fail(JSONArray failures, String code, String detail) {
        failures.put(issue(code, detail));
    }

    private static void warn(JSONArray warnings, String code, String detail) {
        warnings.put(issue(code, detail));
    }

    private static JSONObject issue(String code, String detail) {
        JSONObject issue = new JSONObject();
        try {
            issue.put("code", code);
            issue.put("detail", detail);
        } catch (Exception ignored) {
        }
        return issue;
    }

    private static JSONObject readJson(File file) {
        if (file == null || !file.isFile() || file.length() <= 0L || file.length() > 2 * 1024 * 1024L) return null;
        try (FileInputStream in = new FileInputStream(file)) {
            byte[] bytes = new byte[(int) file.length()];
            int offset = 0;
            while (offset < bytes.length) {
                int read = in.read(bytes, offset, bytes.length - offset);
                if (read < 0) break;
                offset += read;
            }
            return new JSONObject(new String(bytes, 0, offset, StandardCharsets.UTF_8));
        } catch (Exception ignored) {
            return null;
        }
    }
}
