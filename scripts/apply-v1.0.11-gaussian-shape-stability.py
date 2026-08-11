#!/usr/bin/env python3
"""Apply PCS v1.0.11 Gaussian shape correction to app-owned runtime sources."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
MODEL = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/GaussianModel.java"
NATIVE = ROOT / "app/src/main/cpp/native_3dgs.cpp"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"v1.0.11 apply failed: {label}: anchor missing in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# New shader generation because the Slang MCMC/optimizer contract changes.
replace_once(
    JAVA,
    'private static final String SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v8";',
    'private static final String SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v9";',
    "shader cache v9",
)
replace_once(
    JAVA,
    '                            + " numeric=finite_projection+mcmc_relocation+adam_rollback"\n',
    '                            + " numeric=finite_projection+mcmc_relocation+adam_rollback"\n'
    '                            + " shape=metric_scale_cap_6cm+mcmc_cumulative+resume_sanitize"\n',
    "shape diagnostic marker",
)

# Viewer safety: an old v1.0.10 PLY can contain valid-but-huge scales. Display no axis larger than
# 6 cm in original ARCore metric world-space. This changes display only; continuation below repairs
# the checkpoint itself and persists the correction.
replace_once(
    MODEL,
    '    private static final int ROBUST_SAMPLE_LIMIT = 8192;\n',
    '    private static final int ROBUST_SAMPLE_LIMIT = 8192;\n'
    '    private static final float MAX_WORLD_SCALE_METERS = 0.06f;\n',
    "viewer metric scale constant",
)
replace_once(
    MODEL,
    '''            float[] positionsOut = new float[validCount * 3];\n            float[] dcOut = new float[validCount * 3];\n''',
    '''            final float viewerScaleCap = Math.min(0.35f,\n                    MAX_WORLD_SCALE_METERS / Math.max(radius, 1e-6f));\n            int clampedScaleAxes = 0;\n\n            float[] positionsOut = new float[validCount * 3];\n            float[] dcOut = new float[validCount * 3];\n''',
    "viewer scale cap setup",
)
replace_once(
    MODEL,
    '''                scalesOut[dstP] = clamp(worldScales[srcP] / radius, 1e-6f, 0.35f);\n                scalesOut[dstP + 1] = clamp(worldScales[srcP + 1] / radius, 1e-6f, 0.35f);\n                scalesOut[dstP + 2] = clamp(worldScales[srcP + 2] / radius, 1e-6f, 0.35f);\n''',
    '''                float nsx = worldScales[srcP] / radius;\n                float nsy = worldScales[srcP + 1] / radius;\n                float nsz = worldScales[srcP + 2] / radius;\n                if (nsx > viewerScaleCap) clampedScaleAxes++;\n                if (nsy > viewerScaleCap) clampedScaleAxes++;\n                if (nsz > viewerScaleCap) clampedScaleAxes++;\n                scalesOut[dstP] = clamp(nsx, 1e-6f, viewerScaleCap);\n                scalesOut[dstP + 1] = clamp(nsy, 1e-6f, viewerScaleCap);\n                scalesOut[dstP + 2] = clamp(nsz, 1e-6f, viewerScaleCap);\n''',
    "viewer per-axis scale clamp",
)
replace_once(
    MODEL,
    '''                                    + "radius99=%.3f anisotropic=%d SHprops=%d sh1=%d sh2=%d sh3=%d "\n                                    + "viewer=ANISOTROPIC_COVARIANCE_SORTED_ES3 appearance=SH0_SH3",\n                            validCount, header.vertexCount, centerX, centerY, centerZ, radius,\n                            anisotropicCount, availableRest, sh1Count, sh2Count, sh3Count));\n''',
    '''                                    + "radius99=%.3f anisotropic=%d SHprops=%d sh1=%d sh2=%d sh3=%d "\n                                    + "viewerScaleCapM=%.3f clampedScaleAxes=%d "\n                                    + "viewer=ANISOTROPIC_COVARIANCE_SORTED_ES3 appearance=SH0_SH3",\n                            validCount, header.vertexCount, centerX, centerY, centerZ, radius,\n                            anisotropicCount, availableRest, sh1Count, sh2Count, sh3Count,\n                            MAX_WORLD_SCALE_METERS, clampedScaleAxes));\n''',
    "viewer scale diagnostics",
)

# Runtime checkpoint repair. v1.0.10 exact checkpoints are structurally sound; sanitize only scale
# axes and their Adam moments, preserving positions, appearance, rotations, opacity, RNG and steps.
replace_once(
    NATIVE,
    '''constexpr float kMaxSurfaceScale = 5.0e-2f;\n''',
    '''constexpr float kMaxSurfaceScale = 5.0e-2f;\nconstexpr float kMinTrainedGaussianScale = 1.0e-4f;\nconstexpr float kMaxTrainedGaussianScale = 6.0e-2f;\n''',
    "trained Gaussian metric scale bounds",
)

sanitize_anchor = '''size_t maxTileIndicesForBudget() {\n'''
sanitize_method = r'''void sanitizeGaussianShape(VulkanGSTrainer& trainer,
                           VulkanGSPipelineBuffers& buffers,
                           const char* phase) {
    const size_t n = buffers.num_splats;
    if (n == 0 || buffers.scales_opacs.deviceBuffer.buffer == VK_NULL_HANDLE) return;

    trainer.copyFromDevice(buffers.scales_opacs);
    if (buffers.scales_opacs.size() < 4 * n)
        throw std::runtime_error("Gaussian shape sanitizer received a short scale buffer");

    const bool hasMoments = buffers.g_scales_opacs.deviceBuffer.buffer != VK_NULL_HANDLE
            && buffers.g_scales_opacs.deviceSize() >= 8 * n;
    if (hasMoments) trainer.copyFromDevice(buffers.g_scales_opacs);

    size_t clampedGaussians = 0;
    size_t clampedAxes = 0;
    size_t nonFiniteAxes = 0;
    float maxScaleBefore = 0.0f;
    float maxScaleAfter = 0.0f;
    bool scaleChanged = false;
    bool momentsChanged = false;

    for (size_t i = 0; i < n; ++i) {
        bool gaussianChanged = false;
        for (int axis = 0; axis < 3; ++axis) {
            const size_t idx = 4 * i + static_cast<size_t>(axis);
            float value = buffers.scales_opacs[idx];
            if (std::isfinite(value) && value > 0.0f)
                maxScaleBefore = std::max(maxScaleBefore, value);
            else
                nonFiniteAxes++;

            float corrected = value;
            if (!std::isfinite(corrected) || corrected <= 0.0f)
                corrected = kMinTrainedGaussianScale;
            corrected = std::clamp(corrected,
                                   kMinTrainedGaussianScale,
                                   kMaxTrainedGaussianScale);
            if (!std::isfinite(value) || corrected != value) {
                buffers.scales_opacs[idx] = corrected;
                clampedAxes++;
                gaussianChanged = true;
                scaleChanged = true;
                if (hasMoments) {
                    // g_scales_opacs stores two float4 Adam moments per Gaussian.
                    buffers.g_scales_opacs[8 * i + static_cast<size_t>(axis)] = 0.0f;
                    buffers.g_scales_opacs[8 * i + 4 + static_cast<size_t>(axis)] = 0.0f;
                    momentsChanged = true;
                }
            }
            maxScaleAfter = std::max(maxScaleAfter, corrected);
        }
        if (gaussianChanged) clampedGaussians++;
    }

    if (scaleChanged) trainer.copyToDevice(buffers.scales_opacs);
    if (momentsChanged) trainer.copyToDevice(buffers.g_scales_opacs);

    std::ostringstream info;
    info << "Gaussian shape sanitize phase=" << (phase ? phase : "unknown")
         << " gaussians=" << n
         << " clampedGaussians=" << clampedGaussians
         << " clampedAxes=" << clampedAxes
         << " nonFiniteAxes=" << nonFiniteAxes
         << " maxScaleBeforeM=" << maxScaleBefore
         << " maxScaleAfterM=" << maxScaleAfter
         << " capM=" << kMaxTrainedGaussianScale
         << " momentsReset=" << (momentsChanged ? 1 : 0);
    logi(info.str());
}

'''
text = NATIVE.read_text(encoding="utf-8")
if "void sanitizeGaussianShape(VulkanGSTrainer& trainer" not in text:
    if sanitize_anchor not in text:
        raise SystemExit("v1.0.11 apply failed: sanitizer insertion anchor missing")
    text = text.replace(sanitize_anchor, sanitize_method + sanitize_anchor, 1)
    NATIVE.write_text(text, encoding="utf-8")

replace_once(
    NATIVE,
    '''        if (resumed) {\n            progress.send(7, "保存したGaussian・optimizer状態を復元しました…");\n            logi("Training checkpoint restored steps=" + std::to_string(completedBeforeRun)\n                 + " gaussians=" + std::to_string(buffers.num_splats));\n        } else {\n            spatialBudgetInitialCloud(trainer, buffers, kInitialGaussianBudget);\n            progress.send(7, "Depthから表面方向を推定しています…");\n            initializeSurfaceGaussians(trainer, buffers);\n        }\n\n        {\n''',
    '''        if (resumed) {\n            progress.send(7, "保存したGaussian・optimizer状態を復元しました…");\n            logi("Training checkpoint restored steps=" + std::to_string(completedBeforeRun)\n                 + " gaussians=" + std::to_string(buffers.num_splats));\n            // Salvage v1.0.10 exact checkpoints in place; do not throw away learned appearance.\n            sanitizeGaussianShape(trainer, buffers, "checkpoint_restore");\n        } else {\n            spatialBudgetInitialCloud(trainer, buffers, kInitialGaussianBudget);\n            progress.send(7, "Depthから表面方向を推定しています…");\n            initializeSurfaceGaussians(trainer, buffers);\n        }\n\n        {\n''',
    "checkpoint scale repair call",
)
replace_once(
    NATIVE,
    '''        progress.send(96, "学習結果を確認しています…");\n        ValidationMetrics validation = validationMetrics(trainer, uniforms, buffers, dataRoot);\n''',
    '''        // Audit and repair once more before hold-out rendering, PLY output and checkpoint save.\n        sanitizeGaussianShape(trainer, buffers, "pre_validation_output");\n        progress.send(96, "学習結果を確認しています…");\n        ValidationMetrics validation = validationMetrics(trainer, uniforms, buffers, dataRoot);\n''',
    "final scale audit call",
)

print("Applied v1.0.11 Gaussian shape correction (viewer + checkpoint sanitizer + shader generation)")
