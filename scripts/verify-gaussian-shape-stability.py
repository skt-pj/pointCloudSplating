#!/usr/bin/env python3
"""Build-time contract checks for PCS v1.0.11 needle-splat correction."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLANG = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/slang"
MCMC = SLANG / "mcmc.slang"
FUSED = SLANG / "fused_projection_backward_optimizer.slang"
TRAINER_CPP = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_trainer.cpp"
NATIVE = ROOT / "app/src/main/cpp/native_3dgs.cpp"
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
MODEL = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/GaussianModel.java"
HINT = ROOT / "scripts/patch-vksplat-legacy-resume-hint-android.py"
SHAPE_PATCH = ROOT / "scripts/patch-vksplat-shape-stability.py"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"Gaussian shape stability check failed: {why}: missing {needle!r}")


def forbid(text: str, needle: str, why: str) -> None:
    if needle in text:
        raise SystemExit(f"Gaussian shape stability check failed: {why}: found {needle!r}")


def main() -> None:
    for path in (MCMC, FUSED, TRAINER_CPP, NATIVE, JAVA, MODEL, HINT, SHAPE_PATCH, VERSION):
        if not path.is_file():
            raise SystemExit(f"Gaussian shape stability check failed: missing {path}")

    mcmc = MCMC.read_text(encoding="utf-8")
    fused = FUSED.read_text(encoding="utf-8")
    trainer = TRAINER_CPP.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    java = JAVA.read_text(encoding="utf-8")
    model = MODEL.read_text(encoding="utf-8")
    hint = HINT.read_text(encoding="utf-8")
    patch = SHAPE_PATCH.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    require(mcmc, "new_scale = clamp(new_scale, float3(1.0e-4), float3(6.0e-2))",
            "MCMC relocation can still create meter-scale splats")
    require(fused, "clamp(u.xyz, float3(-18.0), float3(-2.8134108))",
            "optimizer does not cap Gaussian axes at 6 cm")
    forbid(fused, "clamp(u.xyz, float3(-18.0), float3(8.0))",
           "v1.0.10 overflow-only scale bound is still active")

    require(trainer, "PCS_CUMULATIVE_MCMC_SCHEDULE",
            "resumed MCMC still restarts its refinement schedule")
    require(trainer, "const int schedule_step = static_cast<int>(resume_training_step) + step",
            "MCMC cumulative step is not derived from checkpoint age")
    require(trainer,
            "schedule_step < config.refine_stop_iter && schedule_step > config.refine_start_iter",
            "MCMC refinement gate still uses local step")
    require(trainer, "schedule_step % config.refine_every == 0",
            "MCMC cadence still uses local step")

    require(native, "kMaxTrainedGaussianScale = 6.0e-2f",
            "runtime checkpoint repair has no 6 cm cap")
    require(native, "void sanitizeGaussianShape(VulkanGSTrainer& trainer",
            "existing checkpoints are not repaired")
    require(native, 'sanitizeGaussianShape(trainer, buffers, "checkpoint_restore")',
            "v1.0.10 checkpoint is not sanitized before continuation")
    require(native, 'sanitizeGaussianShape(trainer, buffers, "pre_validation_output")',
            "final model is not audited before validation/output")
    require(native, "buffers.g_scales_opacs[8 * i + static_cast<size_t>(axis)] = 0.0f",
            "Adam scale moment is not reset when an axis is clamped")
    require(native, "maxScaleBeforeM=", "shape repair diagnostics are incomplete")

    require(model, "MAX_WORLD_SCALE_METERS = 0.06f",
            "viewer cannot immediately suppress old needle splats")
    require(model, "MAX_WORLD_SCALE_METERS / Math.max(radius, 1e-6f)",
            "viewer cap is not converted from metric world-space")
    require(model, "clampedScaleAxes", "viewer does not report scale repairs")

    require(hint, 'runpy.run_path("scripts/patch-vksplat-shape-stability.py"',
            "shape patch is not wired before shader compilation")
    require(patch, "PCS_CUMULATIVE_MCMC_SCHEDULE", "shape patch lacks cumulative schedule marker")
    require(java, 'SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v9"',
            "shape-changing shaders are not isolated in cache v9")
    require(java, "shape=metric_scale_cap_6cm+mcmc_cumulative+resume_sanitize",
            "runtime shape generation marker missing")
    require(version, "VERSION_NAME=1.0.11", "versionName mismatch")
    require(version, "VERSION_CODE=48", "versionCode mismatch")

    print("Gaussian shape stability checks passed: metric 6cm cap, cumulative MCMC, checkpoint/viewer repair")


if __name__ == "__main__":
    main()
