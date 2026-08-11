#!/usr/bin/env python3
"""Keep PCS room-scale Gaussians geometrically bounded during resumed MCMC training.

v1.0.10 prevented NaN/Inf and tile-index underflow, but it intentionally left scale growth
essentially unbounded (exp(8)). On ARCore PCS datasets world units are meters because
normalize_world_space(..., false) is used. The Depth-conditioned initializer starts tangential
scales in [0.0005, 0.05] m, so a 0.06 m training cap prevents meter-long needle splats while
leaving normal surface adaptation room.

The exact-checkpoint path also stores the cumulative user-visible step. MCMC refinement must use
that cumulative step rather than restarting its 0-based schedule every time the user presses
Additional training.
"""
from pathlib import Path

SLANG = Path("app/src/main/cpp/third_party/vksplat/vksplat/slang")
MCMC = SLANG / "mcmc.slang"
FUSED = SLANG / "fused_projection_backward_optimizer.slang"
TRAINER = Path("app/src/main/cpp/third_party/vksplat/vksplat/src/gs_trainer.cpp")

MIN_SCALE_M = "1.0e-4"
MAX_SCALE_M = "6.0e-2"
MAX_LOG_SCALE = "-2.8134108"  # ln(0.06)


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"shape stability patch failed: {label}: anchor missing in {path}")
    path.write_text(text.replace(old, new, 1))


# Relocation is allowed to change scale, but a merely finite denominator can still produce a huge
# coefficient. Clamp the published candidate in meter-space before it can become a long needle.
old_relocation = '''    float3 new_scale = coeff * base_scale;\n    if (!isfinite(coeff) || coeff <= 0.0 ||\n        !isfinite(new_scale.x) || !isfinite(new_scale.y) || !isfinite(new_scale.z) ||\n        new_scale.x <= 0.0 || new_scale.y <= 0.0 || new_scale.z <= 0.0)\n        return;\n\n    opacity = new_opacity;\n    scale = new_scale;\n'''
new_relocation = f'''    float3 new_scale = coeff * base_scale;\n    if (!isfinite(coeff) || coeff <= 0.0 ||\n        !isfinite(new_scale.x) || !isfinite(new_scale.y) || !isfinite(new_scale.z) ||\n        new_scale.x <= 0.0 || new_scale.y <= 0.0 || new_scale.z <= 0.0)\n        return;\n\n    // PCS ARCore world units are meters. Keep relocation inside the same room-scale envelope\n    // as the Depth-conditioned surface initializer instead of accepting arbitrarily large finite values.\n    new_scale = clamp(new_scale, float3({MIN_SCALE_M}), float3({MAX_SCALE_M}));\n    opacity = new_opacity;\n    scale = new_scale;\n'''
replace_once(MCMC, old_relocation, new_relocation, "bounded MCMC relocation scale")

# v1.0.10's +8 upper log bound was only an overflow guard: exp(8) is ~2981 world units. Replace it
# with the actual PCS room-scale geometric contract while keeping the wide lower range for thin axes.
old_map = '''        // map. These bounds are numerical overflow guards, not scene-quality limits:\n        // exp(-18)..exp(8) spans ~1.5e-8..3e3 in normalized world units.\n        u.xyz = clamp(u.xyz, float3(-18.0), float3(8.0));\n        u.w = clamp(u.w, -20.0, 20.0);\n        x.xyz = exp(u.xyz);  // scale: x = exp(u)\n'''
new_map = f'''        // map. PCS keeps ARCore metric world coordinates, so cap each Gaussian axis at 6 cm.\n        // This is a geometric quality guard, not merely an overflow guard.\n        u.xyz = clamp(u.xyz, float3(-18.0), float3({MAX_LOG_SCALE}));\n        u.w = clamp(u.w, -20.0, 20.0);\n        x.xyz = exp(u.xyz);  // scale: x = exp(u)\n'''
replace_once(FUSED, old_map, new_map, "metric optimizer scale cap")

# Use cumulative training age for MCMC refinement/sorting. Adam already has its own persisted
# optimizer age in the checkpoint patch and must remain separate for legacy PLY migration.
source = TRAINER.read_text()
start_marker = "void VulkanGSTrainer::executeMCMCPostBackward("
end_marker = "void VulkanGSTrainer::executeMortonSorting("
start = source.find(start_marker)
end = source.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("shape stability patch failed: MCMC trainer section missing")
section = source[start:end]

schedule_anchor = '''    size_t num_splats = buffers.num_splats;\n\n    Uniform32_t uniforms[5];\n'''
schedule_replacement = '''    size_t num_splats = buffers.num_splats;\n    // PCS_CUMULATIVE_MCMC_SCHEDULE: continuation must not restart relocation/densification at 0.\n    const int schedule_step = static_cast<int>(resume_training_step) + step;\n\n    Uniform32_t uniforms[5];\n'''
if "PCS_CUMULATIVE_MCMC_SCHEDULE" not in section:
    if schedule_anchor not in section:
        raise SystemExit("shape stability patch failed: MCMC schedule anchor missing")
    section = section.replace(schedule_anchor, schedule_replacement, 1)

condition_old = '''        step < config.refine_stop_iter && step > config.refine_start_iter\n        && step % config.refine_every == 0\n'''
condition_new = '''        schedule_step < config.refine_stop_iter && schedule_step > config.refine_start_iter\n        && schedule_step % config.refine_every == 0\n'''
if condition_new not in section:
    if condition_old not in section:
        raise SystemExit("shape stability patch failed: MCMC refinement condition missing")
    section = section.replace(condition_old, condition_new, 1)

sort_old = '''    if (step % config.refine_every == 0) {\n        executeMortonSorting(renderer_uniforms, buffers);\n    }\n'''
sort_new = '''    if (schedule_step % config.refine_every == 0) {\n        executeMortonSorting(renderer_uniforms, buffers);\n    }\n'''
if sort_new not in section:
    if sort_old not in section:
        raise SystemExit("shape stability patch failed: MCMC Morton schedule missing")
    section = section.replace(sort_old, sort_new, 1)

source = source[:start] + section + source[end:]
TRAINER.write_text(source)
print("Patched VkSplat Gaussian shape stability: 6cm scale cap + cumulative MCMC schedule")
