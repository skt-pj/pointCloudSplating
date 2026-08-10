#!/usr/bin/env python3
"""Harden VkSplat long-running Android training against per-Gaussian numerical divergence.

Pixel 10a v1.0.9 evidence showed a resumed 3000-step run stopping at
`projection invariant failed ... tiles=-31`. The upstream projection shader converts floating
bounds directly to uint16 tile coordinates and has no non-finite guard; MCMC relocation can also
produce a non-finite scale when its denominator becomes degenerate. This patch rejects invalid
projections before integer conversion, performs tile subtraction in signed space, makes MCMC
relocation transactional, and prevents Adam/exp/quaternion updates from publishing non-finite
parameters.
"""
from pathlib import Path

ROOT = Path("app/src/main/cpp/third_party/vksplat/vksplat/slang")
VERTEX = ROOT / "vertex_shader.slang"
MCMC = ROOT / "mcmc.slang"
FUSED = ROOT / "fused_projection_backward_optimizer.slang"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"long-training stability patch failed: {label}: anchor missing in {path}")
    path.write_text(text.replace(old, new, 1))


# 1) Projection: NaN/Inf comparisons are false, so the original early-reject path allowed an
# invalid covariance/radius to reach uint16 rectangle conversion. Reject every non-finite field
# that participates in rectangle construction or raster parameters.
replace_once(
    VERTEX,
    '''    if (splat.xyz_vs.z <= NEAR_CLIP ||\n        splat.inv_cov_vs_opac.w <= ALPHA_THRESHOLD ||\n        splat.det <= 0.0\n    ) {\n''',
    '''    if (splat.xyz_vs.z <= NEAR_CLIP ||\n        splat.inv_cov_vs_opac.w <= ALPHA_THRESHOLD ||\n        splat.det <= 0.0 ||\n        !isfinite(splat.xyz_vs.x) || !isfinite(splat.xyz_vs.y) || !isfinite(splat.xyz_vs.z) ||\n        !isfinite(splat.cov_vs.x) || !isfinite(splat.cov_vs.y) || !isfinite(splat.cov_vs.z) ||\n        !isfinite(splat.inv_cov_vs_opac.x) || !isfinite(splat.inv_cov_vs_opac.y) ||\n        !isfinite(splat.inv_cov_vs_opac.z) || !isfinite(splat.inv_cov_vs_opac.w) ||\n        !isfinite(splat.det)\n    ) {\n''',
    "projection non-finite early reject",
)

# Reject invalid radii before float->uint16 conversion and subtract tile coordinates only after
# widening to signed int32. This removes the exact uint16 underflow mechanism that produced -31.
old_rect = '''    rectangle rect_tile_space = get_rectangle_tile_space(\n        splat_dist.xyz_vs.xy, radii, grid_height, grid_width);\n\n    int32_t n_tiles = int32_t(rect_tile_space.max_x - rect_tile_space.min_x) * int32_t(rect_tile_space.max_y - rect_tile_space.min_y);\n    if (n_tiles == 0) {\n'''
new_rect = '''    if (!isfinite(radii.x) || !isfinite(radii.y) || radii.x < 0.0 || radii.y < 0.0) {\n        #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD\n            out_radii[g_idx] = 0;\n            out_tiles_touched[g_idx] = 0;\n          #if USE_EMULATED_INT64\n            out_rect_tile_space[2*g_idx+0] = 0;\n            out_rect_tile_space[2*g_idx+1] = 0;\n          #else\n            out_rect_tile_space[g_idx] = int64_t(0);\n          #endif\n        #endif\n        return;\n    }\n\n    rectangle rect_tile_space = get_rectangle_tile_space(\n        splat_dist.xyz_vs.xy, radii, grid_height, grid_width);\n\n    if (rect_tile_space.max_x < rect_tile_space.min_x ||\n        rect_tile_space.max_y < rect_tile_space.min_y) {\n        #if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD\n            out_radii[g_idx] = 0;\n            out_tiles_touched[g_idx] = 0;\n          #if USE_EMULATED_INT64\n            out_rect_tile_space[2*g_idx+0] = 0;\n            out_rect_tile_space[2*g_idx+1] = 0;\n          #else\n            out_rect_tile_space[g_idx] = int64_t(0);\n          #endif\n        #endif\n        return;\n    }\n\n    int32_t tile_dx = int32_t(rect_tile_space.max_x) - int32_t(rect_tile_space.min_x);\n    int32_t tile_dy = int32_t(rect_tile_space.max_y) - int32_t(rect_tile_space.min_y);\n    int32_t n_tiles = tile_dx * tile_dy;\n    if (n_tiles <= 0 || uint32_t(n_tiles) > grid_width * grid_height) {\n'''
replace_once(VERTEX, old_rect, new_rect, "signed tile rectangle arithmetic")

# 2) MCMC relocation: calculate candidates first and publish only when the denominator,
# coefficient, opacity and scale remain finite/positive. Invalid relocation now becomes a no-op.
old_relocation = '''void relocation(inout float opacity, inout float3 scale, int n_idx) {\n    n_idx = min(n_idx, 50);  // log_factorial only fits to 50\n\n    float new_opacity = 1.0 - pow(1.0-opacity, 1.0 / n_idx);\n\n    float denom_sum = 0.0;\n    for (int i = 1; i <= n_idx; ++i) {\n        for (int k = 0; k <= (i - 1); ++k) {\n            denom_sum += binom(i-1, k) * \n                (cos(M_PI*k) / sqrt(k+1)) *  // (-1)^k / sqrt(k+1)\n                pow(new_opacity, k+1);\n        }\n    }\n    float coeff = (opacity / denom_sum);\n\n    opacity = new_opacity;\n    scale = coeff * scale;\n}\n'''
new_relocation = '''void relocation(inout float opacity, inout float3 scale, int n_idx) {\n    n_idx = max(1, min(n_idx, 50));  // log_factorial only fits to 50\n\n    float base_opacity = clamp(opacity, 1e-6, 1.0-1e-6);\n    float3 base_scale = scale;\n    if (!isfinite(base_opacity) ||\n        !isfinite(base_scale.x) || !isfinite(base_scale.y) || !isfinite(base_scale.z) ||\n        base_scale.x <= 0.0 || base_scale.y <= 0.0 || base_scale.z <= 0.0)\n        return;\n\n    float new_opacity = 1.0 - pow(1.0-base_opacity, 1.0 / n_idx);\n    if (!isfinite(new_opacity) || new_opacity <= 0.0 || new_opacity >= 1.0)\n        return;\n\n    float denom_sum = 0.0;\n    for (int i = 1; i <= n_idx; ++i) {\n        for (int k = 0; k <= (i - 1); ++k) {\n            denom_sum += binom(i-1, k) *\n                (cos(M_PI*k) / sqrt(k+1)) *  // (-1)^k / sqrt(k+1)\n                pow(new_opacity, k+1);\n        }\n    }\n    if (!isfinite(denom_sum) || abs(denom_sum) < 1e-12)\n        return;\n\n    float coeff = base_opacity / denom_sum;\n    float3 new_scale = coeff * base_scale;\n    if (!isfinite(coeff) || coeff <= 0.0 ||\n        !isfinite(new_scale.x) || !isfinite(new_scale.y) || !isfinite(new_scale.z) ||\n        new_scale.x <= 0.0 || new_scale.y <= 0.0 || new_scale.z <= 0.0)\n        return;\n\n    opacity = new_opacity;\n    scale = new_scale;\n}\n'''
replace_once(MCMC, old_relocation, new_relocation, "transactional finite MCMC relocation")

# 3) Adam: a single non-finite gradient must not poison persistent parameters/moments. Roll the
# update back atomically if x/g1/g2 becomes non-finite. Slang supports vector isfinite + all().
old_adam = '''    const float g1_sc = 1.0 / (1.0 - pow(beta_1, step));\n    const float g2_sc = 1.0 / (1.0 - pow(beta_2, step));\n    g1 = lerp(g1, v, 1.0-beta_1);\n    g2 = lerp(g2, v*v, 1.0-beta_2);\n    vector<float,dim> g1_t = g1 * g1_sc;\n    vector<float,dim> g2_t = g2 * g2_sc;\n    x = x - lr * g1_t / (sqrt(g2_t) + eps);\n}\n'''
new_adam = '''    vector<float,dim> old_x = x;\n    vector<float,dim> old_g1 = g1;\n    vector<float,dim> old_g2 = g2;\n\n    const float g1_sc = 1.0 / (1.0 - pow(beta_1, step));\n    const float g2_sc = 1.0 / (1.0 - pow(beta_2, step));\n    g1 = lerp(g1, v, 1.0-beta_1);\n    g2 = lerp(g2, v*v, 1.0-beta_2);\n    vector<float,dim> g1_t = g1 * g1_sc;\n    vector<float,dim> g2_t = g2 * g2_sc;\n    x = x - lr * g1_t / (sqrt(g2_t) + eps);\n\n    if (!all(isfinite(x)) || !all(isfinite(g1)) || !all(isfinite(g2))) {\n        x = old_x;\n        g1 = old_g1;\n        g2 = old_g2;\n    }\n}\n'''
replace_once(FUSED, old_adam, new_adam, "Adam finite rollback")

# Quaternion normalize(0) and exp(very-large-u) are separate post-Adam sources of NaN/Inf.
replace_once(
    FUSED,
    '''        optimizer_update(x, v, g1, g2, uniforms.step, uniforms.lr_quats);\n        rotations[g_idx] = bool(NORMALIZE_QUATS_IN_OPTIMIZER) ? normalize(x) : x;\n''',
    '''        optimizer_update(x, v, g1, g2, uniforms.step, uniforms.lr_quats);\n        float q_len = length(x);\n        rotations[g_idx] = bool(NORMALIZE_QUATS_IN_OPTIMIZER)\n            ? ((isfinite(q_len) && q_len > 1e-8) ? x / q_len : gauss.rotations)\n            : x;\n''',
    "safe quaternion normalization",
)

replace_once(
    FUSED,
    '''        // map\n        x.xyz = exp(u.xyz);  // scale: x = exp(u)\n        x.w = 1.0 / (1.0+exp(-u.w));  // opacity: x = sigmoid(u)\n''',
    '''        // map. These bounds are numerical overflow guards, not scene-quality limits:\n        // exp(-18)..exp(8) spans ~1.5e-8..3e3 in normalized world units.\n        u.xyz = clamp(u.xyz, float3(-18.0), float3(8.0));\n        u.w = clamp(u.w, -20.0, 20.0);\n        x.xyz = exp(u.xyz);  // scale: x = exp(u)\n        x.w = 1.0 / (1.0+exp(-u.w));  // opacity: x = sigmoid(u)\n''',
    "scale/opacity overflow guard",
)

print("Patched VkSplat long-training projection, MCMC and optimizer numerical stability")
