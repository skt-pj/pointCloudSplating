#!/usr/bin/env python3
"""Build-time contract checks for Pixel 10a long-training numerical stability."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLANG = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/slang"
VERTEX = SLANG / "vertex_shader.slang"
MCMC = SLANG / "mcmc.slang"
FUSED = SLANG / "fused_projection_backward_optimizer.slang"
TRAINER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
PATCH = ROOT / "scripts/patch-vksplat-long-training-stability.py"
HINT_PATCH = ROOT / "scripts/patch-vksplat-legacy-resume-hint-android.py"
VERSION = ROOT / "version.properties"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"long-training stability check failed: {why}: missing {needle!r}")


def forbid(text: str, needle: str, why: str) -> None:
    if needle in text:
        raise SystemExit(f"long-training stability check failed: {why}: found {needle!r}")


def main() -> None:
    for path in (VERTEX, MCMC, FUSED, TRAINER, PATCH, HINT_PATCH, VERSION):
        if not path.is_file():
            raise SystemExit(f"long-training stability check failed: missing {path}")

    vertex = VERTEX.read_text(encoding="utf-8")
    mcmc = MCMC.read_text(encoding="utf-8")
    fused = FUSED.read_text(encoding="utf-8")
    trainer = TRAINER.read_text(encoding="utf-8")
    patch = PATCH.read_text(encoding="utf-8")
    hint_patch = HINT_PATCH.read_text(encoding="utf-8")
    version = VERSION.read_text(encoding="utf-8")

    require(vertex, "!isfinite(splat.xyz_vs.x)", "non-finite projection guard missing")
    require(vertex, "!isfinite(radii.x)", "non-finite radius guard missing")
    require(vertex,
            "int32_t tile_dx = int32_t(rect_tile_space.max_x) - int32_t(rect_tile_space.min_x)",
            "tile subtraction is not widened before subtraction")
    require(vertex, "n_tiles <= 0", "invalid tile count reject missing")
    forbid(vertex, "int32_t(rect_tile_space.max_x - rect_tile_space.min_x)",
           "uint16 tile subtraction underflow path returned")

    require(mcmc, "abs(denom_sum) < 1e-12", "degenerate relocation denominator guard missing")
    require(mcmc, "float3 new_scale = coeff * base_scale", "transactional relocation missing")
    require(mcmc, "!isfinite(new_scale.x)", "relocation finite-scale guard missing")

    require(fused, "vector<float,dim> old_x = x", "Adam rollback snapshot missing")
    require(fused, "!all(isfinite(x))", "Adam finite check missing")
    require(fused, "float q_len = length(x)", "quaternion finite normalization guard missing")
    if ("clamp(u.xyz, float3(-18.0), float3(8.0))" not in fused
            and "clamp(u.xyz, float3(-18.0), float3(-2.8134108))" not in fused):
        raise SystemExit("long-training stability check failed: scale exp guard missing")
    require(fused, "u.w = clamp(u.w, -20.0, 20.0)", "opacity logit overflow guard missing")

    require(hint_patch, 'runpy.run_path("scripts/patch-vksplat-long-training-stability.py"',
            "stability patch is not wired before shader compilation")
    require(patch, "uint16 underflow", "failure mechanism is not documented")
    require(trainer, "numeric=finite_projection+mcmc_relocation+adam_rollback",
            "runtime stability diagnostic marker missing")

    values = dict(line.split("=", 1) for line in version.splitlines() if "=" in line)
    try:
        version_code = int(values.get("VERSION_CODE", "0"))
    except ValueError:
        raise SystemExit("long-training stability check failed: invalid VERSION_CODE")
    if version_code < 47:
        raise SystemExit("long-training stability check failed: numerical fix requires VERSION_CODE >= 47")

    print("Long-training numerical-stability checks passed")


if __name__ == "__main__":
    main()
