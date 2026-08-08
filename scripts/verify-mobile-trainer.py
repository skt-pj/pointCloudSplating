#!/usr/bin/env python3
"""Build-time architecture checks for the PCS Android mobile 3DGS trainer.

These checks intentionally validate design invariants rather than a particular crash symptom. They
keep the Android trainer in the mobile-first regime selected from PocketGS/Taming/gsplat references
and make the VkSplat cumsum correctness patch part of the reproducible backend preparation.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "app/src/main/cpp/native_3dgs.cpp"
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
CUMSUM_PATCH = ROOT / "scripts/patch-vksplat-cumsum-android.py"
VENDORED_RENDERER = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp"


def require(text: str, needle: str, reason: str) -> None:
    if needle not in text:
        raise SystemExit(f"mobile trainer architecture check failed: {reason}: missing {needle!r}")


def forbid(text: str, needle: str, reason: str) -> None:
    if needle in text:
        raise SystemExit(f"mobile trainer architecture check failed: {reason}: found {needle!r}")


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def verify_cumsum_hierarchy() -> None:
    # The bug caught on Mali came from using the original element count while scanning the smaller
    # block-sum buffer. For representative sizes and plausible Vulkan subgroup-derived block sizes,
    # every recursive level must dispatch only the number of elements actually allocated there.
    sizes = [1, 16, 17, 1024, 1025, 59_218, 1_048_576]
    blocks = [128, 256, 512, 1024]
    for block in blocks:
        for n in sizes:
            if n <= 1024:
                continue
            level1 = ceil_div(n, block)
            if n <= block * block:
                dispatched = level1
                allocated = level1
                if dispatched > allocated:
                    raise AssertionError((block, n, dispatched, allocated))
            elif n <= block * block * block:
                level2 = ceil_div(level1, block)
                if level1 > level1 or level2 > level2:
                    raise AssertionError((block, n, level1, level2))
                # add_block_offsets at level 1 operates on level1 values, not n values.
                if level1 > ceil_div(n, block):
                    raise AssertionError((block, n, level1))


def main() -> None:
    native = NATIVE.read_text(encoding="utf-8")
    java = JAVA.read_text(encoding="utf-8")
    patch = CUMSUM_PATCH.read_text(encoding="utf-8")

    require(native, "TrainerConfig::Strategy::MCMC", "density control must be budget-aware")
    require(native, "kGaussianBudget = 120'000", "Gaussian count must have an explicit budget")
    require(native, "kNormalNeighbors = 16", "surface normal initialization must use K=16")
    require(native, "kScaleNeighbors = 3", "surface scale initialization must use K=3")
    require(native, "initializeSurfaceGaussians", "surface-aware initialization is required")
    require(native, "validateProjectionInvariants", "GPU projection/cumsum must be checked semantically")
    require(native, "cpuTiles=", "CPU/GPU tile-sum comparison must remain observable")
    require(native, "kTileWorkingSetBudgetBytes", "tile memory must be budgeted before allocation")
    forbid(native, "executeDefaultPostBackward(", "desktop Default densification is not the PCS mobile policy")
    forbid(native, "kAndroidDensifySoftCap", "post-hoc densification soft caps are not allowed")

    require(java, "BASE_TRAIN_STEPS = 750", "mobile schedule baseline changed unexpectedly")
    require(java, "MAX_TRAIN_STEPS = 1_000", "mobile schedule must remain bounded")
    require(java, '"pcs_mobile_vulkan_trainer_v1"', "result metadata must identify the PCS trainer")
    forbid(java, "DEFAULT_TRAIN_STEPS = 6_000", "desktop-length training schedule returned")

    m = re.search(r"MAX_TRAIN_STEPS\s*=\s*([\d_]+)", java)
    if not m or int(m.group(1).replace("_", "")) > 1000:
        raise SystemExit("mobile trainer architecture check failed: MAX_TRAIN_STEPS exceeds 1000")

    require(patch, "num_blocks", "two-level cumsum must use reduced block count")
    require(patch, "level1_uniforms", "three-level cumsum level 1 must use reduced count")
    require(patch, "level2_uniforms", "three-level cumsum level 2 must use reduced count")
    verify_cumsum_hierarchy()

    if VENDORED_RENDERER.is_file():
        renderer = VENDORED_RENDERER.read_text(encoding="utf-8")
        require(renderer, "block_uniforms", "prepared VkSplat backend is missing cumsum fix")
        require(renderer, "level1_uniforms", "prepared VkSplat backend is missing level-1 bounds")
        require(renderer, "level2_uniforms", "prepared VkSplat backend is missing level-2 bounds")

    print("PCS mobile trainer architecture checks passed")


if __name__ == "__main__":
    main()
