#!/usr/bin/env python3
"""Build-time architecture checks for the PCS Android mobile 3DGS trainer.

These checks validate design invariants rather than a particular crash symptom. They keep the
Android trainer in the mobile-first regime and make the VkSplat cumsum correctness patch part of
the reproducible backend preparation.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "app/src/main/cpp/native_3dgs.cpp"
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
CUMSUM_PATCH = ROOT / "scripts/patch-vksplat-cumsum-android.py"
VENDORED_RENDERER = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp"
VENDORED_CUMSUM = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/slang/cumsum.slang"


def require(text: str, needle: str, reason: str) -> None:
    if needle not in text:
        raise SystemExit(f"mobile trainer architecture check failed: {reason}: missing {needle!r}")


def forbid(text: str, needle: str, reason: str) -> None:
    if needle in text:
        raise SystemExit(f"mobile trainer architecture check failed: {reason}: found {needle!r}")


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def verify_cumsum_hierarchy() -> None:
    # VkSplat's block scan reduces one partial sum per subgroup using one subgroup-level prefix scan.
    # Therefore a block may contain at most subgroup_size^2 threads. For Mali subgroup 16 this is
    # 256 threads, not VkSplat desktop's 1024-thread block (which would create 64 partial sums).
    subgroup = 16
    block = subgroup * subgroup
    assert block == 256
    assert block // subgroup == subgroup

    sizes = [1, 16, 17, 256, 257, 1024, 1025, 59_218, 87_777, 1_048_576]
    for n in sizes:
        if n <= 1024:
            continue
        level1_alloc = ceil_div(n, block)
        if n <= block * block:
            scan_uniform = level1_alloc
            assert scan_uniform <= block
            assert scan_uniform <= level1_alloc
            continue
        if n <= block * block * block:
            level2_alloc = ceil_div(level1_alloc, block)
            level1_scan_uniform = level1_alloc
            level2_scan_uniform = level2_alloc
            level1_offset_uniform = level1_alloc
            assert level2_scan_uniform <= block
            assert level1_scan_uniform <= level1_alloc
            assert level2_scan_uniform <= level2_alloc
            assert level1_offset_uniform <= level1_alloc


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

    require(patch, "deviceInfo.subgroupSize*deviceInfo.subgroupSize;",
            "renderer cumsum block must be subgroup squared")
    forbid(patch, "deviceInfo.subgroupSize*deviceInfo.subgroupSize*deviceInfo.subgroupSize;",
           "renderer cumsum block must not use subgroup cubed")
    require(patch, "SUBGROUP_SIZE*SUBGROUP_SIZE)",
            "shader cumsum block must be subgroup squared")
    require(patch, "if (laneId >= offset) {",
            "subgroup scan must not read a negative predecessor lane")
    require(patch, "num_blocks", "two-level cumsum must use reduced block count")
    require(patch, "level1_uniforms", "three-level cumsum level 1 must use reduced count")
    require(patch, "level2_uniforms", "three-level cumsum level 2 must use reduced count")
    verify_cumsum_hierarchy()

    if VENDORED_RENDERER.is_file():
        renderer = VENDORED_RENDERER.read_text(encoding="utf-8")
        require(renderer, "deviceInfo.subgroupSize*deviceInfo.subgroupSize;",
                "prepared renderer still uses invalid subgroup-16 block geometry")
        forbid(renderer, "deviceInfo.subgroupSize*deviceInfo.subgroupSize*deviceInfo.subgroupSize;",
               "prepared renderer still uses subgroup cubed")
        require(renderer, "block_uniforms", "prepared VkSplat backend is missing cumsum fix")
        require(renderer, "level1_uniforms", "prepared VkSplat backend is missing level-1 bounds")
        require(renderer, "level2_uniforms", "prepared VkSplat backend is missing level-2 bounds")

    if VENDORED_CUMSUM.is_file():
        cumsum = VENDORED_CUMSUM.read_text(encoding="utf-8")
        require(cumsum, "SUBGROUP_SIZE*SUBGROUP_SIZE)",
                "prepared cumsum shader does not bound block to one subgroup of subgroup sums")
        forbid(cumsum, "SUBGROUP_SIZE*SUBGROUP_SIZE*SUBGROUP_SIZE)",
               "prepared cumsum shader still uses desktop subgroup-cubed geometry")
        require(cumsum, "if (laneId >= offset) {",
                "prepared subgroup scan may read an invalid predecessor lane")

    print("PCS mobile trainer architecture checks passed")


if __name__ == "__main__":
    main()
