#!/usr/bin/env python3
"""Build-time architecture checks for the PCS Android mobile scanner and 3DGS trainer."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "app/src/main/cpp/native_3dgs.cpp"
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
SCANNER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/ScannerActivity.java"
CAPTURE = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DatasetCaptureManager.java"
CAMERA_CONFIG = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/CameraConfigSelector.java"
FINALIZER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DatasetFinalizer.java"
EXPORTER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/ColmapDatasetExporter.java"
MANIFEST = ROOT / "app/src/main/AndroidManifest.xml"
CUMSUM_PATCH = ROOT / "scripts/patch-vksplat-cumsum-android.py"
PREPARE = ROOT / "scripts/prepare-vksplat-android.sh"
VENDORED_RENDERER = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp"
VENDORED_CUMSUM = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/slang/cumsum.slang"


def require(text: str, needle: str, reason: str) -> None:
    if needle not in text:
        raise SystemExit(f"mobile architecture check failed: {reason}: missing {needle!r}")


def forbid(text: str, needle: str, reason: str) -> None:
    if needle in text:
        raise SystemExit(f"mobile architecture check failed: {reason}: found {needle!r}")


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def verify_cumsum_hierarchy() -> None:
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
            assert level2_alloc <= block
            assert level1_alloc <= level1_alloc


def main() -> None:
    native = NATIVE.read_text(encoding="utf-8")
    java = JAVA.read_text(encoding="utf-8")
    scanner = SCANNER.read_text(encoding="utf-8")
    capture = CAPTURE.read_text(encoding="utf-8")
    camera_config = CAMERA_CONFIG.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    exporter = EXPORTER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    patch = CUMSUM_PATCH.read_text(encoding="utf-8")
    prepare = PREPARE.read_text(encoding="utf-8")

    # Continuous scanner invariants. Capturing a useful RGB frame must not interrupt SharedCamera's
    # repeating preview with a Camera2 still request or force the person holding the phone to stop.
    require(scanner, "continuousCpuOnly=true",
            "SharedCamera session must contain only ARCore surfaces during continuous capture")
    forbid(scanner, "ImageReader.newInstance(",
           "continuous capture must not allocate a legacy JPEG ImageReader")
    forbid(scanner, "sessionSurfaces.add(jpegReader.getSurface())",
           "continuous capture must not attach a legacy JPEG Surface")
    forbid(scanner, "requestHighResolutionStill();",
           "frame loop must not fall back to Camera2 still capture")

    require(capture, "frame.acquireCameraImage()",
            "scanner must sample RGB from the current ARCore frame")
    require(capture, '"arcore_cpu_yuv_continuous"',
            "saved metadata must identify the continuous CPU-frame source")
    require(capture, 'quality.put("motion_is_hard_gate", false)',
            "motion may rank candidates but must not gate capture")
    require(capture, "calculateSharpness",
            "automatic keyframe selection must measure pixel sharpness")
    require(capture, "flushBestCandidateLocked",
            "scanner must select the best frame from a moving window")
    forbid(capture, "MAX_LINEAR_SPEED_MPS",
           "continuous capture must not require the user to slow below a threshold")
    forbid(capture, "MAX_ANGULAR_SPEED_DPS",
           "continuous capture must not require the user to stop rotating")
    forbid(capture, "waiting for lower blur",
           "scanner UI state must not wait for a stop-and-shoot moment")

    require(camera_config, "MAX_CONTINUOUS_CPU_PIXELS = 2_500_000L",
            "ARCore CPU stream needs an explicit continuous-frame bandwidth budget")
    require(camera_config, "max(Comparator",
            "camera selection must prefer useful CPU-frame resolution")

    require(finalizer, "if(capture==null)return null;",
            "ARCore CPU frames must not be remapped through Camera2 still calibration")
    require(finalizer, '"arcore_image_intrinsics"',
            "continuous frames must retain exact ARCore image intrinsics")

    require(exporter, "MAX_TRAIN_LONG_EDGE = 1000",
            "training images must keep the established mobile memory envelope")
    forbid(exporter, "TRAIN_DOWNSAMPLE",
           "continuous source resolution must not be blindly divided by a legacy still factor")
    require(exporter, "never upscale",
            "training export must document adaptive source scaling")

    require(manifest, 'android:icon="@mipmap/ic_launcher"',
            "app must ship an explicit launcher icon")
    require(manifest, 'android:roundIcon="@mipmap/ic_launcher"',
            "app must ship a round/adaptive launcher icon")

    # Mobile trainer invariants.
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
        raise SystemExit("mobile architecture check failed: MAX_TRAIN_STEPS exceeds 1000")

    # Mali subgroup-16 VkSplat cumsum invariants.
    require(patch, "deviceInfo.subgroupSize*deviceInfo.subgroupSize;",
            "renderer cumsum block replacement must use subgroup squared")
    require(patch, "SUBGROUP_SIZE*SUBGROUP_SIZE)",
            "shader cumsum block replacement must use subgroup squared")
    require(patch, "if (laneId >= offset) {",
            "subgroup scan replacement must not read a negative predecessor lane")
    require(patch, "num_blocks", "two-level cumsum must use reduced block count")
    require(patch, "level1_uniforms", "three-level cumsum level 1 must use reduced count")
    require(patch, "level2_uniforms", "three-level cumsum level 2 must use reduced count")
    verify_cumsum_hierarchy()

    patch_pos = prepare.find("python3 scripts/patch-vksplat-cumsum-android.py")
    compile_pos = prepare.find("python3 compile_shaders.py")
    if patch_pos < 0 or compile_pos < 0 or patch_pos >= compile_pos:
        raise SystemExit(
            "mobile architecture check failed: cumsum source patch must run before shader compilation")

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

    print("PCS continuous scanner + mobile trainer architecture checks passed")


if __name__ == "__main__":
    main()
