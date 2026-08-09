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


def require(text: str, needle: str, reason: str) -> None:
    if needle not in text:
        raise SystemExit(f"mobile architecture check failed: {reason}: missing {needle!r}")


def forbid(text: str, needle: str, reason: str) -> None:
    if needle in text:
        raise SystemExit(f"mobile architecture check failed: {reason}: found {needle!r}")


def verify_cpu_cumsum_reference() -> None:
    # Mirrors the inclusive int32 prefix-sum contract used by VkSplat. Exercise the sizes observed on
    # Pixel 10a plus the configured initial and maximum Gaussian budgets.
    for n in [1, 9, 257, 37_635, 90_000, 120_000]:
        values = [((i * 17) % 7) for i in range(n)]
        running = 0
        out = []
        for value in values:
            running += value
            assert -(2**31) <= running < 2**31
            out.append(running)
        assert len(out) == n
        assert out[-1] == sum(values)
        if n > 1:
            assert all(out[i] >= out[i - 1] for i in range(1, n))


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
    require(scanner, "surfaceView.setPreserveEGLContextOnPause(true)",
            "resumable SharedCamera scans must preserve the camera EGL texture")
    require(scanner, "releaseGlForModel",
            "EGL context should be released only when model processing needs the GPU")
    require(scanner, 'append("lastModelError=")',
            "3DGS failure reason must remain visible in copied diagnostics")

    require(capture, "frame.acquireCameraImage()",
            "scanner must sample RGB from the current ARCore frame")
    require(capture, '"arcore_cpu_yuv_continuous"',
            "saved metadata must identify the continuous CPU-frame source")
    require(capture, 'quality.put("motion_is_hard_gate", false)',
            "motion may rank candidates but must not gate capture")
    require(capture, "calculateSharpness",
            "automatic keyframe selection must measure pixel sharpness")
    require(capture, "flushBestCandidateLocked",
            "scanner must select the best frame from each selection window")
    require(capture, "SELECTION_WINDOW_NS = 750_000_000L",
            "candidate images must compete in a fixed 750 ms selection window")
    require(capture, "shouldCloseSelectionWindow(long timestampNs)",
            "selection-window closure must be time based rather than movement triggered")
    forbid(capture, "NEW_WINDOW_TRANSLATION_METERS",
           "camera translation must not close the selection window early")
    forbid(capture, "NEW_WINDOW_ROTATION_DEGREES",
           "camera rotation must not close the selection window early")
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
    require(native, "validateProjectionInvariants", "projection/prefix-sum semantics must be checked")
    require(native, "cpuTiles=", "tile-sum comparison must remain observable on failure")
    require(native, "kTileWorkingSetBudgetBytes", "tile memory must be budgeted before allocation")
    forbid(native, "executeDefaultPostBackward(", "desktop Default densification is not the PCS mobile policy")
    forbid(native, "kAndroidDensifySoftCap", "post-hoc densification soft caps are not allowed")

    require(java, "BASE_TRAIN_STEPS = 750", "mobile schedule baseline changed unexpectedly")
    require(java, "MAX_TRAIN_STEPS = 1_000", "mobile schedule must remain bounded")
    require(java, '"pcs_mobile_vulkan_trainer_v1"', "result metadata must identify the PCS trainer")
    require(java, 'SHADER_CACHE = "vksplat_shader_41cff93b_cpuscan_v4"',
            "CPU cumsum build must use a fresh on-device shader cache generation")
    require(java, '" cumsum=cpu_scan"',
            "runtime diagnostics must identify the CPU cumsum path")
    forbid(java, "DEFAULT_TRAIN_STEPS = 6_000", "desktop-length training schedule returned")

    m = re.search(r"MAX_TRAIN_STEPS\s*=\s*([\d_]+)", java)
    if not m or int(m.group(1).replace("_", "")) > 1000:
        raise SystemExit("mobile architecture check failed: MAX_TRAIN_STEPS exceeds 1000")

    # Pixel 10a cumsum must not create or execute a cumsum compute pipeline. The prior subgroup
    # implementation produced wrong totals, and the replacement Slang shared-memory pipeline caused
    # CRASH_NATIVE during vkCreateComputePipelines on Mali-G715.
    require(patch, "CPU prefix scan", "Android cumsum patch must document the CPU scan")
    require(patch, "copyFromDevice(input_buffer);",
            "CPU cumsum must read the projected counts back from Vulkan")
    require(patch, "int64_t running = 0;",
            "CPU cumsum must accumulate without int32 intermediate overflow")
    require(patch, "copyToDevice(output_buffer);",
            "CPU cumsum must upload inclusive offsets for downstream Vulkan kernels")
    require(patch, "buffers.index_buffer_offset.back()",
            "tile-index count must come from the already computed CPU prefix value")
    require(patch, "Do not create cumsum Vulkan pipelines",
            "cumsum pipeline creation must stay disabled on Android")
    verify_cpu_cumsum_reference()

    patch_pos = prepare.find("python3 scripts/patch-vksplat-cumsum-android.py")
    compile_pos = prepare.find("python3 compile_shaders.py")
    if patch_pos < 0 or compile_pos < 0 or patch_pos >= compile_pos:
        raise SystemExit(
            "mobile architecture check failed: VkSplat renderer patch must run before final backend build")

    if VENDORED_RENDERER.is_file():
        renderer = VENDORED_RENDERER.read_text(encoding="utf-8")
        forbid(renderer, "createComputePipeline(pipeline_cumsum.",
               "prepared renderer still creates the Pixel-crashing cumsum compute pipeline")
        require(renderer, "copyFromDevice(input_buffer);",
                "prepared renderer is missing CPU cumsum device readback")
        require(renderer, "copyToDevice(output_buffer);",
                "prepared renderer is missing CPU cumsum device upload")
        require(renderer, "buffers.index_buffer_offset.back()",
                "prepared renderer still re-reads cumsum total from the GPU")
        start = renderer.find("void VulkanGSRenderer::executeCumsum(")
        end = renderer.find("void VulkanGSRenderer::executeCalculateIndexBufferOffset(", start)
        if start < 0 or end < 0:
            raise SystemExit("mobile architecture check failed: prepared cumsum function not found")
        cumsum_fn = renderer[start:end]
        forbid(cumsum_fn, "executeCompute(",
               "prepared Android cumsum must not dispatch a compute shader")
        require(cumsum_fn, "int64_t running = 0;",
                "prepared Android cumsum must use exact host accumulation")

    print("PCS continuous scanner + mobile trainer architecture checks passed")


if __name__ == "__main__":
    main()
