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
PIPELINE_DIAG_PATCH = ROOT / "scripts/patch-vksplat-pipeline-diagnostics-android.py"
CUMSUM_GLSL = ROOT / "scripts/vksplat-android-shaders/cumsum.comp"
PREPARE = ROOT / "scripts/prepare-vksplat-android.sh"
VENDORED_RENDERER = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp"
VENDORED_PIPELINE = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_pipeline.cpp"


def require(text: str, needle: str, reason: str) -> None:
    if needle not in text:
        raise SystemExit(f"mobile architecture check failed: {reason}: missing {needle!r}")


def forbid(text: str, needle: str, reason: str) -> None:
    if needle in text:
        raise SystemExit(f"mobile architecture check failed: {reason}: found {needle!r}")


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def verify_cumsum_hierarchy() -> None:
    block = 256
    for n in [1, 17, 255, 256, 257, 1025, 37_635, 90_000, 120_000, 1_048_576]:
        if n <= block:
            continue
        level1 = ceil_div(n, block)
        if n <= block * block:
            assert level1 <= block
            continue
        if n <= block * block * block:
            level2 = ceil_div(level1, block)
            assert level1 > block
            assert level2 <= block
            continue
        raise AssertionError(f"test size exceeds supported hierarchy: {n}")


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
    pipeline_diag = PIPELINE_DIAG_PATCH.read_text(encoding="utf-8")
    cumsum_glsl = CUMSUM_GLSL.read_text(encoding="utf-8")
    prepare = PREPARE.read_text(encoding="utf-8")

    # Continuous scanner invariants.
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
    require(java, 'SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_v5"',
            "glslc cumsum build must use a fresh on-device shader cache generation")
    require(java, '" cumsum=glslc256"',
            "runtime diagnostics must identify the GPU cumsum compiler path")
    require(java, "logCumsumShaderIdentity", "runtime must log exact cumsum SPIR-V identities")
    require(java, 'MessageDigest.getInstance("SHA-256")',
            "runtime cumsum diagnostics must include SHA-256")
    forbid(java, "cumsum=cpu_scan", "CPU cumsum workaround must not remain")
    forbid(java, "DEFAULT_TRAIN_STEPS = 6_000", "desktop-length training schedule returned")

    m = re.search(r"MAX_TRAIN_STEPS\s*=\s*([\d_]+)", java)
    if not m or int(m.group(1).replace("_", "")) > 1000:
        raise SystemExit("mobile architecture check failed: MAX_TRAIN_STEPS exceeds 1000")

    # GPU cumsum must be subgroup-independent and its hierarchy must match the shader local size.
    require(patch, "const size_t block = 256;",
            "renderer cumsum must use the shader's fixed workgroup size")
    require(patch, "num_blocks", "two-level cumsum must use ceil-div block count")
    require(patch, "level1_uniforms", "three-level cumsum must pass level-1 element count")
    require(patch, "level2_uniforms", "three-level cumsum must pass level-2 element count")
    require(patch, "GPU cumsum self-test begin block=256",
            "renderer must run a cumsum self-test before training")
    require(patch, "37635, 90000",
            "self-test must cover the observed two-level and three-level Pixel workloads")
    require(patch, "GPU cumsum self-test FAIL",
            "self-test mismatch details must be logged")
    forbid(patch, "CPU prefix scan", "CPU cumsum workaround returned")
    forbid(patch, "copyFromDevice(input_buffer);", "renderer cumsum must remain GPU-side")
    verify_cumsum_hierarchy()

    require(cumsum_glsl, "#version 450", "Android cumsum must be Vulkan GLSL")
    require(cumsum_glsl, "#define PCS_BLOCK_SIZE 256", "GLSL cumsum local size changed")
    require(cumsum_glsl, "layout(local_size_x = PCS_BLOCK_SIZE", "GLSL local workgroup missing")
    require(cumsum_glsl, "shared int sData[PCS_BLOCK_SIZE]", "GLSL cumsum needs workgroup storage")
    require(cumsum_glsl, "barrier();", "GLSL cumsum needs workgroup synchronization")
    require(cumsum_glsl, "inclusiveWorkgroupScan", "GLSL cumsum scan implementation missing")
    forbid(cumsum_glsl, "subgroup", "custom cumsum must not use subgroup operations")
    forbid(cumsum_glsl, "Wave", "custom cumsum must not use wave operations")

    # Pipeline diagnostics must isolate driver aborts to an exact shader and Vulkan call.
    require(pipeline_diag, "Vulkan pipeline begin shader=",
            "pipeline diagnostics must log SPIR-V before module creation")
    require(pipeline_diag, "Vulkan pipeline create begin shader=",
            "pipeline diagnostics must log immediately before vkCreateComputePipelines")
    require(pipeline_diag, "Vulkan pipeline ready shader=",
            "pipeline diagnostics must log successful pipeline creation")
    require(pipeline_diag, "maxWGInvocations=",
            "device diagnostics must include maxComputeWorkGroupInvocations")
    require(pipeline_diag, "OpExecutionMode opcode=16",
            "pipeline diagnostics must decode SPIR-V LocalSize")
    require(pipeline_diag, "Invalid SPIR-V header",
            "runtime must reject malformed SPIR-V before the driver sees it")

    require(prepare, "cumsum.slang", "build must explicitly remove upstream Slang cumsum job")
    require(prepare, "PCS_CUMSUM_PHASE", "build must compile the PCS GLSL cumsum phases")
    require(prepare, "spirv-val", "build must validate cumsum SPIR-V")
    require(prepare, "spirv-dis", "build must inspect cumsum SPIR-V execution mode")
    require(prepare, "LocalSize 256 1 1", "build must assert exact cumsum LocalSize")
    require(prepare, "patch-vksplat-pipeline-diagnostics-android.py",
            "build must apply runtime Vulkan pipeline diagnostics")

    patch_pos = prepare.find("python3 scripts/patch-vksplat-cumsum-android.py")
    diag_pos = prepare.find("python3 scripts/patch-vksplat-pipeline-diagnostics-android.py")
    compile_pos = prepare.find("python3 compile_shaders.py")
    if min(patch_pos, diag_pos, compile_pos) < 0 or not (patch_pos < compile_pos and diag_pos < compile_pos):
        raise SystemExit(
            "mobile architecture check failed: VkSplat patches must run before shader/native build")

    if VENDORED_RENDERER.is_file():
        renderer = VENDORED_RENDERER.read_text(encoding="utf-8")
        require(renderer, "const size_t block = 256;",
                "prepared renderer is not using fixed 256-thread cumsum geometry")
        require(renderer, "if (num_elements <= block)",
                "prepared single-pass threshold must match shader LocalSize")
        require(renderer, "block_uniforms", "prepared renderer is missing two-level bounds")
        require(renderer, "level1_uniforms", "prepared renderer is missing level-1 bounds")
        require(renderer, "level2_uniforms", "prepared renderer is missing level-2 bounds")
        require(renderer, "GPU cumsum self-test COMPLETE",
                "prepared renderer is missing runtime GPU cumsum self-test")
        start = renderer.find("void VulkanGSRenderer::executeCumsum(")
        end = renderer.find("void VulkanGSRenderer::executeCalculateIndexBufferOffset(", start)
        if start < 0 or end < 0:
            raise SystemExit("mobile architecture check failed: prepared cumsum function not found")
        cumsum_fn = renderer[start:end]
        require(cumsum_fn, "executeCompute(", "prepared cumsum is not running on Vulkan")
        forbid(cumsum_fn, "copyFromDevice(input_buffer)", "prepared cumsum fell back to CPU")

    if VENDORED_PIPELINE.is_file():
        pipeline = VENDORED_PIPELINE.read_text(encoding="utf-8")
        require(pipeline, "Vulkan pipeline create begin shader=",
                "prepared VkSplat pipeline lacks driver-abort diagnostics")
        require(pipeline, "maxWGInvocations=",
                "prepared VkSplat pipeline lacks device workgroup limits")

    print("PCS continuous scanner + mobile trainer architecture checks passed")


if __name__ == "__main__":
    main()
