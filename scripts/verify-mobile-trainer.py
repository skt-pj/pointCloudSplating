#!/usr/bin/env python3
"""Build-time architecture checks for the PCS Android mobile scanner and 3DGS trainer."""
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "app/src/main/cpp/native_3dgs.cpp"
CUMSUM_SELFTEST = ROOT / "app/src/main/cpp/cumsum_selftest.cpp"
CMAKE = ROOT / "app/src/main/cpp/CMakeLists.txt"
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
DIAGNOSTIC = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DiagnosticLog.java"
TOMBSTONE_PARSER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeTombstoneParser.java"
SCANNER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/ScannerActivity.java"
CAPTURE = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DatasetCaptureManager.java"
CAMERA_CONFIG = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/CameraConfigSelector.java"
FINALIZER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DatasetFinalizer.java"
EXPORTER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/ColmapDatasetExporter.java"
MANIFEST = ROOT / "app/src/main/AndroidManifest.xml"
CUMSUM_PATCH = ROOT / "scripts/patch-vksplat-cumsum-android.py"
RADIX_PATCH = ROOT / "scripts/patch-vksplat-radix-android.py"
PIPELINE_DIAG_PATCH = ROOT / "scripts/patch-vksplat-pipeline-diagnostics-android.py"
CUMSUM_GLSL = ROOT / "scripts/vksplat-android-shaders/cumsum.comp"
PREPARE = ROOT / "scripts/prepare-vksplat-android.sh"
VENDORED_RENDERER = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp"
VENDORED_PIPELINE = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_pipeline.cpp"
VENDORED_RADIX = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/shader/radix_sort"


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
    for n in [1, 16, 255, 256, 257, 1024, 1025, 37_635, 90_000, 120_000, 1_048_576]:
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


def spirv_local_size(path: Path) -> tuple[int, int, int]:
    raw = path.read_bytes()
    if len(raw) < 20 or len(raw) % 4 != 0:
        raise SystemExit(f"mobile architecture check failed: invalid SPIR-V size: {path}")
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    if words[0] != 0x07230203:
        raise SystemExit(f"mobile architecture check failed: invalid SPIR-V header: {path}")
    i = 5
    while i < len(words):
        first = words[i]
        word_count = first >> 16
        opcode = first & 0xffff
        if word_count == 0 or i + word_count > len(words):
            break
        # OpExecutionMode opcode=16, LocalSize execution mode=17.
        if opcode == 16 and word_count >= 6 and words[i + 2] == 17:
            return words[i + 3], words[i + 4], words[i + 5]
        i += word_count
    raise SystemExit(f"mobile architecture check failed: SPIR-V LocalSize missing: {path}")


def main() -> None:
    native = NATIVE.read_text(encoding="utf-8")
    cumsum_selftest = CUMSUM_SELFTEST.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")
    java = JAVA.read_text(encoding="utf-8")
    diagnostic = DIAGNOSTIC.read_text(encoding="utf-8")
    tombstone_parser = TOMBSTONE_PARSER.read_text(encoding="utf-8")
    scanner = SCANNER.read_text(encoding="utf-8")
    capture = CAPTURE.read_text(encoding="utf-8")
    camera_config = CAMERA_CONFIG.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    exporter = EXPORTER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    patch = CUMSUM_PATCH.read_text(encoding="utf-8")
    radix_patch = RADIX_PATCH.read_text(encoding="utf-8")
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
    require(java, 'SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v6"',
            "Mali radix build must use a fresh shader cache generation")
    require(java, '" cumsum=glslc256 radix=glslc256/subgroup16"',
            "runtime diagnostics must identify both Android GPU shader paths")
    require(java, "logCumsumShaderIdentity", "runtime must log exact cumsum SPIR-V identities")
    require(java, 'MessageDigest.getInstance("SHA-256")',
            "runtime cumsum diagnostics must include SHA-256")
    require(java, "nativeCumsumSelfTest", "cumsum conformance must be a separate JNI path")
    require(java, "ensureCumsumConformance", "training must pass cumsum conformance first")
    require(java, 'DiagnosticLog.i(TAG, "cumsum:selftest Java bridge begin")',
            "native self-test breadcrumbs must persist across process death")
    forbid(java, "cumsum=cpu_scan", "CPU cumsum workaround must not remain")
    forbid(java, "DEFAULT_TRAIN_STEPS = 6_000", "desktop-length training schedule returned")

    m = re.search(r"MAX_TRAIN_STEPS\s*=\s*([\d_]+)", java)
    if not m or int(m.group(1).replace("_", "")) > 1000:
        raise SystemExit("mobile architecture check failed: MAX_TRAIN_STEPS exceeds 1000")

    # GPU cumsum must remain on Vulkan and subgroup-independent.
    require(patch, "const size_t block = 256;",
            "renderer cumsum must use the shader's fixed workgroup size")
    require(patch, "num_blocks", "two-level cumsum must use ceil-div block count")
    require(patch, "level1_uniforms", "three-level cumsum must pass level-1 element count")
    require(patch, "level2_uniforms", "three-level cumsum must pass level-2 element count")
    require(patch, 'spirv_paths.at("cumsum_single_pass"), 0, false',
            "cumsum pipeline must not request a fixed subgroup size")
    forbid(patch, "GPU cumsum self-test", "self-test must not be embedded in full renderer initialization")
    forbid(patch, "CPU prefix scan", "CPU cumsum workaround returned")
    forbid(patch, "copyFromDevice(input_buffer);", "renderer cumsum must remain GPU-side")
    verify_cumsum_hierarchy()

    # Pixel/Mali radix sort must be compiled for the actual native subgroup width. The upstream
    # desktop shader uses a 512-thread workgroup with shared layouts sized for 16 x 32-wide
    # subgroups; on a 16-wide Mali subgroup that produces 32 subgroups and out-of-bounds shared
    # histogram indexing. Keep exactly 16 subgroups by using 256 threads and matching C++ geometry.
    require(radix_patch, '#define SUBGROUP_SIZE 16',
            "radix shader must target the Pixel native subgroup width")
    require(radix_patch, '#define WORKGROUP_SIZE 256',
            "radix shader must use 256 threads on subgroup-16 Mali")
    require(radix_patch, 'SHMEM_SIZE = RADIX * (WORKGROUP_SIZE / SUBGROUP_SIZE)',
            "downsweep shared histogram must be sized from radix x subgroup count")
    require(radix_patch, 'subgroupBallotExclusiveBitCount(mask)',
            "downsweep must use subgroup-width-safe exclusive ballot counting")
    require(radix_patch, 'subgroupBallotBitCount(mask)',
            "downsweep must use subgroup-width-safe ballot counting")
    require(radix_patch, 'const int WORKGROUP_SIZE = 256;',
            "renderer radix partition geometry must match GLSL")
    require(radix_patch, 'ShaderJob("upsweep.comp", {})',
            "Android preparation must recompile radix upsweep")
    require(radix_patch, 'unsupported glslc target',
            "Android radix compilation must remove the slang-only glslc target argument")
    require(pipeline_diag, 'patch-vksplat-radix-android.py',
            "VkSplat preparation must apply the Mali radix port")

    # The conformance test is a standalone Vulkan+cumsum path, not a whole-trainer probe.
    require(cumsum_selftest, "class CumsumSelfTestRenderer final : public VulkanGSRenderer",
            "isolated cumsum renderer is missing")
    require(cumsum_selftest, "VulkanGSPipeline::initialize(deviceId)",
            "self-test must initialize Vulkan without full trainer pipelines")
    require(cumsum_selftest, 'createComputePipeline(pipeline, path, 0, false)',
            "self-test cumsum pipelines must be subgroup-independent")
    require(cumsum_selftest, "{1, 16, 255, 256, 257, 1024, 1025, 37635, 90000}",
            "self-test must cover boundaries and observed Pixel workloads")
    require(cumsum_selftest, "DeviceGuard guard(this)",
            "cumsum dispatch must run inside a valid VkSplat command batch")
    require(cumsum_selftest, "executeCumsum(buffers, input, output)",
            "self-test must exercise the production GPU cumsum implementation")
    require(cumsum_selftest, "copyFromDevice(output)",
            "self-test must compare the full GPU prefix output against CPU reference")
    require(cumsum_selftest, "cumsum self-test mismatch n=",
            "self-test must report the exact failing prefix index")
    require(cumsum_selftest, "cumsum:selftest dispatch begin",
            "dispatch stage must persist before a possible native abort")
    forbid(cumsum_selftest, "VulkanGSTrainer", "cumsum self-test must stay independent of trainer initialization")
    require(cmake, "cumsum_selftest.cpp", "isolated cumsum test must be linked into the JNI library")

    require(cumsum_glsl, "#version 450", "Android cumsum must be Vulkan GLSL")
    require(cumsum_glsl, "#define PCS_BLOCK_SIZE 256", "GLSL cumsum local size changed")
    require(cumsum_glsl, "layout(local_size_x = PCS_BLOCK_SIZE", "GLSL cumsum local workgroup missing")
    require(cumsum_glsl, "shared int sData[PCS_BLOCK_SIZE]", "GLSL cumsum needs workgroup storage")
    require(cumsum_glsl, "barrier();", "GLSL cumsum needs workgroup synchronization")
    require(cumsum_glsl, "inclusiveWorkgroupScan", "GLSL cumsum scan implementation missing")
    forbid(cumsum_glsl, "gl_Subgroup", "custom cumsum must not use subgroup built-ins")
    forbid(cumsum_glsl, "subgroupInclusive", "custom cumsum must not use subgroup arithmetic")
    forbid(cumsum_glsl, "subgroupAdd", "custom cumsum must not use subgroup arithmetic")
    forbid(cumsum_glsl, "WavePrefix", "custom cumsum must not use wave operations")
    forbid(cumsum_glsl, "WaveRead", "custom cumsum must not use wave operations")

    # Native crash diagnostics must preserve the actual Android tombstone, not only its existence.
    require(diagnostic, "PREF_LAST_NATIVE_TRACE_TIMESTAMP",
            "native tombstone capture needs an independent cursor so older unparsed crashes can be recovered")
    require(diagnostic, "exit.getTraceInputStream()",
            "native crash diagnostics must read ApplicationExitInfo tombstones")
    require(diagnostic, "NativeTombstoneParser.parse(trace)",
            "native tombstones must be rendered into the saved text diagnostics")
    require(tombstone_parser, "case 16:", "tombstone parser must read the thread map")
    require(tombstone_parser, "out.frames.add(parseFrame", "tombstone parser must read native backtrace frames")
    require(tombstone_parser, 'append(" rel_pc=0x")', "native frame relative PCs must be retained for symbolization")
    require(tombstone_parser, "abortMessage", "native abort messages must be retained")

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
        require(renderer, 'spirv_paths.at("cumsum_single_pass"), 0, false',
                "prepared cumsum pipeline still requests subgroup compatibility")
        forbid(renderer, "GPU cumsum self-test", "prepared renderer still embeds the self-test")
        start = renderer.find("void VulkanGSRenderer::executeCumsum(")
        end = renderer.find("void VulkanGSRenderer::executeCalculateIndexBufferOffset(", start)
        if start < 0 or end < 0:
            raise SystemExit("mobile architecture check failed: prepared cumsum function not found")
        cumsum_fn = renderer[start:end]
        require(cumsum_fn, "executeCompute(", "prepared cumsum is not running on Vulkan")
        forbid(cumsum_fn, "copyFromDevice(input_buffer)", "prepared cumsum fell back to CPU")

        sort_start = renderer.find("void VulkanGSRenderer::executeSort(")
        if sort_start < 0:
            raise SystemExit("mobile architecture check failed: prepared radix sort function not found")
        sort_fn = renderer[sort_start:]
        require(sort_fn, "const int WORKGROUP_SIZE = 256;",
                "prepared radix C++ geometry is not Mali subgroup-16 safe")

    if VENDORED_PIPELINE.is_file():
        pipeline = VENDORED_PIPELINE.read_text(encoding="utf-8")
        require(pipeline, "Vulkan pipeline create begin shader=",
                "prepared VkSplat pipeline lacks driver-abort diagnostics")
        require(pipeline, "maxWGInvocations=",
                "prepared VkSplat pipeline lacks device workgroup limits")

    if VENDORED_RADIX.is_dir():
        radix_config = (VENDORED_RADIX / "config.glsl").read_text(encoding="utf-8")
        radix_downsweep = (VENDORED_RADIX / "downsweep.comp").read_text(encoding="utf-8")
        require(radix_config, "#define SUBGROUP_SIZE 16",
                "prepared radix shader still targets desktop subgroup width")
        require(radix_config, "#define WORKGROUP_SIZE 256",
                "prepared radix shader still uses 512-thread desktop geometry")
        require(radix_downsweep, "subgroupBallotExclusiveBitCount(mask)",
                "prepared downsweep still uses handcrafted 32-wide ballot masks")
        for name in ("upsweep.spv", "spine.spv", "downsweep.spv"):
            path = VENDORED_RADIX / name
            if not path.is_file():
                raise SystemExit(f"mobile architecture check failed: Android radix SPIR-V missing: {path}")
            local = spirv_local_size(path)
            if local != (256, 1, 1):
                raise SystemExit(
                    f"mobile architecture check failed: {name} LocalSize={local}, expected (256, 1, 1)")

    print("PCS continuous scanner + mobile trainer architecture checks passed")


if __name__ == "__main__":
    main()
