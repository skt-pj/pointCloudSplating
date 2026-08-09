#!/usr/bin/env python3
"""Build-time architecture checks for the PCS Android scanner and resumable 3DGS trainer."""
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "app/src/main/cpp/native_3dgs.cpp"
CUMSUM_SELFTEST = ROOT / "app/src/main/cpp/cumsum_selftest.cpp"
CMAKE = ROOT / "app/src/main/cpp/CMakeLists.txt"
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
JOB = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/GaussianSplatJob.java"
LIBRARY = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/LibraryActivity.java"
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
CHECKPOINT_PATCH = ROOT / "scripts/patch-vksplat-checkpoint-android.py"
PIPELINE_DIAG_PATCH = ROOT / "scripts/patch-vksplat-pipeline-diagnostics-android.py"
CUMSUM_GLSL = ROOT / "scripts/vksplat-android-shaders/cumsum.comp"
PREPARE = ROOT / "scripts/prepare-vksplat-android.sh"
VENDORED_RENDERER = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp"
VENDORED_PIPELINE = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_pipeline.cpp"
VENDORED_TRAINER = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_trainer.cpp"
VENDORED_TRAINER_HEADER = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_trainer.h"
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
        elif n <= block * block * block:
            level2 = ceil_div(level1, block)
            assert level1 > block and level2 <= block
        else:
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
        if opcode == 16 and word_count >= 6 and words[i + 2] == 17:
            return words[i + 3], words[i + 4], words[i + 5]
        i += word_count
    raise SystemExit(f"mobile architecture check failed: SPIR-V LocalSize missing: {path}")


def main() -> None:
    native = NATIVE.read_text(encoding="utf-8")
    cumsum_selftest = CUMSUM_SELFTEST.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")
    java = JAVA.read_text(encoding="utf-8")
    job = JOB.read_text(encoding="utf-8")
    library = LIBRARY.read_text(encoding="utf-8")
    diagnostic = DIAGNOSTIC.read_text(encoding="utf-8")
    tombstone_parser = TOMBSTONE_PARSER.read_text(encoding="utf-8")
    scanner = SCANNER.read_text(encoding="utf-8")
    capture = CAPTURE.read_text(encoding="utf-8")
    camera_config = CAMERA_CONFIG.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    exporter = EXPORTER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    cumsum_patch = CUMSUM_PATCH.read_text(encoding="utf-8")
    radix_patch = RADIX_PATCH.read_text(encoding="utf-8")
    checkpoint_patch = CHECKPOINT_PATCH.read_text(encoding="utf-8")
    pipeline_diag = PIPELINE_DIAG_PATCH.read_text(encoding="utf-8")
    cumsum_glsl = CUMSUM_GLSL.read_text(encoding="utf-8")
    prepare = PREPARE.read_text(encoding="utf-8")

    # Camera/capture invariants: 3DGS work must never regress the scanner path.
    require(scanner, "continuousCpuOnly=true", "continuous SharedCamera must use ARCore-only surfaces")
    forbid(scanner, "sessionSurfaces.add(jpegReader.getSurface())", "legacy JPEG surface must stay out of continuous capture")
    require(scanner, "surfaceView.setPreserveEGLContextOnPause(true)", "normal scan pauses must preserve EGL")
    require(scanner, "releaseGlForModel", "model processing must explicitly release camera GL")
    require(scanner, 'append("lastModelError=")', "model failures must remain in diagnostics")
    require(capture, "frame.acquireCameraImage()", "RGB must come from the current ARCore frame")
    require(capture, '"arcore_cpu_yuv_continuous"', "saved RGB source must stay explicit")
    require(capture, 'quality.put("motion_is_hard_gate", false)', "motion must rank rather than gate")
    require(capture, "calculateSharpness", "keyframe selection must measure sharpness")
    require(capture, "flushBestCandidateLocked", "selection windows must retain only the best candidate")
    require(capture, "SELECTION_WINDOW_NS = 750_000_000L", "selection window must remain 750ms")
    forbid(capture, "NEW_WINDOW_TRANSLATION_METERS", "movement must not close selection windows")
    forbid(capture, "NEW_WINDOW_ROTATION_DEGREES", "rotation must not close selection windows")
    require(camera_config, "MAX_CONTINUOUS_CPU_PIXELS = 2_500_000L", "continuous CPU resolution budget changed")
    require(finalizer, "if(capture==null)return null;", "ARCore frames must not be remapped as Camera2 stills")
    require(finalizer, '"arcore_image_intrinsics"', "ARCore intrinsics must be retained")
    require(exporter, "MAX_TRAIN_LONG_EDGE = 1000", "mobile training memory envelope changed")
    forbid(exporter, "TRAIN_DOWNSAMPLE", "legacy blind downsample returned")
    require(exporter, "never upscale", "adaptive export must never upscale")
    require(manifest, 'android:icon="@mipmap/ic_launcher"', "launcher icon missing")

    # Core mobile trainer semantics.
    require(native, "TrainerConfig::Strategy::MCMC", "density control must remain MCMC")
    require(native, "kGaussianBudget = 120'000", "Gaussian budget changed")
    require(native, "kNormalNeighbors = 16", "surface normal K changed")
    require(native, "kScaleNeighbors = 3", "surface scale K changed")
    require(native, "validateProjectionInvariants", "projection invariants must remain checked")
    require(native, "cpuTiles=", "prefix mismatch must remain observable")
    require(native, "kTileWorkingSetBudgetBytes", "tile working set must remain budgeted")
    forbid(native, "executeDefaultPostBackward(", "desktop Default densification is not allowed")

    # Training iterations are user-controlled. 1000 is a default, not a quality ceiling.
    require(java, "BASE_TRAIN_STEPS = 750", "first-run default changed unexpectedly")
    forbid(java, "MAX_TRAIN_STEPS", "old 1000-step quality ceiling returned")
    require(java, "int requestedSteps", "trainer needs explicit requested steps")
    require(java, "requestedSteps <= 0", "requested steps must be validated")
    require(java, 'result.put("resumable_training", true)', "result must advertise continuation")
    require(java, 'result.put("checkpoint_state", "gaussians+adam+rng+step")', "checkpoint contents must be documented")
    require(java, 'SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v6"', "shader cache generation changed")
    require(java, "nativeCumsumSelfTest", "cumsum self-test gate is required")
    forbid(java, "cumsum=cpu_scan", "CPU cumsum fallback returned")

    # Native continuation must resume the true trainable state and report cumulative progress.
    require(native, "trainer.restoreTrainingCheckpoint(buffers)", "native resume is missing")
    require(native, "trainer.getResumeTrainingStep()", "cumulative resume step is missing")
    require(native, "completedBeforeRun", "previous cumulative step is not retained")
    require(native, "targetStep", "cumulative progress target is missing")
    require(native, "getCompletedTrainingSteps()", "persisted cumulative step is not verified")
    require(native, "added_steps", "native result must report the extension size")
    require(native, "resumed", "native result must identify resumed runs")
    forbid(native, "std::max(300,", "custom runs must not be silently raised to 300 steps")

    require(job, "continueTraining", "job needs an explicit continuation entry point")
    require(job, "canContinueTraining", "legacy PLY-only models must be distinguished")
    require(job, "previousSteps + additionalSteps", "continuation must verify cumulative growth")
    require(job, "optimizerの学習状態", "legacy models must not be mislabeled as exact resumes")
    require(library, 'more.setText("追加学習")', "completed models need an additional-training action")
    require(library, "TRAINING_PRESETS = {300, 1_000, 3_000, 10_000}", "training presets are missing")
    require(library, "showCustomTrainingInput", "arbitrary user step input is missing")
    require(library, "Integer.MAX_VALUE", "custom steps must not have a small product cap")

    # Checkpoint = parameters + Adam moments + RNG + cumulative step, atomically published.
    for needle, reason in [
        ("restoreTrainingCheckpoint", "checkpoint restore patch missing"),
        ("saveTrainingCheckpoint", "checkpoint save patch missing"),
        ("g_xyz_ws", "Adam means moments missing"),
        ("g_sh_coeffs_1", "Adam SH m1 missing"),
        ("g_sh_coeffs_2", "Adam SH m2 missing"),
        ("g_rotations", "Adam rotation moments missing"),
        ("g_scales_opacs", "Adam scale/opacity moments missing"),
        ("rng_out << rng", "MCMC RNG save missing"),
        ("rng_in >> rng", "MCMC RNG restore missing"),
        ("resume_training_step", "cumulative Adam step missing"),
        ('training_checkpoint_path + ".tmp"', "temporary checkpoint file missing"),
        ("std::filesystem::rename(temp_path, training_checkpoint_path", "atomic checkpoint publish missing"),
        ("config.means_lr_final * scene_scale", "resume LR must not jump to its initial value"),
    ]:
        require(checkpoint_patch, needle, reason)
    require(pipeline_diag, "patch-vksplat-checkpoint-android.py", "prepare must apply checkpoint patch")

    # Cumsum remains fixed 256-thread subgroup-independent Vulkan code.
    require(cumsum_patch, "const size_t block = 256;", "cumsum block size changed")
    require(cumsum_patch, "level1_uniforms", "three-level cumsum L1 bounds missing")
    require(cumsum_patch, "level2_uniforms", "three-level cumsum L2 bounds missing")
    require(cumsum_patch, 'spirv_paths.at("cumsum_single_pass"), 0, false', "cumsum fixed subgroup request returned")
    forbid(cumsum_patch, "CPU prefix scan", "CPU cumsum fallback returned")
    verify_cumsum_hierarchy()
    require(cumsum_selftest, "class CumsumSelfTestRenderer final : public VulkanGSRenderer", "isolated cumsum renderer missing")
    require(cumsum_selftest, "VulkanGSPipeline::initialize(deviceId)", "isolated Vulkan init missing")
    require(cumsum_selftest, "executeCumsum(buffers, input, output)", "self-test must use production cumsum")
    require(cumsum_selftest, "{1, 16, 255, 256, 257, 1024, 1025, 37635, 90000}", "cumsum boundary coverage changed")
    forbid(cumsum_selftest, "VulkanGSTrainer", "cumsum test must remain isolated from trainer")
    require(cmake, "cumsum_selftest.cpp", "cumsum self-test must be linked")
    require(cumsum_glsl, "#define PCS_BLOCK_SIZE 256", "GLSL cumsum local size changed")
    require(cumsum_glsl, "shared int sData[PCS_BLOCK_SIZE]", "GLSL cumsum workgroup storage missing")
    forbid(cumsum_glsl, "gl_Subgroup", "cumsum must remain subgroup-independent")

    # Mali radix remains subgroup-16 / 256-thread and NDK-glslc generated.
    require(radix_patch, '#define SUBGROUP_SIZE 16', "radix subgroup width changed")
    require(radix_patch, '#define WORKGROUP_SIZE 256', "radix workgroup size changed")
    require(radix_patch, 'SHMEM_SIZE = RADIX * (WORKGROUP_SIZE / SUBGROUP_SIZE)', "radix shared histogram sizing regressed")
    require(radix_patch, 'subgroupBallotExclusiveBitCount(mask)', "radix exclusive ballot must be subgroup-width-safe")
    require(radix_patch, 'subgroupBallotBitCount(mask)', "radix ballot must be subgroup-width-safe")
    require(pipeline_diag, 'patch-vksplat-radix-android.py', "prepare must apply Mali radix patch")

    # Crash evidence and pipeline diagnostics remain persistent.
    require(diagnostic, "PREF_LAST_NATIVE_TRACE_TIMESTAMP", "native tombstone cursor missing")
    require(diagnostic, "exit.getTraceInputStream()", "native tombstones must be read")
    require(diagnostic, "NativeTombstoneParser.parse(trace)", "native tombstones must be decoded")
    require(tombstone_parser, "case 16:", "tombstone thread map missing")
    require(tombstone_parser, "out.frames.add(parseFrame", "native backtrace frames missing")
    require(pipeline_diag, "Vulkan pipeline begin shader=", "pipeline start diagnostics missing")
    require(pipeline_diag, "Vulkan pipeline create begin shader=", "pipeline creation diagnostics missing")
    require(pipeline_diag, "Vulkan pipeline ready shader=", "pipeline success diagnostics missing")
    require(pipeline_diag, "maxWGInvocations=", "device workgroup diagnostics missing")
    require(pipeline_diag, "Invalid SPIR-V header", "malformed SPIR-V validation missing")

    # Build must patch before compilation and validate generated Android shader geometry.
    require(prepare, "PCS_CUMSUM_PHASE", "prepare must compile custom cumsum")
    require(prepare, "spirv-val", "prepare must validate cumsum SPIR-V")
    require(prepare, "LocalSize 256 1 1", "prepare must assert cumsum LocalSize")
    patch_pos = prepare.find("python3 scripts/patch-vksplat-cumsum-android.py")
    diag_pos = prepare.find("python3 scripts/patch-vksplat-pipeline-diagnostics-android.py")
    compile_pos = prepare.find("python3 compile_shaders.py")
    if min(patch_pos, diag_pos, compile_pos) < 0 or not (patch_pos < compile_pos and diag_pos < compile_pos):
        raise SystemExit("mobile architecture check failed: VkSplat patches must run before shader/native build")

    if VENDORED_RENDERER.is_file():
        renderer = VENDORED_RENDERER.read_text(encoding="utf-8")
        require(renderer, "const size_t block = 256;", "prepared renderer cumsum geometry regressed")
        require(renderer, 'spirv_paths.at("cumsum_single_pass"), 0, false', "prepared cumsum subgroup request regressed")
        sort_start = renderer.find("void VulkanGSRenderer::executeSort(")
        if sort_start < 0:
            raise SystemExit("mobile architecture check failed: prepared radix sort function missing")
        require(renderer[sort_start:], "const int WORKGROUP_SIZE = 256;", "prepared radix C++ geometry regressed")

    if VENDORED_PIPELINE.is_file():
        pipeline = VENDORED_PIPELINE.read_text(encoding="utf-8")
        require(pipeline, "Vulkan pipeline create begin shader=", "prepared pipeline diagnostics missing")
        require(pipeline, "maxWGInvocations=", "prepared device diagnostics missing")

    if VENDORED_TRAINER.is_file() and VENDORED_TRAINER_HEADER.is_file():
        trainer = VENDORED_TRAINER.read_text(encoding="utf-8")
        trainer_header = VENDORED_TRAINER_HEADER.read_text(encoding="utf-8")
        require(trainer_header, "restoreTrainingCheckpoint", "prepared trainer restore API missing")
        require(trainer_header, "completed_training_step", "prepared trainer cumulative step missing")
        require(trainer, "Saved PCS 3DGS checkpoint", "prepared checkpoint save missing")
        require(trainer, "Restored PCS 3DGS checkpoint", "prepared checkpoint restore missing")
        require(trainer, "global_step = resume_training_step", "prepared Adam step is not cumulative")
        require(trainer, "saveTrainingCheckpoint(buffers);", "PLY output does not save checkpoint")

    if VENDORED_RADIX.is_dir():
        radix_config = (VENDORED_RADIX / "config.glsl").read_text(encoding="utf-8")
        radix_downsweep = (VENDORED_RADIX / "downsweep.comp").read_text(encoding="utf-8")
        require(radix_config, "#define SUBGROUP_SIZE 16", "prepared radix subgroup width regressed")
        require(radix_config, "#define WORKGROUP_SIZE 256", "prepared radix workgroup regressed")
        require(radix_downsweep, "subgroupBallotExclusiveBitCount(mask)", "prepared radix ballot regressed")
        for name in ("upsweep.spv", "spine.spv", "downsweep.spv"):
            path = VENDORED_RADIX / name
            if not path.is_file():
                raise SystemExit(f"mobile architecture check failed: Android radix SPIR-V missing: {path}")
            local = spirv_local_size(path)
            if local != (256, 1, 1):
                raise SystemExit(f"mobile architecture check failed: {name} LocalSize={local}, expected (256, 1, 1)")

    print("PCS continuous scanner + resumable mobile trainer architecture checks passed")


if __name__ == "__main__":
    main()
