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
TOMBSTONE = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeTombstoneParser.java"
SCANNER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/ScannerActivity.java"
CAPTURE = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DatasetCaptureManager.java"
CAMERA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/CameraConfigSelector.java"
FINALIZER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/DatasetFinalizer.java"
EXPORTER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/ColmapDatasetExporter.java"
MANIFEST = ROOT / "app/src/main/AndroidManifest.xml"
CUMSUM_PATCH = ROOT / "scripts/patch-vksplat-cumsum-android.py"
RADIX_PATCH = ROOT / "scripts/patch-vksplat-radix-android.py"
CHECKPOINT_PATCH = ROOT / "scripts/patch-vksplat-checkpoint-android.py"
LEGACY_PATCH = ROOT / "scripts/patch-vksplat-legacy-resume-android.py"
HINT_PATCH = ROOT / "scripts/patch-vksplat-legacy-resume-hint-android.py"
PIPELINE_PATCH = ROOT / "scripts/patch-vksplat-pipeline-diagnostics-android.py"
CUMSUM_GLSL = ROOT / "scripts/vksplat-android-shaders/cumsum.comp"
PREPARE = ROOT / "scripts/prepare-vksplat-android.sh"
VENDORED = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"mobile architecture check failed: {why}: missing {needle!r}")


def forbid(text: str, needle: str, why: str) -> None:
    if needle in text:
        raise SystemExit(f"mobile architecture check failed: {why}: found {needle!r}")


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def verify_cumsum_hierarchy() -> None:
    block = 256
    for n in [1, 16, 255, 256, 257, 1024, 1025, 37_635, 90_000, 120_000, 1_048_576]:
        if n <= block:
            continue
        l1 = ceil_div(n, block)
        if n <= block * block:
            assert l1 <= block
        elif n <= block * block * block:
            assert l1 > block and ceil_div(l1, block) <= block
        else:
            raise AssertionError(n)


def spirv_local_size(path: Path) -> tuple[int, int, int]:
    raw = path.read_bytes()
    if len(raw) < 20 or len(raw) % 4:
        raise SystemExit(f"invalid SPIR-V size: {path}")
    words = struct.unpack(f"<{len(raw)//4}I", raw)
    if words[0] != 0x07230203:
        raise SystemExit(f"invalid SPIR-V magic: {path}")
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xffff
        if wc == 0 or i + wc > len(words):
            break
        if op == 16 and wc >= 6 and words[i + 2] == 17:
            return words[i + 3], words[i + 4], words[i + 5]
        i += wc
    raise SystemExit(f"SPIR-V LocalSize missing: {path}")


def main() -> None:
    native = NATIVE.read_text(encoding="utf-8")
    selftest = CUMSUM_SELFTEST.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")
    java = JAVA.read_text(encoding="utf-8")
    job = JOB.read_text(encoding="utf-8")
    library = LIBRARY.read_text(encoding="utf-8")
    diagnostic = DIAGNOSTIC.read_text(encoding="utf-8")
    tombstone = TOMBSTONE.read_text(encoding="utf-8")
    scanner = SCANNER.read_text(encoding="utf-8")
    capture = CAPTURE.read_text(encoding="utf-8")
    camera = CAMERA.read_text(encoding="utf-8")
    finalizer = FINALIZER.read_text(encoding="utf-8")
    exporter = EXPORTER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    cumsum_patch = CUMSUM_PATCH.read_text(encoding="utf-8")
    radix_patch = RADIX_PATCH.read_text(encoding="utf-8")
    checkpoint_patch = CHECKPOINT_PATCH.read_text(encoding="utf-8")
    legacy_patch = LEGACY_PATCH.read_text(encoding="utf-8")
    hint_patch = HINT_PATCH.read_text(encoding="utf-8")
    pipeline_patch = PIPELINE_PATCH.read_text(encoding="utf-8")
    cumsum_glsl = CUMSUM_GLSL.read_text(encoding="utf-8")
    prepare = PREPARE.read_text(encoding="utf-8")

    # Scanner invariants: 3DGS changes must not alter capture behavior.
    require(scanner, "continuousCpuOnly=true", "continuous SharedCamera must stay ARCore-only")
    forbid(scanner, "sessionSurfaces.add(jpegReader.getSurface())", "legacy JPEG surface returned")
    require(scanner, "surfaceView.setPreserveEGLContextOnPause(true)", "normal scan pause must preserve EGL")
    require(scanner, "releaseGlForModel", "model processing must explicitly release GL")
    require(scanner, 'append("lastModelError=")', "model error diagnostics missing")
    require(capture, "frame.acquireCameraImage()", "RGB must come from ARCore CPU frames")
    require(capture, '"arcore_cpu_yuv_continuous"', "continuous RGB source marker missing")
    require(capture, 'quality.put("motion_is_hard_gate", false)', "motion must not become a hard gate")
    require(capture, "calculateSharpness", "sharpness ranking missing")
    require(capture, "flushBestCandidateLocked", "selection-window best-frame logic missing")
    require(capture, "SELECTION_WINDOW_NS = 750_000_000L", "750ms selection window changed")
    forbid(capture, "NEW_WINDOW_TRANSLATION_METERS", "movement-triggered window closure returned")
    forbid(capture, "NEW_WINDOW_ROTATION_DEGREES", "rotation-triggered window closure returned")
    require(camera, "MAX_CONTINUOUS_CPU_PIXELS = 2_500_000L", "continuous CPU resolution budget changed")
    require(finalizer, "if(capture==null)return null;", "ARCore frames must not be Camera2-remapped")
    require(finalizer, '"arcore_image_intrinsics"', "ARCore intrinsics marker missing")
    require(exporter, "MAX_TRAIN_LONG_EDGE = 1000", "mobile training image envelope changed")
    forbid(exporter, "TRAIN_DOWNSAMPLE", "legacy blind training downsample returned")
    require(exporter, "never upscale", "adaptive export must never upscale")
    require(manifest, 'android:icon="@mipmap/ic_launcher"', "launcher icon missing")

    # Mobile 3DGS semantics.
    require(native, "TrainerConfig::Strategy::MCMC", "MCMC density control missing")
    require(native, "kGaussianBudget = 120'000", "Gaussian budget changed")
    require(native, "kNormalNeighbors = 16", "surface normal K changed")
    require(native, "kScaleNeighbors = 3", "surface scale K changed")
    require(native, "validateProjectionInvariants", "projection invariant checks missing")
    require(native, "cpuTiles=", "prefix mismatch evidence missing")
    require(native, "kTileWorkingSetBudgetBytes", "tile working-set budget missing")
    forbid(native, "executeDefaultPostBackward(", "desktop default densification returned")

    # User chooses iterations; 1000 is not a hard ceiling.
    require(java, "BASE_TRAIN_STEPS = 750", "first-run default changed unexpectedly")
    forbid(java, "MAX_TRAIN_STEPS", "1000-step product ceiling returned")
    require(java, "int requestedSteps", "explicit user step count missing")
    require(java, "requestedSteps <= 0", "step validation missing")
    require(java, 'result.put("resumable_training", true)', "resumable metadata missing")
    require(java, 'result.put("checkpoint_state", "gaussians+adam+rng+step")', "checkpoint contents metadata missing")
    require(java, 'SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v6"', "shader cache generation changed")
    require(java, "nativeCumsumSelfTest", "isolated cumsum gate missing")
    forbid(java, "cumsum=cpu_scan", "CPU cumsum fallback returned")

    require(native, "trainer.restoreTrainingCheckpoint(buffers)", "native continuation restore missing")
    require(native, "trainer.getResumeTrainingStep()", "cumulative resume step missing")
    require(native, "targetStep", "cumulative target missing")
    require(native, "getCompletedTrainingSteps()", "persisted total-step verification missing")
    require(native, "added_steps", "extension length missing from native result")
    forbid(native, "std::max(300,", "custom runs must not be silently raised to 300")

    require(job, "continueTraining", "explicit continuation job missing")
    require(job, "canContinueTraining", "continuation capability missing")
    require(job, "hasExactTrainingCheckpoint", "exact-vs-legacy continuation distinction missing")
    require(job, "writeLegacyResumeHint", "legacy PLY migration path missing")
    require(job, "previousSteps + additionalSteps", "cumulative continuation verification missing")
    require(job, "LEGACY_PLY_MIGRATED", "legacy migration diagnostics missing")
    require(library, 'more.setText("追加学習")', "completed-model continuation button missing")
    require(library, "TRAINING_PRESETS = {300, 1_000, 3_000, 10_000}", "training presets missing")
    require(library, "showCustomTrainingInput", "arbitrary step input missing")
    require(library, "Integer.MAX_VALUE", "custom steps have an artificial small cap")

    # Full checkpoint and one-time legacy migration.
    for needle in ["restoreTrainingCheckpoint", "saveTrainingCheckpoint", "g_xyz_ws",
                   "g_sh_coeffs_1", "g_sh_coeffs_2", "g_rotations", "g_scales_opacs",
                   "rng_out << rng", "rng_in >> rng", 'training_checkpoint_path + ".tmp"',
                   "std::filesystem::rename(temp_path, training_checkpoint_path"]:
        require(checkpoint_patch, needle, "full optimizer checkpoint is incomplete")
    require(legacy_patch, "importLegacyTrainingPly", "legacy PLY importer missing")
    require(legacy_patch, "resume_optimizer_step", "legacy Adam-age separation missing")
    require(legacy_patch, "completed_optimizer_step", "optimizer age is not persisted")
    require(legacy_patch, "buffers.g_xyz_ws.assign(2 * 3 * n, 0.0f)", "legacy Adam reset missing")
    require(legacy_patch, "total_training_step = resume_training_step + local_step", "legacy total-step continuation missing")
    require(legacy_patch, "global_step = resume_optimizer_step + local_step", "legacy Adam bias age is wrong")
    require(hint_patch, '"legacy_resume_step.txt"', "legacy migration hint bridge missing")
    require(hint_patch, "return importLegacyTrainingPly", "legacy hint does not trigger PLY import")
    require(pipeline_patch, "patch-vksplat-checkpoint-android.py", "checkpoint patch is not wired")
    require(pipeline_patch, "patch-vksplat-legacy-resume-android.py", "legacy importer is not wired")
    require(pipeline_patch, "patch-vksplat-legacy-resume-hint-android.py", "legacy hint bridge is not wired")

    # Cumsum remains subgroup-independent and radix remains Pixel/Mali subgroup-16 safe.
    require(cumsum_patch, "const size_t block = 256;", "cumsum block size changed")
    require(cumsum_patch, "level1_uniforms", "cumsum level-1 bounds missing")
    require(cumsum_patch, "level2_uniforms", "cumsum level-2 bounds missing")
    require(cumsum_patch, 'spirv_paths.at("cumsum_single_pass"), 0, false', "cumsum fixed-subgroup request returned")
    forbid(cumsum_patch, "CPU prefix scan", "CPU cumsum fallback returned")
    verify_cumsum_hierarchy()
    require(selftest, "class CumsumSelfTestRenderer final : public VulkanGSRenderer", "isolated cumsum renderer missing")
    require(selftest, "executeCumsum(buffers, input, output)", "self-test must exercise production cumsum")
    require(selftest, "{1, 16, 255, 256, 257, 1024, 1025, 37635, 90000}", "cumsum test coverage changed")
    forbid(selftest, "VulkanGSTrainer", "cumsum self-test must stay independent of trainer")
    require(cmake, "cumsum_selftest.cpp", "cumsum self-test is not linked")
    require(cumsum_glsl, "#define PCS_BLOCK_SIZE 256", "GLSL cumsum local size changed")
    forbid(cumsum_glsl, "gl_Subgroup", "custom cumsum must stay subgroup-independent")
    require(radix_patch, '#define SUBGROUP_SIZE 16', "radix subgroup width changed")
    require(radix_patch, '#define WORKGROUP_SIZE 256', "radix workgroup size changed")
    require(radix_patch, 'SHMEM_SIZE = RADIX * (WORKGROUP_SIZE / SUBGROUP_SIZE)', "radix shared histogram sizing regressed")
    require(radix_patch, 'subgroupBallotExclusiveBitCount(mask)', "radix subgroup ballot regressed")

    # Persistent native failure evidence remains available.
    require(diagnostic, "PREF_LAST_NATIVE_TRACE_TIMESTAMP", "native tombstone cursor missing")
    require(diagnostic, "exit.getTraceInputStream()", "native tombstones are not read")
    require(diagnostic, "NativeTombstoneParser.parse(trace)", "native tombstones are not decoded")
    require(tombstone, "out.frames.add(parseFrame", "native backtrace frames missing")
    require(pipeline_patch, "Vulkan pipeline create begin shader=", "pipeline diagnostics missing")
    require(pipeline_patch, "maxWGInvocations=", "device workgroup diagnostics missing")
    require(pipeline_patch, "Invalid SPIR-V header", "SPIR-V validation missing")

    # Preparation order and generated shader geometry.
    patch_pos = prepare.find("python3 scripts/patch-vksplat-cumsum-android.py")
    diag_pos = prepare.find("python3 scripts/patch-vksplat-pipeline-diagnostics-android.py")
    compile_pos = prepare.find("python3 compile_shaders.py")
    if min(patch_pos, diag_pos, compile_pos) < 0 or not (patch_pos < compile_pos and diag_pos < compile_pos):
        raise SystemExit("mobile architecture check failed: VkSplat patches must run before shader compilation")

    renderer_path = VENDORED / "src/gs_renderer.cpp"
    pipeline_path = VENDORED / "src/gs_pipeline.cpp"
    trainer_path = VENDORED / "src/gs_trainer.cpp"
    trainer_h_path = VENDORED / "src/gs_trainer.h"
    radix_dir = VENDORED / "shader/radix_sort"
    if renderer_path.is_file():
        renderer = renderer_path.read_text(encoding="utf-8")
        require(renderer, "const size_t block = 256;", "prepared cumsum geometry regressed")
        require(renderer, 'spirv_paths.at("cumsum_single_pass"), 0, false', "prepared cumsum subgroup contract regressed")
        sort_start = renderer.find("void VulkanGSRenderer::executeSort(")
        if sort_start < 0:
            raise SystemExit("prepared radix sort function missing")
        require(renderer[sort_start:], "const int WORKGROUP_SIZE = 256;", "prepared radix geometry regressed")
    if pipeline_path.is_file():
        prepared_pipeline = pipeline_path.read_text(encoding="utf-8")
        require(prepared_pipeline, "Vulkan pipeline create begin shader=", "prepared pipeline diagnostics missing")
    if trainer_path.is_file() and trainer_h_path.is_file():
        prepared_trainer = trainer_path.read_text(encoding="utf-8") + trainer_h_path.read_text(encoding="utf-8")
        require(prepared_trainer, "restoreTrainingCheckpoint", "prepared checkpoint restore missing")
        require(prepared_trainer, "importLegacyTrainingPly", "prepared legacy PLY importer missing")
        require(prepared_trainer, "legacy_resume_step.txt", "prepared legacy hint bridge missing")
        require(prepared_trainer, "resume_optimizer_step", "prepared optimizer-age separation missing")
        require(prepared_trainer, "saveTrainingCheckpoint(buffers);", "prepared checkpoint save missing")
    if radix_dir.is_dir():
        cfg = (radix_dir / "config.glsl").read_text(encoding="utf-8")
        down = (radix_dir / "downsweep.comp").read_text(encoding="utf-8")
        require(cfg, "#define SUBGROUP_SIZE 16", "prepared radix subgroup width regressed")
        require(cfg, "#define WORKGROUP_SIZE 256", "prepared radix workgroup regressed")
        require(down, "subgroupBallotExclusiveBitCount(mask)", "prepared radix ballot regressed")
        for name in ("upsweep.spv", "spine.spv", "downsweep.spv"):
            path = radix_dir / name
            if not path.is_file() or spirv_local_size(path) != (256, 1, 1):
                raise SystemExit(f"prepared Android radix SPIR-V invalid: {path}")

    print("PCS scanner + user-controlled resumable 3DGS architecture checks passed")


if __name__ == "__main__":
    main()
