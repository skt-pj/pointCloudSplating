#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts/patch-vksplat-pipeline-diagnostics-android.py"
CHECKPOINT = ROOT / "scripts/patch-vksplat-checkpoint-android.py"
LEGACY = ROOT / "scripts/patch-vksplat-legacy-resume-android.py"
VALIDITY = ROOT / "scripts/patch-vksplat-legacy-ply-validity-android.py"
HINT = ROOT / "scripts/patch-vksplat-legacy-resume-hint-android.py"
JOB = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/GaussianSplatJob.java"
LIBRARY = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/LibraryActivity.java"
TRAINER_CPP = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_trainer.cpp"
TRAINER_H = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat/src/gs_trainer.h"
MAGIC_V2 = "kPcsCheckpointMagic[8] = {'P','C','S','3','D','G','S','2'}"


def require(text: str, needle: str, why: str) -> None:
    if needle not in text:
        raise SystemExit(f"resumable training check failed: {why}: missing {needle!r}")


def main() -> None:
    pipeline = PIPELINE.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    legacy = LEGACY.read_text(encoding="utf-8")
    validity = VALIDITY.read_text(encoding="utf-8")
    hint = HINT.read_text(encoding="utf-8")
    job = JOB.read_text(encoding="utf-8")
    library = LIBRARY.read_text(encoding="utf-8")

    require(pipeline, 'patch-vksplat-checkpoint-android.py', "exact checkpoint patch is not wired")
    require(pipeline, 'patch-vksplat-legacy-resume-android.py', "legacy PLY importer is not wired")
    require(pipeline, 'patch-vksplat-legacy-ply-validity-android.py', "legacy PLY validity normalization is not wired")
    require(pipeline, 'patch-vksplat-legacy-resume-hint-android.py', "legacy resume hint bridge is not wired")
    if not (pipeline.find('patch-vksplat-checkpoint-android.py')
            < pipeline.find('patch-vksplat-legacy-resume-android.py')
            < pipeline.find('patch-vksplat-legacy-ply-validity-android.py')
            < pipeline.find('patch-vksplat-legacy-resume-hint-android.py')):
        raise SystemExit("resumable training check failed: checkpoint/legacy/validity/hint patch order is invalid")

    for needle, why in [
        ("restoreTrainingCheckpoint", "checkpoint restore API missing"),
        ("saveTrainingCheckpoint", "checkpoint save API missing"),
        ("g_xyz_ws", "Adam means moments not saved"),
        ("g_sh_coeffs_1", "Adam SH first moments not saved"),
        ("g_sh_coeffs_2", "Adam SH second moments not saved"),
        ("g_rotations", "Adam rotation moments not saved"),
        ("g_scales_opacs", "Adam scale/opacity moments not saved"),
        ("rng_out << rng", "MCMC RNG state not saved"),
        ("rng_in >> rng", "MCMC RNG state not restored"),
        ('training_checkpoint_path + ".tmp"', "checkpoint does not use a temp file"),
        ("std::filesystem::rename(temp_path, training_checkpoint_path", "checkpoint is not atomically published"),
    ]:
        require(checkpoint, needle, why)

    for needle, why in [
        (MAGIC_V2, "v2 checkpoint magic missing"),
        ("kPcsCheckpointVersion = 2", "v2 checkpoint version missing"),
        ("importLegacyTrainingPly", "legacy PLY import API missing"),
        ("resume_optimizer_step", "legacy migration does not separate Adam age from total steps"),
        ("completed_optimizer_step", "optimizer age is not checkpointed"),
        ('"opacity", "scale_0", "scale_1", "scale_2"', "legacy PLY property layout is not validated"),
        ("std::exp(std::clamp(record[52]", "legacy log-scales are not inverted before validity normalization"),
        ("1.0f / (1.0f + std::exp(-opacity_logit))", "legacy opacity logit is not inverted before validity normalization"),
        ("buffers.g_xyz_ws.assign(2 * 3 * n, 0.0f)", "unavailable legacy Adam means must reset once"),
        ("buffers.g_sh_coeffs_1.assign(sh_n, 0.0f)", "unavailable legacy SH Adam state must reset once"),
        ("total_training_step = resume_training_step + local_step", "user-visible cumulative steps are not preserved"),
        ("global_step = resume_optimizer_step + local_step", "Adam bias correction age is not separated for legacy migration"),
    ]:
        require(legacy, needle, why)

    # Viewer-compatible migration: only invalid world positions make a Gaussian unusable. Appearance,
    # scale, opacity and quaternion channels are recoverable and must be made finite deterministically.
    # The patch source intentionally contains the old all-or-nothing loop as its replacement anchor,
    # so absence of that old error is asserted only against the fully prepared trainer below.
    for needle, why in [
        ("dropped_nonfinite_position", "invalid-position Gaussians are not compacted"),
        ("continue;", "invalid-position record is not skipped"),
        ("finiteOrZero", "non-finite SH/rotation channels are not normalized"),
        ("positiveScale", "legacy scale normalization missing"),
        ("return 0.004f", "non-finite legacy scale fallback differs from viewer"),
        ("opacityFromLogit", "legacy opacity normalization missing"),
        ("return 0.5f", "non-finite legacy opacity fallback differs from viewer"),
        ("q_len > 1e-8f", "legacy quaternion validity threshold differs from viewer"),
        ("qw = 1.0f", "invalid legacy quaternion does not become identity"),
        ("const size_t n = xyz.size() / 3", "valid Gaussians are not compacted before GPU upload"),
        ("legacy 3DGS PLY has too few valid Gaussian positions", "migration does not reject a fundamentally unusable model"),
        ("sanitizedSH=", "migration diagnostics do not report repaired channels"),
    ]:
        require(validity, needle, why)

    for needle, why in [
        ('"legacy_resume_step.txt"', "legacy resume hint filename missing"),
        ("legacy_training_ply_path = config.output_ply", "legacy importer is not bound to the actual final PLY"),
        ("return importLegacyTrainingPly", "missing checkpoint does not fall back to the legacy importer"),
        ("std::filesystem::remove(hint_path", "legacy hint is not cleared after checkpoint publication"),
    ]:
        require(hint, needle, why)

    for needle, why in [
        ("canContinueTraining", "public continuation capability missing"),
        ("hasExactTrainingCheckpoint", "UI cannot distinguish legacy migration from exact checkpoint resume"),
        ("writeLegacyResumeHint", "legacy completed models are not armed for import"),
        ("previousSteps + additionalSteps", "cumulative step growth is not verified"),
        ("out.getFD().sync()", "legacy resume hint is not flushed before native training starts"),
        ("LEGACY_PLY_MIGRATED", "legacy migration mode is not diagnostic"),
    ]:
        require(job, needle, why)

    require(library, 'more.setText("追加学習")', "completed model card lacks an additional-training action")
    require(library, "TRAINING_PRESETS = {300, 1_000, 3_000, 10_000}", "training presets are missing")
    require(library, "showCustomTrainingInput", "arbitrary user-selected steps are missing")

    if TRAINER_CPP.is_file() and TRAINER_H.is_file():
        prepared = TRAINER_CPP.read_text(encoding="utf-8") + TRAINER_H.read_text(encoding="utf-8")
        for needle, why in [
            ("importLegacyTrainingPly", "prepared trainer has no legacy importer"),
            ("legacy_resume_step.txt", "prepared trainer has no legacy hint bridge"),
            (MAGIC_V2, "prepared trainer is not using checkpoint v2"),
            ("resume_optimizer_step", "prepared trainer does not separate Adam age"),
            ("dropped_nonfinite_position", "prepared legacy importer does not compact invalid positions"),
            ("sanitizedSH=", "prepared legacy importer lacks finite-value repair diagnostics"),
            ("Saved PCS 3DGS checkpoint", "prepared trainer does not save continuation state"),
        ]:
            require(prepared, needle, why)
        if "legacy 3DGS PLY contains non-finite values" in prepared:
            raise SystemExit("resumable training check failed: prepared importer still rejects the whole PLY for one non-finite channel")

    print("PCS exact checkpoint + viewer-compatible legacy PLY 3DGS continuation checks passed")


if __name__ == "__main__":
    main()
