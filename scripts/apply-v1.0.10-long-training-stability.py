#!/usr/bin/env python3
"""Apply v1.0.10 runtime identifiers for long-training numerical stability."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"
PHASE3_VERIFY = ROOT / "scripts/verify-phase3.py"

text = TRAINER.read_text(encoding="utf-8")
old = 'private static final String SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v7";'
new = 'private static final String SHADER_CACHE = "vksplat_shader_41cff93b_glslc256_radix16_v8";'
if new not in text:
    if old not in text:
        raise SystemExit("v1.0.10 patch failed: shader cache v7 anchor missing")
    text = text.replace(old, new, 1)

old_diag = ' + " cumsum=glslc256 radix=glslc256/subgroup16/two_stage_no_global_atomic"\n'
new_diag = (' + " cumsum=glslc256 radix=glslc256/subgroup16/two_stage_no_global_atomic"\n'
            '                            + " numeric=finite_projection+mcmc_relocation+adam_rollback"\n')
if "numeric=finite_projection+mcmc_relocation+adam_rollback" not in text:
    if old_diag not in text:
        raise SystemExit("v1.0.10 patch failed: trainer diagnostic anchor missing")
    text = text.replace(old_diag, new_diag, 1)
TRAINER.write_text(text, encoding="utf-8")

verify = PHASE3_VERIFY.read_text(encoding="utf-8")
old_version = '''    require(version, "VERSION_NAME=1.0.9", "versionName mismatch")\n    require(version, "VERSION_CODE=46", "versionCode mismatch")\n'''
new_version = '''    values = dict(\n        line.split("=", 1) for line in version.splitlines() if "=" in line\n    )\n    try:\n        version_code = int(values.get("VERSION_CODE", "0"))\n    except ValueError:\n        raise SystemExit("Phase 3 check failed: invalid VERSION_CODE")\n    if version_code < 46:\n        raise SystemExit("Phase 3 check failed: Phase 3 requires VERSION_CODE >= 46")\n'''
if new_version not in verify:
    if old_version not in verify:
        raise SystemExit("v1.0.10 patch failed: Phase 3 version verifier anchor missing")
    verify = verify.replace(old_version, new_version, 1)
PHASE3_VERIFY.write_text(verify, encoding="utf-8")

print("Applied v1.0.10 long-training runtime identifiers and verifier compatibility")
