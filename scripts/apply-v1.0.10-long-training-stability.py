#!/usr/bin/env python3
"""Apply v1.0.10 runtime identifiers for long-training numerical stability."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"

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
print("Applied v1.0.10 long-training runtime identifiers")
