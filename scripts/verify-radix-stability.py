#!/usr/bin/env python3
from pathlib import Path
import os
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VKSPLAT = ROOT / "app/src/main/cpp/third_party/vksplat/vksplat"
UPS = VKSPLAT / "shader/radix_sort/upsweep.comp"
SPINE = VKSPLAT / "shader/radix_sort/spine.comp"
DOWN = VKSPLAT / "shader/radix_sort/downsweep.comp"
RENDERER = VKSPLAT / "src/gs_renderer.cpp"
JAVA = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/NativeGaussianTrainer.java"


def fail(message: str) -> None:
    raise SystemExit(f"radix stability check failed: {message}")


def local_size(path: Path) -> tuple[int, int, int]:
    raw = path.read_bytes()
    if len(raw) < 20 or len(raw) % 4:
        fail(f"invalid SPIR-V size: {path}")
    words = struct.unpack(f"<{len(raw)//4}I", raw)
    if words[0] != 0x07230203:
        fail(f"invalid SPIR-V magic: {path}")
    i = 5
    while i < len(words):
        wc = words[i] >> 16
        op = words[i] & 0xffff
        if wc == 0 or i + wc > len(words):
            break
        if op == 16 and wc >= 6 and words[i + 2] == 17:
            return words[i + 3], words[i + 4], words[i + 5]
        i += wc
    fail(f"LocalSize missing: {path}")


def main() -> None:
    upsweep = UPS.read_text(encoding="utf-8")
    spine = SPINE.read_text(encoding="utf-8")
    downsweep = DOWN.read_text(encoding="utf-8")
    renderer = RENDERER.read_text(encoding="utf-8")
    java = JAVA.read_text(encoding="utf-8")

    if "atomicAdd(globalHistogram" in upsweep:
        fail("upsweep still performs contended global histogram atomics")
    if "partitionHistogram.data[RADIX * partitionIndex + index] = localHistogram[index]" not in upsweep:
        fail("upsweep no longer publishes per-partition histograms")
    if "globalHistogram.data[RADIX * pass + radix] = reduction" not in spine:
        fail("first spine stage does not publish one total per radix")
    if "gl_NumWorkGroups.x == 1" not in spine:
        fail("second synchronized global-prefix spine stage is missing")
    if "subgroupExclusiveAdd" not in spine:
        fail("spine exclusive scan missing")
    if "subgroupBallotExclusiveBitCount" not in downsweep or "subgroupBallotBitCount" not in downsweep:
        fail("downsweep is not subgroup-size-safe")
    if renderer.count("pipeline_sorting.spine") < 2:
        fail("renderer does not dispatch both spine stages")
    if "{{1, 1}}" not in renderer:
        fail("single-workgroup global prefix dispatch missing")

    active_cache = re.search(r'^\s*private static final String SHADER_CACHE = "([^"]+)";', java, re.M)
    expected_cache = "vksplat_shader_41cff93b_glslc256_radix16_v9"
    if not active_cache or active_cache.group(1) != expected_cache:
        fail(f"active shader cache is not v9: {active_cache.group(1) if active_cache else 'missing'}")
    if "two_stage_no_global_atomic" not in java:
        fail("runtime radix-generation diagnostic marker missing")
    if "numeric=finite_projection+mcmc_relocation+adam_rollback" not in java:
        fail("numerical-stability cache generation marker missing")
    if "shape=metric_scale_cap_6cm+mcmc_cumulative+resume_sanitize" not in java:
        fail("v1.0.11 shape-stability cache generation marker missing")

    shader_dir = VKSPLAT / "shader/radix_sort"
    spv_paths = [shader_dir / "upsweep.spv", shader_dir / "spine.spv", shader_dir / "downsweep.spv"]
    for path in spv_paths:
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"compiled shader missing: {path.name}")
        if local_size(path) != (256, 1, 1):
            fail(f"unexpected LocalSize for {path.name}: {local_size(path)}")

    ndk = os.environ.get("ANDROID_NDK_HOME") or os.environ.get("ANDROID_NDK_ROOT")
    if ndk:
        validator = Path(ndk) / "shader-tools/linux-x86_64/spirv-val"
        if validator.is_file():
            for path in spv_paths:
                subprocess.run([str(validator), "--target-env", "vulkan1.2", str(path)], check=True)

    print("Radix stability checks passed: subgroup16 local256, no upsweep global atomics, two-stage spine prefix, cache v9")


if __name__ == "__main__":
    main()
