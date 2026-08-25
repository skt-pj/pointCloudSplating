#!/usr/bin/env python3
from pathlib import Path

checkout = Path("app/src/main/cpp/third_party/vksplat")
root = checkout / "vksplat"
config_path = root / "shader/radix_sort/config.glsl"
downsweep_path = root / "shader/radix_sort/downsweep.comp"
renderer_path = root / "src/gs_renderer.cpp"
compiler_path = checkout / "compile_shaders.py"

config = config_path.read_text()
old_config = '''#define SUBGROUP_SIZE 32

// #define KEY_BITS 64
// #define keyType uint64_t
#define KEY_BITS 32
#define keyType uint32_t

const int RADIX = 256;
#define WORKGROUP_SIZE 512
#define PARTITION_DIVISION 8
const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
'''
new_config = '''// Pixel 10a Mali-G715 runs this backend with native subgroup size 16. Keep the radix
// workgroup at exactly 16 subgroups so the shared-memory layout remains bounded and deterministic.
#define SUBGROUP_SIZE 16

// #define KEY_BITS 64
// #define keyType uint64_t
#define KEY_BITS 32
#define keyType uint32_t

const int RADIX = 256;
#define WORKGROUP_SIZE 256
#define PARTITION_DIVISION 8
const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
'''
if old_config not in config:
    raise SystemExit("radix config patch anchor not found")
config = config.replace(old_config, new_config, 1)
config_path.write_text(config)

downsweep = downsweep_path.read_text()
old_shmem = '''const uint SHMEM_SIZE = PARTITION_SIZE;

shared uint32_t localHistogram[(KEY_BITS/32)*SHMEM_SIZE];  // (R, S=16)=4096, (P) for alias. take maximum.
shared uint32_t localHistogramSum[RADIX];
'''
new_shmem = '''// The histogram needs RADIX * number-of-subgroups entries. With 256 threads and native
// subgroup size 16 that is 4096 uints, which is larger than the 2048-element partition alias.
// Allocate for the histogram explicitly; this stays below the Pixel 10a 32 KiB shared-memory limit.
const uint SHMEM_SIZE = RADIX * (WORKGROUP_SIZE / SUBGROUP_SIZE);

shared uint32_t localHistogram[(KEY_BITS/32)*SHMEM_SIZE];
shared uint32_t localHistogramSum[RADIX];
'''
if old_shmem not in downsweep:
    raise SystemExit("radix downsweep shared-memory patch anchor not found")
downsweep = downsweep.replace(old_shmem, new_shmem, 1)

old_helpers = '''// returns 0b00000....11111, where msb is id-1.
uvec4 GetExclusiveSubgroupMask(uint id) {
  return uvec4(
    (1 << id) - 1,
    (1 << (id - 32)) - 1,
    (1 << (id - 64)) - 1,
    (1 << (id - 96)) - 1
  );
}

uint GetBitCount(uvec4 value) {
  uvec4 result = bitCount(value);
  return result[0] + result[1] + result[2] + result[3];
}

'''
if old_helpers not in downsweep:
    raise SystemExit("radix downsweep ballot-helper patch anchor not found")
downsweep = downsweep.replace(old_helpers, "", 1)

old_mask = '''  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;
  uvec4 subgroupMask = GetExclusiveSubgroupMask(threadIndex);

  uint partitionIndex = gl_WorkGroupID.x;
'''
new_mask = '''  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;

  uint partitionIndex = gl_WorkGroupID.x;
'''
if old_mask not in downsweep:
    raise SystemExit("radix downsweep subgroup mask anchor not found")
downsweep = downsweep.replace(old_mask, new_mask, 1)

old_counts = '''    // subgroup level offset for radix
    uint subgroupOffset = GetBitCount(subgroupMask & mask);
    uint radixCount = GetBitCount(mask);
'''
new_counts = '''    // subgroup level offset for radix. Use the Vulkan subgroup ballot operations directly;
    // unlike the old handcrafted 32/64/96-bit shifts this is valid for native subgroup size 16.
    uint subgroupOffset = subgroupBallotExclusiveBitCount(mask);
    uint radixCount = subgroupBallotBitCount(mask);
'''
if old_counts not in downsweep:
    raise SystemExit("radix downsweep subgroup count anchor not found")
downsweep = downsweep.replace(old_counts, new_counts, 1)
downsweep_path.write_text(downsweep)

renderer = renderer_path.read_text()
old_geometry = '''    const int RADIX = 256;
    const int WORKGROUP_SIZE = 512;
    const int PARTITION_DIVISION = 8;
    const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
'''
new_geometry = '''    const int RADIX = 256;
    const int WORKGROUP_SIZE = 256;
    const int PARTITION_DIVISION = 8;
    const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
'''
if old_geometry not in renderer:
    raise SystemExit("radix renderer geometry patch anchor not found")
renderer = renderer.replace(old_geometry, new_geometry, 1)
renderer_path.write_text(renderer)

# NDK glslc intentionally does not implement GL_ARB_shading_language_include. Make each radix
# shader self-contained by replacing the single local include with the already patched Android
# config. This is preprocessing only; the shader algorithm remains the VkSplat radix implementation.
inlined_config = "// inlined config.glsl for Android NDK glslc\n" + config.rstrip() + "\n"
for name in ("upsweep.comp", "spine.comp", "downsweep.comp"):
    shader_path = root / "shader/radix_sort" / name
    shader = shader_path.read_text()
    extension_line = "#extension GL_ARB_shading_language_include : require\n"
    include_line = '#include "./config.glsl"\n'
    if extension_line not in shader or include_line not in shader:
        raise SystemExit(f"radix include patch anchor not found: {name}")
    shader = shader.replace(extension_line, "", 1)
    shader = shader.replace(include_line, inlined_config, 1)
    shader_path.write_text(shader)

# prepare-vksplat-android.sh deliberately removes the upstream radix job because VkSplat's glslc
# wrapper passes a slang-only '-target spirv' argument. Re-add the job after fixing that wrapper so
# the Android NDK glslc recompiles all three radix shaders from the patched subgroup-16 GLSL.
compiler = compiler_path.read_text()
compile_anchor = '''        # Add target and other args
        if target is not None:
            cmd.extend(["-target", target])
        cmd.extend(self.config.glslc_compile_args.split())
'''
compile_replacement = '''        # glslc emits SPIR-V directly for compute GLSL. '-target' is a slangc option and must not
        # be forwarded to the Android NDK glslc path.
        if target not in (None, "spirv"):
            raise RuntimeError(f"unsupported glslc target: {target}")
        cmd.extend(self.config.glslc_compile_args.split())
'''
if compile_anchor not in compiler:
    raise SystemExit("radix glslc target patch anchor not found")
compiler = compiler.replace(compile_anchor, compile_replacement, 1)

job_anchor = '''        # Morton sorting Functions
'''
radix_job = '''        # Radix sort - Android recompiles this from subgroup-16 GLSL with NDK glslc.
        jobs.append(("radix_sort", [
            ShaderJob("upsweep.comp", {}),
            ShaderJob("spine.comp", {}),
            ShaderJob("downsweep.comp", {}),
        ], []))

'''
if job_anchor not in compiler:
    raise SystemExit("radix job insertion anchor not found")
if 'ShaderJob("upsweep.comp", {})' in compiler:
    raise SystemExit("radix job already present before Android patch")
compiler = compiler.replace(job_anchor, radix_job + job_anchor, 1)
compiler_path.write_text(compiler)

print("Patched VkSplat radix sort for Mali native subgroup 16 and NDK glslc compilation")
