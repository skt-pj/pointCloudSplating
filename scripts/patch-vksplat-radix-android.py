#!/usr/bin/env python3
from pathlib import Path

checkout = Path("app/src/main/cpp/third_party/vksplat")
root = checkout / "vksplat"
config_path = root / "shader/radix_sort/config.glsl"
upsweep_path = root / "shader/radix_sort/upsweep.comp"
spine_path = root / "shader/radix_sort/spine.comp"
downsweep_path = root / "shader/radix_sort/downsweep.comp"
prefix_path = root / "shader/radix_sort/prefix.comp"
renderer_path = root / "src/gs_renderer.cpp"
renderer_header_path = root / "src/gs_renderer.h"
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
new_config = '''// Pixel 10a Tensor G4 GPU reports native subgroup size 16. Keep the radix workgroup at exactly
// 16 subgroups so the shared-memory layout is bounded and deterministic on the target device.
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

# The desktop shader performs one global atomicAdd per radix bin and partition. On the Pixel 10a
# this creates a highly contended storage-buffer atomic hotspot and can end in VK_ERROR_DEVICE_LOST
# while the upsweep dispatch is completing. The Android path only writes per-partition histograms;
# global totals are produced by spine and prefixed by a separate single-workgroup kernel below.
upsweep = upsweep_path.read_text()
old_global_atomic = '''    // add to global histogram
    atomicAdd(globalHistogram.data[RADIX * pass + index], localHistogram[index]);
    // atomicAdd(globalHistogram.data[RADIX * pass + index], 1);
'''
new_global_atomic = '''    // Android: do not contend on one global histogram from every partition. spine.comp reduces
    // these partition histograms and prefix.comp performs the final 256-bin prefix scan.
'''
if old_global_atomic not in upsweep:
    raise SystemExit("radix upsweep global-atomic patch anchor not found")
upsweep = upsweep.replace(old_global_atomic, new_global_atomic, 1)
upsweep_path.write_text(upsweep)

spine = spine_path.read_text()
old_global_prefix = '''  if (gl_WorkGroupID.x == 0) {
    // one workgroup is responsible for global histogram prefix sum
    uint32_t value, excl, sum;
    if (index < RADIX) {
      value = globalHistogram.data[RADIX * pass + index];
      excl = subgroupExclusiveAdd(value);
      sum = subgroupAdd(value);

      if (subgroupElect()) {
        intermediate[subgroupIndex] = sum;
      }
    }
    barrier();

    if (index < RADIX / gl_SubgroupSize) {
      uint32_t excl = subgroupExclusiveAdd(intermediate[index]);
      intermediate[index] = excl;
    }
    barrier();

    if (index < RADIX) {
      excl += intermediate[subgroupIndex];
      globalHistogram.data[RADIX * pass + index] = excl;
    }
  }
'''
new_global_prefix = '''  // Each workgroup owns one radix bin. After scanning every partition, reduction is the total
  // count for this bin. Store totals without atomics; prefix.comp will scan all 256 totals after
  // this dispatch has completed, giving Vulkan an explicit inter-dispatch synchronization point.
  if (index == 0) {
    globalHistogram.data[RADIX * pass + radix] = reduction;
  }
'''
if old_global_prefix not in spine:
    raise SystemExit("radix spine global-prefix patch anchor not found")
spine = spine.replace(old_global_prefix, new_global_prefix, 1)
spine_path.write_text(spine)

# Single-workgroup exclusive scan of the 256 global radix totals. This replaces the cross-workgroup
# dependency that would otherwise be required if spine tried to publish totals and prefix them in
# the same dispatch. No global atomics are used.
prefix_path.write_text(r'''#version 460 core

#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_KHR_shader_subgroup_basic: enable
#extension GL_KHR_shader_subgroup_arithmetic: enable

#extension GL_ARB_shading_language_include : require
#include "./config.glsl"

layout (local_size_x = WORKGROUP_SIZE) in;

layout (std430, binding = 0) restrict buffer GlobalHistogram {
  uint32_t data[];
} globalHistogram;

layout (push_constant) uniform PushConstant {
  uint32_t pass;
  uint32_t elementCount;
} uniforms;

shared uint32_t subgroupTotals[WORKGROUP_SIZE / SUBGROUP_SIZE];

void main() {
  uint index = gl_LocalInvocationID.x;
  uint subgroupIndex = gl_SubgroupID;
  uint value = index < RADIX ? globalHistogram.data[RADIX * uniforms.pass + index] : 0;
  uint excl = subgroupExclusiveAdd(value);
  uint total = subgroupAdd(value);

  if (subgroupElect()) {
    subgroupTotals[subgroupIndex] = total;
  }
  barrier();

  if (index < gl_NumSubgroups) {
    subgroupTotals[index] = subgroupExclusiveAdd(subgroupTotals[index]);
  }
  barrier();

  if (index < RADIX) {
    globalHistogram.data[RADIX * uniforms.pass + index] = excl + subgroupTotals[subgroupIndex];
  }
}
''')

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
new_counts = '''    // subgroup level offset for radix. Use Vulkan subgroup ballot operations directly; unlike
    // the old handcrafted 32/64/96-bit shifts this is valid for native subgroup size 16.
    uint subgroupOffset = subgroupBallotExclusiveBitCount(mask);
    uint radixCount = subgroupBallotBitCount(mask);
'''
if old_counts not in downsweep:
    raise SystemExit("radix downsweep subgroup count anchor not found")
downsweep = downsweep.replace(old_counts, new_counts, 1)
downsweep_path.write_text(downsweep)

header = renderer_header_path.read_text()
old_pipeline_struct = '''    struct _RadixSortComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(3);
        _ComputePipeline spine = _ComputePipeline(2);
        _ComputePipeline downsweep = _ComputePipeline(6);
    } pipeline_sorting_1, pipeline_sorting_2;
'''
new_pipeline_struct = '''    struct _RadixSortComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(3);
        _ComputePipeline spine = _ComputePipeline(2);
        _ComputePipeline prefix = _ComputePipeline(1);
        _ComputePipeline downsweep = _ComputePipeline(6);
    } pipeline_sorting_1, pipeline_sorting_2;
'''
if old_pipeline_struct not in header:
    raise SystemExit("radix renderer header pipeline anchor not found")
header = header.replace(old_pipeline_struct, new_pipeline_struct, 1)
renderer_header_path.write_text(header)

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

old_pipeline_init = '''    createComputePipeline(pipeline_sorting_1.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_1.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_1.downsweep, spirv_paths.at("radix_sort/downsweep"));
    createComputePipeline(pipeline_sorting_2.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_2.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_2.downsweep, spirv_paths.at("radix_sort/downsweep"));
'''
new_pipeline_init = '''    createComputePipeline(pipeline_sorting_1.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_1.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_1.prefix, spirv_paths.at("radix_sort/prefix"));
    createComputePipeline(pipeline_sorting_1.downsweep, spirv_paths.at("radix_sort/downsweep"));
    createComputePipeline(pipeline_sorting_2.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_2.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_2.prefix, spirv_paths.at("radix_sort/prefix"));
    createComputePipeline(pipeline_sorting_2.downsweep, spirv_paths.at("radix_sort/downsweep"));
'''
if old_pipeline_init not in renderer:
    raise SystemExit("radix renderer pipeline-init anchor not found")
renderer = renderer.replace(old_pipeline_init, new_pipeline_init, 1)

old_sort_sequence = '''        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{RADIX, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.spine,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ);
        executeCompute(
'''
new_sort_sequence = '''        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{RADIX, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.spine,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
            }
        );

        // spine publishes one total per radix bin. Prefix those 256 totals only after the spine
        // dispatch has completed; this removes the contended global atomics from upsweep and avoids
        // relying on cross-workgroup visibility inside one dispatch.
        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{1, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.prefix,
            {
                globalHistogram.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { globalHistogram.deviceBuffer, COMPUTE_SHADER_WRITE },
            { partitionHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ);
        executeCompute(
'''
if old_sort_sequence not in renderer:
    raise SystemExit("radix renderer sort-sequence anchor not found")
renderer = renderer.replace(old_sort_sequence, new_sort_sequence, 1)
renderer_path.write_text(renderer)

# NDK glslc intentionally does not implement GL_ARB_shading_language_include. Make each radix
# shader self-contained by replacing the single local include with the already patched Android
# config. This is preprocessing only; the shader algorithm remains VkSplat-compatible.
inlined_config = "// inlined config.glsl for Android NDK glslc\n" + config.rstrip() + "\n"
for name in ("upsweep.comp", "spine.comp", "prefix.comp", "downsweep.comp"):
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
# Android NDK glslc recompiles all radix shaders from the patched subgroup-16 GLSL.
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
radix_job = '''        # Radix sort - Android recompiles subgroup-16 GLSL and uses a separate global-prefix pass
        # to avoid high-contention global atomics on mobile GPUs.
        jobs.append(("radix_sort", [
            ShaderJob("upsweep.comp", {}),
            ShaderJob("spine.comp", {}),
            ShaderJob("prefix.comp", {}),
            ShaderJob("downsweep.comp", {}),
        ], []))

'''
if job_anchor not in compiler:
    raise SystemExit("radix job insertion anchor not found")
if 'ShaderJob("upsweep.comp", {})' in compiler:
    raise SystemExit("radix job already present before Android patch")
compiler = compiler.replace(job_anchor, radix_job + job_anchor, 1)
compiler_path.write_text(compiler)

print("Patched VkSplat radix sort for subgroup16 with atomic-free two-stage global histogram scan")
