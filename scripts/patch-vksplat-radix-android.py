#!/usr/bin/env python3
from pathlib import Path

root = Path("app/src/main/cpp/third_party/vksplat/vksplat")
config_path = root / "shader/radix_sort/config.glsl"
downsweep_path = root / "shader/radix_sort/downsweep.comp"
renderer_path = root / "src/gs_renderer.cpp"

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

print("Patched VkSplat radix sort for Mali native subgroup 16 and 256-thread workgroups")
