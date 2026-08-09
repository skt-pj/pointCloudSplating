#!/usr/bin/env python3
from pathlib import Path

renderer_path = Path("app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp")
renderer = renderer_path.read_text()

# Android uses a fixed 256-invocation cumsum workgroup. Do not derive cumsum geometry from
# VkPhysicalDeviceSubgroupProperties: Mali subgroup width is an implementation detail and the
# prefix sum must remain correct regardless of whether a driver exposes 16- or 32-lane subgroups.
old_geometry = '''    const size_t block_0 = 1024;
    const size_t block_limit = deviceInfo.subgroupSize*deviceInfo.subgroupSize*deviceInfo.subgroupSize;
    const size_t block = std::min(block_0, block_limit);
'''
new_geometry = '''    const size_t block = 256;
'''
if old_geometry not in renderer:
    raise SystemExit("cumsum geometry patch anchor not found")
renderer = renderer.replace(old_geometry, new_geometry, 1)

old_single_pass = '''    if (num_elements <= block_0) {
        executeCompute(
            {{num_elements, block_0}},
'''
new_single_pass = '''    if (num_elements <= block) {
        executeCompute(
            {{num_elements, block}},
'''
if old_single_pass not in renderer:
    raise SystemExit("cumsum single-pass geometry patch anchor not found")
renderer = renderer.replace(old_single_pass, new_single_pass, 1)

old_two_level = '''        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.scan_block_sums,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );
'''
new_two_level = '''        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        const uint32_t num_blocks = (uint32_t)_CEIL_DIV(num_elements, block);
        uint32_t block_uniforms[1] = { num_blocks };
        executeCompute(
            {{num_blocks, block}},
            block_uniforms, uniform_size,
            pipeline_cumsum.scan_block_sums,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            }
        );
'''
if old_two_level not in renderer:
    raise SystemExit("two-level cumsum patch anchor not found")
renderer = renderer.replace(old_two_level, new_two_level, 1)

old_three_level = '''        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.block_scan,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements_1/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.scan_block_sums,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements/block, block}},
            uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );
'''
new_three_level = '''        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        uint32_t level1_uniforms[1] = { (uint32_t)num_elements_1 };
        executeCompute(
            {{num_elements_1, block}},
            level1_uniforms, uniform_size,
            pipeline_cumsum.block_scan,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
            { buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        const size_t num_elements_2 = _CEIL_DIV(num_elements_1, block);
        uint32_t level2_uniforms[1] = { (uint32_t)num_elements_2 };
        executeCompute(
            {{num_elements_2, block}},
            level2_uniforms, uniform_size,
            pipeline_cumsum.scan_block_sums,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );

        bufferMemoryBarrier({
            { buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
        }, COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_elements_1, block}},
            level1_uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            }
        );
'''
if old_three_level not in renderer:
    raise SystemExit("three-level cumsum patch anchor not found")
renderer = renderer.replace(old_three_level, new_three_level, 1)
renderer_path.write_text(renderer)

# Replace the pinned desktop-oriented wave/subgroup scan with a subgroup-independent workgroup
# scan. All 256 invocations participate in every GroupMemoryBarrierWithGroupSync(). Partial blocks
# are zero padded, so the final lane always carries the inclusive block sum.
shader_path = Path("app/src/main/cpp/third_party/vksplat/vksplat/slang/cumsum.slang")
shader_path.write_text(r'''#define CUMSUM_PHASE_blockScan 1
#define CUMSUM_PHASE_scanBlockSums 2
#define CUMSUM_PHASE_addBlockOffsets 3
#define CUMSUM_PHASE_singlePassPrefixSum 0

#ifndef CUMSUM_PHASE
#define CUMSUM_PHASE -1
#endif

#if CUMSUM_PHASE == CUMSUM_PHASE_blockScan
#define blockScan main
#elif CUMSUM_PHASE == CUMSUM_PHASE_scanBlockSums
#define scanBlockSums main
#elif CUMSUM_PHASE == CUMSUM_PHASE_addBlockOffsets
#define addBlockOffsets main
#elif CUMSUM_PHASE == CUMSUM_PHASE_singlePassPrefixSum
#define singlePassPrefixSum main
#endif

static const uint BLOCK_SIZE = 256;

layout(binding=0) StructuredBuffer<int32_t> g_input;
layout(binding=1) RWStructuredBuffer<int32_t> g_output;
layout(binding=2) RWStructuredBuffer<int32_t> g_blockSums;

groupshared int32_t s_data[BLOCK_SIZE];

struct Uniforms {
    uint32_t numElements;
};

void inclusiveWorkgroupScan(uint tid) {
    [ForceUnroll]
    for (uint offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int32_t addend = (tid >= offset) ? s_data[tid - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        if (tid >= offset)
            s_data[tid] += addend;
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(BLOCK_SIZE, 1, 1)]
void blockScan(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    uint tid = localId.x;
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;

    s_data[tid] = (gid < uniforms.numElements) ? g_input[gid] : 0;
    GroupMemoryBarrierWithGroupSync();
    inclusiveWorkgroupScan(tid);

    if (gid < uniforms.numElements)
        g_output[gid] = s_data[tid];
    if (tid == BLOCK_SIZE - 1)
        g_blockSums[blockId] = s_data[tid];
}

[numthreads(BLOCK_SIZE, 1, 1)]
void scanBlockSums(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    uint tid = localId.x;
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;

    s_data[tid] = (gid < uniforms.numElements) ? g_blockSums[gid] : 0;
    GroupMemoryBarrierWithGroupSync();
    inclusiveWorkgroupScan(tid);

    if (gid < uniforms.numElements)
        g_blockSums[gid] = s_data[tid];
}

[numthreads(BLOCK_SIZE, 1, 1)]
void addBlockOffsets(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    uint tid = localId.x;
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;
    if (gid < uniforms.numElements && blockId > 0)
        g_output[gid] += g_blockSums[blockId - 1];
}

[numthreads(BLOCK_SIZE, 1, 1)]
void singlePassPrefixSum(
    uint3 localId : SV_GroupThreadID,
    uint3 globalId : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint tid = localId.x;
    uint gid = globalId.x;

    s_data[tid] = (gid < uniforms.numElements) ? g_input[gid] : 0;
    GroupMemoryBarrierWithGroupSync();
    inclusiveWorkgroupScan(tid);

    if (gid < uniforms.numElements)
        g_output[gid] = s_data[tid];
}
''')

print("Patched VkSplat cumsum for Android with subgroup-independent 256-thread scan")
