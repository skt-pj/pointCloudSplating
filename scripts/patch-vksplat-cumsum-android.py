#!/usr/bin/env python3
from pathlib import Path

renderer_path = Path("app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp")
renderer = renderer_path.read_text()

# VkSplat's block scan assumes the list of subgroup sums fits in exactly one subgroup. That is true
# for its desktop default (1024 threads / subgroup 32 = 32 sums), but false on Mali subgroup 16
# (1024 / 16 = 64 sums). Bound the block to subgroup^2 so the second-level WavePrefixSum always
# operates on at most one subgroup worth of partial sums.
old_block_limit = (
    "    const size_t block_limit = deviceInfo.subgroupSize*deviceInfo.subgroupSize*deviceInfo.subgroupSize;\n"
)
new_block_limit = (
    "    const size_t block_limit = deviceInfo.subgroupSize*deviceInfo.subgroupSize;\n"
)
if old_block_limit not in renderer:
    raise SystemExit("cumsum block-limit patch anchor not found")
renderer = renderer.replace(old_block_limit, new_block_limit, 1)

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

shader_path = Path("app/src/main/cpp/third_party/vksplat/vksplat/slang/cumsum.slang")
shader = shader_path.read_text()
old_shader_block = "    #define BLOCK_SIZE min(BLOCK_SIZE_0, SUBGROUP_SIZE*SUBGROUP_SIZE*SUBGROUP_SIZE)\n"
new_shader_block = "    #define BLOCK_SIZE min(BLOCK_SIZE_0, SUBGROUP_SIZE*SUBGROUP_SIZE)\n"
if old_shader_block not in shader:
    raise SystemExit("cumsum shader block-size patch anchor not found")
shader = shader.replace(old_shader_block, new_shader_block, 1)

# Avoid an out-of-range WaveReadLaneAt on lanes smaller than offset. Desktop drivers commonly mask
# this benignly; Mali is not required to. Only read a predecessor lane when it actually exists.
old_subgroup_scan = '''    for (uint offset = 1; offset < SUBGROUP_SIZE; offset <<= 1) {
        // int32_t temp = WaveReadLaneAt(val, (laneId - offset) & (SUBGROUP_SIZE - 1));
        int32_t temp = WaveReadLaneAt(val, laneId - offset);
        if (laneId >= offset)
            val += temp;
    }
'''
new_subgroup_scan = '''    for (uint offset = 1; offset < SUBGROUP_SIZE; offset <<= 1) {
        if (laneId >= offset) {
            int32_t temp = WaveReadLaneAt(val, laneId - offset);
            val += temp;
        }
    }
'''
if old_subgroup_scan not in shader:
    raise SystemExit("cumsum subgroup scan patch anchor not found")
shader = shader.replace(old_subgroup_scan, new_subgroup_scan, 1)
shader_path.write_text(shader)

print("Patched VkSplat cumsum for Android subgroup-16 correctness")
