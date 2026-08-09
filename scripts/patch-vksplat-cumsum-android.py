#!/usr/bin/env python3
from pathlib import Path

renderer_path = Path("app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp")
renderer = renderer_path.read_text()

# Android uses a fixed 256-invocation cumsum workgroup. Keep the C++ dispatch geometry exactly
# aligned with the platform GLSL shader compiled by NDK glslc. Cumsum itself remains on the GPU.
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

# The PCS GLSL cumsum implementation is subgroup-independent. Do not ask Vulkan to force a
# particular subgroup width for these four pipelines. This keeps the pipeline contract aligned with
# the shader and avoids coupling cumsum validity to the rasterizer's subgroup configuration.
old_pipelines = '''    createComputePipeline(pipeline_cumsum.single_pass, spirv_paths.at("cumsum_single_pass"));
    createComputePipeline(pipeline_cumsum.block_scan, spirv_paths.at("cumsum_block_scan"));
    createComputePipeline(pipeline_cumsum.scan_block_sums, spirv_paths.at("cumsum_scan_block_sums"));
    createComputePipeline(pipeline_cumsum.add_block_offsets, spirv_paths.at("cumsum_add_block_offsets"));
'''
new_pipelines = '''    createComputePipeline(pipeline_cumsum.single_pass, spirv_paths.at("cumsum_single_pass"), 0, false);
    createComputePipeline(pipeline_cumsum.block_scan, spirv_paths.at("cumsum_block_scan"), 0, false);
    createComputePipeline(pipeline_cumsum.scan_block_sums, spirv_paths.at("cumsum_scan_block_sums"), 0, false);
    createComputePipeline(pipeline_cumsum.add_block_offsets, spirv_paths.at("cumsum_add_block_offsets"), 0, false);
'''
if old_pipelines not in renderer:
    raise SystemExit("cumsum pipeline creation anchor not found")
renderer = renderer.replace(old_pipelines, new_pipelines, 1)

renderer_path.write_text(renderer)
print("Patched VkSplat cumsum hierarchy and subgroup-independent pipeline contract for Android")
