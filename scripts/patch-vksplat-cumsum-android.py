#!/usr/bin/env python3
from pathlib import Path

renderer_path = Path("app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp")
renderer = renderer_path.read_text()

include_anchor = '#include <csignal>\n'
include_replacement = '''#include <csignal>\n\n#ifdef __ANDROID__\n#include <android/log.h>\n#define PCS_CUMSUM_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "Native3DGS", __VA_ARGS__)\n#define PCS_CUMSUM_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "Native3DGS", __VA_ARGS__)\n#else\n#define PCS_CUMSUM_LOGI(...) ((void)0)\n#define PCS_CUMSUM_LOGE(...) ((void)0)\n#endif\n'''
if include_anchor not in renderer:
    raise SystemExit("renderer Android logging include anchor not found")
renderer = renderer.replace(include_anchor, include_replacement, 1)

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

# Run a deterministic GPU-only cumsum self-test as soon as renderer pipelines exist. This exercises
# the single-pass, two-level, and three-level paths before any dataset allocations or training work.
# A driver/pipeline failure is therefore separated from model-data failures in diagnostics.
self_test_anchor = '''    createComputePipeline(pipeline_sum, spirv_paths.at("sum"));
    createComputePipeline(pipeline_where, spirv_paths.at("where"));

}
'''
self_test_replacement = r'''    createComputePipeline(pipeline_sum, spirv_paths.at("sum"));
    createComputePipeline(pipeline_where, spirv_paths.at("where"));

    PCS_CUMSUM_LOGI("GPU cumsum self-test begin block=256");
    VulkanGSPipelineBuffers pcsTestBuffers;
    Buffer<int32_t> pcsTestInput;
    Buffer<int32_t> pcsTestOutput;
    auto pcsCleanupSelfTest = [&]() {
        if (pcsTestInput.deviceBuffer.buffer != VK_NULL_HANDLE)
            destroyBuffer(pcsTestInput.deviceBuffer);
        if (pcsTestOutput.deviceBuffer.buffer != VK_NULL_HANDLE)
            destroyBuffer(pcsTestOutput.deviceBuffer);
        cleanupBuffers(pcsTestBuffers);
    };
    try {
        const size_t pcsTestSizes[] = {1, 17, 255, 256, 257, 1025, 37635, 90000};
        for (size_t n : pcsTestSizes) {
            pcsTestInput.resize(n);
            std::vector<int32_t> expected(n);
            int64_t running = 0;
            for (size_t i = 0; i < n; ++i) {
                const int32_t value = (i % 13 == 0) ? 0 : (1 + static_cast<int32_t>(i % 5));
                pcsTestInput[i] = value;
                running += value;
                expected[i] = static_cast<int32_t>(running);
            }
            copyToDevice(pcsTestInput);
            executeCumsum(pcsTestBuffers, pcsTestInput, pcsTestOutput);
            copyFromDevice(pcsTestOutput);
            if (pcsTestOutput.size() != n) {
                PCS_CUMSUM_LOGE("GPU cumsum self-test size mismatch n=%zu got=%zu", n, pcsTestOutput.size());
                throw std::runtime_error("GPU cumsum self-test output size mismatch");
            }
            for (size_t i = 0; i < n; ++i) {
                if (pcsTestOutput[i] != expected[i]) {
                    PCS_CUMSUM_LOGE(
                        "GPU cumsum self-test FAIL n=%zu index=%zu expected=%d actual=%d",
                        n, i, expected[i], pcsTestOutput[i]);
                    throw std::runtime_error(
                        "GPU cumsum self-test mismatch n=" + std::to_string(n) +
                        " index=" + std::to_string(i) +
                        " expected=" + std::to_string(expected[i]) +
                        " actual=" + std::to_string(pcsTestOutput[i]));
                }
            }
            PCS_CUMSUM_LOGI("GPU cumsum self-test PASS n=%zu last=%d", n, expected.back());
        }
    } catch (...) {
        pcsCleanupSelfTest();
        throw;
    }
    pcsCleanupSelfTest();
    PCS_CUMSUM_LOGI("GPU cumsum self-test COMPLETE");

}
'''
if self_test_anchor not in renderer:
    raise SystemExit("renderer self-test insertion anchor not found")
renderer = renderer.replace(self_test_anchor, self_test_replacement, 1)

renderer_path.write_text(renderer)
print("Patched VkSplat cumsum dispatch hierarchy and GPU self-test for Android")
