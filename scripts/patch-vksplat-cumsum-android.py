#!/usr/bin/env python3
from pathlib import Path

renderer_path = Path("app/src/main/cpp/third_party/vksplat/vksplat/src/gs_renderer.cpp")
renderer = renderer_path.read_text()

# Pixel 10a / Mali showed two independent failures in the pinned desktop cumsum path:
# 1) the subgroup implementation returned deterministic wrong totals;
# 2) a subgroup-independent Slang shared-memory replacement caused a native driver abort while
#    creating the compute pipeline, before training started.
#
# Correctness is more important than keeping this small scan on the GPU. Android therefore uses a
# CPU prefix scan for cumsum only. Projection/rasterization/loss/backward/optimizer remain Vulkan.
# At the current 90k initial / 120k maximum Gaussian budget this transfers at most about 480 KiB in
# each direction per scan and completely removes cumsum shader/subgroup/driver-compiler dependence.

old_pipeline_init = '''    createComputePipeline(pipeline_cumsum.single_pass, spirv_paths.at("cumsum_single_pass"));
    createComputePipeline(pipeline_cumsum.block_scan, spirv_paths.at("cumsum_block_scan"));
    createComputePipeline(pipeline_cumsum.scan_block_sums, spirv_paths.at("cumsum_scan_block_sums"));
    createComputePipeline(pipeline_cumsum.add_block_offsets, spirv_paths.at("cumsum_add_block_offsets"));
'''
new_pipeline_init = '''    // PCS Android: cumsum uses the CPU prefix scan below. Do not create cumsum Vulkan pipelines;
    // Mali-G715 aborted inside compute-pipeline creation for the replacement shared-memory SPIR-V.
'''
if old_pipeline_init not in renderer:
    raise SystemExit("cumsum pipeline-init patch anchor not found")
renderer = renderer.replace(old_pipeline_init, new_pipeline_init, 1)

start_marker = '''void VulkanGSRenderer::executeCumsum(
'''
end_marker = '''void VulkanGSRenderer::executeCalculateIndexBufferOffset(
'''
start = renderer.find(start_marker)
end = renderer.find(end_marker, start)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("cumsum function patch anchors not found")

cpu_cumsum = r'''void VulkanGSRenderer::executeCumsum(
    VulkanGSPipelineBuffers &buffers,
    Buffer<int32_t> &input_buffer,
    Buffer<int32_t> &output_buffer
) {
    PerfTimer::Timer<PerfTimer::_Cumsum> timer(this);

    // PCS Android CPU prefix scan. copyFromDevice/copyToDevice use VkSplat's staging buffer and
    // HostGuard, so an enclosing command batch is safely submitted, synchronized, and resumed.
    const size_t num_elements = input_buffer.deviceSize();
    if (num_elements == 0) {
        output_buffer.clear();
        output_buffer.deviceBuffer.size = 0;
        return;
    }

    copyFromDevice(input_buffer);
    if (input_buffer.size() != num_elements)
        _THROW_ERROR("CPU cumsum input size changed during device readback");

    output_buffer.resize(num_elements);
    int64_t running = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        running += static_cast<int64_t>(input_buffer[i]);
        if (running > 2147483647LL || running < -2147483648LL)
            _THROW_ERROR("CPU cumsum exceeded int32 range");
        output_buffer[i] = static_cast<int32_t>(running);
    }

    copyToDevice(output_buffer);
}

'''
renderer = renderer[:start] + cpu_cumsum + renderer[end:]

# The CPU scan already leaves the inclusive prefix values in the host Buffer. Avoid an extra
# one-element GPU readback and take num_indices directly from the last prefix value.
old_num_indices = '''    if (commandBatchInProgress) bufferMemoryBarrier({
        { buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_READ_WRITE },
    }, TRANSFER_READ);
    int num_indices = readElement<int32_t>(buffers.index_buffer_offset.deviceBuffer, num_elements-1);
    buffers.num_indices = (size_t)num_indices;
'''
new_num_indices = '''    if (buffers.index_buffer_offset.empty()) {
        buffers.num_indices = 0;
    } else {
        int32_t num_indices = buffers.index_buffer_offset.back();
        if (num_indices < 0)
            _THROW_ERROR("CPU cumsum produced a negative tile-index count");
        buffers.num_indices = static_cast<size_t>(num_indices);
    }
'''
if old_num_indices not in renderer:
    raise SystemExit("cumsum num_indices readback patch anchor not found")
renderer = renderer.replace(old_num_indices, new_num_indices, 1)

renderer_path.write_text(renderer)
print("Patched VkSplat cumsum for Android with CPU prefix scan; no cumsum Vulkan pipeline is created")
