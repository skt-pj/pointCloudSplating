#!/usr/bin/env python3
from pathlib import Path

root = Path("app/src/main/cpp/third_party/vksplat/vksplat/src")
header_path = root / "gs_pipeline.h"
pipeline_path = root / "gs_pipeline.cpp"
buffer_path = root / "buffer.cpp"
renderer_path = root / "gs_renderer.cpp"
trainer_path = root / "gs_trainer.cpp"

h = header_path.read_text()
s = pipeline_path.read_text()
b = buffer_path.read_text()
r = renderer_path.read_text()
t = trainer_path.read_text()

# This patch is intentionally applied after patch-vksplat-pipeline-diagnostics-android.py so
# failures can be reported through the same Native3DGS Android log tag.
if "PCS_VK_LOGE" not in s:
    raise SystemExit("pipeline diagnostics must be applied before command-batch safety patch")

include_anchor = '#include <functional>\n'
if include_anchor not in h:
    raise SystemExit("gs_pipeline.h include anchor not found")
h = h.replace(include_anchor, include_anchor + '#include <exception>\n', 1)

public_anchor = '''    void beginCommandBatch();
    void endCommandBatch(bool use_fence = true);
    bool isCommandBatchInProgress() const {
'''
public_replacement = '''    void beginCommandBatch();
    void endCommandBatch(bool use_fence = true);
    // Drop a recording batch during exception unwinding. Destructors must never submit or throw
    // while another exception is already active.
    void discardCommandBatchNoThrow(const char* reason) noexcept;
    bool isCommandBatchInProgress() const {
'''
if public_anchor not in h:
    raise SystemExit("gs_pipeline.h command-batch public anchor not found")
h = h.replace(public_anchor, public_replacement, 1)

state_anchor = '''    bool commandBatchInProgress = false;
    uint32_t timestampNumWritten = 0;
    uint32_t timestampStackDepth = 0;
    std::vector<std::function<void(const std::vector<std::pair<size_t, double>>&)>> timerCallbacks;
'''
state_replacement = '''    bool commandBatchInProgress = false;
    uint32_t timestampNumWritten = 0;
    uint32_t timestampStackDepth = 0;
    uint64_t pcsBatchSerial = 0;
    std::string pcsLastOperation = "none";
    std::vector<std::function<void(const std::vector<std::pair<size_t, double>>&)>> timerCallbacks;
'''
if state_anchor not in h:
    raise SystemExit("gs_pipeline.h command-batch state anchor not found")
h = h.replace(state_anchor, state_replacement, 1)

pipeline_member_anchor = '''        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;
        std::vector<int> buffer_layouts;
'''
pipeline_member_replacement = '''        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;
        std::vector<int> buffer_layouts;
        std::string debug_name;
'''
if pipeline_member_anchor not in h:
    raise SystemExit("gs_pipeline.h compute-pipeline member anchor not found")
h = h.replace(pipeline_member_anchor, pipeline_member_replacement, 1)

device_guard_old = '''    ~DeviceGuard() noexcept(false) {
        // printf("DeviceGuard destructor\\n");
        if (!cbip) {
            pipeline->endCommandBatch();
            if (debugInfo1) {
                printf("DeviceGuard freed: %s:%d\\n", debugInfo1, debugInfo2);
            }
        }
        else if (cbip != pipeline->isCommandBatchInProgress()) {
            fprintf(stderr, "commandBatchInProgress changed during DeviceGuard (originally %d)\\n", (int)cbip);
            std::terminate();
        }

    }
'''
device_guard_new = '''    ~DeviceGuard() noexcept(false) {
        // A submit failure or any other exception inside a guarded scope must not cause a second
        // throwing submit from this destructor. Preserve the original exception instead.
        if (!cbip) {
            if (std::uncaught_exceptions() > 0) {
                pipeline->discardCommandBatchNoThrow("DeviceGuard exception unwind");
            } else {
                pipeline->endCommandBatch();
            }
            if (debugInfo1) {
                printf("DeviceGuard freed: %s:%d\\n", debugInfo1, debugInfo2);
            }
        }
        else if (cbip != pipeline->isCommandBatchInProgress()) {
            if (std::uncaught_exceptions() > 0) {
                pipeline->discardCommandBatchNoThrow("DeviceGuard nested state change");
            } else {
                _THROW_ERROR_ALWAYS("commandBatchInProgress changed during DeviceGuard");
            }
        }

    }
'''
if device_guard_old not in h:
    raise SystemExit("DeviceGuard destructor anchor not found")
h = h.replace(device_guard_old, device_guard_new, 1)

host_guard_old = '''    ~HostGuard() noexcept(false) {
        // printf("HostGuard destructor\\n");
        if (cbip) {
            pipeline->beginCommandBatch();
            if (debugInfo1) {
                printf("HostGuard freed: %s:%d\\n", debugInfo1, debugInfo2);
            }
        }
        else if (cbip != pipeline->isCommandBatchInProgress()) {
            fprintf(stderr, "commandBatchInProgress changed during HostGuard (originally %d)\\n", (int)cbip);
            std::terminate();
        }

    }
'''
host_guard_new = '''    ~HostGuard() noexcept(false) {
        // Do not reopen a GPU batch while unwinding an exception. The enclosing DeviceGuard will
        // observe the inactive state and discard without submitting.
        if (cbip) {
            if (std::uncaught_exceptions() == 0) {
                pipeline->beginCommandBatch();
            }
            if (debugInfo1) {
                printf("HostGuard freed: %s:%d\\n", debugInfo1, debugInfo2);
            }
        }
        else if (cbip != pipeline->isCommandBatchInProgress()) {
            if (std::uncaught_exceptions() > 0) {
                pipeline->discardCommandBatchNoThrow("HostGuard nested state change");
            } else {
                _THROW_ERROR_ALWAYS("commandBatchInProgress changed during HostGuard");
            }
        }

    }
'''
if host_guard_old not in h:
    raise SystemExit("HostGuard destructor anchor not found")
h = h.replace(host_guard_old, host_guard_new, 1)

base_destructor_old = '''VulkanGSPipeline::~VulkanGSPipeline() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    cleanup();
}
'''
base_destructor_new = '''VulkanGSPipeline::~VulkanGSPipeline() {
    discardCommandBatchNoThrow("VulkanGSPipeline destructor");
    try {
        cleanup();
    } catch (const std::exception& error) {
        PCS_VK_LOGE("Vulkan cleanup ignored during destructor: %s", error.what());
    } catch (...) {
        PCS_VK_LOGE("Vulkan cleanup ignored unknown exception during destructor");
    }
}
'''
if base_destructor_old not in s:
    raise SystemExit("VulkanGSPipeline destructor anchor not found")
s = s.replace(base_destructor_old, base_destructor_new, 1)

begin_start = s.find('void VulkanGSPipeline::beginCommandBatch() {')
begin_end = s.find('\nvoid VulkanGSPipeline::endCommandBatch(bool use_fence) {', begin_start)
if begin_start < 0 or begin_end < 0:
    raise SystemExit("beginCommandBatch function anchor not found")
new_begin = r'''void VulkanGSPipeline::beginCommandBatch() {
    if (commandBatchInProgress)
        _THROW_ERROR("Command batch already in progress");

    PerfTimer::hostToc();

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    const VkResult begin_result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (begin_result != VK_SUCCESS) {
        _THROW_ERROR("vkBeginCommandBuffer failed VkResult=" +
                     std::to_string(static_cast<int>(begin_result)));
    }
    commandBatchInProgress = true;
    ++pcsBatchSerial;
    pcsLastOperation = "batch begin";

    try {
        vkCmdResetQueryPool(command_buffer, timestamp_query_pool, 0, MAX_TIMESTAMP_QUERY_COUNT);
        PerfTimer::popMarkers(this);
    } catch (...) {
        discardCommandBatchNoThrow("beginCommandBatch setup exception");
        throw;
    }
}
'''
s = s[:begin_start] + new_begin + s[begin_end+1:]

end_start = s.find('void VulkanGSPipeline::endCommandBatch(bool use_fence) {')
end_end = s.find('\nbool VulkanGSPipeline::writeTimestamp(int delta) {', end_start)
if end_start < 0 or end_end < 0:
    raise SystemExit("endCommandBatch function anchor not found")
new_end = r'''void VulkanGSPipeline::endCommandBatch(bool use_fence) {
    if (!commandBatchInProgress)
        _THROW_ERROR("No command batch in progress");

    if (timestampNumWritten > 0) {
        while (timestampStackDepth > 0)
            PerfTimer::pushMarker(this);
    }

    const VkResult end_result = vkEndCommandBuffer(command_buffer);
    if (end_result != VK_SUCCESS) {
        commandBatchInProgress = false;
        timestampNumWritten = 0;
        timestampStackDepth = 0;
        PCS_VK_LOGE("vkEndCommandBuffer failed result=%d batch=%llu op=%s",
                    static_cast<int>(end_result),
                    static_cast<unsigned long long>(pcsBatchSerial),
                    pcsLastOperation.c_str());
        _THROW_ERROR("vkEndCommandBuffer failed VkResult=" +
                     std::to_string(static_cast<int>(end_result)) +
                     " batch=" + std::to_string(pcsBatchSerial) +
                     " op=" + pcsLastOperation);
    }

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    const VkResult submit_result = vkQueueSubmit(
        command_queue, 1, &submit_info, use_fence ? fence : VK_NULL_HANDLE);
    // A failed submit must never leave the object claiming that a recording batch is active. The
    // old state caused destructors to submit the same batch again while unwinding, masking the
    // original Vulkan error with std::terminate/SIGABRT.
    commandBatchInProgress = false;
    if (submit_result != VK_SUCCESS) {
        timestampNumWritten = 0;
        timestampStackDepth = 0;
        PCS_VK_LOGE("vkQueueSubmit failed result=%d batch=%llu op=%s",
                    static_cast<int>(submit_result),
                    static_cast<unsigned long long>(pcsBatchSerial),
                    pcsLastOperation.c_str());
        _THROW_ERROR("vkQueueSubmit failed VkResult=" +
                     std::to_string(static_cast<int>(submit_result)) +
                     " batch=" + std::to_string(pcsBatchSerial) +
                     " op=" + pcsLastOperation);
    }

    if (use_fence) {
      #if SSE2_AVAILABLE
      #if ENABLE_ASSERTION
        constexpr unsigned long long kTimeout = 0x100000000ull;
        auto time0 = __rdtsc();
      #endif
        VkResult wait_result = VK_NOT_READY;
        while ((wait_result = vkGetFenceStatus(device, fence)) == VK_NOT_READY) {
            _mm_pause();
          #if ENABLE_ASSERTION
            if (__rdtsc() - time0 >= kTimeout) {
                wait_result = VK_TIMEOUT;
                break;
            }
          #endif
        }
      #else
        const VkResult wait_result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
      #endif
        if (wait_result != VK_SUCCESS) {
            timestampNumWritten = 0;
            timestampStackDepth = 0;
            PCS_VK_LOGE("Vulkan fence wait failed result=%d batch=%llu op=%s",
                        static_cast<int>(wait_result),
                        static_cast<unsigned long long>(pcsBatchSerial),
                        pcsLastOperation.c_str());
            _THROW_ERROR("Vulkan fence wait failed VkResult=" +
                         std::to_string(static_cast<int>(wait_result)) +
                         " batch=" + std::to_string(pcsBatchSerial) +
                         " op=" + pcsLastOperation);
        }
        const VkResult reset_result = vkResetFences(device, 1, &fence);
        if (reset_result != VK_SUCCESS) {
            timestampNumWritten = 0;
            timestampStackDepth = 0;
            _THROW_ERROR("vkResetFences failed VkResult=" +
                         std::to_string(static_cast<int>(reset_result)) +
                         " batch=" + std::to_string(pcsBatchSerial) +
                         " op=" + pcsLastOperation);
        }
    }
    else {
        const VkResult idle_result = vkQueueWaitIdle(command_queue);
        if (idle_result != VK_SUCCESS) {
            timestampNumWritten = 0;
            timestampStackDepth = 0;
            _THROW_ERROR("vkQueueWaitIdle failed VkResult=" +
                         std::to_string(static_cast<int>(idle_result)) +
                         " batch=" + std::to_string(pcsBatchSerial) +
                         " op=" + pcsLastOperation);
        }
    }

    PerfTimer::hostTic();

    if (timestampNumWritten > 0) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physical_device, &deviceProperties);
        double timestampPeriod = deviceProperties.limits.timestampPeriod;

        std::vector<uint64_t> timestamps(timestampNumWritten);
        const VkResult query_result = vkGetQueryPoolResults(
            device, timestamp_query_pool,
            0, timestampNumWritten,
            sizeof(uint64_t) * timestampNumWritten,
            timestamps.data(), sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
        );
        if (query_result != VK_SUCCESS) {
            timestampNumWritten = 0;
            timestampStackDepth = 0;
            _THROW_ERROR("vkGetQueryPoolResults failed VkResult=" +
                         std::to_string(static_cast<int>(query_result)) +
                         " batch=" + std::to_string(pcsBatchSerial) +
                         " op=" + pcsLastOperation);
        }
        std::vector<double> times(timestampNumWritten);
        for (uint32_t i = 0; i < timestampNumWritten; i++)
            times[i] = 1e-9 * double(timestamps[i] - timestamps[0]) * timestampPeriod;
        auto time_updates = PerfTimer::update(times);
        for (auto& callback : timerCallbacks)
            callback(time_updates);

        timestampNumWritten = 0;
    }
    timestampStackDepth = 0;
}

void VulkanGSPipeline::discardCommandBatchNoThrow(const char* reason) noexcept {
    if (!commandBatchInProgress)
        return;
    PCS_VK_LOGE("Discarding Vulkan command batch batch=%llu op=%s reason=%s",
                static_cast<unsigned long long>(pcsBatchSerial),
                pcsLastOperation.c_str(), reason ? reason : "unknown");
    if (command_buffer != VK_NULL_HANDLE) {
        const VkResult reset_result = vkResetCommandBuffer(command_buffer, 0);
        if (reset_result != VK_SUCCESS) {
            PCS_VK_LOGE("vkResetCommandBuffer while discarding failed result=%d batch=%llu",
                        static_cast<int>(reset_result),
                        static_cast<unsigned long long>(pcsBatchSerial));
        }
    }
    commandBatchInProgress = false;
    timestampNumWritten = 0;
    timestampStackDepth = 0;
}
'''
s = s[:end_start] + new_end + s[end_end+1:]

create_anchor = '''void VulkanGSPipeline::createComputePipeline(_ComputePipeline &pipeline, const std::string& spirv_path, uint32_t min_shared_memory, bool compatible_subgroup_size) {

'''
create_replacement = '''void VulkanGSPipeline::createComputePipeline(_ComputePipeline &pipeline, const std::string& spirv_path, uint32_t min_shared_memory, bool compatible_subgroup_size) {

    pipeline.debug_name = spirv_path;
'''
if create_anchor not in s:
    raise SystemExit("createComputePipeline debug-name anchor not found")
s = s.replace(create_anchor, create_replacement, 1)

execute_anchor = '''void VulkanGSPipeline::executeCompute(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline &pipeline,
    const std::vector<_VulkanBuffer> &buffers
) {
    if (uniformSize > MAX_UNIFORM_SIZE)
'''
execute_replacement = '''void VulkanGSPipeline::executeCompute(
    std::vector<std::pair<size_t, size_t>> dims,
    const void* uniformsPtr, size_t uniformSize,
    _ComputePipeline &pipeline,
    const std::vector<_VulkanBuffer> &buffers
) {
    pcsLastOperation = std::string("compute shader=") +
            (pipeline.debug_name.empty() ? std::string("<unknown>") : pipeline.debug_name);
    if (uniformSize > MAX_UNIFORM_SIZE)
'''
if execute_anchor not in s:
    raise SystemExit("executeCompute operation anchor not found")
s = s.replace(execute_anchor, execute_replacement, 1)

# Label transfer-only batches too, so a failure before the first compute dispatch is still useful.
def replace_once(text: str, old: str, new: str, name: str) -> str:
    if old not in text:
        raise SystemExit(f"{name} anchor not found")
    return text.replace(old, new, 1)

b = replace_once(b,
'''_VulkanBuffer& VulkanGSPipeline::copyToDevice(Buffer<T>& buffer) {

    resizeDeviceBuffer(buffer, buffer.size());
''',
'''_VulkanBuffer& VulkanGSPipeline::copyToDevice(Buffer<T>& buffer) {

    pcsLastOperation = "copyToDevice bytes=" + std::to_string(buffer.byteLength());
    resizeDeviceBuffer(buffer, buffer.size());
''', "copyToDevice")
b = replace_once(b,
'''void VulkanGSPipeline::copyFromDevice(Buffer<T>& buffer) {

    auto& deviceBuffer = buffer.deviceBuffer;
''',
'''void VulkanGSPipeline::copyFromDevice(Buffer<T>& buffer) {

    pcsLastOperation = "copyFromDevice bytes=" + std::to_string(buffer.deviceBuffer.size);
    auto& deviceBuffer = buffer.deviceBuffer;
''', "copyFromDevice")
b = replace_once(b,
'''T VulkanGSPipeline::readElement(const _VulkanBuffer& buffer, size_t index) {

    const size_t elementSize = sizeof(T);
''',
'''T VulkanGSPipeline::readElement(const _VulkanBuffer& buffer, size_t index) {

    pcsLastOperation = "readElement index=" + std::to_string(index) +
            " bytes=" + std::to_string(buffer.size);
    const size_t elementSize = sizeof(T);
''', "readElement")
b = replace_once(b,
'''void VulkanGSPipeline::copyFromDeviceToDevice(const _VulkanBuffer& srcBuffer, _VulkanBuffer& dstBuffer) {
    
''',
'''void VulkanGSPipeline::copyFromDeviceToDevice(const _VulkanBuffer& srcBuffer, _VulkanBuffer& dstBuffer) {
    pcsLastOperation = "copyDeviceToDevice bytes=" + std::to_string(srcBuffer.size);
    
''', "copyFromDeviceToDevice")

renderer_destructor_old = '''VulkanGSRenderer::~VulkanGSRenderer() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    cleanup();
}
'''
renderer_destructor_new = '''VulkanGSRenderer::~VulkanGSRenderer() {
    discardCommandBatchNoThrow("VulkanGSRenderer destructor");
    try {
        cleanup();
    } catch (const std::exception& error) {
        fprintf(stderr, "VulkanGSRenderer cleanup ignored during destructor: %s\\n", error.what());
    } catch (...) {
        fprintf(stderr, "VulkanGSRenderer cleanup ignored unknown exception during destructor\\n");
    }
}
'''
if renderer_destructor_old not in r:
    raise SystemExit("VulkanGSRenderer destructor anchor not found")
r = r.replace(renderer_destructor_old, renderer_destructor_new, 1)

trainer_destructor_old = '''VulkanGSTrainer::~VulkanGSTrainer() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    cleanup();
}
'''
trainer_destructor_new = '''VulkanGSTrainer::~VulkanGSTrainer() {
    discardCommandBatchNoThrow("VulkanGSTrainer destructor");
    try {
        cleanup();
    } catch (const std::exception& error) {
        fprintf(stderr, "VulkanGSTrainer cleanup ignored during destructor: %s\\n", error.what());
    } catch (...) {
        fprintf(stderr, "VulkanGSTrainer cleanup ignored unknown exception during destructor\\n");
    }
}
'''
if trainer_destructor_old not in t:
    raise SystemExit("VulkanGSTrainer destructor anchor not found")
t = t.replace(trainer_destructor_old, trainer_destructor_new, 1)

header_path.write_text(h)
pipeline_path.write_text(s)
buffer_path.write_text(b)
renderer_path.write_text(r)
trainer_path.write_text(t)
print("Patched VkSplat Android command batches for exception-safe Vulkan error propagation")
