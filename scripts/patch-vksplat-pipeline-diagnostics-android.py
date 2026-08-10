#!/usr/bin/env python3
from pathlib import Path
import runpy

path = Path("app/src/main/cpp/third_party/vksplat/vksplat/src/gs_pipeline.cpp")
s = path.read_text()

include_anchor = '#include <fstream>\n'
include_replacement = '''#include <fstream>\n\n#ifdef __ANDROID__\n#include <android/log.h>\n#define PCS_VK_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "Native3DGS", __VA_ARGS__)\n#define PCS_VK_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "Native3DGS", __VA_ARGS__)\n#else\n#define PCS_VK_LOGI(...) ((void)0)\n#define PCS_VK_LOGE(...) ((void)0)\n#endif\n'''
if include_anchor not in s:
    raise SystemExit("gs_pipeline include anchor not found")
s = s.replace(include_anchor, include_replacement, 1)

load_anchor = '''    std::streamsize fileSize = file.tellg();\n    file.seekg(0, std::ios::beg);\n\n    std::vector<uint32_t> spirv_code(fileSize / sizeof(uint32_t));\n    if (!file.read(reinterpret_cast<char*>(spirv_code.data()), fileSize))\n        throw std::runtime_error("Failed to read file: " + spirv_path);\n    \n    return spirv_code;\n'''
load_replacement = '''    std::streamsize fileSize = file.tellg();\n    file.seekg(0, std::ios::beg);\n    if (fileSize <= 0 || (fileSize % sizeof(uint32_t)) != 0)\n        throw std::runtime_error("Invalid SPIR-V byte size: " + spirv_path);\n\n    std::vector<uint32_t> spirv_code(fileSize / sizeof(uint32_t));\n    if (!file.read(reinterpret_cast<char*>(spirv_code.data()), fileSize))\n        throw std::runtime_error("Failed to read file: " + spirv_path);\n    if (spirv_code.size() < 5 || spirv_code[0] != 0x07230203u)\n        throw std::runtime_error("Invalid SPIR-V header: " + spirv_path);\n\n    return spirv_code;\n'''
if load_anchor not in s:
    raise SystemExit("loadSpirv validation anchor not found")
s = s.replace(load_anchor, load_replacement, 1)

device_anchor = '''    this->physical_device = device.device;\n    this->queue_family_index = device.queueFamilyIdx;\n    this->deviceInfo = device.deviceInfo;\n'''
device_replacement = '''    this->physical_device = device.device;\n    this->queue_family_index = device.queueFamilyIdx;\n    this->deviceInfo = device.deviceInfo;\n    VkPhysicalDeviceProperties pcsDeviceProperties{};\n    vkGetPhysicalDeviceProperties(this->physical_device, &pcsDeviceProperties);\n    PCS_VK_LOGI(\n        "Vulkan device name=%s vendor=0x%x api=%u.%u.%u subgroup=%u maxWGInvocations=%u maxWGSize=%ux%ux%u maxShared=%u",\n        pcsDeviceProperties.deviceName, pcsDeviceProperties.vendorID,\n        VK_VERSION_MAJOR(pcsDeviceProperties.apiVersion),\n        VK_VERSION_MINOR(pcsDeviceProperties.apiVersion),\n        VK_VERSION_PATCH(pcsDeviceProperties.apiVersion),\n        this->deviceInfo.subgroupSize,\n        pcsDeviceProperties.limits.maxComputeWorkGroupInvocations,\n        pcsDeviceProperties.limits.maxComputeWorkGroupSize[0],\n        pcsDeviceProperties.limits.maxComputeWorkGroupSize[1],\n        pcsDeviceProperties.limits.maxComputeWorkGroupSize[2],\n        pcsDeviceProperties.limits.maxComputeSharedMemorySize);\n'''
if device_anchor not in s:
    raise SystemExit("device diagnostics anchor not found")
s = s.replace(device_anchor, device_replacement, 1)

pipeline_load_anchor = '''    createShaderModule(loadSpirv(spirv_path), &pipeline.shader);\n    createComputeDescriptorSetLayout(pipeline);\n'''
pipeline_load_replacement = '''    const auto pcs_spirv_code = loadSpirv(spirv_path);\n    uint64_t pcs_fnv64 = 14695981039346656037ull;\n    for (uint32_t word : pcs_spirv_code) {\n        for (unsigned shift = 0; shift < 32; shift += 8) {\n            pcs_fnv64 ^= static_cast<uint8_t>((word >> shift) & 0xffu);\n            pcs_fnv64 *= 1099511628211ull;\n        }\n    }\n    uint32_t pcs_local_x = 0, pcs_local_y = 0, pcs_local_z = 0;\n    for (size_t i = 5; i < pcs_spirv_code.size();) {\n        const uint32_t first = pcs_spirv_code[i];\n        const uint32_t word_count = first >> 16;\n        const uint32_t opcode = first & 0xffffu;\n        if (word_count == 0 || i + word_count > pcs_spirv_code.size()) break;\n        // SPIR-V OpExecutionMode opcode=16, ExecutionMode LocalSize=17.\n        if (opcode == 16u && word_count >= 6u && pcs_spirv_code[i + 2] == 17u) {\n            pcs_local_x = pcs_spirv_code[i + 3];\n            pcs_local_y = pcs_spirv_code[i + 4];\n            pcs_local_z = pcs_spirv_code[i + 5];\n            break;\n        }\n        i += word_count;\n    }\n    PCS_VK_LOGI(\n        "Vulkan pipeline begin shader=%s bytes=%zu fnv64=%016llx local=%ux%ux%u descriptors=%zu minShared=%u subgroupCompatible=%d",\n        spirv_path.c_str(), pcs_spirv_code.size() * sizeof(uint32_t),\n        static_cast<unsigned long long>(pcs_fnv64), pcs_local_x, pcs_local_y, pcs_local_z,\n        pipeline.buffer_layouts.size(), min_shared_memory, compatible_subgroup_size ? 1 : 0);\n    createShaderModule(pcs_spirv_code, &pipeline.shader);\n    PCS_VK_LOGI("Vulkan shader module ready shader=%s", spirv_path.c_str());\n    createComputeDescriptorSetLayout(pipeline);\n    PCS_VK_LOGI("Vulkan descriptor layout ready shader=%s", spirv_path.c_str());\n'''
if pipeline_load_anchor not in s:
    raise SystemExit("compute pipeline SPIR-V load anchor not found")
s = s.replace(pipeline_load_anchor, pipeline_load_replacement, 1)

layout_anchor = '''    if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline.pipeline_layout) != VK_SUCCESS) {\n        _THROW_ERROR("Failed to create pipeline set layout");\n    }\n\n    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req = {};\n'''
layout_replacement = '''    if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline.pipeline_layout) != VK_SUCCESS) {\n        PCS_VK_LOGE("Vulkan pipeline layout failed shader=%s", spirv_path.c_str());\n        _THROW_ERROR("Failed to create pipeline set layout");\n    }\n    PCS_VK_LOGI("Vulkan pipeline layout ready shader=%s", spirv_path.c_str());\n\n    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT req = {};\n'''
if layout_anchor not in s:
    raise SystemExit("pipeline layout diagnostics anchor not found")
s = s.replace(layout_anchor, layout_replacement, 1)

create_anchor = '''    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline.pipeline) != VK_SUCCESS)\n        _THROW_ERROR("Failed to create compute pipeline");\n\n    createComputeDescriptorPool(pipeline);\n'''
create_replacement = '''    PCS_VK_LOGI("Vulkan pipeline create begin shader=%s", spirv_path.c_str());\n    const VkResult pcs_pipeline_result =\n        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline.pipeline);\n    if (pcs_pipeline_result != VK_SUCCESS) {\n        PCS_VK_LOGE("Vulkan pipeline create failed shader=%s VkResult=%d",\n                    spirv_path.c_str(), static_cast<int>(pcs_pipeline_result));\n        _THROW_ERROR("Failed to create compute pipeline VkResult=" +\n                     std::to_string(static_cast<int>(pcs_pipeline_result)) +\n                     " shader=" + spirv_path);\n    }\n    PCS_VK_LOGI("Vulkan pipeline ready shader=%s", spirv_path.c_str());\n\n    createComputeDescriptorPool(pipeline);\n'''
if create_anchor not in s:
    raise SystemExit("vkCreateComputePipelines diagnostics anchor not found")
s = s.replace(create_anchor, create_replacement, 1)

path.write_text(s)
print("Patched VkSplat Vulkan pipeline creation with Android diagnostics")

# Apply the Mali radix port while the vendored source tree is still being prepared. This also
# restores the radix GLSL jobs using the Android NDK glslc path after prepare removed the desktop job.
runpy.run_path("scripts/patch-vksplat-radix-android.py", run_name="__main__")
# Fold the global-prefix pass back into a second synchronized spine dispatch so native_3dgs keeps
# the same stable shader-path contract while upsweep remains free of contended global atomics.
runpy.run_path("scripts/patch-vksplat-radix-two-stage-fixup.py", run_name="__main__")

# The command-batch patch depends on the Android log macros injected above, so keep the ordering
# explicit in one entry point used by the build.
runpy.run_path("scripts/patch-vksplat-command-batch-android.py", run_name="__main__")

# Persist the full trainable state after successful PLY output so later runs continue from the
# exact Gaussian parameters, Adam moments, RNG state and cumulative optimizer step.
runpy.run_path("scripts/patch-vksplat-checkpoint-android.py", run_name="__main__")

# Existing PCS models from builds before resumable checkpoints still contain the converged Gaussian
# parameters in splat.ply. Import those parameters once, initialize only the unavailable Adam
# moments, and apply the same finite-value validity contract used by the viewer before publishing
# a v2 checkpoint for every later exact continuation.
runpy.run_path("scripts/patch-vksplat-legacy-resume-android.py", run_name="__main__")
runpy.run_path("scripts/patch-vksplat-legacy-ply-validity-android.py", run_name="__main__")
runpy.run_path("scripts/patch-vksplat-legacy-resume-hint-android.py", run_name="__main__")
