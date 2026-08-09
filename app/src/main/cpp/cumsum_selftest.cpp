#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gs_renderer.h"
#include "buffer.h"

namespace {
constexpr const char* kTag = "Native3DGS";

void logi(const std::string& text) {
    __android_log_print(ANDROID_LOG_INFO, kTag, "%s", text.c_str());
}
void loge(const std::string& text) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", text.c_str());
}

std::string jstringToString(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* raw = env->GetStringUTFChars(value, nullptr);
    if (!raw) return {};
    std::string out(raw);
    env->ReleaseStringUTFChars(value, raw);
    return out;
}

std::string jsonEscape(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

struct ProgressCallback {
    JNIEnv* env = nullptr;
    jobject listener = nullptr;
    jmethodID method = nullptr;

    ProgressCallback(JNIEnv* e, jobject l) : env(e), listener(l) {
        if (!listener) return;
        jclass cls = env->GetObjectClass(listener);
        if (cls) method = env->GetMethodID(cls, "onProgress", "(ILjava/lang/String;)V");
    }

    void send(const std::string& message) const {
        logi(message);
        if (!env || !listener || !method) return;
        jstring jmsg = env->NewStringUTF(message.c_str());
        env->CallVoidMethod(listener, method, 1, jmsg);
        env->DeleteLocalRef(jmsg);
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
    }
};

std::map<std::string, std::string> makeCumsumShaderPaths(const std::string& root) {
    return {
        {"cumsum_single_pass", root + "generated/cumsum_single_pass.spv"},
        {"cumsum_block_scan", root + "generated/cumsum_block_scan.spv"},
        {"cumsum_scan_block_sums", root + "generated/cumsum_scan_block_sums.spv"},
        {"cumsum_add_block_offsets", root + "generated/cumsum_add_block_offsets.spv"},
    };
}

class CumsumSelfTestRenderer final : public VulkanGSRenderer {
public:
    void initializeCumsumOnly(const std::map<std::string, std::string>& paths,
                              int deviceId,
                              const ProgressCallback& progress) {
        progress.send("cumsum:selftest vulkan_init begin");
        VulkanGSPipeline::initialize(deviceId);
        progress.send("cumsum:selftest vulkan_init ready");

        createOne("single_pass", pipeline_cumsum.single_pass,
                  paths.at("cumsum_single_pass"), progress);
        createOne("block_scan", pipeline_cumsum.block_scan,
                  paths.at("cumsum_block_scan"), progress);
        createOne("scan_block_sums", pipeline_cumsum.scan_block_sums,
                  paths.at("cumsum_scan_block_sums"), progress);
        createOne("add_block_offsets", pipeline_cumsum.add_block_offsets,
                  paths.at("cumsum_add_block_offsets"), progress);
        progress.send("cumsum:selftest pipelines ready");
    }

    void runConformance(const ProgressCallback& progress) {
        VulkanGSPipelineBuffers buffers;
        Buffer<int32_t> input;
        Buffer<int32_t> output;

        try {
            const size_t sizes[] = {1, 16, 255, 256, 257, 1024, 1025, 37635, 90000};
            for (size_t n : sizes) {
                const int patternCount = n <= 1025 ? 3 : 1;
                for (int pattern = 0; pattern < patternCount; ++pattern) {
                    runCase(buffers, input, output, n, pattern, progress);
                }
            }
            progress.send("cumsum:selftest COMPLETE");
        } catch (...) {
            cleanupCaseBuffers(buffers, input, output);
            throw;
        }
        cleanupCaseBuffers(buffers, input, output);
    }

protected:
    DeviceRequirement getDeviceRequirement() override {
        // The isolated conformance path only needs the cumsum kernel's actual execution envelope.
        // Keep it independent from rasterizer-only workgroup/shared-memory requirements.
        return DeviceRequirement{{1024, 1, 1}, {256, 1, 1}, 256u * sizeof(int32_t)};
    }

private:
    void createOne(const char* name,
                   _ComputePipeline& pipeline,
                   const std::string& path,
                   const ProgressCallback& progress) {
        progress.send(std::string("cumsum:selftest pipeline ") + name + " begin");
        // PCS cumsum GLSL contains no subgroup instructions. Do not request a fixed subgroup size.
        createComputePipeline(pipeline, path, 0, false);
        progress.send(std::string("cumsum:selftest pipeline ") + name + " ready");
    }

    static int32_t testValue(size_t i, int pattern) {
        switch (pattern) {
            case 0: return 1;
            case 1: return (i % 13 == 0) ? 0 : (1 + static_cast<int32_t>(i % 5));
            case 2: return (i & 1u) ? 7 : 0;
            default: return 0;
        }
    }

    void runCase(VulkanGSPipelineBuffers& buffers,
                 Buffer<int32_t>& input,
                 Buffer<int32_t>& output,
                 size_t n,
                 int pattern,
                 const ProgressCallback& progress) {
        input.resize(n);
        std::vector<int32_t> expected(n);
        int64_t running = 0;
        for (size_t i = 0; i < n; ++i) {
            const int32_t value = testValue(i, pattern);
            input[i] = value;
            running += value;
            if (running > INT32_MAX) {
                throw std::runtime_error("cumsum self-test reference overflow");
            }
            expected[i] = static_cast<int32_t>(running);
        }

        const std::string caseId = " n=" + std::to_string(n)
                + " pattern=" + std::to_string(pattern);
        progress.send("cumsum:selftest upload begin" + caseId);
        copyToDevice(input);
        progress.send("cumsum:selftest upload ready" + caseId);

        progress.send("cumsum:selftest dispatch begin" + caseId);
        {
            DeviceGuard guard(this);
            executeCumsum(buffers, input, output);
        }
        progress.send("cumsum:selftest dispatch ready" + caseId);

        progress.send("cumsum:selftest readback begin" + caseId);
        copyFromDevice(output);
        progress.send("cumsum:selftest readback ready" + caseId);

        if (output.size() != n) {
            throw std::runtime_error("cumsum self-test output size mismatch n="
                    + std::to_string(n) + " got=" + std::to_string(output.size()));
        }
        for (size_t i = 0; i < n; ++i) {
            if (output[i] != expected[i]) {
                std::ostringstream error;
                error << "cumsum self-test mismatch n=" << n
                      << " pattern=" << pattern
                      << " index=" << i
                      << " expected=" << expected[i]
                      << " actual=" << output[i];
                throw std::runtime_error(error.str());
            }
        }
        progress.send("cumsum:selftest PASS" + caseId
                + " last=" + std::to_string(expected.back()));
    }

    void cleanupCaseBuffers(VulkanGSPipelineBuffers& buffers,
                            Buffer<int32_t>& input,
                            Buffer<int32_t>& output) {
        if (isCommandBatchInProgress()) {
            endCommandBatch();
        }
        if (input.deviceBuffer.buffer != VK_NULL_HANDLE) {
            destroyBuffer(input.deviceBuffer);
        }
        if (output.deviceBuffer.buffer != VK_NULL_HANDLE) {
            destroyBuffer(output.deviceBuffer);
        }
        cleanupBuffers(buffers);
    }
};

std::string successJson() {
    return "{\"success\":true,\"message\":\"GPU cumsum conformance passed\"}";
}
std::string failureJson(const std::string& message) {
    return std::string("{\"success\":false,\"message\":\"")
            + jsonEscape(message) + "\"}";
}
} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_sktpj_pointcloudsplatting_NativeGaussianTrainer_nativeCumsumSelfTest(
        JNIEnv* env,
        jclass,
        jstring jShaderDir,
        jobject listener) {
    const std::string shaderDir = jstringToString(env, jShaderDir);
    ProgressCallback progress(env, listener);

    try {
        progress.send("cumsum:selftest BEGIN");
        CumsumSelfTestRenderer renderer;
        renderer.initializeCumsumOnly(makeCumsumShaderPaths(shaderDir), -1, progress);
        renderer.runConformance(progress);
        return env->NewStringUTF(successJson().c_str());
    } catch (const std::exception& error) {
        const std::string message = std::string("cumsum:selftest FAIL ") + error.what();
        loge(message);
        progress.send(message);
        const std::string json = failureJson(error.what());
        return env->NewStringUTF(json.c_str());
    } catch (...) {
        const std::string message = "cumsum:selftest FAIL unknown native exception";
        loge(message);
        progress.send(message);
        const std::string json = failureJson("unknown cumsum self-test error");
        return env->NewStringUTF(json.c_str());
    }
}
