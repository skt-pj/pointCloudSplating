#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "gs_trainer.h"
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

    void send(int percent, const std::string& message) const {
        if (!env || !listener || !method) return;
        jstring jmsg = env->NewStringUTF(message.c_str());
        env->CallVoidMethod(listener, method, static_cast<jint>(std::clamp(percent, 0, 100)), jmsg);
        env->DeleteLocalRef(jmsg);
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
    }
};

std::map<std::string, std::string> makeShaderPaths(const std::string& root) {
    const std::vector<std::string> names = {
        "projection_forward", "generate_keys", "compute_tile_ranges", "rasterize_forward",
        "rasterize_backward_0", "rasterize_backward_1", "rasterize_backward_2",
        "rasterize_backward_3", "rasterize_backward_4", "cumsum_single_pass",
        "cumsum_block_scan", "cumsum_scan_block_sums", "cumsum_add_block_offsets",
        "radix_sort/upsweep", "radix_sort/spine", "radix_sort/downsweep", "ssim_forward",
        "ssim_backward", "fused_projection_backward_optimizer", "sum", "where",
        "default_update_state", "default_compute_grow_mask", "default_duplicate",
        "default_split", "default_compute_prune_mask", "default_prune", "default_prune_mean",
        "default_prune_sh", "default_reset_opa", "mcmc_inject_noise", "mcmc_compute_probs",
        "mcmc_compute_relocation_index_map", "mcmc_compute_relocation",
        "mcmc_update_relocation", "mcmc_compute_add_index_map", "mcmc_compute_add",
        "mcmc_update_add", "morton_sort_compute_stats", "morton_sort_generate_keys",
        "morton_sort_apply_indices", "morton_sort_apply_indices_sh", "morton_sort_update_buffer",
        "morton_sort_update_buffer_sh"
    };
    std::map<std::string, std::string> result;
    for (const auto& name : names) {
        if (name.rfind("radix_sort/", 0) == 0) result[name] = root + name + ".spv";
        else result[name] = root + "generated/" + name + ".spv";
    }
    return result;
}

TrainerConfig makeConfig(const std::string& dataRoot,
                         const std::string& imageDir,
                         const std::string& sparseDir,
                         const std::string& outputPly,
                         int frameCount,
                         int trainSteps) {
    TrainerConfig c{};
    c.output_dir = dataRoot;
    c.output_ply = outputPly;
    c.dataset_dir = dataRoot;
    c.image_dir = imageDir;
    c.mask_dir = "";
    c.sparse_dir = sparseDir;
    c.eval_interval = std::max(2, frameCount + 1); // first view is validation; all others train
    c.image_cache_device = TrainerConfig::CacheImage::CPU;
    c.global_scale = 1.0f;
    c.init_scale = 1.0f;
    c.init_opacity = 0.10f;
    c.strategy = TrainerConfig::Strategy::Default;

    c.max_steps = trainSteps;
    c.ssim_lambda = 0.20f;
    c.means_lr = 1.6e-4f;
    c.means_lr_final = 1.6e-6f;
    c.features_dc_lr = 0.0025f;
    c.features_rest_lr = 0.0025f / 20.0f;
    c.opacities_lr = 0.05f;
    c.scales_lr = 0.005f;
    c.quats_lr = 0.001f;
    c.scale_reg = 0.0f;
    c.opacity_reg = 0.0f;

    c.refine_start_iter = std::min(500, std::max(100, trainSteps / 10));
    c.refine_stop_iter = std::max(c.refine_start_iter + 200, trainSteps - 800);
    c.refine_every = 100;
    c.prune_opa = 0.005f;
    c.grow_grad2d = 0.0002f;
    c.grow_scale3d = 0.01f;
    c.grow_scale2d = 0.05f;
    c.prune_scale3d = 0.10f;
    c.prune_scale2d = 0.15f;
    c.refine_scale2d_stop_iter = 0;
    c.reset_every = 3000;
    c.stop_reset_at = -1;
    c.pause_refine_after_reset = 0;

    // MCMC values are initialized even though the default densification strategy is selected.
    c.noise_lr = 5e5f;
    c.min_opacity = 0.005f;
    c.grow_factor = 1.05f;
    c.cap_max = 350000;
    return c;
}

void processTiles(VulkanGSTrainer& trainer,
                  VulkanGSRendererUniforms& uniforms,
                  VulkanGSPipelineBuffers& buffers) {
    trainer.executeCalculateIndexBufferOffset(buffers);
    if (buffers.num_indices == 0) return;
    trainer.executeGenerateKeys(uniforms, buffers);
    trainer.executeSort(uniforms, buffers, -1);
    trainer.executeComputeTileRanges(uniforms, buffers);
}

void forward(VulkanGSTrainer& trainer,
             VulkanGSRendererUniforms& uniforms,
             VulkanGSPipelineBuffers& buffers) {
    uniforms.num_splats = static_cast<uint32_t>(buffers.num_splats);
    trainer.executeProjectionForward(uniforms, buffers);
    processTiles(trainer, uniforms, buffers);
    trainer.executeRasterizeForward(uniforms, buffers);
}

void trainOneStep(VulkanGSTrainer& trainer,
                  const TrainerConfig& config,
                  VulkanGSRendererUniforms& uniforms,
                  VulkanGSPipelineBuffers& buffers,
                  size_t imageIndex,
                  int step) {
    trainer.get_train_camera(imageIndex, uniforms);
    uniforms.active_sh = static_cast<uint32_t>(std::min(step / 1000, 3));
    uniforms.step = static_cast<uint32_t>(step);

    auto guard = DeviceGuard(&trainer);
    forward(trainer, uniforms, buffers);
    if (buffers.num_indices == 0) return;
    trainer.executeComputeSSIMGradient(config, uniforms, buffers, imageIndex);
    trainer.executeRasterizeBackward(uniforms, buffers);
    trainer.executeFusedProjectionBackwardOptimizerStep(config, uniforms, buffers, step + 1);
    trainer.executeDefaultPostBackward(config, uniforms, buffers, step);
}

double validationPsnr(VulkanGSTrainer& trainer,
                      VulkanGSRendererUniforms& uniforms,
                      VulkanGSPipelineBuffers& buffers) {
    if (trainer.num_val() == 0) return NAN;
    trainer.get_val_camera(0, uniforms);
    uniforms.active_sh = 3;
    uniforms.step = 0;
    {
        auto guard = DeviceGuard(&trainer);
        forward(trainer, uniforms, buffers);
    }
    trainer.copyFromDevice(buffers.pixel_state);
    auto& reference = trainer.get_val_image(0).buffer;
    size_t pixelCount = static_cast<size_t>(uniforms.image_width) * uniforms.image_height;
    if (buffers.pixel_state.size() < pixelCount * 4 || reference.size() < pixelCount * 4) return NAN;
    double mse = 0.0;
    size_t samples = 0;
    for (size_t i = 0; i < pixelCount; ++i) {
        for (int c = 0; c < 3; ++c) {
            double predicted = std::clamp(static_cast<double>(buffers.pixel_state[4*i+c]), 0.0, 1.0);
            double expected = static_cast<double>(reference[4*i+c]) / 255.0;
            double d = predicted - expected;
            mse += d * d;
            ++samples;
        }
    }
    if (samples == 0) return NAN;
    mse /= static_cast<double>(samples);
    if (mse <= 1e-12) return 120.0;
    return 10.0 * std::log10(1.0 / mse);
}

std::string deviceName(VulkanGSTrainer& trainer) {
    auto info = trainer.get_device_info();
    auto it = info.find("name");
    if (it == info.end()) return "Vulkan device";
    if (auto value = std::get_if<std::string>(&it->second)) return *value;
    return "Vulkan device";
}

std::string successJson(size_t count, int steps, double psnr, const std::string& device) {
    std::ostringstream out;
    out << "{\"success\":true,\"message\":\"3DGS training complete\",\"gaussian_count\":"
        << count << ",\"steps\":" << steps << ",\"device\":\"" << jsonEscape(device) << "\"";
    if (std::isfinite(psnr)) out << ",\"validation_psnr\":" << psnr;
    out << "}";
    return out.str();
}

std::string failureJson(const std::string& message) {
    return std::string("{\"success\":false,\"message\":\"") + jsonEscape(message) + "\"}";
}
} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_sktpj_pointcloudsplatting_NativeGaussianTrainer_nativeTrain(
        JNIEnv* env,
        jclass,
        jstring jDataRoot,
        jstring jImageDir,
        jstring jSparseDir,
        jstring jShaderDir,
        jstring jOutputPly,
        jint frameCount,
        jint trainSteps,
        jobject listener) {
    const std::string dataRoot = jstringToString(env, jDataRoot);
    const std::string imageDir = jstringToString(env, jImageDir);
    const std::string sparseDir = jstringToString(env, jSparseDir);
    const std::string shaderDir = jstringToString(env, jShaderDir);
    const std::string outputPly = jstringToString(env, jOutputPly);
    ProgressCallback progress(env, listener);

    try {
        progress.send(1, "Vulkan学習器を開始しています…");
        VulkanGSTrainer trainer;
        VulkanGSPipelineBuffers buffers;
        VulkanGSRendererUniforms uniforms{};
        TrainerConfig config = makeConfig(dataRoot, imageDir, sparseDir, outputPly,
                                          static_cast<int>(frameCount), static_cast<int>(trainSteps));

        trainer.initialize(makeShaderPaths(shaderDir), -1);
        std::string gpu = deviceName(trainer);
        logi("VkSplat initialized device=" + gpu);
        progress.send(4, "端末のGPUで3DGS学習を準備しています…");

        trainer.load_colmap_dataset(config, buffers);
        if (trainer.num_train() == 0) throw std::runtime_error("no training camera views");
        if (buffers.num_splats < 64) throw std::runtime_error("initial 3D points are insufficient");

        {
            std::ostringstream info;
            info << "Training start views=" << trainer.num_train() << " initial=" << buffers.num_splats
                 << " steps=" << trainSteps << " loss=L1+SSIM densityControl=default";
            logi(info.str());
        }

        const int steps = std::max(1000, static_cast<int>(trainSteps));
        for (int step = 0; step < steps; ++step) {
            size_t imageIndex = static_cast<size_t>(step) % trainer.num_train();
            trainOneStep(trainer, config, uniforms, buffers, imageIndex, step);
            if (step == 0 || (step + 1) % 100 == 0 || step + 1 == steps) {
                int percent = 5 + static_cast<int>(90.0 * (step + 1) / steps);
                std::ostringstream message;
                message << "写真と3Dモデルを比較して学習しています… " << (step + 1) << "/" << steps;
                progress.send(percent, message.str());
                std::ostringstream log;
                log << "step=" << (step + 1) << "/" << steps << " gaussians=" << buffers.num_splats
                    << " activeSH=" << uniforms.active_sh;
                logi(log.str());
            }
        }

        progress.send(96, "学習結果を確認しています…");
        double psnr = validationPsnr(trainer, uniforms, buffers);
        progress.send(98, "3DGSモデルを書き出しています…");
        const size_t finalCount = buffers.num_splats;
        trainer.writePLY(outputPly, buffers);
        logi("Training COMPLETE gaussians=" + std::to_string(finalCount)
             + " psnr=" + (std::isfinite(psnr) ? std::to_string(psnr) : std::string("n/a")));
        trainer.cleanupBuffers(buffers);
        trainer.cleanup();
        progress.send(100, "3DGS学習が完了しました");
        std::string json = successJson(finalCount, steps, psnr, gpu);
        return env->NewStringUTF(json.c_str());
    } catch (const std::exception& error) {
        loge(std::string("3DGS training failed: ") + error.what());
        std::string json = failureJson(error.what());
        return env->NewStringUTF(json.c_str());
    } catch (...) {
        loge("3DGS training failed: unknown native exception");
        std::string json = failureJson("unknown native 3DGS error");
        return env->NewStringUTF(json.c_str());
    }
}
