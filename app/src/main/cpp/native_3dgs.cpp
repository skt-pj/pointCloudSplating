#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <glm/glm.hpp>

#include "gs_trainer.h"
#include "gs_renderer.h"
#include "buffer.h"
#include "knn.h"

namespace {
constexpr const char* kTag = "Native3DGS";

// Mobile trainer budgets. These are hard working-set budgets, not emergency caps applied after
// allocations have already happened.
constexpr size_t kInitialGaussianBudget = 90'000;
constexpr int kGaussianBudget = 120'000;
constexpr size_t kTileWorkingSetBudgetBytes = 128ull * 1024ull * 1024ull;

// PocketGS prior-conditioned initialization: K_n=16 for stable normals and K_s=3 for local scale.
constexpr int kNormalNeighbors = 16;
constexpr int kScaleNeighbors = 3;
// PocketGS defines s_n=r_normal*s_t but does not prescribe a universal r_normal. 0.20 is a PCS
// engineering choice for Raw Depth: thin enough to start surface-aligned without becoming singular.
constexpr float kNormalThicknessRatio = 0.20f;
constexpr float kMinSurfaceScale = 5.0e-4f;
constexpr float kMaxSurfaceScale = 5.0e-2f;

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
    // Keep roughly 20% of sufficiently large datasets out of optimization for real hold-out evaluation.
    c.eval_interval = frameCount >= 8 ? 5 : std::max(2, frameCount + 1);
    c.image_cache_device = TrainerConfig::CacheImage::CPU;
    c.global_scale = 1.0f;
    c.init_scale = 1.0f;
    c.init_opacity = 0.10f;

    c.strategy = TrainerConfig::Strategy::MCMC;
    c.cap_max = kGaussianBudget;

    c.max_steps = trainSteps;
    c.ssim_lambda = 0.20f;
    c.means_lr = 1.6e-4f;
    c.means_lr_final = 1.6e-6f;
    c.features_dc_lr = 0.0025f;
    c.features_rest_lr = 0.0025f / 20.0f;
    c.opacities_lr = 0.05f;
    c.scales_lr = 0.005f;
    c.quats_lr = 0.001f;
    c.scale_reg = 0.01f;
    c.opacity_reg = 0.01f;

    c.refine_start_iter = std::min(100, std::max(40, trainSteps / 8));
    c.refine_stop_iter = std::max(c.refine_start_iter + 100, trainSteps - 100);
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

    c.noise_lr = 5e5f;
    c.min_opacity = 0.005f;
    c.grow_factor = 1.05f;
    return c;
}

uint32_t expandMorton10(uint32_t v) {
    v &= 0x000003ffu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

uint32_t mortonCode(const glm::vec3& p, const glm::vec3& lo, const glm::vec3& hi) {
    glm::vec3 extent = glm::max(hi - lo, glm::vec3(1e-6f));
    glm::vec3 u = glm::clamp((p - lo) / extent, glm::vec3(0.0f), glm::vec3(1.0f));
    uint32_t x = static_cast<uint32_t>(std::lround(u.x * 1023.0f));
    uint32_t y = static_cast<uint32_t>(std::lround(u.y * 1023.0f));
    uint32_t z = static_cast<uint32_t>(std::lround(u.z * 1023.0f));
    return expandMorton10(x) | (expandMorton10(y) << 1) | (expandMorton10(z) << 2);
}

void spatialBudgetInitialCloud(VulkanGSTrainer& trainer,
                               VulkanGSPipelineBuffers& buffers,
                               size_t budget) {
    const size_t oldCount = buffers.num_splats;
    if (oldCount <= budget) return;
    if (budget < 64) throw std::runtime_error("mobile Gaussian budget is too small");

    buffers.undoReorderSH(buffers.sh_coeffs, oldCount);

    glm::vec3 lo(std::numeric_limits<float>::infinity());
    glm::vec3 hi(-std::numeric_limits<float>::infinity());
    std::vector<glm::vec3> points(oldCount);
    for (size_t i = 0; i < oldCount; ++i) {
        glm::vec3 p(buffers.xyz_ws[3*i], buffers.xyz_ws[3*i+1], buffers.xyz_ws[3*i+2]);
        points[i] = p;
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }

    std::vector<std::pair<uint32_t, size_t>> ordered;
    ordered.reserve(oldCount);
    for (size_t i = 0; i < oldCount; ++i) ordered.emplace_back(mortonCode(points[i], lo, hi), i);
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    Buffer<float> xyzNew;
    Buffer<float> shNew;
    Buffer<float> rotNew;
    Buffer<float> scaleNew;
    xyzNew.resize(3 * budget);
    shNew.resize(16 * 3 * budget);
    rotNew.resize(4 * budget);
    scaleNew.resize(4 * budget);

    for (size_t out = 0; out < budget; ++out) {
        size_t pos = std::min(oldCount - 1,
                static_cast<size_t>(((out + 0.5) * static_cast<double>(oldCount)) / budget));
        size_t src = ordered[pos].second;
        std::copy_n(&buffers.xyz_ws[3*src], 3, &xyzNew[3*out]);
        std::copy_n(&buffers.sh_coeffs[16*3*src], 16*3, &shNew[16*3*out]);
        std::copy_n(&buffers.rotations[4*src], 4, &rotNew[4*out]);
        std::copy_n(&buffers.scales_opacs[4*src], 4, &scaleNew[4*out]);
    }

    buffers.xyz_ws.assign(xyzNew.begin(), xyzNew.end());
    buffers.sh_coeffs.assign(shNew.begin(), shNew.end());
    buffers.rotations.assign(rotNew.begin(), rotNew.end());
    buffers.scales_opacs.assign(scaleNew.begin(), scaleNew.end());
    buffers.num_splats = budget;
    buffers.reorderSH(buffers.sh_coeffs);

    trainer.destroyBuffer(buffers.xyz_ws.deviceBuffer);
    trainer.destroyBuffer(buffers.sh_coeffs.deviceBuffer);
    trainer.destroyBuffer(buffers.rotations.deviceBuffer);
    trainer.destroyBuffer(buffers.scales_opacs.deviceBuffer);
    buffers.xyz_ws.deviceBuffer = _VulkanBuffer();
    buffers.sh_coeffs.deviceBuffer = _VulkanBuffer();
    buffers.rotations.deviceBuffer = _VulkanBuffer();
    buffers.scales_opacs.deviceBuffer = _VulkanBuffer();
    trainer.copyToDevice(buffers.xyz_ws);
    trainer.copyToDevice(buffers.sh_coeffs);
    trainer.copyToDevice(buffers.rotations);
    trainer.copyToDevice(buffers.scales_opacs);

    logi("Spatially budgeted initial cloud " + std::to_string(oldCount) + " -> "
         + std::to_string(budget) + " Gaussians");
}

struct EigenFrame {
    std::array<double, 3> values{};
    std::array<glm::dvec3, 3> vectors{};
};

EigenFrame symmetricEigen3(double a[3][3]) {
    double v[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    for (int iter = 0; iter < 16; ++iter) {
        int p = 0, q = 1;
        double maxOff = std::abs(a[0][1]);
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                double off = std::abs(a[i][j]);
                if (off > maxOff) { maxOff = off; p = i; q = j; }
            }
        }
        if (maxOff < 1e-12) break;

        double app = a[p][p], aqq = a[q][q], apq = a[p][q];
        double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        double c = std::cos(phi), s = std::sin(phi);

        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q) continue;
            double akp = a[k][p], akq = a[k][q];
            a[k][p] = a[p][k] = c * akp - s * akq;
            a[k][q] = a[q][k] = s * akp + c * akq;
        }
        a[p][p] = c*c*app - 2.0*s*c*apq + s*s*aqq;
        a[q][q] = s*s*app + 2.0*s*c*apq + c*c*aqq;
        a[p][q] = a[q][p] = 0.0;

        for (int k = 0; k < 3; ++k) {
            double vkp = v[k][p], vkq = v[k][q];
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }

    std::array<int, 3> order = {0, 1, 2};
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) { return a[lhs][lhs] < a[rhs][rhs]; });
    EigenFrame out;
    for (int k = 0; k < 3; ++k) {
        int col = order[k];
        out.values[k] = a[col][col];
        glm::dvec3 vec(v[0][col], v[1][col], v[2][col]);
        double len = glm::length(vec);
        out.vectors[k] = len > 1e-12 ? vec / len : glm::dvec3(k == 0, k == 1, k == 2);
    }
    return out;
}

std::array<float, 4> quaternionFromBasis(const glm::vec3& xAxis,
                                         const glm::vec3& yAxis,
                                         const glm::vec3& zAxis) {
    const float r00=xAxis.x, r01=yAxis.x, r02=zAxis.x;
    const float r10=xAxis.y, r11=yAxis.y, r12=zAxis.y;
    const float r20=xAxis.z, r21=yAxis.z, r22=zAxis.z;
    float w, x, y, z;
    float trace = r00 + r11 + r22;
    if (trace > 0.0f) {
        float s = std::sqrt(trace + 1.0f) * 2.0f;
        w = 0.25f * s;
        x = (r21 - r12) / s;
        y = (r02 - r20) / s;
        z = (r10 - r01) / s;
    } else if (r00 > r11 && r00 > r22) {
        float s = std::sqrt(std::max(1e-12f, 1.0f + r00 - r11 - r22)) * 2.0f;
        w = (r21 - r12) / s;
        x = 0.25f * s;
        y = (r01 + r10) / s;
        z = (r02 + r20) / s;
    } else if (r11 > r22) {
        float s = std::sqrt(std::max(1e-12f, 1.0f + r11 - r00 - r22)) * 2.0f;
        w = (r02 - r20) / s;
        x = (r01 + r10) / s;
        y = 0.25f * s;
        z = (r12 + r21) / s;
    } else {
        float s = std::sqrt(std::max(1e-12f, 1.0f + r22 - r00 - r11)) * 2.0f;
        w = (r10 - r01) / s;
        x = (r02 + r20) / s;
        y = (r12 + r21) / s;
        z = 0.25f * s;
    }
    float inv = 1.0f / std::max(1e-12f, std::sqrt(w*w+x*x+y*y+z*z));
    return {w*inv, x*inv, y*inv, z*inv};
}

void initializeSurfaceGaussians(VulkanGSTrainer& trainer, VulkanGSPipelineBuffers& buffers) {
    const size_t n = buffers.num_splats;
    if (n < static_cast<size_t>(kNormalNeighbors + 1)) {
        throw std::runtime_error("not enough geometry for surface-aware initialization");
    }

    std::vector<glm::vec3> points(n);
    for (size_t i = 0; i < n; ++i) {
        points[i] = glm::vec3(buffers.xyz_ws[3*i], buffers.xyz_ws[3*i+1], buffers.xyz_ws[3*i+2]);
    }

    NearestNeighbors3D knn;
    knn.fit(points);
    auto [distances, indices] = knn.kneighbors(points, kNormalNeighbors + 1);
    (void)distances;

    std::vector<float> tangentialScales;
    tangentialScales.reserve(n);
    size_t validFrames = 0;

    for (size_t i = 0; i < n; ++i) {
        std::array<size_t, kNormalNeighbors> neighborIdx{};
        int neighborCount = 0;
        std::array<float, kScaleNeighbors> scaleDist{};
        int scaleCount = 0;
        for (size_t j = 0; j < indices[i].size(); ++j) {
            size_t idx = static_cast<size_t>(indices[i][j]);
            if (idx == i || idx >= n) continue;
            if (neighborCount < kNormalNeighbors) neighborIdx[neighborCount++] = idx;
            if (scaleCount < kScaleNeighbors) {
                float d = glm::length(points[idx] - points[i]);
                if (std::isfinite(d) && d > 0.0f) scaleDist[scaleCount++] = d;
            }
            if (neighborCount >= kNormalNeighbors && scaleCount >= kScaleNeighbors) break;
        }

        float fallback = std::clamp(buffers.scales_opacs[4*i], kMinSurfaceScale, kMaxSurfaceScale);
        float st = fallback;
        if (scaleCount == kScaleNeighbors) {
            st = std::accumulate(scaleDist.begin(), scaleDist.end(), 0.0f) / kScaleNeighbors;
            st = std::clamp(st, kMinSurfaceScale, kMaxSurfaceScale);
        }

        glm::vec3 tangent1(1,0,0), tangent2(0,1,0), normal(0,0,1);
        bool valid = neighborCount == kNormalNeighbors;
        if (valid) {
            glm::dvec3 centroid(0.0);
            for (int k = 0; k < neighborCount; ++k) centroid += glm::dvec3(points[neighborIdx[k]]);
            centroid /= static_cast<double>(neighborCount);
            double cov[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
            for (int k = 0; k < neighborCount; ++k) {
                glm::dvec3 d = glm::dvec3(points[neighborIdx[k]]) - centroid;
                cov[0][0] += d.x*d.x; cov[0][1] += d.x*d.y; cov[0][2] += d.x*d.z;
                cov[1][0] += d.y*d.x; cov[1][1] += d.y*d.y; cov[1][2] += d.y*d.z;
                cov[2][0] += d.z*d.x; cov[2][1] += d.z*d.y; cov[2][2] += d.z*d.z;
            }
            for (int r=0;r<3;++r) for (int c=0;c<3;++c) cov[r][c] /= neighborCount;
            EigenFrame eig = symmetricEigen3(cov);
            normal = glm::vec3(eig.vectors[0]);
            tangent1 = glm::vec3(eig.vectors[2]);
            if (!std::isfinite(normal.x) || glm::length(normal) < 0.5f) valid = false;
            if (valid) {
                normal = glm::normalize(normal);
                tangent1 -= normal * glm::dot(tangent1, normal);
                if (!std::isfinite(tangent1.x) || glm::length(tangent1) < 1e-5f) valid = false;
            }
            if (valid) {
                tangent1 = glm::normalize(tangent1);
                tangent2 = glm::normalize(glm::cross(normal, tangent1));
                if (glm::dot(glm::cross(tangent1, tangent2), normal) < 0.0f) tangent2 = -tangent2;
                validFrames++;
            } else {
                tangent1 = glm::vec3(1,0,0);
                tangent2 = glm::vec3(0,1,0);
                normal = glm::vec3(0,0,1);
            }
        }

        auto q = quaternionFromBasis(tangent1, tangent2, normal);
        buffers.rotations[4*i] = q[0];
        buffers.rotations[4*i+1] = q[1];
        buffers.rotations[4*i+2] = q[2];
        buffers.rotations[4*i+3] = q[3];
        buffers.scales_opacs[4*i] = st;
        buffers.scales_opacs[4*i+1] = st;
        buffers.scales_opacs[4*i+2] = std::max(kMinSurfaceScale, st * kNormalThicknessRatio);
        buffers.scales_opacs[4*i+3] = 0.10f;
        tangentialScales.push_back(st);
    }

    trainer.copyToDevice(buffers.rotations);
    trainer.copyToDevice(buffers.scales_opacs);

    std::sort(tangentialScales.begin(), tangentialScales.end());
    float median = tangentialScales[tangentialScales.size()/2];
    std::ostringstream info;
    info << "Surface init Knormal=" << kNormalNeighbors << " Kscale=" << kScaleNeighbors
         << " validNormals=" << validFrames << "/" << n
         << " scaleMedian=" << median << " thicknessRatio=" << kNormalThicknessRatio;
    logi(info.str());
}

size_t maxTileIndicesForBudget() {
    const size_t bytesPerIndex = 2 * sizeof(sortingKey_t) + 2 * sizeof(int32_t);
    return std::max<size_t>(1, kTileWorkingSetBudgetBytes / bytesPerIndex);
}

void validateProjectionInvariants(VulkanGSTrainer& trainer,
                                  const VulkanGSRendererUniforms& uniforms,
                                  VulkanGSPipelineBuffers& buffers) {
    trainer.copyFromDevice(buffers.tiles_touched);
    const int64_t gridTiles = static_cast<int64_t>(uniforms.grid_width) * uniforms.grid_height;
    int64_t cpuSum = 0;
    size_t visible = 0;
    if (buffers.tiles_touched.size() < buffers.num_splats) {
        throw std::runtime_error("projection tiles_touched buffer is shorter than Gaussian count");
    }
    for (size_t i = 0; i < buffers.num_splats; ++i) {
        int32_t touched = buffers.tiles_touched[i];
        if (touched < 0 || static_cast<int64_t>(touched) > gridTiles) {
            std::ostringstream error;
            error << "projection invariant failed at Gaussian " << i << ": tiles=" << touched
                  << " gridTiles=" << gridTiles;
            throw std::runtime_error(error.str());
        }
        cpuSum += touched;
        if (touched > 0) visible++;
    }
    if (cpuSum != static_cast<int64_t>(buffers.num_indices)) {
        std::ostringstream error;
        error << "prefix-sum invariant failed: cpuTiles=" << cpuSum
              << " gpuCumsum=" << buffers.num_indices << " gaussians=" << buffers.num_splats;
        throw std::runtime_error(error.str());
    }
    std::ostringstream info;
    info << "Projection validated visible=" << visible << "/" << buffers.num_splats
         << " tileIndices=" << buffers.num_indices << " gridTiles=" << gridTiles;
    logi(info.str());
}

void processTiles(VulkanGSTrainer& trainer,
                  VulkanGSRendererUniforms& uniforms,
                  VulkanGSPipelineBuffers& buffers,
                  bool validate) {
    trainer.executeCalculateIndexBufferOffset(buffers);
    if (buffers.num_indices == 0) return;

    const size_t safeIndices = maxTileIndicesForBudget();
    if (buffers.num_indices > safeIndices) {
        std::ostringstream error;
        error << "projected tile working set exceeds mobile budget: indices=" << buffers.num_indices
              << " limit=" << safeIndices << " budgetMB=" << (kTileWorkingSetBudgetBytes >> 20)
              << " gaussians=" << buffers.num_splats
              << " image=" << uniforms.image_width << "x" << uniforms.image_height;
        loge(error.str());
        throw std::runtime_error(error.str());
    }
    if (validate) validateProjectionInvariants(trainer, uniforms, buffers);

    trainer.executeGenerateKeys(uniforms, buffers);
    trainer.executeSort(uniforms, buffers, -1);
    trainer.executeComputeTileRanges(uniforms, buffers);
}

void forward(VulkanGSTrainer& trainer,
             VulkanGSRendererUniforms& uniforms,
             VulkanGSPipelineBuffers& buffers,
             bool validate) {
    uniforms.num_splats = static_cast<uint32_t>(buffers.num_splats);
    trainer.executeProjectionForward(uniforms, buffers);
    processTiles(trainer, uniforms, buffers, validate);
    trainer.executeRasterizeForward(uniforms, buffers);
}

bool trainOneStep(VulkanGSTrainer& trainer,
                  const TrainerConfig& config,
                  VulkanGSRendererUniforms& uniforms,
                  VulkanGSPipelineBuffers& buffers,
                  size_t imageIndex,
                  int localStep,
                  uint32_t cumulativeStep) {
    trainer.get_train_camera(imageIndex, uniforms);
    const int shInterval = std::max(1, config.max_steps / 4);
    uniforms.active_sh = static_cast<uint32_t>(std::min<int>(cumulativeStep / shInterval, 3));
    uniforms.step = cumulativeStep;

    const bool validate = localStep == 0 || (localStep + 1) % 100 == 0;
    auto guard = DeviceGuard(&trainer);
    forward(trainer, uniforms, buffers, validate);
    if (buffers.num_indices == 0) return false;
    trainer.executeComputeSSIMGradient(config, uniforms, buffers, imageIndex);
    trainer.executeRasterizeBackward(uniforms, buffers);
    trainer.executeFusedProjectionBackwardOptimizerStep(config, uniforms, buffers, localStep + 1);
    trainer.executeMCMCPostBackward(config, uniforms, buffers, localStep);
    if (buffers.num_splats > static_cast<size_t>(config.cap_max)) {
        throw std::runtime_error("MCMC exceeded configured Gaussian budget");
    }
    return true;
}

struct ValidationMetrics {
    double psnr = NAN;
    double ssim = NAN;
    int views = 0;
};

void writeValidationPpm(const std::string& path, uint32_t width, uint32_t height,
                        const Buffer<float>& pixels) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("could not write hold-out render");
    out << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i) {
        unsigned char rgb[3];
        for (int c = 0; c < 3; ++c) {
            double value = std::clamp(static_cast<double>(pixels[4*i+c]), 0.0, 1.0);
            rgb[c] = static_cast<unsigned char>(std::lround(value * 255.0));
        }
        out.write(reinterpret_cast<const char*>(rgb), 3);
    }
}

double blockSsim(const Buffer<float>& predicted, const Buffer<uint8_t>& reference,
                 uint32_t width, uint32_t height) {
    constexpr int block = 8;
    constexpr double c1 = 0.01 * 0.01;
    constexpr double c2 = 0.03 * 0.03;
    double total = 0.0;
    size_t blocks = 0;
    for (uint32_t by = 0; by < height; by += block) {
        for (uint32_t bx = 0; bx < width; bx += block) {
            double meanP = 0.0, meanR = 0.0;
            size_t count = 0;
            for (uint32_t y = by; y < std::min<uint32_t>(height, by + block); ++y) {
                for (uint32_t x = bx; x < std::min<uint32_t>(width, bx + block); ++x) {
                    size_t i = static_cast<size_t>(y) * width + x;
                    double py = 0.299 * std::clamp<double>(predicted[4*i], 0.0, 1.0)
                              + 0.587 * std::clamp<double>(predicted[4*i+1], 0.0, 1.0)
                              + 0.114 * std::clamp<double>(predicted[4*i+2], 0.0, 1.0);
                    double ry = (0.299 * reference[4*i] + 0.587 * reference[4*i+1]
                               + 0.114 * reference[4*i+2]) / 255.0;
                    meanP += py; meanR += ry; count++;
                }
            }
            if (count < 2) continue;
            meanP /= count; meanR /= count;
            double varP = 0.0, varR = 0.0, cov = 0.0;
            for (uint32_t y = by; y < std::min<uint32_t>(height, by + block); ++y) {
                for (uint32_t x = bx; x < std::min<uint32_t>(width, bx + block); ++x) {
                    size_t i = static_cast<size_t>(y) * width + x;
                    double py = 0.299 * std::clamp<double>(predicted[4*i], 0.0, 1.0)
                              + 0.587 * std::clamp<double>(predicted[4*i+1], 0.0, 1.0)
                              + 0.114 * std::clamp<double>(predicted[4*i+2], 0.0, 1.0);
                    double ry = (0.299 * reference[4*i] + 0.587 * reference[4*i+1]
                               + 0.114 * reference[4*i+2]) / 255.0;
                    double dp = py - meanP, dr = ry - meanR;
                    varP += dp * dp; varR += dr * dr; cov += dp * dr;
                }
            }
            double denom = static_cast<double>(count - 1);
            varP /= denom; varR /= denom; cov /= denom;
            double value = ((2.0 * meanP * meanR + c1) * (2.0 * cov + c2))
                         / ((meanP * meanP + meanR * meanR + c1) * (varP + varR + c2));
            if (std::isfinite(value)) { total += value; blocks++; }
        }
    }
    return blocks ? total / static_cast<double>(blocks) : NAN;
}

ValidationMetrics validationMetrics(VulkanGSTrainer& trainer,
                                    VulkanGSRendererUniforms& uniforms,
                                    VulkanGSPipelineBuffers& buffers,
                                    const std::string& dataRoot) {
    ValidationMetrics result;
    double squaredError = 0.0;
    size_t samples = 0;
    double ssimTotal = 0.0;
    int ssimViews = 0;
    for (size_t view = 0; view < trainer.num_val(); ++view) {
        trainer.get_val_camera(view, uniforms);
        uniforms.active_sh = 3;
        uniforms.step = trainer.getCompletedTrainingSteps();
        {
            auto guard = DeviceGuard(&trainer);
            forward(trainer, uniforms, buffers, true);
        }
        trainer.copyFromDevice(buffers.pixel_state);
        auto& reference = trainer.get_val_image(view).buffer;
        size_t pixelCount = static_cast<size_t>(uniforms.image_width) * uniforms.image_height;
        if (buffers.pixel_state.size() < pixelCount * 4 || reference.size() < pixelCount * 4) continue;
        for (size_t i = 0; i < pixelCount; ++i) {
            for (int c = 0; c < 3; ++c) {
                double predicted = std::clamp(static_cast<double>(buffers.pixel_state[4*i+c]), 0.0, 1.0);
                double expected = static_cast<double>(reference[4*i+c]) / 255.0;
                double d = predicted - expected;
                squaredError += d * d; samples++;
            }
        }
        double ssim = blockSsim(buffers.pixel_state, reference, uniforms.image_width, uniforms.image_height);
        if (std::isfinite(ssim)) { ssimTotal += ssim; ssimViews++; }
        writeValidationPpm(dataRoot + "/phase3_holdout_render_" + std::to_string(view) + ".ppm",
                           uniforms.image_width, uniforms.image_height, buffers.pixel_state);
        result.views++;
    }
    if (samples > 0) {
        double mse = squaredError / static_cast<double>(samples);
        result.psnr = mse <= 1e-12 ? 120.0 : 10.0 * std::log10(1.0 / mse);
    }
    if (ssimViews > 0) result.ssim = ssimTotal / ssimViews;
    return result;
}

std::string deviceName(VulkanGSTrainer& trainer) {
    auto info = trainer.get_device_info();
    auto it = info.find("name");
    if (it == info.end()) return "Vulkan device";
    if (auto value = std::get_if<std::string>(&it->second)) return *value;
    return "Vulkan device";
}

std::string successJson(size_t count, uint32_t cumulativeSteps, int addedSteps,
                        int optimizedSteps, bool resumed, double psnr, double ssim,
                        int validationViews, const std::string& device, size_t peakBytes) {
    std::ostringstream out;
    out << "{\"success\":true,\"message\":\"mobile 3DGS training complete\",\"gaussian_count\":"
        << count << ",\"steps\":" << cumulativeSteps
        << ",\"added_steps\":" << addedSteps
        << ",\"optimized_steps\":" << optimizedSteps
        << ",\"resumed\":" << (resumed ? "true" : "false")
        << ",\"gaussian_budget\":" << kGaussianBudget
        << ",\"peak_vram_mb\":" << (peakBytes / (1024.0 * 1024.0))
        << ",\"strategy\":\"bounded_mcmc\",\"initialization\":\"surface_knn_16_3\""
        << ",\"device\":\"" << jsonEscape(device) << "\"";
    if (std::isfinite(psnr)) out << ",\"validation_psnr\":" << psnr;
    if (std::isfinite(ssim)) out << ",\"validation_ssim\":" << ssim;
    out << ",\"validation_view_count\":" << validationViews;
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
        const int steps = std::max(1, static_cast<int>(trainSteps));
        progress.send(1, "Vulkan学習器を開始しています…");
        VulkanGSTrainer trainer;
        VulkanGSPipelineBuffers buffers;
        VulkanGSRendererUniforms uniforms{};
        TrainerConfig config = makeConfig(dataRoot, imageDir, sparseDir, outputPly,
                                          static_cast<int>(frameCount), steps);

        trainer.initialize(makeShaderPaths(shaderDir), -1);
        std::string gpu = deviceName(trainer);
        logi("Mobile trainer initialized device=" + gpu);
        progress.send(4, "端末向け3DGS学習を準備しています…");

        trainer.load_colmap_dataset(config, buffers);
        if (trainer.num_train() == 0) throw std::runtime_error("no training camera views");
        if (buffers.num_splats < 64) throw std::runtime_error("initial 3D points are insufficient");

        const bool resumed = trainer.restoreTrainingCheckpoint(buffers);
        const uint32_t completedBeforeRun = trainer.getResumeTrainingStep();
        if (resumed) {
            progress.send(7, "保存したGaussian・optimizer状態を復元しました…");
            logi("Training checkpoint restored steps=" + std::to_string(completedBeforeRun)
                 + " gaussians=" + std::to_string(buffers.num_splats));
        } else {
            spatialBudgetInitialCloud(trainer, buffers, kInitialGaussianBudget);
            progress.send(7, "Depthから表面方向を推定しています…");
            initializeSurfaceGaussians(trainer, buffers);
        }

        {
            std::ostringstream info;
            info << "Training start views=" << trainer.num_train() << " initial=" << buffers.num_splats
                 << " additionalSteps=" << steps << " resumeStep=" << completedBeforeRun
                 << " loss=L1+SSIM strategy=bounded_mcmc"
                 << " gaussianBudget=" << config.cap_max
                 << " tileBudgetMB=" << (kTileWorkingSetBudgetBytes >> 20)
                 << " surfaceInit=" << (resumed ? "checkpoint" : "K16/K3");
            logi(info.str());
        }

        int optimizedSteps = 0;
        const uint64_t targetStep = static_cast<uint64_t>(completedBeforeRun)
                + static_cast<uint64_t>(steps);
        if (targetStep > UINT32_MAX) throw std::runtime_error("cumulative training step overflow");

        for (int step = 0; step < steps; ++step) {
            const uint32_t cumulativeStep = completedBeforeRun + static_cast<uint32_t>(step);
            size_t imageIndex = static_cast<size_t>(cumulativeStep) % trainer.num_train();
            if (trainOneStep(trainer, config, uniforms, buffers, imageIndex, step, cumulativeStep)) {
                optimizedSteps++;
            }
            if (step == 0 || (step + 1) % 50 == 0 || step + 1 == steps) {
                int percent = 10 + static_cast<int>(85.0 * (step + 1) / steps);
                const uint32_t current = completedBeforeRun + static_cast<uint32_t>(step + 1);
                std::ostringstream message;
                message << "端末向け3DGSを最適化しています… " << current << "/" << targetStep;
                progress.send(percent, message.str());
                std::ostringstream log;
                log << "step=" << current << "/" << targetStep
                    << " added=" << (step + 1) << "/" << steps
                    << " optimized=" << optimizedSteps
                    << " gaussians=" << buffers.num_splats
                    << " tileIndices=" << buffers.num_indices
                    << " activeSH=" << uniforms.active_sh
                    << " peakMB=" << (trainer.getPeakAllocSize() / (1024.0 * 1024.0));
                logi(log.str());
            }
        }

        const int minOptimizedSteps = std::max(1, steps * 3 / 4);
        if (optimizedSteps < minOptimizedSteps) {
            std::ostringstream error;
            error << "insufficient projected training steps: optimized=" << optimizedSteps
                  << " required=" << minOptimizedSteps << " added=" << steps
                  << ". Check camera pose/intrinsics and geometry coverage.";
            throw std::runtime_error(error.str());
        }

        progress.send(96, "学習結果を確認しています…");
        ValidationMetrics validation = validationMetrics(trainer, uniforms, buffers, dataRoot);
        double psnr = validation.psnr;
        double ssim = validation.ssim;
        progress.send(98, "3DGSモデルと追加学習checkpointを書き出しています…");
        const size_t finalCount = buffers.num_splats;
        const size_t peakBytes = trainer.getPeakAllocSize();
        trainer.writePLY(outputPly, buffers);
        const uint32_t cumulativeSteps = trainer.getCompletedTrainingSteps();
        if (cumulativeSteps != static_cast<uint32_t>(targetStep)) {
            throw std::runtime_error("checkpoint cumulative step mismatch");
        }
        logi("Training COMPLETE gaussians=" + std::to_string(finalCount)
             + " totalSteps=" + std::to_string(cumulativeSteps)
             + " addedSteps=" + std::to_string(steps)
             + " resumed=" + std::string(resumed ? "true" : "false")
             + " optimizedSteps=" + std::to_string(optimizedSteps)
             + " peakMB=" + std::to_string(peakBytes / (1024.0 * 1024.0))
             + " psnr=" + (std::isfinite(psnr) ? std::to_string(psnr) : std::string("n/a"))
             + " ssim=" + (std::isfinite(ssim) ? std::to_string(ssim) : std::string("n/a"))
             + " holdoutViews=" + std::to_string(validation.views));
        trainer.cleanupBuffers(buffers);
        trainer.cleanup();
        progress.send(100, resumed ? "3DGS追加学習が完了しました" : "3DGS学習が完了しました");
        std::string json = successJson(finalCount, cumulativeSteps, steps, optimizedSteps,
                                       resumed, psnr, ssim, validation.views, gpu, peakBytes);
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
