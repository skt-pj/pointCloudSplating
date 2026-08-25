#!/usr/bin/env python3
from pathlib import Path

root = Path("app/src/main/cpp/third_party/vksplat/vksplat/src")
header_path = root / "gs_trainer.h"
source_path = root / "gs_trainer.cpp"

header = header_path.read_text()
old_public = '''    void writePLY(std::string filename, VulkanGSPipelineBuffers& buffers);\n\nprivate:\n    std::default_random_engine rng;\n'''
new_public = '''    void writePLY(std::string filename, VulkanGSPipelineBuffers& buffers);\n\n    // PCS Android persists the full trainable state so a completed model can be refined later.\n    bool restoreTrainingCheckpoint(VulkanGSPipelineBuffers& buffers);\n    uint32_t getResumeTrainingStep() const { return resume_training_step; }\n    uint32_t getCompletedTrainingSteps() const { return completed_training_step; }\n\nprivate:\n    void saveTrainingCheckpoint(VulkanGSPipelineBuffers& buffers);\n\n    std::string training_checkpoint_path;\n    uint32_t resume_training_step = 0;\n    uint32_t completed_training_step = 0;\n    std::default_random_engine rng;\n'''
if old_public not in header:
    raise SystemExit("gs_trainer checkpoint header anchor not found")
header = header.replace(old_public, new_public, 1)
header_path.write_text(header)

source = source_path.read_text()
include_anchor = '''#include <filesystem>\n'''
include_replacement = '''#include <filesystem>\n#include <fstream>\n#include <sstream>\n#include <system_error>\n'''
if include_anchor not in source:
    raise SystemExit("gs_trainer checkpoint include anchor not found")
source = source.replace(include_anchor, include_replacement, 1)

load_anchor = '''void VulkanGSTrainer::load_colmap_dataset(\n    const TrainerConfig &config,\n    VulkanGSPipelineBuffers& buffers\n) {\n    ColmapReader reader;\n'''
load_replacement = '''void VulkanGSTrainer::load_colmap_dataset(\n    const TrainerConfig &config,\n    VulkanGSPipelineBuffers& buffers\n) {\n    training_checkpoint_path = config.output_dir + "/3dgs_checkpoint.bin";\n    resume_training_step = 0;\n    completed_training_step = 0;\n\n    ColmapReader reader;\n'''
if load_anchor not in source:
    raise SystemExit("gs_trainer checkpoint dataset anchor not found")
source = source.replace(load_anchor, load_replacement, 1)

camera_anchor = '''void VulkanGSTrainer::camera_to_uniforms(const Camera &cam, VulkanGSRendererUniforms &uniforms) const {\n'''
checkpoint_methods = r'''namespace {
constexpr char kPcsCheckpointMagic[8] = {'P','C','S','3','D','G','S','1'};
constexpr uint32_t kPcsCheckpointVersion = 1;

void pcsWriteExact(std::ofstream& out, const void* data, size_t bytes, const char* what) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!out.good()) throw std::runtime_error(std::string("checkpoint write failed: ") + what);
}

void pcsReadExact(std::ifstream& in, void* data, size_t bytes, const char* what) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in.good()) throw std::runtime_error(std::string("checkpoint read failed: ") + what);
}
}

bool VulkanGSTrainer::restoreTrainingCheckpoint(VulkanGSPipelineBuffers& buffers) {
    if (training_checkpoint_path.empty() || !std::filesystem::exists(training_checkpoint_path))
        return false;

    std::ifstream in(training_checkpoint_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open 3DGS training checkpoint");

    char magic[8]{};
    uint32_t version = 0;
    uint32_t completed = 0;
    uint64_t num_splats_u64 = 0;
    uint64_t rng_bytes = 0;
    pcsReadExact(in, magic, sizeof(magic), "magic");
    pcsReadExact(in, &version, sizeof(version), "version");
    pcsReadExact(in, &completed, sizeof(completed), "completed step");
    pcsReadExact(in, &num_splats_u64, sizeof(num_splats_u64), "Gaussian count");
    pcsReadExact(in, &rng_bytes, sizeof(rng_bytes), "RNG length");
    if (std::memcmp(magic, kPcsCheckpointMagic, sizeof(magic)) != 0 ||
        version != kPcsCheckpointVersion)
        throw std::runtime_error("unsupported 3DGS training checkpoint format");
    if (completed == 0 || num_splats_u64 < 64 || num_splats_u64 > static_cast<uint64_t>(UINT32_MAX))
        throw std::runtime_error("invalid 3DGS training checkpoint metadata");
    if (rng_bytes == 0 || rng_bytes > 1024 * 1024)
        throw std::runtime_error("invalid 3DGS training checkpoint RNG state");

    std::string rng_state(static_cast<size_t>(rng_bytes), '\0');
    pcsReadExact(in, rng_state.data(), rng_state.size(), "RNG state");

    const size_t n = static_cast<size_t>(num_splats_u64);
    const size_t sh_n = 16 * 3 * _CEIL_ROUND(n, SH_REORDER_SIZE);
    auto readBuffer = [&](Buffer<float>& buffer, size_t expected, const char* name) {
        uint64_t count_u64 = 0;
        pcsReadExact(in, &count_u64, sizeof(count_u64), name);
        if (count_u64 != expected)
            throw std::runtime_error(std::string("checkpoint buffer length mismatch: ") + name);
        buffer.resize(expected);
        if (expected > 0)
            pcsReadExact(in, buffer.data(), expected * sizeof(float), name);
    };

    readBuffer(buffers.xyz_ws, 3 * n, "xyz_ws");
    readBuffer(buffers.sh_coeffs, sh_n, "sh_coeffs");
    readBuffer(buffers.rotations, 4 * n, "rotations");
    readBuffer(buffers.scales_opacs, 4 * n, "scales_opacs");
    readBuffer(buffers.g_xyz_ws, 2 * 3 * n, "g_xyz_ws");
    readBuffer(buffers.g_sh_coeffs_1, sh_n, "g_sh_coeffs_1");
    readBuffer(buffers.g_sh_coeffs_2, sh_n, "g_sh_coeffs_2");
    readBuffer(buffers.g_rotations, 2 * 4 * n, "g_rotations");
    readBuffer(buffers.g_scales_opacs, 2 * 4 * n, "g_scales_opacs");

    std::istringstream rng_in(rng_state);
    rng_in >> rng;
    if (rng_in.fail()) throw std::runtime_error("invalid 3DGS training checkpoint RNG data");

    // Do not modify live GPU state until the whole file has been validated and decoded.
    auto replaceDeviceBuffer = [&](Buffer<float>& buffer) {
        destroyBuffer(buffer.deviceBuffer);
        buffer.deviceBuffer = _VulkanBuffer();
        copyToDevice(buffer);
    };
    replaceDeviceBuffer(buffers.xyz_ws);
    replaceDeviceBuffer(buffers.sh_coeffs);
    replaceDeviceBuffer(buffers.rotations);
    replaceDeviceBuffer(buffers.scales_opacs);
    replaceDeviceBuffer(buffers.g_xyz_ws);
    replaceDeviceBuffer(buffers.g_sh_coeffs_1);
    replaceDeviceBuffer(buffers.g_sh_coeffs_2);
    replaceDeviceBuffer(buffers.g_rotations);
    replaceDeviceBuffer(buffers.g_scales_opacs);

    buffers.num_splats = n;
    buffers.num_indices = 0;
    buffers.is_unsorted_1 = true;
    resume_training_step = completed;
    completed_training_step = completed;
    printf("Restored PCS 3DGS checkpoint: %u steps, %zu Gaussians\n", completed, n);
    fflush(stdout);
    return true;
}

void VulkanGSTrainer::saveTrainingCheckpoint(VulkanGSPipelineBuffers& buffers) {
    if (training_checkpoint_path.empty() || completed_training_step == 0)
        return;

    const size_t n = buffers.num_splats;
    const size_t sh_n = 16 * 3 * _CEIL_ROUND(n, SH_REORDER_SIZE);
    const std::string temp_path = training_checkpoint_path + ".tmp";
    std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create 3DGS training checkpoint");

    std::ostringstream rng_out;
    rng_out << rng;
    const std::string rng_state = rng_out.str();
    const uint64_t n_u64 = static_cast<uint64_t>(n);
    const uint64_t rng_bytes = static_cast<uint64_t>(rng_state.size());
    pcsWriteExact(out, kPcsCheckpointMagic, sizeof(kPcsCheckpointMagic), "magic");
    pcsWriteExact(out, &kPcsCheckpointVersion, sizeof(kPcsCheckpointVersion), "version");
    pcsWriteExact(out, &completed_training_step, sizeof(completed_training_step), "completed step");
    pcsWriteExact(out, &n_u64, sizeof(n_u64), "Gaussian count");
    pcsWriteExact(out, &rng_bytes, sizeof(rng_bytes), "RNG length");
    pcsWriteExact(out, rng_state.data(), rng_state.size(), "RNG state");

    auto writeBuffer = [&](Buffer<float>& buffer, size_t expected, const char* name) {
        if (buffer.deviceBuffer.buffer == VK_NULL_HANDLE || buffer.deviceSize() < expected)
            throw std::runtime_error(std::string("checkpoint GPU buffer missing: ") + name);
        copyFromDevice(buffer);
        if (buffer.size() < expected)
            throw std::runtime_error(std::string("checkpoint host buffer too short: ") + name);
        const uint64_t count_u64 = static_cast<uint64_t>(expected);
        pcsWriteExact(out, &count_u64, sizeof(count_u64), name);
        if (expected > 0)
            pcsWriteExact(out, buffer.data(), expected * sizeof(float), name);
        // The device copy remains alive. Drop the temporary CPU mirror to keep phone RAM bounded.
        buffer.clear();
        buffer.shrink_to_fit();
    };

    writeBuffer(buffers.xyz_ws, 3 * n, "xyz_ws");
    writeBuffer(buffers.sh_coeffs, sh_n, "sh_coeffs");
    writeBuffer(buffers.rotations, 4 * n, "rotations");
    writeBuffer(buffers.scales_opacs, 4 * n, "scales_opacs");
    writeBuffer(buffers.g_xyz_ws, 2 * 3 * n, "g_xyz_ws");
    writeBuffer(buffers.g_sh_coeffs_1, sh_n, "g_sh_coeffs_1");
    writeBuffer(buffers.g_sh_coeffs_2, sh_n, "g_sh_coeffs_2");
    writeBuffer(buffers.g_rotations, 2 * 4 * n, "g_rotations");
    writeBuffer(buffers.g_scales_opacs, 2 * 4 * n, "g_scales_opacs");
    out.flush();
    if (!out.good()) throw std::runtime_error("failed to flush 3DGS training checkpoint");
    out.close();

    std::error_code rename_error;
    std::filesystem::rename(temp_path, training_checkpoint_path, rename_error);
    if (rename_error) {
        std::filesystem::remove(temp_path);
        throw std::runtime_error("failed to publish 3DGS training checkpoint: " + rename_error.message());
    }
    printf("Saved PCS 3DGS checkpoint: %u steps, %zu Gaussians\n", completed_training_step, n);
    fflush(stdout);
}

'''
if camera_anchor not in source:
    raise SystemExit("gs_trainer checkpoint method insertion anchor not found")
source = source.replace(camera_anchor, checkpoint_methods + camera_anchor, 1)

step_anchor = '''    shaderUniforms.step = step;\n'''
step_replacement = '''    const uint32_t global_step = resume_training_step + static_cast<uint32_t>(std::max(step, 1));\n    shaderUniforms.step = global_step;\n'''
if step_anchor not in source:
    raise SystemExit("optimizer step anchor not found")
source = source.replace(step_anchor, step_replacement, 1)

lr_anchor = '''    shaderUniforms.lr_means = config.get_means_lr(step, scene_scale);\n'''
lr_replacement = '''    // A resumed run keeps the converged means LR instead of jumping back to the beginning\n    // of the decay schedule. Adam bias correction still uses the cumulative global step above.\n    shaderUniforms.lr_means = resume_training_step > 0\n        ? config.means_lr_final * scene_scale\n        : config.get_means_lr(global_step, scene_scale);\n'''
if lr_anchor not in source:
    raise SystemExit("optimizer means LR anchor not found")
source = source.replace(lr_anchor, lr_replacement, 1)

optimizer_end = '''            resizeAndCopyDeviceBuffer(buffers.g_scales_opacs, 2*4*alloc_size, true),\n        }\n    );\n}\n\n\nvoid VulkanGSTrainer::barrierAllGaussParams'''
optimizer_replacement = '''            resizeAndCopyDeviceBuffer(buffers.g_scales_opacs, 2*4*alloc_size, true),\n        }\n    );\n    completed_training_step = std::max(completed_training_step, global_step);\n}\n\n\nvoid VulkanGSTrainer::barrierAllGaussParams'''
if optimizer_end not in source:
    raise SystemExit("optimizer completion anchor not found")
source = source.replace(optimizer_end, optimizer_replacement, 1)

noise_anchor = '''    uniforms[4].f = config.noise_lr * config.get_means_lr(step, scene_scale);\n'''
noise_replacement = '''    uniforms[4].f = config.noise_lr * (resume_training_step > 0\n        ? config.means_lr_final * scene_scale\n        : config.get_means_lr(step, scene_scale));\n'''
if noise_anchor not in source:
    raise SystemExit("MCMC noise LR anchor not found")
source = source.replace(noise_anchor, noise_replacement, 1)

ply_end = '''    fclose(fp);\n}'''
ply_replacement = '''    fclose(fp);\n\n    // Persist trainable parameters, Adam moments and RNG only after the public PLY was written.\n    // writePLY mutates only CPU mirrors; saveTrainingCheckpoint re-reads canonical GPU buffers.\n    saveTrainingCheckpoint(buffers);\n}'''
pos = source.rfind(ply_end)
if pos < 0:
    raise SystemExit("writePLY checkpoint anchor not found")
source = source[:pos] + source[pos:].replace(ply_end, ply_replacement, 1)

source_path.write_text(source)
print("Patched VkSplat with resumable PCS 3DGS checkpoints (parameters + Adam + RNG + cumulative step)")
