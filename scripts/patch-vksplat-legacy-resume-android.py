#!/usr/bin/env python3
from pathlib import Path

root = Path("app/src/main/cpp/third_party/vksplat/vksplat/src")
header_path = root / "gs_trainer.h"
source_path = root / "gs_trainer.cpp"

header = header_path.read_text()
old_public = '''    bool restoreTrainingCheckpoint(VulkanGSPipelineBuffers& buffers);\n    uint32_t getResumeTrainingStep() const { return resume_training_step; }\n    uint32_t getCompletedTrainingSteps() const { return completed_training_step; }\n'''
new_public = '''    bool restoreTrainingCheckpoint(VulkanGSPipelineBuffers& buffers);\n    bool importLegacyTrainingPly(const std::string& filename, uint32_t cumulativeStep,\n                                 VulkanGSPipelineBuffers& buffers);\n    uint32_t getResumeTrainingStep() const { return resume_training_step; }\n    uint32_t getCompletedTrainingSteps() const { return completed_training_step; }\n'''
if old_public not in header:
    raise SystemExit("legacy resume public API anchor not found")
header = header.replace(old_public, new_public, 1)

old_fields = '''    std::string training_checkpoint_path;\n    uint32_t resume_training_step = 0;\n    uint32_t completed_training_step = 0;\n    std::default_random_engine rng;\n'''
new_fields = '''    std::string training_checkpoint_path;\n    // Total user-visible optimization iterations and Adam moment age are separate. They are equal\n    // for native checkpoints. A legacy PLY has its cumulative total but no saved moments, so only\n    // that one migration begins Adam at age zero while preserving the Gaussian model itself.\n    uint32_t resume_training_step = 0;\n    uint32_t completed_training_step = 0;\n    uint32_t resume_optimizer_step = 0;\n    uint32_t completed_optimizer_step = 0;\n    std::default_random_engine rng;\n'''
if old_fields not in header:
    raise SystemExit("legacy resume trainer-field anchor not found")
header = header.replace(old_fields, new_fields, 1)
header_path.write_text(header)

source = source_path.read_text()
source = source.replace(
'''    resume_training_step = 0;\n    completed_training_step = 0;\n\n    ColmapReader reader;\n''',
'''    resume_training_step = 0;\n    completed_training_step = 0;\n    resume_optimizer_step = 0;\n    completed_optimizer_step = 0;\n\n    ColmapReader reader;\n''', 1)

old_magic = '''constexpr char kPcsCheckpointMagic[8] = {'P','C','S','3','D','G','S','1'};\nconstexpr uint32_t kPcsCheckpointVersion = 1;\n'''
new_magic = '''constexpr char kPcsCheckpointMagic[8] = {'P','C','S','3','D','G','S','2'};\nconstexpr uint32_t kPcsCheckpointVersion = 2;\n'''
if old_magic not in source:
    raise SystemExit("legacy resume checkpoint-version anchor not found")
source = source.replace(old_magic, new_magic, 1)

old_read_meta = '''    uint32_t completed = 0;\n    uint64_t num_splats_u64 = 0;\n    uint64_t rng_bytes = 0;\n    pcsReadExact(in, magic, sizeof(magic), "magic");\n    pcsReadExact(in, &version, sizeof(version), "version");\n    pcsReadExact(in, &completed, sizeof(completed), "completed step");\n    pcsReadExact(in, &num_splats_u64, sizeof(num_splats_u64), "Gaussian count");\n'''
new_read_meta = '''    uint32_t completed = 0;\n    uint32_t optimizer_completed = 0;\n    uint64_t num_splats_u64 = 0;\n    uint64_t rng_bytes = 0;\n    pcsReadExact(in, magic, sizeof(magic), "magic");\n    pcsReadExact(in, &version, sizeof(version), "version");\n    pcsReadExact(in, &completed, sizeof(completed), "completed step");\n    pcsReadExact(in, &optimizer_completed, sizeof(optimizer_completed), "optimizer step");\n    pcsReadExact(in, &num_splats_u64, sizeof(num_splats_u64), "Gaussian count");\n'''
if old_read_meta not in source:
    raise SystemExit("legacy resume checkpoint-read anchor not found")
source = source.replace(old_read_meta, new_read_meta, 1)

old_validate = '''    if (completed == 0 || num_splats_u64 < 64 || num_splats_u64 > static_cast<uint64_t>(UINT32_MAX))\n        throw std::runtime_error("invalid 3DGS training checkpoint metadata");\n'''
new_validate = '''    if (completed == 0 || optimizer_completed == 0 || optimizer_completed > completed ||\n        num_splats_u64 < 64 || num_splats_u64 > static_cast<uint64_t>(UINT32_MAX))\n        throw std::runtime_error("invalid 3DGS training checkpoint metadata");\n'''
if old_validate not in source:
    raise SystemExit("legacy resume checkpoint-validation anchor not found")
source = source.replace(old_validate, new_validate, 1)

old_restore_tail = '''    resume_training_step = completed;\n    completed_training_step = completed;\n    printf("Restored PCS 3DGS checkpoint: %u steps, %zu Gaussians\\n", completed, n);\n'''
new_restore_tail = '''    resume_training_step = completed;\n    completed_training_step = completed;\n    resume_optimizer_step = optimizer_completed;\n    completed_optimizer_step = optimizer_completed;\n    printf("Restored PCS 3DGS checkpoint: %u total steps, %u optimizer steps, %zu Gaussians\\n",\n           completed, optimizer_completed, n);\n'''
if old_restore_tail not in source:
    raise SystemExit("legacy resume restore-tail anchor not found")
source = source.replace(old_restore_tail, new_restore_tail, 1)

save_anchor = '''void VulkanGSTrainer::saveTrainingCheckpoint(VulkanGSPipelineBuffers& buffers) {\n'''
legacy_import = r'''bool VulkanGSTrainer::importLegacyTrainingPly(
    const std::string& filename,
    uint32_t cumulativeStep,
    VulkanGSPipelineBuffers& buffers
) {
    if (cumulativeStep == 0 || filename.empty() || !std::filesystem::exists(filename))
        return false;

    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open legacy 3DGS PLY for continuation");

    std::string line;
    if (!std::getline(in, line) || line != "ply")
        throw std::runtime_error("legacy 3DGS PLY has invalid header");
    if (!std::getline(in, line) || line != "format binary_little_endian 1.0")
        throw std::runtime_error("legacy 3DGS PLY must be binary little endian");

    uint64_t vertex_count = 0;
    std::vector<std::string> properties;
    bool in_vertex = false;
    bool ended = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("element vertex ", 0) == 0) {
            vertex_count = std::stoull(line.substr(std::string("element vertex ").size()));
            in_vertex = true;
            continue;
        }
        if (line.rfind("element ", 0) == 0) {
            in_vertex = false;
            continue;
        }
        if (in_vertex && line.rfind("property float ", 0) == 0) {
            properties.push_back(line.substr(std::string("property float ").size()));
            continue;
        }
        if (line == "end_header") {
            ended = true;
            break;
        }
    }
    if (!ended || vertex_count < 64 || vertex_count > UINT32_MAX)
        throw std::runtime_error("legacy 3DGS PLY has invalid Gaussian count");

    const std::vector<std::string> expected = {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "f_rest_0", "f_rest_1", "f_rest_2", "f_rest_3", "f_rest_4", "f_rest_5",
        "f_rest_6", "f_rest_7", "f_rest_8", "f_rest_9", "f_rest_10", "f_rest_11",
        "f_rest_12", "f_rest_13", "f_rest_14", "f_rest_15", "f_rest_16", "f_rest_17",
        "f_rest_18", "f_rest_19", "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_23",
        "f_rest_24", "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29",
        "f_rest_30", "f_rest_31", "f_rest_32", "f_rest_33", "f_rest_34", "f_rest_35",
        "f_rest_36", "f_rest_37", "f_rest_38", "f_rest_39", "f_rest_40", "f_rest_41",
        "f_rest_42", "f_rest_43", "f_rest_44",
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"
    };
    if (properties != expected)
        throw std::runtime_error("legacy 3DGS PLY property layout is not a PCS/VkSplat model");

    const size_t n = static_cast<size_t>(vertex_count);
    Buffer<float> xyz;
    Buffer<float> sh;
    Buffer<float> rotations;
    Buffer<float> scales_opacs;
    xyz.resize(3 * n);
    sh.resize(48 * n);
    rotations.resize(4 * n);
    scales_opacs.resize(4 * n);

    static const int sh_permute[48] = {
        0, 1, 2, 3, 18, 33, 4, 19, 34, 5, 20, 35, 6, 21, 36, 7, 22, 37,
        8, 23, 38, 9, 24, 39, 10, 25, 40, 11, 26, 41, 12, 27, 42, 13, 28, 43,
        14, 29, 44, 15, 30, 45, 16, 31, 46, 17, 32, 47
    };
    std::array<float, 59> record{};
    for (size_t i = 0; i < n; ++i) {
        pcsReadExact(in, record.data(), record.size() * sizeof(float), "legacy PLY vertex");
        for (float value : record) {
            if (!std::isfinite(value))
                throw std::runtime_error("legacy 3DGS PLY contains non-finite values");
        }
        xyz[3*i] = record[0];
        xyz[3*i+1] = record[1];
        xyz[3*i+2] = record[2];
        for (int c = 0; c < 48; ++c)
            sh[48*i+c] = record[3 + sh_permute[c]];
        const float opacity_logit = std::clamp(record[51], -30.0f, 30.0f);
        scales_opacs[4*i] = std::exp(std::clamp(record[52], -30.0f, 30.0f));
        scales_opacs[4*i+1] = std::exp(std::clamp(record[53], -30.0f, 30.0f));
        scales_opacs[4*i+2] = std::exp(std::clamp(record[54], -30.0f, 30.0f));
        scales_opacs[4*i+3] = 1.0f / (1.0f + std::exp(-opacity_logit));
        for (int c = 0; c < 4; ++c)
            rotations[4*i+c] = record[55+c];
    }

    buffers.xyz_ws.assign(xyz.begin(), xyz.end());
    buffers.sh_coeffs.assign(sh.begin(), sh.end());
    buffers.rotations.assign(rotations.begin(), rotations.end());
    buffers.scales_opacs.assign(scales_opacs.begin(), scales_opacs.end());
    buffers.num_splats = n;
    buffers.reorderSH(buffers.sh_coeffs);

    const size_t sh_n = 16 * 3 * _CEIL_ROUND(n, SH_REORDER_SIZE);
    buffers.g_xyz_ws.assign(2 * 3 * n, 0.0f);
    buffers.g_sh_coeffs_1.assign(sh_n, 0.0f);
    buffers.g_sh_coeffs_2.assign(sh_n, 0.0f);
    buffers.g_rotations.assign(2 * 4 * n, 0.0f);
    buffers.g_scales_opacs.assign(2 * 4 * n, 0.0f);

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

    buffers.num_indices = 0;
    buffers.is_unsorted_1 = true;
    resume_training_step = cumulativeStep;
    completed_training_step = cumulativeStep;
    resume_optimizer_step = 0;
    completed_optimizer_step = 0;
    printf("Imported legacy PCS 3DGS PLY: %u previous steps, Adam reset, %zu Gaussians\n",
           cumulativeStep, n);
    fflush(stdout);
    return true;
}

'''
if save_anchor not in source:
    raise SystemExit("legacy PLY import insertion anchor not found")
source = source.replace(save_anchor, legacy_import + save_anchor, 1)

old_save_meta = '''    pcsWriteExact(out, &kPcsCheckpointVersion, sizeof(kPcsCheckpointVersion), "version");\n    pcsWriteExact(out, &completed_training_step, sizeof(completed_training_step), "completed step");\n    pcsWriteExact(out, &n_u64, sizeof(n_u64), "Gaussian count");\n'''
new_save_meta = '''    pcsWriteExact(out, &kPcsCheckpointVersion, sizeof(kPcsCheckpointVersion), "version");\n    pcsWriteExact(out, &completed_training_step, sizeof(completed_training_step), "completed step");\n    pcsWriteExact(out, &completed_optimizer_step, sizeof(completed_optimizer_step), "optimizer step");\n    pcsWriteExact(out, &n_u64, sizeof(n_u64), "Gaussian count");\n'''
if old_save_meta not in source:
    raise SystemExit("legacy resume checkpoint-write anchor not found")
source = source.replace(old_save_meta, new_save_meta, 1)

old_save_log = '''    printf("Saved PCS 3DGS checkpoint: %u steps, %zu Gaussians\\n", completed_training_step, n);\n'''
new_save_log = '''    printf("Saved PCS 3DGS checkpoint: %u total steps, %u optimizer steps, %zu Gaussians\\n",\n           completed_training_step, completed_optimizer_step, n);\n'''
if old_save_log not in source:
    raise SystemExit("legacy resume checkpoint-log anchor not found")
source = source.replace(old_save_log, new_save_log, 1)

old_global_step = '''    const uint32_t global_step = resume_training_step + static_cast<uint32_t>(std::max(step, 1));\n    shaderUniforms.step = global_step;\n'''
new_global_step = '''    const uint32_t local_step = static_cast<uint32_t>(std::max(step, 1));\n    const uint32_t global_step = resume_optimizer_step + local_step;\n    const uint32_t total_training_step = resume_training_step + local_step;\n    shaderUniforms.step = global_step;\n'''
if old_global_step not in source:
    raise SystemExit("legacy resume Adam-step anchor not found")
source = source.replace(old_global_step, new_global_step, 1)

old_completed = '''    completed_training_step = std::max(completed_training_step, global_step);\n'''
new_completed = '''    completed_optimizer_step = std::max(completed_optimizer_step, global_step);\n    completed_training_step = std::max(completed_training_step, total_training_step);\n'''
if old_completed not in source:
    raise SystemExit("legacy resume completed-step anchor not found")
source = source.replace(old_completed, new_completed, 1)

source_path.write_text(source)
print("Patched VkSplat legacy PLY continuation with Adam-reset migration into v2 checkpoints")
