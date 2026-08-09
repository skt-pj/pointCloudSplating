#!/usr/bin/env python3
from pathlib import Path

root = Path("app/src/main/cpp/third_party/vksplat/vksplat/src")
header_path = root / "gs_trainer.h"
source_path = root / "gs_trainer.cpp"

header = header_path.read_text()
old_field = '''    std::string training_checkpoint_path;\n    // Total user-visible optimization iterations and Adam moment age are separate. They are equal\n'''
new_field = '''    std::string training_checkpoint_path;\n    std::string legacy_training_ply_path;\n    // Total user-visible optimization iterations and Adam moment age are separate. They are equal\n'''
if old_field not in header:
    raise SystemExit("legacy resume hint header anchor not found")
header = header.replace(old_field, new_field, 1)
header_path.write_text(header)

source = source_path.read_text()
include_anchor = '''#include <system_error>\n'''
include_replacement = '''#include <system_error>\n#include <array>\n#include <cmath>\n#include <cstring>\n#include <cstdint>\n'''
if include_anchor not in source:
    raise SystemExit("legacy resume explicit includes anchor not found")
source = source.replace(include_anchor, include_replacement, 1)

old_load = '''    training_checkpoint_path = config.output_dir + "/3dgs_checkpoint.bin";\n    resume_training_step = 0;\n'''
new_load = '''    training_checkpoint_path = config.output_dir + "/3dgs_checkpoint.bin";\n    legacy_training_ply_path = config.output_ply;\n    resume_training_step = 0;\n'''
if old_load not in source:
    raise SystemExit("legacy resume hint dataset anchor not found")
source = source.replace(old_load, new_load, 1)

old_restore = '''bool VulkanGSTrainer::restoreTrainingCheckpoint(VulkanGSPipelineBuffers& buffers) {\n    if (training_checkpoint_path.empty() || !std::filesystem::exists(training_checkpoint_path))\n        return false;\n\n    std::ifstream in(training_checkpoint_path, std::ios::binary);\n'''
new_restore = '''bool VulkanGSTrainer::restoreTrainingCheckpoint(VulkanGSPipelineBuffers& buffers) {\n    if (training_checkpoint_path.empty())\n        return false;\n    if (!std::filesystem::exists(training_checkpoint_path)) {\n        const std::filesystem::path hint_path =\n            std::filesystem::path(training_checkpoint_path).parent_path() / "legacy_resume_step.txt";\n        if (!std::filesystem::exists(hint_path))\n            return false;\n        std::ifstream hint(hint_path);\n        uint64_t total_step = 0;\n        hint >> total_step;\n        hint >> std::ws;\n        if (!hint || !hint.eof() || total_step == 0 || total_step > UINT32_MAX)\n            throw std::runtime_error("invalid legacy 3DGS resume step hint");\n        if (legacy_training_ply_path.empty())\n            throw std::runtime_error("legacy 3DGS PLY path is unavailable");\n        return importLegacyTrainingPly(legacy_training_ply_path,\n                                       static_cast<uint32_t>(total_step), buffers);\n    }\n\n    std::ifstream in(training_checkpoint_path, std::ios::binary);\n'''
if old_restore not in source:
    raise SystemExit("legacy resume hint restore anchor not found")
source = source.replace(old_restore, new_restore, 1)

old_publish = '''    if (rename_error) {\n        std::filesystem::remove(temp_path);\n        throw std::runtime_error("failed to publish 3DGS training checkpoint: " + rename_error.message());\n    }\n    printf("Saved PCS 3DGS checkpoint: %u total steps, %u optimizer steps, %zu Gaussians\\n",\n'''
new_publish = '''    if (rename_error) {\n        std::filesystem::remove(temp_path);\n        throw std::runtime_error("failed to publish 3DGS training checkpoint: " + rename_error.message());\n    }\n    std::error_code hint_remove_error;\n    const std::filesystem::path hint_path =\n        std::filesystem::path(training_checkpoint_path).parent_path() / "legacy_resume_step.txt";\n    std::filesystem::remove(hint_path, hint_remove_error);\n    printf("Saved PCS 3DGS checkpoint: %u total steps, %u optimizer steps, %zu Gaussians\\n",\n'''
if old_publish not in source:
    raise SystemExit("legacy resume hint checkpoint publish anchor not found")
source = source.replace(old_publish, new_publish, 1)

source_path.write_text(source)
print("Patched VkSplat legacy PLY continuation hint bridge")
