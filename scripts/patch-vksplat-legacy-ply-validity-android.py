#!/usr/bin/env python3
from pathlib import Path

path = Path("app/src/main/cpp/third_party/vksplat/vksplat/src/gs_trainer.cpp")
s = path.read_text()

old_alloc = '''    const size_t n = static_cast<size_t>(vertex_count);\n    Buffer<float> xyz;\n    Buffer<float> sh;\n    Buffer<float> rotations;\n    Buffer<float> scales_opacs;\n    xyz.resize(3 * n);\n    sh.resize(48 * n);\n    rotations.resize(4 * n);\n    scales_opacs.resize(4 * n);\n'''
new_alloc = '''    const size_t source_n = static_cast<size_t>(vertex_count);\n    Buffer<float> xyz;\n    Buffer<float> sh;\n    Buffer<float> rotations;\n    Buffer<float> scales_opacs;\n    xyz.reserve(3 * source_n);\n    sh.reserve(48 * source_n);\n    rotations.reserve(4 * source_n);\n    scales_opacs.reserve(4 * source_n);\n'''
if old_alloc not in s:
    raise SystemExit("legacy PLY compact-allocation anchor not found")
s = s.replace(old_alloc, new_alloc, 1)

old_loop = '''    std::array<float, 59> record{};\n    for (size_t i = 0; i < n; ++i) {\n        pcsReadExact(in, record.data(), record.size() * sizeof(float), "legacy PLY vertex");\n        for (float value : record) {\n            if (!std::isfinite(value))\n                throw std::runtime_error("legacy 3DGS PLY contains non-finite values");\n        }\n        xyz[3*i] = record[0];\n        xyz[3*i+1] = record[1];\n        xyz[3*i+2] = record[2];\n        for (int c = 0; c < 48; ++c)\n            sh[48*i+c] = record[3 + sh_permute[c]];\n        const float opacity_logit = std::clamp(record[51], -30.0f, 30.0f);\n        scales_opacs[4*i] = std::exp(std::clamp(record[52], -30.0f, 30.0f));\n        scales_opacs[4*i+1] = std::exp(std::clamp(record[53], -30.0f, 30.0f));\n        scales_opacs[4*i+2] = std::exp(std::clamp(record[54], -30.0f, 30.0f));\n        scales_opacs[4*i+3] = 1.0f / (1.0f + std::exp(-opacity_logit));\n        for (int c = 0; c < 4; ++c)\n            rotations[4*i+c] = record[55+c];\n    }\n\n    buffers.xyz_ws.assign(xyz.begin(), xyz.end());\n    buffers.sh_coeffs.assign(sh.begin(), sh.end());\n    buffers.rotations.assign(rotations.begin(), rotations.end());\n    buffers.scales_opacs.assign(scales_opacs.begin(), scales_opacs.end());\n    buffers.num_splats = n;\n    buffers.reorderSH(buffers.sh_coeffs);\n\n    const size_t sh_n = 16 * 3 * _CEIL_ROUND(n, SH_REORDER_SIZE);\n    buffers.g_xyz_ws.assign(2 * 3 * n, 0.0f);\n    buffers.g_sh_coeffs_1.assign(sh_n, 0.0f);\n    buffers.g_sh_coeffs_2.assign(sh_n, 0.0f);\n    buffers.g_rotations.assign(2 * 4 * n, 0.0f);\n    buffers.g_scales_opacs.assign(2 * 4 * n, 0.0f);\n'''
new_loop = r'''    std::array<float, 59> record{};
    size_t dropped_nonfinite_position = 0;
    size_t sanitized_sh = 0;
    size_t sanitized_scale = 0;
    size_t sanitized_opacity = 0;
    size_t sanitized_rotation = 0;

    auto finiteOrZero = [](float value, size_t& sanitized) {
        if (std::isfinite(value)) return value;
        ++sanitized;
        return 0.0f;
    };
    auto positiveScale = [](float log_scale, size_t& sanitized) {
        if (!std::isfinite(log_scale)) {
            ++sanitized;
            return 0.004f;
        }
        const float value = std::exp(std::clamp(log_scale, -20.0f, 2.0f));
        if (!std::isfinite(value) || !(value > 0.0f)) {
            ++sanitized;
            return 0.004f;
        }
        return value;
    };
    auto opacityFromLogit = [](float logit, size_t& sanitized) {
        if (!std::isfinite(logit)) {
            ++sanitized;
            return 0.5f;
        }
        if (logit >= 0.0f) {
            const float z = std::exp(-std::min(logit, 30.0f));
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(std::max(logit, -30.0f));
        return z / (1.0f + z);
    };

    for (size_t i = 0; i < source_n; ++i) {
        pcsReadExact(in, record.data(), record.size() * sizeof(float), "legacy PLY vertex");

        // This is the same validity boundary as GaussianPlyModelReader: a Gaussian with an
        // invalid world position cannot participate in rendering or optimization, so drop only
        // that record. Non-positional trainable channels are recoverable and are normalized below.
        if (!std::isfinite(record[0]) || !std::isfinite(record[1]) || !std::isfinite(record[2])) {
            ++dropped_nonfinite_position;
            continue;
        }

        xyz.push_back(record[0]);
        xyz.push_back(record[1]);
        xyz.push_back(record[2]);

        for (int c = 0; c < 48; ++c) {
            float value = record[3 + sh_permute[c]];
            sh.push_back(finiteOrZero(value, sanitized_sh));
        }

        scales_opacs.push_back(positiveScale(record[52], sanitized_scale));
        scales_opacs.push_back(positiveScale(record[53], sanitized_scale));
        scales_opacs.push_back(positiveScale(record[54], sanitized_scale));
        scales_opacs.push_back(opacityFromLogit(record[51], sanitized_opacity));

        float qw = finiteOrZero(record[55], sanitized_rotation);
        float qx = finiteOrZero(record[56], sanitized_rotation);
        float qy = finiteOrZero(record[57], sanitized_rotation);
        float qz = finiteOrZero(record[58], sanitized_rotation);
        const float q_len = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        if (!(q_len > 1e-8f) || !std::isfinite(q_len)) {
            ++sanitized_rotation;
            qw = 1.0f;
            qx = qy = qz = 0.0f;
        } else {
            const float inv = 1.0f / q_len;
            qw *= inv;
            qx *= inv;
            qy *= inv;
            qz *= inv;
        }
        rotations.push_back(qw);
        rotations.push_back(qx);
        rotations.push_back(qy);
        rotations.push_back(qz);
    }

    const size_t n = xyz.size() / 3;
    if (n < 64)
        throw std::runtime_error("legacy 3DGS PLY has too few valid Gaussian positions");
    if (sh.size() != 48*n || rotations.size() != 4*n || scales_opacs.size() != 4*n)
        throw std::runtime_error("legacy 3DGS PLY compaction produced inconsistent buffer lengths");

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
'''
if old_loop not in s:
    raise SystemExit("legacy PLY validity-policy loop anchor not found")
s = s.replace(old_loop, new_loop, 1)

old_log = '''    printf("Imported legacy PCS 3DGS PLY: %u previous steps, Adam reset, %zu Gaussians\\n",\n           cumulativeStep, n);\n'''
new_log = '''    printf("Imported legacy PCS 3DGS PLY: %u previous steps, Adam reset, %zu/%zu Gaussians "\n           "droppedPosition=%zu sanitizedSH=%zu sanitizedScale=%zu sanitizedOpacity=%zu "\n           "sanitizedRotation=%zu\\n",\n           cumulativeStep, n, source_n, dropped_nonfinite_position, sanitized_sh, sanitized_scale,\n           sanitized_opacity, sanitized_rotation);\n'''
if old_log not in s:
    raise SystemExit("legacy PLY validity diagnostics anchor not found")
s = s.replace(old_log, new_log, 1)

path.write_text(s)
print("Patched VkSplat legacy PLY validity to match the viewer contract")
