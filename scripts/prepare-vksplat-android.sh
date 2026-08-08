#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPP_DIR="$ROOT/app/src/main/cpp"
ASSET_DIR="$ROOT/app/src/main/assets/vksplat_shader"
THIRD="$CPP_DIR/third_party"
VKSPLAT_COMMIT="41cff93b79145dec314488d4313bc3a6d737038b"
SLANG_VERSION="2026.14.1"

rm -rf "$THIRD/vksplat" "$THIRD/glm" "$ASSET_DIR"
mkdir -p "$THIRD" "$ASSET_DIR"

git clone -q https://github.com/harry7557558/vksplat.git "$THIRD/vksplat"
git -C "$THIRD/vksplat" checkout -q "$VKSPLAT_COMMIT"
git clone -q --depth 1 --branch 0.9.9.8 https://github.com/g-truc/glm.git "$THIRD/glm"

# Mobile GPUs do not consistently expose shaderInt64 or VK_EXT_shader_atomic_float.
# VkSplat already ships emulation paths; compile the packaged SPIR-V with both enabled.
python3 - "$THIRD/vksplat/vksplat/slang/config.slang" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace('#define USE_EMULATED_INT64 0', '#define USE_EMULATED_INT64 1')
s = s.replace('#define USE_EMULATED_F32_ATOMIC 0', '#define USE_EMULATED_F32_ATOMIC 1')
p.write_text(s)
PY

# The desktop source unconditionally enables optional device extensions even when the emulation
# shaders do not need them. Android must not request extensions that the phone does not expose.
python3 - "$THIRD/vksplat/vksplat/src/gs_pipeline.cpp" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = '''    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features = {};
    atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    atomic_float_features.shaderBufferFloat32AtomicAdd = VK_TRUE;
    atomic_float_features.pNext = VK_NULL_HANDLE;

    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupSizeControlFeatures = {};
    subgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    subgroupSizeControlFeatures.subgroupSizeControl = VK_TRUE;
    subgroupSizeControlFeatures.computeFullSubgroups = VK_TRUE;
    subgroupSizeControlFeatures.pNext = VK_NULL_HANDLE;
    if (deviceInfo.hasFloat32AtomicAdd)
        subgroupSizeControlFeatures.pNext = &atomic_float_features;

    VkDeviceCreateInfo create_info = {};
'''
new = '''    // Android build uses emulated Int64/F32 atomic shaders and the native subgroup size.
    // Do not require optional desktop extensions at vkCreateDevice time.
    VkDeviceCreateInfo create_info = {};
'''
if old not in s:
    raise SystemExit('VkSplat createDevice patch anchor not found')
s = s.replace(old, new)
s = s.replace('''    std::vector<const char*> device_extensions = {
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
    };
    create_info.enabledExtensionCount = (uint32_t)device_extensions.size();
    create_info.ppEnabledExtensionNames = device_extensions.data();
    create_info.pNext = &subgroupSizeControlFeatures;
''', '''    create_info.enabledExtensionCount = 0;
    create_info.ppEnabledExtensionNames = nullptr;
    create_info.pNext = nullptr;
''')
# Required subgroup-size pNext also depends on VK_EXT_subgroup_size_control. The Pixel path uses
# the physical device's native subgroup. If it is not 32, initialization reports it and fails
# cleanly rather than requesting an unsupported extension.
s = s.replace('''    if (compatible_subgroup_size && (
        deviceInfo.subgroupSize != SUBGROUP_SIZE ||
        deviceInfo.vendor == DeviceVendor::Intel_R_
    ))
        compute_shader_stage_info.pNext = &req;
''', '''    if (compatible_subgroup_size && deviceInfo.subgroupSize != SUBGROUP_SIZE)
        _THROW_ERROR_ALWAYS("Android 3DGS requires native subgroup size 32; device reports " + std::to_string(deviceInfo.subgroupSize));
''')
p.write_text(s)
PY

# Keep dataset coordinates in the app's saved root-anchor coordinate system. This is important for
# later AR alignment and lets progressive stages reuse a previous PLY without an unknown transform.
python3 - "$THIRD/vksplat/vksplat/src/gs_trainer.cpp" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace('dataparser_transform = ColmapReader::normalize_world_space(c2w_poses, points);',
              'dataparser_transform = ColmapReader::normalize_world_space(c2w_poses, points, false);')
p.write_text(s)
PY

# Turn VkSplat's guarded errors back into real exceptions so JNI can report failures rather than
# silently continuing with invalid Vulkan objects.
sed -i 's/#define ENABLE_ASSERTION 0/#define ENABLE_ASSERTION 1/' "$THIRD/vksplat/vksplat/src/config.h"

SLANG_ROOT="$RUNNER_TEMP/slang-$SLANG_VERSION"
if [[ ! -x "$SLANG_ROOT/bin/slangc" ]]; then
  rm -rf "$SLANG_ROOT"
  mkdir -p "$SLANG_ROOT"
  curl -L --retry 3 -o "$RUNNER_TEMP/slang.tar.gz" \
    "https://github.com/shader-slang/slang/releases/download/v$SLANG_VERSION/slang-$SLANG_VERSION-linux-x86_64-glibc-2.27.tar.gz"
  tar -xzf "$RUNNER_TEMP/slang.tar.gz" -C "$SLANG_ROOT" --strip-components=1
fi

GLSLC="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}/shader-tools/linux-x86_64/glslc"
if [[ ! -x "$GLSLC" ]]; then
  echo "glslc not found under Android NDK: $GLSLC" >&2
  exit 1
fi

pushd "$THIRD/vksplat" >/dev/null
python3 compile_shaders.py --force --slangc "$SLANG_ROOT/bin/slangc" --glslc "$GLSLC"
popd >/dev/null

cp -R "$THIRD/vksplat/vksplat/shader/." "$ASSET_DIR/"
printf '%s\n' "$VKSPLAT_COMMIT" > "$ASSET_DIR/VKSPLAT_COMMIT.txt"
printf '%s\n' "$SLANG_VERSION" > "$ASSET_DIR/SLANG_VERSION.txt"

echo "Prepared VkSplat $VKSPLAT_COMMIT for Android"
