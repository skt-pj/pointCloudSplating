#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CPP_DIR="$ROOT/app/src/main/cpp"
ASSET_DIR="$ROOT/app/src/main/assets/vksplat_shader"
NOTICE_DIR="$ROOT/app/src/main/assets/third_party"
THIRD="$CPP_DIR/third_party"
VKSPLAT_COMMIT="41cff93b79145dec314488d4313bc3a6d737038b"
SLANG_VERSION="2026.14.1"

rm -rf "$THIRD/vksplat" "$THIRD/glm" "$ASSET_DIR" "$NOTICE_DIR"
mkdir -p "$THIRD" "$ASSET_DIR" "$NOTICE_DIR"

git clone -q https://github.com/harry7557558/vksplat.git "$THIRD/vksplat"
git -C "$THIRD/vksplat" checkout -q "$VKSPLAT_COMMIT"
git clone -q --depth 1 --branch 0.9.9.8 https://github.com/g-truc/glm.git "$THIRD/glm"

python3 - "$THIRD/vksplat/vksplat/slang/config.slang" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace('#define SUBGROUP_SIZE 32', '#define SUBGROUP_SIZE 16')
s = s.replace('#define USE_EMULATED_INT64 0', '#define USE_EMULATED_INT64 1')
s = s.replace('#define USE_EMULATED_F32_ATOMIC 0', '#define USE_EMULATED_F32_ATOMIC 1')
p.write_text(s)
PY

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
new = '''    // Android uses emulation shaders; optional desktop Vulkan extensions are not required.
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
s = s.replace('''    if (compatible_subgroup_size && (
        deviceInfo.subgroupSize != SUBGROUP_SIZE ||
        deviceInfo.vendor == DeviceVendor::Intel_R_
    ))
        compute_shader_stage_info.pNext = &req;
''', '''    if (compatible_subgroup_size && deviceInfo.subgroupSize != SUBGROUP_SIZE)
        _THROW_ERROR_ALWAYS("Pixel 10a 3DGS expects Mali native subgroup size 16; device reports " + std::to_string(deviceInfo.subgroupSize));
''')

props_call = '        vkGetPhysicalDeviceProperties2(device, &deviceProperties2);\n'
props_dyn = '''        auto getProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
        if (getProperties2) {
            getProperties2(device, &deviceProperties2);
        } else {
            auto getProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(
                vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2KHR"));
            if (!getProperties2KHR)
                _THROW_ERROR_ALWAYS("Vulkan physical-device properties2 query is unavailable");
            getProperties2KHR(device, reinterpret_cast<VkPhysicalDeviceProperties2KHR*>(&deviceProperties2));
        }
'''
if props_call not in s:
    raise SystemExit('VkSplat properties2 call anchor not found')
s = s.replace(props_call, props_dyn)
features_call = '        vkGetPhysicalDeviceFeatures2(device, &deviceFeatures2);\n'
features_dyn = '''        auto getFeatures2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
        if (getFeatures2) {
            getFeatures2(device, &deviceFeatures2);
        } else {
            auto getFeatures2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
                vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
            if (!getFeatures2KHR)
                _THROW_ERROR_ALWAYS("Vulkan physical-device features2 query is unavailable");
            getFeatures2KHR(device, reinterpret_cast<VkPhysicalDeviceFeatures2KHR*>(&deviceFeatures2));
        }
'''
if features_call not in s:
    raise SystemExit('VkSplat features2 call anchor not found')
s = s.replace(features_call, features_dyn)
p.write_text(s)
PY

python3 - "$THIRD/vksplat/vksplat/src/gs_trainer.cpp" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace('dataparser_transform = ColmapReader::normalize_world_space(c2w_poses, points);',
              'dataparser_transform = ColmapReader::normalize_world_space(c2w_poses, points, false);')
# PCS writes intrinsics for the exact resized training JPEG dimensions. Disable VkSplat's
# Mip-NeRF-specific images_2/images_4 directory-name rescaling; otherwise the already-scaled
# Camera2 intrinsics are scaled twice before the image-dimension correction path runs.
needle = '''      #if 1
        // to be consistent with gsplat, especially on Mip-NeRF 360; TODO: refactor instead of hard code
'''
replacement = '''      #if 0
        // Disabled on Android PCS datasets: cameras.txt already matches the actual working JPEGs.
'''
if needle not in s:
    raise SystemExit('VkSplat image factor patch anchor not found')
s = s.replace(needle, replacement, 1)
# Desktop VkSplat loads images using all CPU threads. On a phone this creates a large transient
# RGBA decode spike exactly when Vulkan allocates its training buffers. Keep Android image loading
# bounded to two workers; this changes only preprocessing concurrency, not training resolution or loss.
threads = '    const unsigned int numThreads = std::thread::hardware_concurrency();\n'
threads_android = '''    const unsigned int numThreads = std::min(
        2u, std::max(1u, std::thread::hardware_concurrency()));
'''
if threads not in s:
    raise SystemExit('VkSplat image loader thread-count patch anchor not found')
s = s.replace(threads, threads_android, 1)
p.write_text(s)
PY

sed -i 's/#define ENABLE_ASSERTION 0/#define ENABLE_ASSERTION 1/' "$THIRD/vksplat/vksplat/src/config.h"

python3 - "$THIRD/vksplat/compile_shaders.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
radix_block = '''        # Radix sort
        jobs.append(("radix_sort", [
            ShaderJob("upsweep.comp", {}),
            ShaderJob("spine.comp", {}),
            ShaderJob("downsweep.comp", {}),
        ], []))

'''
if radix_block not in s:
    raise SystemExit('compile_shaders radix block not found')
s = s.replace(radix_block, '')

# PCS owns its Android cumsum shader as Vulkan GLSL and compiles it with the NDK reference path.
# Do not let Slang produce another SPIR-V module with the same output names.
cumsum_block = '''        # Prefix Sum
        jobs.append(("cumsum.slang", [
            ShaderJob(
                f"cumsum_{phase_name}",
                {"CUMSUM_PHASE": phase_id}
            ) for phase_name, phase_id in [
                ("block_scan", 1), ("scan_block_sums", 2),
                ("add_block_offsets", 3), ("single_pass", 0)
            ]
        ], []))

'''
if cumsum_block not in s:
    raise SystemExit('compile_shaders cumsum block not found')
s = s.replace(cumsum_block, '')

warning_block = '''            if output.stdout != "" or output.stderr != "":
                return False, f"O Compiled {job_name} with warning: {output.stdout} {output.stderr}"
            return True, f"✓ Compiled {job_name}"
'''
warning_fixed = '''            if output.stdout != "" or output.stderr != "":
                return True, f"O Compiled {job_name} with warning: {output.stdout} {output.stderr}"
            return True, f"✓ Compiled {job_name}"
'''
if warning_block not in s:
    raise SystemExit('compile_shaders warning handling anchor not found')
s = s.replace(warning_block, warning_fixed)
p.write_text(s)
PY

# Patch renderer hierarchy and add per-pipeline Android diagnostics before native compilation.
pushd "$ROOT" >/dev/null
python3 scripts/patch-vksplat-cumsum-android.py
python3 scripts/patch-vksplat-pipeline-diagnostics-android.py
popd >/dev/null

SLANG_ROOT="$RUNNER_TEMP/slang-$SLANG_VERSION"
rm -rf "$SLANG_ROOT"
mkdir -p "$SLANG_ROOT"
curl -L --retry 3 -o "$RUNNER_TEMP/slang.tar.gz" \
  "https://github.com/shader-slang/slang/releases/download/v$SLANG_VERSION/slang-$SLANG_VERSION-linux-x86_64-glibc-2.27.tar.gz"
tar -xzf "$RUNNER_TEMP/slang.tar.gz" -C "$SLANG_ROOT"
SLANGC="$(find "$SLANG_ROOT" -type f -name slangc -perm -u+x | head -n 1 || true)"
if [[ -z "$SLANGC" ]]; then
  SLANGC="$(find "$SLANG_ROOT" -type f -name slangc | head -n 1 || true)"
fi
if [[ -z "$SLANGC" ]]; then
  echo "slangc not found after extracting Slang $SLANG_VERSION" >&2
  exit 1
fi
chmod +x "$SLANGC"
echo "Using slangc: $SLANGC"

GLSLC="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}/shader-tools/linux-x86_64/glslc"
SHADER_TOOLS="$(dirname "$GLSLC")"
SPIRV_VAL="$SHADER_TOOLS/spirv-val"
SPIRV_DIS="$SHADER_TOOLS/spirv-dis"
for tool in "$GLSLC" "$SPIRV_VAL" "$SPIRV_DIS"; do
  if [[ ! -x "$tool" ]]; then
    echo "required Android NDK shader tool not found: $tool" >&2
    exit 1
  fi
done

pushd "$THIRD/vksplat" >/dev/null
python3 compile_shaders.py --slangc "$SLANGC" --glslc "$GLSLC"
popd >/dev/null

# Compile the subgroup-independent Android cumsum through the NDK's Vulkan GLSL reference path.
CUMSUM_SRC="$ROOT/scripts/vksplat-android-shaders/cumsum.comp"
CUMSUM_DST="$THIRD/vksplat/vksplat/shader/generated"
mkdir -p "$CUMSUM_DST"
declare -a CUMSUM_PHASES=(
  "0:cumsum_single_pass"
  "1:cumsum_block_scan"
  "2:cumsum_scan_block_sums"
  "3:cumsum_add_block_offsets"
)
for item in "${CUMSUM_PHASES[@]}"; do
  phase="${item%%:*}"
  name="${item#*:}"
  out="$CUMSUM_DST/$name.spv"
  "$GLSLC" -fshader-stage=compute -O --target-env=vulkan1.2 --target-spv=spv1.5 \
    -DPCS_CUMSUM_PHASE="$phase" "$CUMSUM_SRC" -o "$out"
  "$SPIRV_VAL" --target-env vulkan1.2 "$out"
  dis="$RUNNER_TEMP/$name.spvasm"
  "$SPIRV_DIS" "$out" -o "$dis"
  if ! grep -Eq 'OpExecutionMode .* LocalSize 256 1 1' "$dis"; then
    echo "$name does not declare LocalSize 256 1 1" >&2
    exit 1
  fi
  if [[ "$phase" != "3" ]] && ! grep -q 'Workgroup' "$dis"; then
    echo "$name is missing Workgroup storage" >&2
    exit 1
  fi
  sha256sum "$out"
done

echo "Validated Android cumsum SPIR-V with NDK spirv-val"

cp -R "$THIRD/vksplat/vksplat/shader/." "$ASSET_DIR/"
printf '%s\n' "$VKSPLAT_COMMIT" > "$ASSET_DIR/VKSPLAT_COMMIT.txt"
printf '%s\n' "$SLANG_VERSION" > "$ASSET_DIR/SLANG_VERSION.txt"

cp "$THIRD/vksplat/LICENSE" "$NOTICE_DIR/VkSplat-LICENSE.txt"
printf 'VkSplat source commit: %s\nhttps://github.com/harry7557558/vksplat\n' "$VKSPLAT_COMMIT" > "$NOTICE_DIR/VkSplat-NOTICE.txt"
if [[ -f "$THIRD/glm/copying.txt" ]]; then cp "$THIRD/glm/copying.txt" "$NOTICE_DIR/GLM-LICENSE.txt"; fi

echo "Prepared VkSplat $VKSPLAT_COMMIT for Android Mali subgroup 16 with glslc256 GPU cumsum"
