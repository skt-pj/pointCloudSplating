#!/usr/bin/env python3
from pathlib import Path

root = Path("app/src/main/cpp/third_party/vksplat/vksplat")
spine_path = root / "shader/radix_sort/spine.comp"
header_path = root / "src/gs_renderer.h"
renderer_path = root / "src/gs_renderer.cpp"
compiler_path = root.parent / "compile_shaders.py"

# Reuse the already mapped radix_sort/spine SPIR-V for both stages. A one-workgroup dispatch means
# "prefix the 256 global totals"; the normal 256-workgroup dispatch means "scan one radix bin over
# all partitions and publish that bin's total". This avoids adding a new shader-path key to PCS.
spine = spine_path.read_text()
main_anchor = '''  uint elementCount = uniforms.elementCount;
  int pass = int(uniforms.pass);

  uint partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;
'''
main_replacement = '''  uint elementCount = uniforms.elementCount;
  int pass = int(uniforms.pass);

  // Second stage: one workgroup performs an exclusive scan over the 256 radix totals written by
  // the first spine dispatch. This is deliberately a separate dispatch so Vulkan barriers provide
  // inter-workgroup visibility without any contended global atomics.
  if (gl_NumWorkGroups.x == 1) {
    uint32_t value = index < RADIX ? globalHistogram.data[RADIX * pass + index] : 0;
    uint32_t excl = subgroupExclusiveAdd(value);
    uint32_t sum = subgroupAdd(value);
    if (subgroupElect()) {
      intermediate[subgroupIndex] = sum;
    }
    barrier();

    if (index < gl_NumSubgroups) {
      intermediate[index] = subgroupExclusiveAdd(intermediate[index]);
    }
    barrier();

    if (index < RADIX) {
      globalHistogram.data[RADIX * pass + index] = excl + intermediate[subgroupIndex];
    }
    return;
  }

  uint partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;
'''
if main_anchor not in spine:
    raise SystemExit("two-stage spine main anchor not found")
spine = spine.replace(main_anchor, main_replacement, 1)
spine_path.write_text(spine)

header = header_path.read_text()
with_prefix = '''    struct _RadixSortComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(3);
        _ComputePipeline spine = _ComputePipeline(2);
        _ComputePipeline prefix = _ComputePipeline(1);
        _ComputePipeline downsweep = _ComputePipeline(6);
    } pipeline_sorting_1, pipeline_sorting_2;
'''
without_prefix = '''    struct _RadixSortComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(3);
        _ComputePipeline spine = _ComputePipeline(2);
        _ComputePipeline downsweep = _ComputePipeline(6);
    } pipeline_sorting_1, pipeline_sorting_2;
'''
if with_prefix not in header:
    raise SystemExit("two-stage renderer header anchor not found")
header = header.replace(with_prefix, without_prefix, 1)
header_path.write_text(header)

renderer = renderer_path.read_text()
renderer = renderer.replace(
    '    createComputePipeline(pipeline_sorting_1.prefix, spirv_paths.at("radix_sort/prefix"));\n',
    '', 1)
renderer = renderer.replace(
    '    createComputePipeline(pipeline_sorting_2.prefix, spirv_paths.at("radix_sort/prefix"));\n',
    '', 1)

prefix_call = '''        executeCompute(
            {{1, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.prefix,
            {
                globalHistogram.deviceBuffer,
            }
        );
'''
spine_call = '''        executeCompute(
            {{1, 1}},
            uniforms, 2*sizeof(int32_t),
            pipeline_sorting.spine,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
            }
        );
'''
if prefix_call not in renderer:
    raise SystemExit("two-stage renderer prefix-dispatch anchor not found")
renderer = renderer.replace(prefix_call, spine_call, 1)
renderer_path.write_text(renderer)

# prefix.comp was created by the first Android radix patch while the design was being split. It is
# no longer needed because the existing spine shader now handles both stages, so do not compile or
# package an unused shader.
compiler = compiler_path.read_text()
prefix_job = '            ShaderJob("prefix.comp", {}),\n'
if prefix_job not in compiler:
    raise SystemExit("two-stage compiler prefix job anchor not found")
compiler = compiler.replace(prefix_job, '', 1)
compiler_path.write_text(compiler)

print("Folded Android radix global-prefix stage into a second synchronized spine dispatch")
