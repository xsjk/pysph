"""Tests for the retained fused CUDA codegen paths."""

from dataclasses import replace

import numpy as np
import pytest

from pysph.sph.fused_cuda_codegen import (
    CudaPairPrecompute,
    _force_cuda_source_fp32,
    generate_csr_pair_stage_outline_from_equations,
    generate_pointwise_kernel_outline_with_equation_calls,
    generate_snapshot_csr_pair_window_outline_from_equations,
    csr_context_argument_declarations,
    CudaLaunchConfig,
    launch_cuda_kernel,
    snapshot_csr_pair_window_stage,
)
from pysph.sph.equation import CUDAGroup, KnownType
from pysph.sph.fused_cuda_stage_backend import (
    _snapshot_pair_window_fields,
    _snapshot_pair_windows,
)
from pysph.sph.fused_cuda_stage_plan import (
    analyze_equation_method,
    MethodDeps,
    MethodKind,
    StageKind,
    StageNode,
)
from pysph.sph.tests.fused_cuda_codegen_equations import (
    AddMass,
    CopyAcceleration,
    InitLoopPost,
    ReadDestAndSourceAcceleration,
)


def _deps(equation_name, method_kind):
    return MethodDeps(
        equation_name=equation_name,
        method_kind=method_kind,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )


def _sum_reduction_deps(equation_name, field):
    return replace(
        _deps(equation_name, MethodKind.LOOP),
        dest_reads=frozenset((field,)),
        dest_writes=frozenset((field,)),
        dest_reduction_writes=frozenset((field,)),
        dest_reduction_reads=frozenset((field,)),
    )


def _stage(kind, methods):
    return StageNode(
        kind=kind,
        dest="fluid",
        sources=("fluid",),
        methods=methods,
        reason="test",
        convergence_policy=None,
    )


def test_pointwise_kernel_outline_can_reuse_pysph_cuda_equation_wrapper():
    equation = CopyAcceleration(dest="fluid", sources=None)
    group = CUDAGroup([equation])
    known_types = {
        "d_u": KnownType("GLOBAL_MEM float*"),
        "d_au": KnownType("GLOBAL_MEM float*"),
    }
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = (_call(equation, "loop", known_types, frozenset()),)

    outline = generate_pointwise_kernel_outline_with_equation_calls(
        "plan0",
        _stage(StageKind.POINTWISE, (_deps("CopyAcceleration", MethodKind.LOOP),)),
        wrapper_source,
        calls,
    )

    assert outline.name == "fused_plan0_fluid_pointwise"
    assert "WITHIN_KERNEL void CopyAcceleration_loop" in outline.source
    assert "int dst = blockIdx.x * blockDim.x + threadIdx.x;" in outline.source
    assert (
        "CopyAcceleration_loop(copy_acceleration0, dst, d_u, d_au);" in outline.source
    )
    assert "sorted_ids" not in outline.source


def test_fp32_wrapper_uses_fast_dim_power_helpers():
    source = "\n".join(
        (
            "double rho = d_m[d_idx] / pow((d_h[d_idx] / self->k), self->dim);",
            "double h = self->k * pow((d_m[d_idx] / d_rho[d_idx]), (1.0f / self->dim));",
        )
    )

    fp32_source = _force_cuda_source_fp32(source)

    assert "double" not in fp32_source
    assert "fused_codegen_pow_dim(d_h[d_idx] / self->k, self->dim)" in fp32_source
    assert (
        "fused_codegen_pow_inv_dim(d_m[d_idx] / d_rho[d_idx], self->dim)" in fp32_source
    )


def test_csr_pair_outline_keeps_variable_h_bucket_traversal():
    deps = _sum_reduction_deps("AddMass", "au")
    stage = _stage(StageKind.PAIR_RATE, (deps,))
    equation = AddMass(dest="fluid", sources=["fluid"])

    outline = generate_csr_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (equation,),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        {},
        False,
    )

    for declaration in csr_context_argument_declarations():
        assert declaration in outline.source
    assert "int dst = fused_dst_linear;" in outline.source
    assert "unsigned long long begin = start_idx[dst];" in outline.source
    assert "int src = neighbors[pos];" in outline.source
    assert "bucket_count" not in outline.source
    assert "float fused_acc_d_au = 0.0f;" in outline.source
    assert "AddMass_loop(add_mass0, dst, src, fused_local_d_au, s_m);" in outline.source
    assert "d_au[dst] += fused_acc_d_au;" in outline.source


def test_csr_pair_outline_uses_known_integer_array_type():
    deps = _sum_reduction_deps("AddMass", "au")
    stage = _stage(StageKind.PAIR_RATE, (deps,))
    equation = AddMass(dest="fluid", sources=["fluid"])

    outline = generate_csr_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (equation,),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        {"s_m": KnownType("GLOBAL_MEM long*")},
        False,
    )

    assert "GLOBAL_MEM long* s_m" in outline.source


def test_snapshot_pair_window_snapshots_left_read_before_right_write():
    left = _stage(
        StageKind.PAIR_RATE,
        _method_deps(ReadDestAndSourceAcceleration(dest="fluid", sources=["fluid"])),
    )
    right = _stage(
        StageKind.PAIR_RATE,
        _method_deps(InitLoopPost(dest="fluid", sources=["fluid"])),
    )

    assert _snapshot_pair_windows((left, right)) == ((0, 1),)
    assert _snapshot_pair_window_fields((left, right)) == ("au",)

    combined = snapshot_csr_pair_window_stage((left, right))

    assert [
        (method.equation_name, method.method_kind) for method in combined.methods
    ] == [
        ("ReadDestAndSourceAcceleration", MethodKind.INITIALIZE),
        ("InitLoopPost", MethodKind.INITIALIZE),
        ("ReadDestAndSourceAcceleration", MethodKind.LOOP),
        ("InitLoopPost", MethodKind.LOOP),
        ("ReadDestAndSourceAcceleration", MethodKind.POST_LOOP),
        ("InitLoopPost", MethodKind.POST_LOOP),
    ]


def test_snapshot_pair_window_outline_reads_snapshotted_left_arguments():
    left_equation = ReadDestAndSourceAcceleration(dest="fluid", sources=["fluid"])
    right_equation = InitLoopPost(dest="fluid", sources=["fluid"])
    left = _stage(StageKind.PAIR_RATE, _method_deps(left_equation))
    right = _stage(StageKind.PAIR_RATE, _method_deps(right_equation))

    outline = generate_snapshot_csr_pair_window_outline_from_equations(
        "plan0",
        (left, right),
        ((left_equation,), (right_equation,)),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        ("au",),
        {},
        False,
    )

    assert outline.name == "fused_plan0_fluid_pair_rate"
    assert "GLOBAL_MEM float* fused_snapshot_au" in outline.source
    assert (
        "ReadDestAndSourceAcceleration_loop("
        "read_dest_and_source_acceleration0, dst, src, fused_local_d_alpha, "
        "fused_snapshot_au, fused_snapshot_au);"
    ) in outline.source
    assert (
        "InitLoopPost_loop(init_loop_post0, dst, src, fused_local_d_au, s_m);"
        in outline.source
    )


def test_backend_records_pair_launch_config_counts():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.pair_launch_config_counts = {}

    backend._record_pair_launch_config(
        CudaLaunchConfig(traversal="csr", n=1000, block_size=256, grid_x=4)
    )
    backend._record_pair_launch_config(
        CudaLaunchConfig(traversal="csr", n=1000, block_size=256, grid_x=4)
    )

    assert backend.pair_launch_config_counts == {
        ("csr", 1000, 256, 4): 2,
    }


@pytest.mark.parametrize("size", [1, 255, 256, 257])
@pytest.mark.parametrize("block_size", [128, 256])
def test_launch_uses_configured_block_and_covers_tail(size, block_size):
    calls = []

    def kernel(*arguments, **options):
        calls.append((arguments, options))

    arguments = (np.int32(size), np.float32(2.0))
    config = launch_cuda_kernel(kernel, size, None, arguments, "pointwise", block_size)
    grid = (size + block_size - 1) // block_size
    assert config == CudaLaunchConfig("pointwise", size, block_size, grid)
    assert calls == [
        (
            arguments,
            {
                "block": (block_size, 1, 1),
                "grid": (grid, 1, 1),
                "stream": None,
            },
        )
    ]


def _call(equation, method_name, known_types, precomputed_symbols):
    from pysph.sph.fused_cuda_codegen import (
        cuda_equation_method_call_from_equation_with_precomputed,
    )

    return cuda_equation_method_call_from_equation_with_precomputed(
        equation, method_name, known_types, precomputed_symbols
    )


def _method_deps(equation):
    return tuple(
        analyze_equation_method(equation, method_name)
        for method_name in ("initialize", "loop", "post_loop")
    )


def test_pointwise_and_pair_outlines_emit_shared_table_helpers_once():
    from pysph.sph.fused_cuda_codegen import (
        generate_pointwise_stage_outline_from_equations,
    )
    from pysph.sph.tests.fused_cuda_codegen_equations import (
        TableLookup,
        SourceTableLookup,
    )

    pointwise = TableLookup(dest="fluid", sources=None)
    deps = analyze_equation_method(pointwise, "loop")
    stage = StageNode(
        kind=StageKind.POINTWISE,
        dest="fluid",
        sources=(),
        methods=(deps,),
        reason="test",
        convergence_policy=None,
    )
    known_types = {"d_index": KnownType("GLOBAL_MEM int*")}
    outline = generate_pointwise_stage_outline_from_equations(
        "table", stage, (pointwise,), known_types
    )
    assert outline.source.count("WITHIN_KERNEL float table_element(") == 1
    assert outline.source.count("WITHIN_KERNEL float table_lookup(") == 1
    assert (
        outline.source.index("WITHIN_KERNEL float table_element(")
        < outline.source.index("WITHIN_KERNEL float table_lookup(")
        < outline.source.index("WITHIN_KERNEL void TableLookup_loop(")
    )
    pair = SourceTableLookup(dest="fluid", sources=["fluid"])
    deps = analyze_equation_method(pair, "loop")
    stage = _stage(StageKind.PAIR_RATE, (deps,))
    outline = generate_csr_pair_stage_outline_from_equations(
        "table",
        stage,
        (pair,),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        {"s_index": KnownType("GLOBAL_MEM int*")},
        False,
    )
    assert outline.source.count("WITHIN_KERNEL float table_lookup(") == 1
