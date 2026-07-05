"""Tests for the retained fused CUDA codegen paths."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from pysph.sph.fused_cuda_codegen import (
    CudaPairPrecompute,
    _force_cuda_source_fp32,
    generate_hbucket_pair_stage_outline_from_equations,
    generate_pointwise_kernel_outline_with_equation_calls,
    generate_snapshot_hbucket_pair_window_outline_from_equations,
    hbucket_context_argument_declarations,
    PairLaunchConfig,
    snapshot_hbucket_pair_window_stage,
)
from pysph.sph.equation import CUDAGroup, KnownType
from pysph.sph.fused_cuda_stage_backend import (
    _neighbor_context_bounds_and_periodicity,
    _pair_traversal_for_stage,
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


def test_hbucket_pair_outline_keeps_variable_h_bucket_traversal():
    deps = _sum_reduction_deps("AddMass", "au")
    stage = _stage(StageKind.PAIR_RATE, (deps,))
    equation = AddMass(dest="fluid", sources=["fluid"])

    outline = generate_hbucket_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (equation,),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
    )

    for declaration in hbucket_context_argument_declarations():
        assert declaration in outline.source
    assert "for (int bucket = 0; bucket < bucket_count; ++bucket)" in outline.source
    assert "int dst = sorted_ids[fused_dst_linear];" in outline.source
    assert "cell_bucket_h_max_bits[flat]" in outline.source
    assert "cell_bucket_starts[flat]" in outline.source
    assert "float fused_acc_d_au = 0.0f;" in outline.source
    assert "AddMass_loop(add_mass0, dst, src, fused_local_d_au, s_m);" in outline.source
    assert "d_au[dst] += fused_acc_d_au;" in outline.source


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

    combined = snapshot_hbucket_pair_window_stage((left, right))

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

    outline = generate_snapshot_hbucket_pair_window_outline_from_equations(
        "plan0",
        (left, right),
        ((left_equation,), (right_equation,)),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        ("au",),
    )

    assert outline.name == "fused_plan0_fluid_snapshot_hbucket_pair_window"
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


def test_pair_traversal_uses_hbucket_context():
    assert (
        _pair_traversal_for_stage(SimpleNamespace(cell_bucket_counts=object()))
        == "hbucket"
    )


def test_backend_records_pair_launch_config_counts():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.pair_launch_config_counts = {}

    backend._record_pair_launch_config(
        PairLaunchConfig(traversal="hbucket", n=1000, block_size=256, grid_x=4)
    )
    backend._record_pair_launch_config(
        PairLaunchConfig(traversal="hbucket", n=1000, block_size=256, grid_x=4)
    )

    assert backend.pair_launch_config_counts == {
        ("hbucket", 1000, 256, 4): 2,
    }


def test_neighbor_context_uses_domain_bounds_for_minimum_image_periodic():
    manager = SimpleNamespace(
        xmin=-1.0,
        xmax=1.0,
        ymin=-0.5,
        ymax=0.5,
        zmin=-0.125,
        zmax=0.125,
        minimum_image_periodic=True,
        periodic_in_x=True,
        periodic_in_y=True,
        periodic_in_z=True,
    )
    nnps = SimpleNamespace(
        xmin=np.array([-0.996, -0.496, -0.121], dtype=np.float64),
        xmax=np.array([0.996, 0.496, 0.121], dtype=np.float64),
        domain=SimpleNamespace(manager=manager),
    )

    lower, upper, periodic = _neighbor_context_bounds_and_periodicity(nnps)

    np.testing.assert_allclose(lower, np.array([-1.0, -0.5, -0.125], dtype=np.float32))
    np.testing.assert_allclose(upper, np.array([1.0, 0.5, 0.125], dtype=np.float32))
    np.testing.assert_array_equal(
        periodic, np.array([True, True, True], dtype=np.bool_)
    )


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
