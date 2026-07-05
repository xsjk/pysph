"""Tests for the retained fused CUDA codegen paths."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from pysph.sph.fused_cuda_codegen import (
    CudaPairPrecompute,
    fused_kernel_specs,
    generate_fused_kernel_outline,
    generate_hbucket_pair_stage_outline_from_equations,
    generate_pointwise_kernel_outline_with_equation_calls,
    generate_resident_hbucket_pair_window_outline_from_equations,
    hbucket_context_argument_declarations,
    launch_budget_for_specs,
    PairLaunchConfig,
)
from pysph.sph.equation import CUDAGroup, KnownType
from pysph.sph.fused_cuda_stage_backend import (
    _neighbor_context_bounds_and_periodicity,
    _pair_traversal_for_stage,
)
from pysph.sph.fused_cuda_stage_plan import (
    CudaStagePlan,
    MethodDeps,
    MethodKind,
    StageKind,
    StageNode,
)
from pysph.sph.tests.fused_cuda_codegen_equations import (
    AddMass,
    CopyAcceleration,
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


def test_common_rhs_specs_insert_one_neighbor_build_and_four_core_kernels():
    plan = CudaStagePlan(
        stages=(
            _stage(StageKind.PAIR_DENSITY, (_deps("Density", MethodKind.LOOP),)),
            _stage(StageKind.POINTWISE, (_deps("EOS", MethodKind.LOOP),)),
            _stage(StageKind.PAIR_RATE, (_deps("Rate", MethodKind.LOOP),)),
            _stage(StageKind.REDUCTION, (_deps("Timestep", MethodKind.INITIALIZE),)),
        ),
        strict=True,
    )

    specs = fused_kernel_specs("plan0", plan)
    budget = launch_budget_for_specs(specs)

    assert [spec.name for spec in specs] == [
        "fused_plan0_fluid_neighbor_build",
        "fused_plan0_fluid_pair_density",
        "fused_plan0_fluid_pointwise",
        "fused_plan0_fluid_pair_rate",
        "fused_plan0_fluid_reduction",
    ]
    assert [spec.uses_neighbors for spec in specs] == [False, True, False, True, False]
    assert budget.neighbor_build_count == 1
    assert budget.rhs_core_kernel_count == 4
    assert budget.total_launch_count == 5


def test_kernel_outline_preserves_equation_method_order():
    stage = _stage(
        StageKind.POINTWISE,
        (
            _deps("Alpha", MethodKind.INITIALIZE),
            _deps("Beta", MethodKind.LOOP),
            _deps("Gamma", MethodKind.POST_LOOP),
        ),
    )

    outline = generate_fused_kernel_outline("plan0", stage)

    assert outline.name == "fused_plan0_fluid_pointwise"
    alpha = outline.source.index("Alpha.initialize")
    beta = outline.source.index("Beta.loop")
    gamma = outline.source.index("Gamma.post_loop")
    assert alpha < beta < gamma


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
    assert "cell_bucket_h_max_bits[flat]" in outline.source
    assert "cell_bucket_starts[flat]" in outline.source
    assert "AddMass_loop(add_mass0, dst, src, d_au, s_m);" in outline.source


def test_resident_hbucket_pair_window_uses_cooperative_grid_sync():
    deps = _sum_reduction_deps("AddMass", "au")
    stage0 = _stage(StageKind.PAIR_DENSITY, (deps,))
    stage1 = _stage(StageKind.PAIR_RATE, (deps,))
    equation = AddMass(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())

    outline = generate_resident_hbucket_pair_window_outline_from_equations(
        "plan0",
        (stage0, stage1),
        ((equation,), (equation,)),
        (precompute, precompute),
    )

    assert outline.name == "fused_plan0_fluid_resident_hbucket_pair_window"
    assert "#include <cooperative_groups.h>" in outline.source
    assert "cooperative_groups::this_grid()" in outline.source
    assert "grid.sync();" in outline.source
    assert outline.source.count("AddMass_loop(add_mass0, dst, src, d_au, s_m);") == 2


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
