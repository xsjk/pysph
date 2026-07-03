"""Tests for generic fused CUDA kernel planning."""

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pysph.base.fused_cuda_nnps import (
    brute_force_neighbor_indices,
    build_fused_cuda_context_from_device_arrays,
    minimum_image_delta,
)
from pysph.sph.fused_cuda_codegen import (
    build_cuda_equation_struct_argument,
    CudaInlineMethodBody,
    CudaPairPrecompute,
    cubic_spline_gradient_precompute,
    cubic_spline_pair_precompute_for_symbols,
    cubic_spline_pair_precompute,
    cubic_spline_wij_precompute,
    cuda_equation_method_call_from_equation,
    cuda_equation_method_call_from_equation_with_precomputed,
    generate_direct_pair_stage_outline_from_equations,
    generate_direct_pair_stage_outline_from_equations_with_convergence_flag,
    generate_cluster_pair_stage_outline_from_equations,
    generate_hbucket_old_state_pair_window_outline_from_equations,
    generate_hbucket_old_state_pair_stage_outline_from_equations,
    hbucket_old_state_pair_window_argument_names,
    hbucket_old_state_pair_stage_argument_names,
    generate_hbucket_source_parallel_pair_stage_outline_from_equations,
    generate_source_parallel_pair_stage_outline_from_equations,
    cluster_context_argument_declarations,
    fused_context_argument_declarations,
    fused_kernel_specs,
    generate_direct_pair_loop_outline,
    generate_direct_pair_loop_outline_with_equation_calls_and_precompute,
    generate_direct_pair_loop_outline_with_equation_calls,
    generate_direct_pair_loop_outline_with_inline_bodies,
    generate_pointwise_stage_outline_from_equations,
    generate_pointwise_kernel_outline_with_equation_calls,
    generate_fused_kernel_outline,
    launch_budget_for_specs,
    launch_direct_pair_kernel_with_context,
    launch_hbucket_pair_kernel_with_context,
    launch_hbucket_source_parallel_pair_kernel_with_context,
    launch_source_parallel_pair_kernel_with_context,
    launch_pointwise_kernel,
    lower_equation_method_to_cuda,
    pair_traversal_mode,
    precompute_argument_names,
    _pair_cluster_size,
    _direct_pair_block_size,
    quintic_spline_gradient_precompute,
    quintic_spline_pair_precompute_for_symbols,
    quintic_spline_wij_precompute,
)
from pysph.sph.equation import CUDAGroup, KnownType
from pysph.sph.basic_equations import (
    IsothermalEOS,
    MonaghanArtificialViscosity,
    SummationDensity,
)
from pysph.sph.fused_cuda_stage_plan import (
    CudaStagePlan,
    DeviceConvergencePolicy,
    MethodDeps,
    MethodKind,
    StageKind,
    StageNode,
)
from pysph.sph.fused_cuda_stage_backend import (
    _launch_segments_for_stage,
    _neighbor_context_bounds_and_periodicity,
    _split_source_inline_pair_window_stages,
)
from pysph.sph.tests.fused_cuda_codegen_equations import (
    AddMass,
    AddScaledMass,
    AssignMassToAcceleration,
    CopyAcceleration,
    PreserveThenSetDensity,
    PrepPressure,
    ReadDestAndSourceAcceleration,
    ReadSourceAcceleration,
    SetDensity,
)
from pysph.sph.tests.fused_cuda_codegen_equations import (
    AccumulateDWIAndDWJ,
    AccumulateDWIJAndMaxSignal,
    AccumulateDWIJ,
    AccumulateGradientH,
    AccumulateRhoIJ,
    DensityConvergenceFlag,
    InitLoopPost,
    InitializeTimeStepCandidate,
    PressureAcceleration,
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


def test_direct_pair_loop_outline_uses_context_metadata_without_csr_arrays():
    stage = _stage(
        StageKind.PAIR_RATE,
        (
            _deps("DensityGradient", MethodKind.LOOP),
            _deps("Momentum", MethodKind.LOOP),
        ),
    )

    outline = generate_direct_pair_loop_outline("plan0", stage)

    assert outline.name == "fused_plan0_fluid_pair_rate"
    for declaration in fused_context_argument_declarations():
        assert declaration in outline.source
    assert "nbr_lengths" not in outline.source
    assert "start_idx" not in outline.source
    assert "neighbors" not in outline.source
    assert "int search_radius_cells" in outline.source
    assert (
        "for (int oz = -search_radius_cells; oz <= search_radius_cells; ++oz)"
        in outline.source
    )
    assert "for (int pos = begin; pos < end; ++pos)" in outline.source
    assert "fused_codegen_in_support_xyz" in outline.source
    density = outline.source.index("DensityGradient.loop")
    momentum = outline.source.index("Momentum.loop")
    support = outline.source.index("fused_codegen_in_support_xyz")
    assert support < density < momentum


def test_fused_context_argument_declarations_are_unique():
    declarations = fused_context_argument_declarations()

    assert len(declarations) == len(set(declarations))
    assert declarations.count("int nz") == 1


def test_generated_stage_backend_keeps_device_convergence_for_pair_post_loop_stage():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    post_loop = replace(
        _deps("Density", MethodKind.POST_LOOP),
        dest_writes=frozenset(("h", "converged")),
    )
    stage = _stage(
        StageKind.PAIR_DENSITY,
        (
            _deps("Density", MethodKind.INITIALIZE),
            _deps("Density", MethodKind.LOOP),
            post_loop,
        ),
    )
    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.stage_by_group = {(0, -1): stage}

    backend._begin_device_convergence_super_stage(({"stage_group": (0, -1)},))

    assert backend.device_convergence_uses_particle_flag


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


def test_neighbor_context_keeps_nnps_bounds_for_nonperiodic_axes():
    manager = SimpleNamespace(
        xmin=-1.0,
        xmax=1.0,
        ymin=-10.0,
        ymax=10.0,
        zmin=-20.0,
        zmax=20.0,
        minimum_image_periodic=True,
        periodic_in_x=True,
        periodic_in_y=False,
        periodic_in_z=False,
    )
    nnps = SimpleNamespace(
        xmin=np.array([-0.996, -0.25, -0.05], dtype=np.float64),
        xmax=np.array([0.996, 0.25, 0.05], dtype=np.float64),
        domain=SimpleNamespace(manager=manager),
    )

    lower, upper, periodic = _neighbor_context_bounds_and_periodicity(nnps)

    np.testing.assert_allclose(lower, np.array([-1.0, -0.25, -0.05], dtype=np.float32))
    np.testing.assert_allclose(upper, np.array([1.0, 0.25, 0.05], dtype=np.float32))
    np.testing.assert_array_equal(
        periodic, np.array([True, False, False], dtype=np.bool_)
    )


def test_quintic_wij_precompute_uses_quintic_helper():
    precompute = quintic_spline_wij_precompute(np.int32(3))

    assert precompute.symbols == frozenset(("HIJ", "XIJ", "R2IJ", "RIJ", "WIJ"))
    assert "fused_codegen_quintic_spline_wij" in precompute.helper_source
    assert any("fused_codegen_quintic_spline_wij" in line for line in precompute.lines)


def test_quintic_pair_precompute_expands_common_symbols():
    precompute = quintic_spline_pair_precompute_for_symbols(
        np.int32(3), frozenset(("DWIJ", "GHIJ", "RHOIJ1"))
    )

    assert "HIJ" in precompute.symbols
    assert "DWIJ" in precompute.symbols
    assert "GHIJ" in precompute.symbols
    assert "RHOIJ" in precompute.symbols
    assert "RHOIJ1" in precompute.symbols
    assert "fused_codegen_quintic_spline_gradient" in precompute.helper_source
    assert any(
        "fused_codegen_quintic_spline_gradient(DWIJ" in line
        for line in precompute.lines
    )
    assert any(
        "fused_codegen_quintic_spline_gradient_h(RIJ, HIJ" in line
        for line in precompute.lines
    )


def test_stage_backend_selects_quintic_precompute_for_quintic_kernel():
    from pysph.base.kernels import QuinticSpline
    from pysph.sph.fused_cuda_stage_backend import _precompute_for_stage

    method = replace(
        _deps("Density", MethodKind.LOOP),
        precomputed_symbols=frozenset(("WIJ",)),
        precomputed_writes=frozenset(),
    )
    stage = _stage(StageKind.PAIR_DENSITY, (method,))

    precompute = _precompute_for_stage(stage, QuinticSpline(dim=3))

    assert "fused_codegen_quintic_spline_wij" in precompute.helper_source


def test_lower_equation_loop_body_to_cuda_array_augassign():
    body = lower_equation_method_to_cuda(
        AddMass(dest="fluid", sources=["fluid"]), "loop"
    )

    assert body.equation_name == "AddMass"
    assert body.method_kind is MethodKind.LOOP
    assert body.argument_declarations == ("float *d_au", "const float *s_m")
    assert body.lines == ("d_au[dst] += s_m[src];",)


def test_lower_equation_loop_body_to_cuda_array_assignment():
    body = lower_equation_method_to_cuda(
        CopyAcceleration(dest="fluid", sources=None), "loop"
    )

    assert body.argument_declarations == ("float *d_u", "const float *d_au")
    assert body.lines == ("d_u[dst] = d_au[dst];",)


def test_direct_pair_loop_outline_can_inline_lowered_equation_body():
    stage = _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),))
    body = CudaInlineMethodBody(
        equation_name="AddMass",
        method_kind=MethodKind.LOOP,
        argument_declarations=("float *d_au", "const float *s_m"),
        lines=("d_au[dst] += s_m[src];",),
    )

    outline = generate_direct_pair_loop_outline_with_inline_bodies(
        "plan0", stage, (body,)
    )

    assert "float *d_au" in outline.source
    assert "const float *s_m" in outline.source
    assert "d_au[dst] += s_m[src];" in outline.source
    assert "// AddMass.loop" not in outline.source


def test_direct_pair_loop_outline_can_reuse_pysph_cuda_equation_wrapper():
    equation = AddMass(dest="fluid", sources=["fluid"])
    group = CUDAGroup([equation])
    known_types = {
        "d_au": KnownType("GLOBAL_MEM float*"),
        "s_m": KnownType("GLOBAL_MEM float*"),
    }
    wrapper_source = group.get_equation_wrappers(known_types)
    call = cuda_equation_method_call_from_equation(equation, "loop", known_types)
    stage = _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),))

    outline = generate_direct_pair_loop_outline_with_equation_calls(
        "plan0", stage, wrapper_source, (call,)
    )

    assert "WITHIN_KERNEL void AddMass_loop" in outline.source
    assert "GLOBAL_MEM AddMass* add_mass0" in outline.source
    assert "GLOBAL_MEM float* d_au" in outline.source
    assert "GLOBAL_MEM float* s_m" in outline.source
    assert "AddMass_loop(add_mass0, dst, src, d_au, s_m);" in outline.source
    assert "nbr_length" not in outline.source
    assert "start_idx" not in outline.source
    assert "neighbors" not in outline.source


def test_cluster_pair_loop_outline_uses_sorted_cell_cluster_metadata():
    equation = AddMass(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())

    outline = generate_cluster_pair_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),)),
        (equation,),
        precompute,
    )

    for declaration in cluster_context_argument_declarations():
        assert declaration in outline.source
    assert "int dst_cluster = blockIdx.x;" in outline.source
    assert "int lane = threadIdx.x;" in outline.source
    assert "if (dst_cluster >= cluster_total)" in outline.source
    assert "if (lane >= cluster_count[dst_cluster])" in outline.source
    assert "int cell0 = cluster_cell[dst_cluster];" in outline.source
    assert "int dst = sorted_ids[cluster_begin[dst_cluster] + lane];" in outline.source
    assert "AddMass_loop(add_mass0, dst, src, d_au, s_m);" in outline.source
    assert "int dst = blockIdx.x * blockDim.x + threadIdx.x;" not in outline.source


def test_source_parallel_pair_outline_maps_warp_to_destination():
    equation = AddMass(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())

    outline = generate_source_parallel_pair_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),)),
        (equation,),
        precompute,
    )

    assert "int lane = threadIdx.x & 31;" in outline.source
    assert "int warp = threadIdx.x >> 5;" in outline.source
    assert "int warps_per_block = blockDim.x >> 5;" in outline.source
    assert "int dst = blockIdx.x * warps_per_block + warp;" in outline.source
    assert "for (int pos = begin + lane; pos < end; pos += 32)" in outline.source
    assert "__shfl_down_sync(0xffffffff, fused_reduce_d_au, offset)" in outline.source
    assert "if (lane == 0)" in outline.source
    assert "d_au[dst] += fused_reduce_d_au;" in outline.source


def test_source_parallel_pair_outline_passes_partial_output_to_loop_wrapper():
    equation = AddMass(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())

    outline = generate_source_parallel_pair_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),)),
        (equation,),
        precompute,
    )

    assert "WITHIN_KERNEL void AddMass_loop" in outline.source
    assert "GLOBAL_MEM float* fused_partial_d_au" not in outline.source
    assert "float *fused_partial_d_au_lane = &fused_acc_d_au;" in outline.source
    assert "d_au[0] += s_m[s_idx];" in outline.source
    assert "AddMass_loop(add_mass0, dst, src, fused_partial_d_au_lane, s_m);" in (
        outline.source
    )
    assert "AddMass_loop(add_mass0, dst, src, d_au, s_m);" not in outline.source


def test_source_parallel_pair_outline_uses_thread_local_accumulator():
    equation = AddMass(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())

    outline = generate_source_parallel_pair_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),)),
        (equation,),
        precompute,
    )

    assert "__shared__ float fused_shared_d_au" not in outline.source
    assert "float fused_acc_d_au = 0.0f;" in outline.source
    assert "float *fused_partial_d_au_lane = &fused_acc_d_au;" in outline.source
    assert "float fused_reduce_d_au = fused_acc_d_au;" in outline.source


def test_source_parallel_pair_outline_rejects_non_additive_destination_write():
    equation = AssignMassToAcceleration(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())

    with pytest.raises(AssertionError):
        generate_source_parallel_pair_stage_outline_from_equations(
            "plan0",
            _stage(
                StageKind.PAIR_RATE,
                (_deps("AssignMassToAcceleration", MethodKind.LOOP),),
            ),
            (equation,),
            precompute,
        )


def test_source_parallel_pair_outline_runs_initialize_and_post_loop_on_lane_zero():
    equation = InitLoopPost(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())
    stage = _stage(
        StageKind.PAIR_RATE,
        (
            _deps("InitLoopPost", MethodKind.INITIALIZE),
            _deps("InitLoopPost", MethodKind.LOOP),
            _deps("InitLoopPost", MethodKind.POST_LOOP),
        ),
    )

    outline = generate_source_parallel_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (equation,),
        precompute,
    )

    init_call = "InitLoopPost_initialize(init_loop_post0, dst, d_u, d_au);"
    loop_call = (
        "InitLoopPost_loop(init_loop_post0, dst, src, fused_partial_d_au_lane, s_m);"
    )
    commit = "d_au[dst] += fused_reduce_d_au;"
    post_call = "InitLoopPost_post_loop(init_loop_post0, dst, d_u, d_au);"
    assert f"if (lane == 0) {{\n            {init_call}\n        }}" in outline.source
    assert f"if (lane == 0) {{\n            {post_call}\n        }}" in outline.source
    assert outline.source.index(init_call) < outline.source.index(loop_call)
    assert outline.source.index(loop_call) < outline.source.index(commit)
    assert outline.source.index(commit) < outline.source.index(post_call)


def test_source_parallel_pair_outline_supports_sum_and_max_reductions():
    equation = AccumulateDWIJAndMaxSignal(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(
        symbols=frozenset(("DWIJ",)),
        helper_source="",
        lines=("float DWIJ[3];", "DWIJ[0] = 2.0f;"),
    )

    outline = generate_source_parallel_pair_stage_outline_from_equations(
        "plan0",
        _stage(
            StageKind.PAIR_RATE,
            (_deps("AccumulateDWIJAndMaxSignal", MethodKind.LOOP),),
        ),
        (equation,),
        precompute,
    )

    assert "__shared__ float fused_shared_d_au" not in outline.source
    assert "__shared__ float fused_shared_d_dt_cfl" not in outline.source
    assert "GLOBAL_MEM float* fused_partial_d_au" not in outline.source
    assert "GLOBAL_MEM float* fused_partial_d_dt_cfl" not in outline.source
    assert "float fused_acc_d_au = 0.0f;" in outline.source
    assert "float fused_acc_d_dt_cfl = d_dt_cfl[dst];" in outline.source
    assert "float *fused_partial_d_au_lane = &fused_acc_d_au;" in outline.source
    assert "float *fused_partial_d_dt_cfl_lane = &fused_acc_d_dt_cfl;" in outline.source
    assert "d_au[0] += (s_m[s_idx] * DWIJ[0]);" in outline.source
    assert "d_dt_cfl[0] = max(d_dt_cfl[0], abs(DWIJ[0]));" in outline.source
    assert (
        "AccumulateDWIJAndMaxSignal_loop(accumulate_dwij_and_max_signal0, dst, src, fused_partial_d_au_lane, fused_partial_d_dt_cfl_lane, s_m, DWIJ);"
        in outline.source
    )
    assert "d_au[dst] += fused_reduce_d_au;" in outline.source
    assert (
        "fused_reduce_d_dt_cfl = fmaxf(fused_reduce_d_dt_cfl, __shfl_down_sync(0xffffffff, fused_reduce_d_dt_cfl, offset));"
        in outline.source
    )
    assert "d_dt_cfl[dst] = fmaxf(d_dt_cfl[dst], fused_reduce_d_dt_cfl);" in (
        outline.source
    )


def test_source_parallel_pair_launch_uses_context_stream_and_warp_grid():
    class FakeGpuArray:
        gpudata = 7

    class FakeKernel:
        def __call__(self, *args, block, grid, stream):
            self.args = args
            self.block = block
            self.grid = grid
            self.stream = stream

    class FakeModule:
        def __init__(self):
            self.kernel = FakeKernel()
            self.requested_name = ""

        def get_function(self, name):
            self.requested_name = name
            return self.kernel

    stream = object()
    context = SimpleNamespace(
        x=FakeGpuArray(),
        y=FakeGpuArray(),
        z=FakeGpuArray(),
        h=FakeGpuArray(),
        n=10,
        lower=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        periodic=np.array([True, False, False], dtype=np.bool_),
        radius_scale=np.float32(2.0),
        search_radius_cells=np.int32(1),
        cell_counts=np.array([7, 2, 1], dtype=np.int32),
        cell_particle_counts=FakeGpuArray(),
        cell_starts=FakeGpuArray(),
        sorted_ids=FakeGpuArray(),
        stream=stream,
    )
    module = FakeModule()
    normal_arg = FakeGpuArray()

    launch_source_parallel_pair_kernel_with_context(
        module,
        "kernel0",
        context,
        (normal_arg,),
    )

    assert module.requested_name == "kernel0"
    assert module.kernel.block == (128, 1, 1)
    assert module.kernel.grid == (3, 1, 1)
    assert module.kernel.stream is stream
    assert module.kernel.args[-1:] == (normal_arg.gpudata,)


def test_pointwise_kernel_outline_can_reuse_pysph_cuda_equation_wrapper():
    equation = CopyAcceleration(dest="fluid", sources=None)
    group = CUDAGroup([equation])
    known_types = {
        "d_u": KnownType("GLOBAL_MEM float*"),
        "d_au": KnownType("GLOBAL_MEM float*"),
    }
    wrapper_source = group.get_equation_wrappers(known_types)
    call = cuda_equation_method_call_from_equation(equation, "loop", known_types)

    outline = generate_pointwise_kernel_outline_with_equation_calls(
        "plan0",
        _stage(StageKind.POINTWISE, (_deps("CopyAcceleration", MethodKind.LOOP),)),
        wrapper_source,
        (call,),
    )

    assert outline.name == "fused_plan0_fluid_pointwise"
    assert "WITHIN_KERNEL void CopyAcceleration_loop" in outline.source
    assert "int dst = blockIdx.x * blockDim.x + threadIdx.x;" in outline.source
    assert "if (dst >= n)" in outline.source
    assert (
        "CopyAcceleration_loop(copy_acceleration0, dst, d_u, d_au);" in outline.source
    )
    assert "sorted_ids" not in outline.source
    assert "cell_starts" not in outline.source


def test_reduction_stage_without_reduce_method_uses_pointwise_wrapper_kernel():
    equation = InitializeTimeStepCandidate(dest="fluid", sources=None)

    outline = generate_pointwise_stage_outline_from_equations(
        "plan0",
        _stage(
            StageKind.REDUCTION,
            (_deps("InitializeTimeStepCandidate", MethodKind.INITIALIZE),),
        ),
        (equation,),
    )

    assert outline.name == "fused_plan0_fluid_reduction"
    assert "int dst = blockIdx.x * blockDim.x + threadIdx.x;" in outline.source
    assert (
        "InitializeTimeStepCandidate_initialize(initialize_time_step_candidate0, dst, d_dt_adapt, d_au);"
        in outline.source
    )
    assert "sorted_ids" not in outline.source


def test_pointwise_kernel_outline_uses_fp32_equation_struct_fields():
    equation = IsothermalEOS(dest="fluid", sources=None, rho0=1.0, c0=10.0, p0=0.5)
    group = CUDAGroup([equation])
    known_types = {
        "d_rho": KnownType("GLOBAL_MEM float*"),
        "d_p": KnownType("GLOBAL_MEM float*"),
    }
    wrapper_source = group.get_equation_wrappers(known_types)
    call = cuda_equation_method_call_from_equation(equation, "loop", known_types)

    outline = generate_pointwise_kernel_outline_with_equation_calls(
        "plan0",
        _stage(StageKind.POINTWISE, (_deps("IsothermalEOS", MethodKind.LOOP),)),
        wrapper_source,
        (call,),
    )

    assert "typedef struct IsothermalEOS" in outline.source
    assert "float c02;" in outline.source
    assert "float p0;" in outline.source
    assert "float rho0;" in outline.source
    assert "double" not in outline.source
    assert "IsothermalEOS_loop(isothermal_eos0, dst, d_rho, d_p);" in outline.source


def test_pair_stage_outline_from_equations_infers_wrapper_calls_and_precompute():
    equation = SummationDensity(dest="fluid", sources=["fluid"])
    precompute = cubic_spline_wij_precompute(np.int32(1))

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.PAIR_DENSITY, (_deps("SummationDensity", MethodKind.LOOP),)),
        (equation,),
        precompute,
    )

    assert "WITHIN_KERNEL void SummationDensity_loop" in outline.source
    assert "float WIJ;" in outline.source
    assert (
        "SummationDensity_loop(summation_density0, dst, d_rho, src, s_m, WIJ);"
        in outline.source
    )


def test_pointwise_stage_outline_from_equations_infers_struct_wrapper_calls():
    equation = IsothermalEOS(dest="fluid", sources=None, rho0=1.0, c0=10.0, p0=0.5)

    outline = generate_pointwise_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.POINTWISE, (_deps("IsothermalEOS", MethodKind.LOOP),)),
        (equation,),
    )

    assert "typedef struct IsothermalEOS" in outline.source
    assert "float c02;" in outline.source
    assert "IsothermalEOS_loop(isothermal_eos0, dst, d_rho, d_p);" in outline.source


def test_pair_stage_places_initialize_and_post_loop_outside_neighbor_loop():
    equation = InitLoopPost(dest="fluid", sources=["fluid"])
    precompute = cubic_spline_wij_precompute(np.int32(1))

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        _stage(
            StageKind.PAIR_RATE,
            (
                _deps("InitLoopPost", MethodKind.INITIALIZE),
                _deps("InitLoopPost", MethodKind.LOOP),
                _deps("InitLoopPost", MethodKind.POST_LOOP),
            ),
        ),
        (equation,),
        precompute,
    )

    initialize = "InitLoopPost_initialize(init_loop_post0, dst, d_u, d_au);"
    loop = "InitLoopPost_loop(init_loop_post0, dst, src, d_au, s_m);"
    post_loop = "InitLoopPost_post_loop(init_loop_post0, dst, d_u, d_au);"
    support = "if (fused_codegen_in_support_xyz"
    first_source_loop = "for (int pos = begin; pos < end; ++pos)"
    source = outline.source

    assert initialize in source
    assert loop in source
    assert post_loop in source
    assert source.index(initialize) < source.index(first_source_loop)
    assert source.index(first_source_loop) < source.index(loop)
    assert source.index(loop) < source.index(post_loop)
    assert source.index(support) < source.index(loop)
    assert source.index(post_loop) > source.rindex("                    }")


def test_pair_stage_places_source_free_pointwise_tail_after_neighbor_loop():
    pair_equation = AddMass(dest="fluid", sources=["fluid"])
    tail_equation = CopyAcceleration(dest="fluid", sources=None)
    stage = _stage(
        StageKind.PAIR_RATE,
        (
            _deps("AddMass", MethodKind.LOOP),
            MethodDeps(
                equation_name="CopyAcceleration",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=(),
                dest_reads=frozenset(("au",)),
                source_reads=frozenset(),
                dest_writes=frozenset(("u",)),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (pair_equation, tail_equation),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
    )

    pair_call = "AddMass_loop(add_mass0, dst, src, d_au, s_m);"
    tail_call = "CopyAcceleration_loop(copy_acceleration0, dst, d_u, d_au);"
    source = outline.source

    assert pair_call in source
    assert tail_call in source
    assert source.index(pair_call) < source.index(tail_call)
    assert source.index(tail_call) > source.rindex("                    }")


def test_pair_stage_places_source_free_pointwise_head_before_neighbor_loop():
    head_equation = CopyAcceleration(dest="fluid", sources=None)
    pair_equation = AddMass(dest="fluid", sources=["fluid"])
    stage = _stage(
        StageKind.PAIR_RATE,
        (
            MethodDeps(
                equation_name="CopyAcceleration",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=(),
                dest_reads=frozenset(("au",)),
                source_reads=frozenset(),
                dest_writes=frozenset(("u",)),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
            _deps("AddMass", MethodKind.LOOP),
        ),
    )

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (head_equation, pair_equation),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
    )

    head_call = "CopyAcceleration_loop(copy_acceleration0, dst, d_u, d_au);"
    pair_call = "AddMass_loop(add_mass0, dst, src, d_au, s_m);"
    first_source_loop = "for (int pos = begin; pos < end; ++pos)"
    source = outline.source

    assert head_call in source
    assert pair_call in source
    assert source.index(head_call) < source.index(first_source_loop)
    assert source.index(first_source_loop) < source.index(pair_call)


def test_pair_stage_with_method_segments_generates_one_neighbor_traversal_per_segment():
    first_equation = AddMass(dest="fluid", sources=["fluid"])
    second_equation = AccumulateDWIJ(dest="fluid", sources=["fluid"])
    first_method = _deps("AddMass", MethodKind.LOOP)
    second_method = _deps("AccumulateDWIJ", MethodKind.LOOP)
    second_method = replace(second_method, precomputed_symbols=frozenset(("DWIJ",)))
    stage = replace(
        _stage(StageKind.PAIR_RATE, (first_method, second_method)),
        method_segments=((first_method,), (second_method,)),
    )

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (first_equation, second_equation),
        cubic_spline_gradient_precompute(np.int32(1)),
    )

    first_call = "AddMass_loop(add_mass0, dst, src, d_au, s_m);"
    second_call = "AccumulateDWIJ_loop(accumulate_dwij0, dst, src, d_au, s_m, DWIJ);"
    first_source_loop = "for (int pos = begin; pos < end; ++pos)"
    source = outline.source

    assert source.count(first_source_loop) == 2
    assert source.index(first_call) < source.index(second_call)
    assert source.index(second_call) > source.index(
        first_source_loop, source.index(first_call)
    )


def test_pair_stage_with_method_segments_scopes_each_neighbor_traversal():
    first_equation = AddMass(dest="fluid", sources=["fluid"])
    second_equation = AccumulateDWIJ(dest="fluid", sources=["fluid"])
    first_method = _deps("AddMass", MethodKind.LOOP)
    second_method = replace(
        _deps("AccumulateDWIJ", MethodKind.LOOP),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
    )
    stage = replace(
        _stage(StageKind.PAIR_RATE, (first_method, second_method)),
        method_segments=((first_method,), (second_method,)),
    )

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (first_equation, second_equation),
        cubic_spline_gradient_precompute(np.int32(1)),
    )

    source = outline.source

    assert "\n    int base_cx =" not in source
    assert source.count("\n    {\n        int base_cx =") == 2


def test_direct_pair_block_size_accepts_supported_values(monkeypatch):
    assert _direct_pair_block_size() == 256
    monkeypatch.setenv("PYSPH_FUSED_PAIR_BLOCK_SIZE", "128")
    assert _direct_pair_block_size() == 128
    monkeypatch.setenv("PYSPH_FUSED_PAIR_BLOCK_SIZE", "96")
    with pytest.raises(AssertionError):
        _direct_pair_block_size()


def test_pair_block_size_for_count_increases_blocks_when_particle_grid_underfills_device(
    monkeypatch,
):
    from pysph.sph import fused_cuda_codegen as codegen

    monkeypatch.setattr(
        codegen, "_cuda_multiprocessor_count", lambda: 170, raising=False
    )

    assert codegen._pair_block_size_for_count(13824) == 128
    assert codegen._pair_block_size_for_count(71688) == 256


def test_pair_block_size_for_count_preserves_explicit_override(monkeypatch):
    from pysph.sph import fused_cuda_codegen as codegen

    monkeypatch.setattr(
        codegen, "_cuda_multiprocessor_count", lambda: 170, raising=False
    )
    monkeypatch.setenv("PYSPH_FUSED_PAIR_BLOCK_SIZE", "64")

    assert codegen._pair_block_size_for_count(13824) == 64


def test_pair_cluster_size_accepts_supported_values(monkeypatch):
    assert _pair_cluster_size() == 64
    monkeypatch.setenv("PYSPH_FUSED_PAIR_CLUSTER_SIZE", "32")
    assert _pair_cluster_size() == 32
    monkeypatch.setenv("PYSPH_FUSED_PAIR_CLUSTER_SIZE", "96")
    with pytest.raises(AssertionError):
        _pair_cluster_size()


def test_pair_traversal_mode_accepts_source_parallel(monkeypatch):
    assert pair_traversal_mode() == "direct"
    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "source_parallel")
    assert pair_traversal_mode() == "source_parallel"


def test_pair_traversal_mode_accepts_hbucket(monkeypatch):
    assert pair_traversal_mode() == "direct"
    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    assert pair_traversal_mode() == "hbucket"


def test_pair_traversal_mode_accepts_hbucket_source_parallel(monkeypatch):
    assert pair_traversal_mode() == "direct"
    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket_source_parallel")
    assert pair_traversal_mode() == "hbucket_source_parallel"


def test_hbucket_pair_outline_uses_bucketed_cell_ranges():
    from pysph.sph.fused_cuda_codegen import (
        generate_hbucket_pair_stage_outline_from_equations,
    )

    stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AddMass", MethodKind.LOOP),),
    )
    outline = generate_hbucket_pair_stage_outline_from_equations(
        "test",
        stage,
        (AddMass(dest="fluid", sources=["fluid"]),),
        cubic_spline_wij_precompute(np.int32(1)),
    )

    assert "bucket_h_max" in outline.source
    assert "cell_bucket_h_max" in outline.source
    assert "cell_bucket_counts" in outline.source
    assert "cell_bucket_starts" in outline.source
    assert "for (int bucket = 0; bucket < bucket_count; ++bucket)" in outline.source
    assert "float dst_x = x[dst];" in outline.source
    assert "fused_codegen_in_support_xyz_cached" in outline.source


def test_hbucket_pair_outline_uses_local_accumulator_for_shared_reduction_writes():
    from pysph.sph.fused_cuda_codegen import (
        generate_hbucket_pair_stage_outline_from_equations,
    )

    stage = replace(
        _stage(
            StageKind.PAIR_RATE,
            (
                _sum_reduction_deps("AddMass", "au"),
                _sum_reduction_deps("AddScaledMass", "au"),
            ),
        ),
        method_segments=(
            (
                _sum_reduction_deps("AddMass", "au"),
                _sum_reduction_deps("AddScaledMass", "au"),
            ),
        ),
    )

    outline = generate_hbucket_pair_stage_outline_from_equations(
        "test",
        stage,
        (
            AddMass(dest="fluid", sources=["fluid"]),
            AddScaledMass(dest="fluid", sources=["fluid"]),
        ),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
    )

    assert "float fused_acc_d_au = 0.0f;" in outline.source
    assert "float *fused_local_d_au = &fused_acc_d_au;" in outline.source
    assert "d_au[0] += s_m[s_idx];" in outline.source
    assert "d_au[0] += (2.0f * s_m[s_idx]);" in outline.source
    assert "AddMass_loop(add_mass0, dst, src, fused_local_d_au, s_m);" in (
        outline.source
    )
    assert (
        "AddScaledMass_loop(add_scaled_mass0, dst, src, fused_local_d_au, s_m);"
        in outline.source
    )
    assert "d_au[dst] += fused_acc_d_au;" in outline.source


def test_hbucket_source_parallel_outline_uses_bucketed_lane_ranges():
    stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AddMass", MethodKind.LOOP),),
    )

    outline = generate_hbucket_source_parallel_pair_stage_outline_from_equations(
        "test",
        stage,
        (AddMass(dest="fluid", sources=["fluid"]),),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
    )

    assert "bucket_h_max" in outline.source
    assert "cell_bucket_h_max" in outline.source
    assert "cell_bucket_counts" in outline.source
    assert "float dst_x = x[dst];" in outline.source
    assert "float cell_bucket_h = cell_bucket_h_max[flat];" in outline.source
    assert "fused_codegen_in_support_xyz_cached" in outline.source
    assert "for (int pos = begin + lane; pos < end; pos += 32)" in outline.source
    assert "__shfl_down_sync" in outline.source


def test_hbucket_pair_launch_uses_bucket_context_stream_and_grid(monkeypatch):
    from pysph.sph import fused_cuda_codegen as codegen

    monkeypatch.setattr(
        codegen, "_pair_block_size_for_count", lambda n: 128, raising=False
    )

    class FakeGpuArray:
        gpudata = 7

    class FakeKernel:
        def __call__(self, *args, block, grid, stream):
            self.args = args
            self.block = block
            self.grid = grid
            self.stream = stream

    class FakeModule:
        def __init__(self):
            self.kernel = FakeKernel()
            self.requested_name = ""

        def get_function(self, name):
            self.requested_name = name
            return self.kernel

    stream = object()
    context = SimpleNamespace(
        x=FakeGpuArray(),
        y=FakeGpuArray(),
        z=FakeGpuArray(),
        h=FakeGpuArray(),
        n=513,
        lower=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        periodic=np.array([True, False, False], dtype=np.bool_),
        radius_scale=np.float32(2.0),
        cell_counts=np.array([7, 2, 1], dtype=np.int32),
        total_cells=14,
        bucket_count=4,
        cell_width=np.array([1.0 / 7.0, 0.5, 1.0], dtype=np.float32),
        bucket_h_max=FakeGpuArray(),
        cell_bucket_h_max=FakeGpuArray(),
        cell_bucket_counts=FakeGpuArray(),
        cell_bucket_starts=FakeGpuArray(),
        sorted_ids=FakeGpuArray(),
        stream=stream,
    )
    module = FakeModule()
    normal_arg = FakeGpuArray()

    launch_config = launch_hbucket_pair_kernel_with_context(
        module,
        "kernel0",
        context,
        (normal_arg,),
    )

    assert module.requested_name == "kernel0"
    assert launch_config.traversal == "hbucket"
    assert launch_config.n == 513
    assert launch_config.block_size == 128
    assert launch_config.grid_x == 5
    assert module.kernel.block == (128, 1, 1)
    assert module.kernel.grid == (5, 1, 1)
    assert module.kernel.stream is stream
    assert module.kernel.args[-1:] == (normal_arg.gpudata,)


def test_generated_stage_backend_counts_pair_launch_configs():
    from pysph.sph.fused_cuda_codegen import PairLaunchConfig
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.pair_launch_config_counts = {}

    backend._record_pair_launch_config(
        PairLaunchConfig(traversal="hbucket", n=513, block_size=128, grid_x=5)
    )
    backend._record_pair_launch_config(
        PairLaunchConfig(traversal="hbucket", n=513, block_size=128, grid_x=5)
    )
    backend._record_pair_launch_config(
        PairLaunchConfig(
            traversal="resident_hbucket", n=13824, block_size=128, grid_x=108
        )
    )

    assert backend.pair_launch_config_counts == {
        ("hbucket", 513, 128, 5): 2,
        ("resident_hbucket", 13824, 128, 108): 1,
    }


def test_hbucket_source_parallel_launch_uses_bucket_context_stream_and_warp_grid():
    class FakeGpuArray:
        gpudata = 7

    class FakeKernel:
        def __call__(self, *args, block, grid, stream):
            self.args = args
            self.block = block
            self.grid = grid
            self.stream = stream

    class FakeModule:
        def __init__(self):
            self.kernel = FakeKernel()
            self.requested_name = ""

        def get_function(self, name):
            self.requested_name = name
            return self.kernel

    stream = object()
    context = SimpleNamespace(
        x=FakeGpuArray(),
        y=FakeGpuArray(),
        z=FakeGpuArray(),
        h=FakeGpuArray(),
        n=513,
        lower=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        periodic=np.array([True, False, False], dtype=np.bool_),
        radius_scale=np.float32(2.0),
        cell_counts=np.array([7, 2, 1], dtype=np.int32),
        total_cells=14,
        bucket_count=4,
        cell_width=np.array([1.0 / 7.0, 0.5, 1.0], dtype=np.float32),
        bucket_h_max=FakeGpuArray(),
        cell_bucket_h_max=FakeGpuArray(),
        cell_bucket_counts=FakeGpuArray(),
        cell_bucket_starts=FakeGpuArray(),
        sorted_ids=FakeGpuArray(),
        stream=stream,
    )
    module = FakeModule()
    normal_arg = FakeGpuArray()

    launch_hbucket_source_parallel_pair_kernel_with_context(
        module,
        "kernel0",
        context,
        (normal_arg,),
    )

    assert module.requested_name == "kernel0"
    assert module.kernel.block == (128, 1, 1)
    assert module.kernel.grid == (129, 1, 1)
    assert module.kernel.stream is stream
    assert module.kernel.args[-1:] == (normal_arg.gpudata,)


def test_stage_backend_uses_source_parallel_only_for_supported_pair_stage(
    monkeypatch,
):
    from pysph.sph.fused_cuda_stage_backend import _pair_traversal_for_stage

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "source_parallel")
    supported_stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AddMass", MethodKind.LOOP),),
    )
    unsupported_stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AssignMassToAcceleration", MethodKind.LOOP),),
    )

    assert (
        _pair_traversal_for_stage(
            supported_stage,
            (AddMass(dest="fluid", sources=["fluid"]),),
            None,
        )
        == "source_parallel"
    )
    assert (
        _pair_traversal_for_stage(
            supported_stage,
            (AddMass(dest="fluid", sources=["fluid"]),),
            "converged",
        )
        == "direct"
    )
    assert (
        _pair_traversal_for_stage(
            unsupported_stage,
            (AssignMassToAcceleration(dest="fluid", sources=["fluid"]),),
            None,
        )
        == "direct"
    )


def test_stage_backend_uses_hbucket_source_parallel_for_supported_pair_stage(
    monkeypatch,
):
    from pysph.sph.fused_cuda_stage_backend import _pair_traversal_for_stage

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket_source_parallel")
    supported_stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AddMass", MethodKind.LOOP),),
    )
    unsupported_stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AssignMassToAcceleration", MethodKind.LOOP),),
    )

    assert (
        _pair_traversal_for_stage(
            supported_stage,
            (AddMass(dest="fluid", sources=["fluid"]),),
            None,
        )
        == "hbucket_source_parallel"
    )
    assert (
        _pair_traversal_for_stage(
            supported_stage,
            (AddMass(dest="fluid", sources=["fluid"]),),
            "converged",
        )
        == "hbucket"
    )
    assert (
        _pair_traversal_for_stage(
            unsupported_stage,
            (AssignMassToAcceleration(dest="fluid", sources=["fluid"]),),
            None,
        )
        == "hbucket"
    )


def test_stage_backend_uses_hbucket_when_requested(monkeypatch):
    from pysph.sph.fused_cuda_stage_backend import _pair_traversal_for_stage

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AssignMassToAcceleration", MethodKind.LOOP),),
    )

    assert (
        _pair_traversal_for_stage(
            stage,
            (AssignMassToAcceleration(dest="fluid", sources=["fluid"]),),
            "converged",
        )
        == "hbucket"
    )


def test_generated_stage_backend_records_pair_traversal_launch_counts():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.traversal_launch_counts = {"direct": 2}

    backend._record_pair_traversal_launch("source_parallel")
    backend._record_pair_traversal_launch("direct")

    assert backend.traversal_launch_counts == {
        "direct": 3,
        "source_parallel": 1,
    }


def test_direct_pair_stage_outline_can_mark_device_convergence_flag():
    equation = DensityConvergenceFlag(dest="fluid", sources=["fluid"])
    stage = _stage(
        StageKind.PAIR_DENSITY,
        (
            MethodDeps(
                equation_name="DensityConvergenceFlag",
                method_kind=MethodKind.INITIALIZE,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(),
                source_reads=frozenset(),
                dest_writes=frozenset(("rho", "converged")),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
            MethodDeps(
                equation_name="DensityConvergenceFlag",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(("rho",)),
                source_reads=frozenset(("m",)),
                dest_writes=frozenset(("rho",)),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(("WIJ",)),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
            MethodDeps(
                equation_name="DensityConvergenceFlag",
                method_kind=MethodKind.POST_LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(("rho",)),
                source_reads=frozenset(),
                dest_writes=frozenset(("converged",)),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )

    outline = generate_direct_pair_stage_outline_from_equations_with_convergence_flag(
        "plan0",
        stage,
        (equation,),
        cubic_spline_wij_precompute(np.int32(1)),
        "converged",
    )

    assert "int *fused_convergence_flag" in outline.source
    assert "if (d_converged[dst] == 0.0f) {" in outline.source
    assert "atomicExch(fused_convergence_flag, 0);" in outline.source
    assert outline.source.index(
        "DensityConvergenceFlag_post_loop"
    ) < outline.source.index("atomicExch(fused_convergence_flag, 0);")


def test_resident_hbucket_pair_window_outline_uses_cooperative_grid_sync():
    from pysph.sph import fused_cuda_codegen as codegen

    assert hasattr(
        codegen, "generate_resident_hbucket_pair_window_outline_from_equations"
    )
    add_mass = AddMass(dest="fluid", sources=["fluid"])
    pressure = PressureAcceleration(dest="fluid", sources=["fluid"])
    first_stage = _stage(
        StageKind.PAIR_RATE,
        (
            MethodDeps(
                equation_name="AddMass",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(("au",)),
                source_reads=frozenset(("m",)),
                dest_writes=frozenset(("au",)),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )
    second_stage = _stage(
        StageKind.PAIR_RATE,
        (
            MethodDeps(
                equation_name="PressureAcceleration",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(("au", "p", "rho")),
                source_reads=frozenset(("p", "rho", "m")),
                dest_writes=frozenset(("au",)),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(("DWIJ",)),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )

    outline = codegen.generate_resident_hbucket_pair_window_outline_from_equations(
        "plan0",
        (first_stage, second_stage),
        ((add_mass,), (pressure,)),
        (
            CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
            cubic_spline_gradient_precompute(np.int32(1)),
        ),
    )

    assert outline.name == "fused_plan0_fluid_resident_hbucket_pair_window"
    assert "#include <cooperative_groups.h>" in outline.source
    assert (
        "cooperative_groups::grid_group grid = cooperative_groups::this_grid();"
        in outline.source
    )
    assert (
        "for (int dst = blockIdx.x * blockDim.x + threadIdx.x; dst < n; dst += blockDim.x * gridDim.x)"
    ) in outline.source
    barrier = outline.source.index("grid.sync();")
    first_call = outline.source.rindex("AddMass_loop", 0, barrier)
    second_call = outline.source.index("PressureAcceleration_loop", barrier)
    assert first_call < barrier < second_call


def test_old_state_single_pass_hbucket_window_uses_one_traversal_and_snapshots():
    class CubicSpline:
        dim = 1

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_init = MethodDeps(
        equation_name="ReadDestAndSourceAcceleration",
        method_kind=MethodKind.INITIALIZE,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_loop = MethodDeps(
        equation_name="ReadDestAndSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha", "au")),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_post = MethodDeps(
        equation_name="ReadDestAndSourceAcceleration",
        method_kind=MethodKind.POST_LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("beta",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(
        StageKind.PAIR_RATE,
        (prep_method, diagnostic_init, diagnostic_loop, diagnostic_post),
    )
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    precompute = cubic_spline_pair_precompute_for_symbols(
        np.int32(1), frozenset(("DWIJ",))
    )

    outline = generate_hbucket_old_state_pair_window_outline_from_equations(
        "plan0",
        (left_stage, right_stage),
        (
            (
                PrepPressure(dest="fluid", sources=None),
                ReadDestAndSourceAcceleration(dest="fluid", sources=["fluid"]),
            ),
            (PressureAcceleration(dest="fluid", sources=["fluid"]),),
        ),
        precompute,
        (prep_method,),
        (diagnostic_loop,),
        ("au",),
    )

    assert outline.name == "fused_plan0_fluid_hbucket_old_state_pair_window"
    assert outline.source.count("grid.sync();") == 1
    assert (
        outline.source.count("for (int bucket = 0; bucket < bucket_count; ++bucket)")
        == 1
    )
    assert "GLOBAL_MEM float* d_old_au" in outline.source
    assert "GLOBAL_MEM float* s_old_au" in outline.source
    assert "ReadDestAndSourceAcceleration_loop" in outline.source
    assert "PressureAcceleration_loop" in outline.source
    assert "d_old_au" in outline.source
    assert "s_old_au" in outline.source


def test_old_state_single_pass_hbucket_window_argument_names_include_snapshots():
    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_loop = MethodDeps(
        equation_name="ReadDestAndSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha", "au")),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_loop))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    precompute = cubic_spline_pair_precompute_for_symbols(
        np.int32(1), frozenset(("DWIJ",))
    )

    names = hbucket_old_state_pair_window_argument_names(
        (left_stage, right_stage),
        (
            (
                PrepPressure(dest="fluid", sources=None),
                ReadDestAndSourceAcceleration(dest="fluid", sources=["fluid"]),
            ),
            (PressureAcceleration(dest="fluid", sources=["fluid"]),),
        ),
        precompute,
        (prep_method,),
        (diagnostic_loop,),
        ("au",),
    )

    assert "d_old_au" in names
    assert "s_old_au" in names
    assert "d_p" in names
    assert "s_p" in names


def test_old_state_hbucket_pair_stage_uses_one_normal_traversal_and_snapshots():
    diagnostic_loop = MethodDeps(
        equation_name="ReadDestAndSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha", "au")),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (diagnostic_loop,))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    precompute = cubic_spline_pair_precompute_for_symbols(
        np.int32(1), frozenset(("DWIJ",))
    )

    outline = generate_hbucket_old_state_pair_stage_outline_from_equations(
        "plan0",
        (left_stage, right_stage),
        (
            (ReadDestAndSourceAcceleration(dest="fluid", sources=["fluid"]),),
            (PressureAcceleration(dest="fluid", sources=["fluid"]),),
        ),
        precompute,
        (diagnostic_loop,),
        ("au",),
    )
    names = hbucket_old_state_pair_stage_argument_names(
        (left_stage, right_stage),
        (
            (ReadDestAndSourceAcceleration(dest="fluid", sources=["fluid"]),),
            (PressureAcceleration(dest="fluid", sources=["fluid"]),),
        ),
        precompute,
        (diagnostic_loop,),
        ("au",),
    )

    assert outline.name == "fused_plan0_fluid_hbucket_old_state_pair_stage"
    assert "#include <cooperative_groups.h>" not in outline.source
    assert "grid.sync();" not in outline.source
    assert (
        outline.source.count("for (int bucket = 0; bucket < bucket_count; ++bucket)")
        == 1
    )
    assert "GLOBAL_MEM float* d_old_au" in outline.source
    assert "GLOBAL_MEM float* s_old_au" in outline.source
    assert "d_old_au" in names
    assert "s_old_au" in names


def test_fused_stage_backend_launches_group_once_and_skips_legacy_calls():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"], tuple(extra_args)))

    stage = _stage(
        StageKind.PAIR_RATE,
        (
            _deps("InitLoopPost", MethodKind.INITIALIZE),
            _deps("InitLoopPost", MethodKind.LOOP),
            _deps("InitLoopPost", MethodKind.POST_LOOP),
        ),
    )
    helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(stages=(stage,)),
        calls=(
            {
                "type": "kernel",
                "stage_group": (0, -1),
                "stage_method_kind": "initialize",
            },
            {"type": "kernel", "stage_group": (0, -1), "stage_method_kind": "loop"},
            {
                "type": "kernel",
                "stage_group": (0, -1),
                "stage_method_kind": "post_loop",
            },
        ),
    )
    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)

    first = backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    second = backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)
    third = backend.handle_call("evaluator", helper.calls[2], ("t", "dt"), 0.0, 0.1)

    assert first
    assert second
    assert third
    assert backend.launched == [(StageKind.PAIR_RATE, (0, -1), ("t", "dt"))]
    assert not backend.end_compute("evaluator", 0.0, 0.1)


def test_fused_stage_backend_launches_resident_rhs_window_from_first_group():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []
            self.windows = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"], tuple(extra_args)))

        def _launch_resident_window(self, evaluator, stage_indices, extra_args):
            stages = tuple(
                self.helper.cuda_stage_plan.stages[index] for index in stage_indices
            )
            infos = tuple(
                self._kernel_info_for_plan_stage_index(index) for index in stage_indices
            )
            self.windows.append(
                (
                    tuple(stage.kind for stage in stages),
                    tuple(info["stage_group"] for info in infos),
                )
            )
            super()._launch_resident_window(evaluator, stage_indices, extra_args)

    density_stage = _stage(
        StageKind.PAIR_DENSITY,
        (_deps("Density", MethodKind.LOOP),),
    )
    eos_stage = _stage(
        StageKind.POINTWISE,
        (_deps("EOS", MethodKind.LOOP),),
    )
    rate_stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("Rate", MethodKind.LOOP),),
    )
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(density_stage, eos_stage, rate_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
            {"type": "kernel", "stage_group": (2, -1)},
        ),
    )
    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)

    first = backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    second = backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)
    third = backend.handle_call("evaluator", helper.calls[2], ("t", "dt"), 0.0, 0.1)

    assert first
    assert second
    assert third
    assert backend.windows == [
        (
            (StageKind.PAIR_DENSITY, StageKind.POINTWISE, StageKind.PAIR_RATE),
            ((0, -1), (1, -1), (2, -1)),
        )
    ]
    assert backend.launched == [
        (StageKind.PAIR_DENSITY, (0, -1), ("t", "dt")),
        (StageKind.POINTWISE, (1, -1), ("t", "dt")),
        (StageKind.PAIR_RATE, (2, -1), ("t", "dt")),
    ]


def test_fused_stage_backend_launches_cooperative_pair_window_from_resident_window():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []
            self.cooperative_windows = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"], tuple(extra_args)))

        def _launch_cooperative_grid_sync_window(
            self, evaluator, stage_indices, extra_args
        ):
            infos = tuple(
                self._kernel_info_for_plan_stage_index(index) for index in stage_indices
            )
            self.cooperative_windows.append(
                (
                    stage_indices,
                    tuple(info["stage_group"] for info in infos),
                    tuple(extra_args),
                )
            )

    first_method = MethodDeps(
        equation_name="Predict",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(("u",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    second_method = MethodDeps(
        equation_name="Rate",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(("u",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    first_stage = _stage(StageKind.PAIR_RATE, (first_method,))
    second_stage = _stage(StageKind.PAIR_RATE, (second_method,))
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(first_stage, second_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
    )
    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)

    first = backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    second = backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)

    assert first
    assert second
    assert backend.cooperative_windows == [((0, 1), ((0, -1), (1, -1)), ("t", "dt"))]
    assert backend.launched == []


def test_fused_stage_backend_launches_source_inline_precompute_window_from_resident_window():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []
            self.inline_windows = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"], tuple(extra_args)))

        def _launch_cooperative_grid_sync_window(
            self, evaluator, stage_indices, extra_args
        ):
            infos = tuple(
                self._kernel_info_for_plan_stage_index(index) for index in stage_indices
            )
            self.inline_windows.append(
                (
                    stage_indices,
                    tuple(info["stage_group"] for info in infos),
                    tuple(extra_args),
                )
            )

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p")),
        source_reads=frozenset(("p",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    prep_stage = replace(_stage(StageKind.POINTWISE, (prep_method,)), sources=())
    rate_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(prep_stage, rate_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
    )
    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)

    first = backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    second = backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)

    assert first
    assert second
    assert backend.inline_windows == [((0, 1), ((0, -1), (1, -1)), ("t", "dt"))]
    assert backend.launched == []


def test_fused_stage_backend_hoists_source_visible_pointwise_methods_before_pair_window(
    monkeypatch,
):
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    monkeypatch.setenv("PYSPH_FUSED_HOIST_SOURCE_VISIBLE_PAIR_WINDOWS", "1")

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append(
                (
                    stage.kind,
                    tuple(method.equation_name for method in stage.methods),
                    info["stage_group"],
                    tuple(extra_args),
                )
            )

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="Diagnostic",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(("m",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="Acceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(("p",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(left_stage, right_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
    )
    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)

    first = backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    second = backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)

    assert first
    assert second
    assert backend.launched == [
        (StageKind.POINTWISE, ("PrepPressure",), (0, -1), ("t", "dt")),
        (
            StageKind.PAIR_RATE,
            ("Diagnostic", "Acceleration"),
            (0, -1),
            ("t", "dt"),
        ),
    ]


def test_fused_stage_backend_does_not_hoist_when_right_writes_left_source_read(
    monkeypatch,
):
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    monkeypatch.setenv("PYSPH_FUSED_HOIST_SOURCE_VISIBLE_PAIR_WINDOWS", "1")

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append(
                (
                    stage.kind,
                    tuple(method.equation_name for method in stage.methods),
                    info["stage_group"],
                )
            )

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="Diagnostic",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_init_method = MethodDeps(
        equation_name="Acceleration",
        method_kind=MethodKind.INITIALIZE,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_loop_method = MethodDeps(
        equation_name="Acceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(("p",)),
        dest_writes=frozenset(("av",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(
        StageKind.PAIR_RATE, (acceleration_init_method, acceleration_loop_method)
    )
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(left_stage, right_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
    )
    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)

    first = backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    second = backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)

    assert first
    assert second
    assert backend.launched == [
        (StageKind.PAIR_RATE, ("PrepPressure", "Diagnostic"), (0, -1)),
        (StageKind.PAIR_RATE, ("Acceleration", "Acceleration"), (1, -1)),
    ]


def test_hbucket_source_inline_pair_window_replaces_source_field_reads():
    from pysph.sph import fused_cuda_codegen as codegen

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="AddMass",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au",)),
        source_reads=frozenset(("m",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    precomputes = (
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        cubic_spline_gradient_precompute(np.int32(1)),
    )

    outline = codegen.generate_hbucket_source_inline_pair_window_outline_from_equations(
        "plan0",
        (left_stage, right_stage),
        (
            (
                PrepPressure(dest="fluid", sources=None),
                AddMass(dest="fluid", sources=["fluid"]),
            ),
            (PressureAcceleration(dest="fluid", sources=["fluid"]),),
        ),
        precomputes,
        (prep_method,),
        (),
    )

    assert "PrepPressure_loop(prep_pressure0, dst, d_p, d_rho);" in outline.source
    assert "float fused_inline_s_p[1];" in outline.source
    assert "float fused_inline_s_rho[1];" in outline.source
    assert "fused_inline_s_rho[0] = s_rho[src];" in outline.source
    assert (
        "PrepPressure_loop(prep_pressure0, 0, fused_inline_s_p, fused_inline_s_rho);"
        in outline.source
    )
    assert "PressureAcceleration_loop_source_inline" in outline.source
    assert "fused_inline_s_p_value" in outline.source
    assert "s_p[s_idx]" not in outline.source


def test_hbucket_source_inline_pair_window_accepts_pointwise_producer():
    from pysph.sph import fused_cuda_codegen as codegen

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    producer_stage = replace(
        _stage(StageKind.POINTWISE, (prep_method,)),
        sources=(),
    )
    consumer_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    precomputes = (
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        cubic_spline_gradient_precompute(np.int32(1)),
    )

    outline = codegen.generate_hbucket_source_inline_pair_window_outline_from_equations(
        "plan0",
        (producer_stage, consumer_stage),
        (
            (PrepPressure(dest="fluid", sources=None),),
            (PressureAcceleration(dest="fluid", sources=["fluid"]),),
        ),
        precomputes,
        (prep_method,),
        (),
    )

    assert "PrepPressure_loop(prep_pressure0, dst, d_p, d_rho);" in outline.source
    assert "float fused_inline_s_p[1];" in outline.source
    assert "float fused_inline_s_rho[1];" in outline.source
    assert "fused_inline_s_rho[0] = s_rho[src];" in outline.source
    assert (
        "PrepPressure_loop(prep_pressure0, 0, fused_inline_s_p, fused_inline_s_rho);"
        in outline.source
    )
    assert "PressureAcceleration_loop_source_inline" in outline.source
    assert "s_p[s_idx]" not in outline.source


def test_hbucket_source_inline_pair_window_reads_old_source_snapshots():
    from pysph.sph import fused_cuda_codegen as codegen

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="ReadSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha",)),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    precomputes = (
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        cubic_spline_gradient_precompute(np.int32(1)),
    )
    equations_by_stage = (
        (
            PrepPressure(dest="fluid", sources=None),
            ReadSourceAcceleration(dest="fluid", sources=["fluid"]),
        ),
        (PressureAcceleration(dest="fluid", sources=["fluid"]),),
    )

    outline = codegen.generate_hbucket_source_inline_pair_window_outline_from_equations(
        "plan0",
        (left_stage, right_stage),
        equations_by_stage,
        precomputes,
        (prep_method,),
        ("au",),
    )
    argument_names = codegen.hbucket_source_inline_pair_window_argument_names(
        (left_stage, right_stage),
        equations_by_stage,
        precomputes,
        (prep_method,),
        ("au",),
    )

    assert "GLOBAL_MEM float* s_old_au" in outline.source
    assert "s_old_au" in argument_names
    assert (
        "ReadSourceAcceleration_loop(read_source_acceleration0, dst, src, d_alpha, s_old_au);"
        in outline.source
    )
    assert (
        "ReadSourceAcceleration_loop(read_source_acceleration0, dst, src, d_alpha, s_au);"
        not in outline.source
    )
    assert "PressureAcceleration_loop_source_inline" in outline.source
    assert "s_p[s_idx]" not in outline.source


def test_hbucket_source_inline_pair_window_uses_local_slot_in_precompute():
    from pysph.sph import fused_cuda_codegen as codegen

    prep_method = MethodDeps(
        equation_name="SetDensity",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(("rho",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="AddMass",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au",)),
        source_reads=frozenset(("m",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="AccumulateRhoIJ",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au",)),
        source_reads=frozenset(("m", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("RHOIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    precomputes = (
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        cubic_spline_pair_precompute_for_symbols(np.int32(1), frozenset(("RHOIJ",))),
    )

    outline = codegen.generate_hbucket_source_inline_pair_window_outline_from_equations(
        "plan0",
        (left_stage, right_stage),
        (
            (
                SetDensity(dest="fluid", sources=None),
                AddMass(dest="fluid", sources=["fluid"]),
            ),
            (AccumulateRhoIJ(dest="fluid", sources=["fluid"]),),
        ),
        precomputes,
        (prep_method,),
        (),
    )

    assert "float fused_inline_s_rho[1];" in outline.source
    assert "RHOIJ = 0.5f * (d_rho[dst] + fused_inline_s_rho[0]);" in outline.source
    assert (
        "RHOIJ = 0.5f * (d_rho[dst] + fused_inline_s_rho_value);" not in outline.source
    )


def test_hbucket_source_inline_pair_window_initializes_read_written_locals():
    from pysph.sph import fused_cuda_codegen as codegen

    prep_method = MethodDeps(
        equation_name="PreserveThenSetDensity",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("rho", "rho_sum")),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="AddMass",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au",)),
        source_reads=frozenset(("m",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="AccumulateRhoIJ",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au",)),
        source_reads=frozenset(("m", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("RHOIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    precomputes = (
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        cubic_spline_pair_precompute_for_symbols(np.int32(1), frozenset(("RHOIJ",))),
    )

    outline = codegen.generate_hbucket_source_inline_pair_window_outline_from_equations(
        "plan0",
        (left_stage, right_stage),
        (
            (
                PreserveThenSetDensity(dest="fluid", sources=None),
                AddMass(dest="fluid", sources=["fluid"]),
            ),
            (AccumulateRhoIJ(dest="fluid", sources=["fluid"]),),
        ),
        precomputes,
        (prep_method,),
        (),
    )

    initialization = "fused_inline_s_rho[0] = s_rho[src];"
    call = "PreserveThenSetDensity_loop(preserve_then_set_density0, 0, fused_inline_s_rho, fused_inline_s_rho_sum);"
    assert initialization in outline.source
    assert call in outline.source
    assert outline.source.index(initialization) < outline.source.index(call)


def test_generated_stage_backend_launches_resident_hbucket_pair_window(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_RESIDENT_GRID_SYNC", "1")
    resident_launches = []

    def fake_launch(module, name, context, blocks, args, stream):
        resident_launches.append((module, name, context, blocks, args, stream))

    monkeypatch.setattr(
        backend_module,
        "_launch_cooperative_hbucket_pair_window_kernel",
        fake_launch,
        raising=False,
    )
    monkeypatch.setattr(
        backend_module,
        "_resident_hbucket_pair_window_extra_arg_names",
        lambda stages, equations, precomputes: ("t",),
        raising=False,
    )

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.stage_launches.append((stage.kind, info["stage_group"]))

        def _neighbor_context_for_stage(self, evaluator, info):
            return "context"

        def _cooperative_grid_block_count(self, context):
            return 7

        def _cooperative_module_for_outline(self, outline):
            self.outline = outline
            return "module"

        def _record_pair_traversal_launch(self, traversal):
            self.traversals.append(traversal)

        def _record_stage_timing(self, stage, traversal, timer):
            self.timings.append((stage.kind, traversal, timer))

        def _finish_launched_stage(self, stage):
            self.finished.append(stage.kind)

        def _old_source_snapshot_values(self, info, fields):
            assert fields == ("au",)
            return {"s_old_au": "old-au"}

    first_method = MethodDeps(
        equation_name="AddMass",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au",)),
        source_reads=frozenset(("m",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    second_method = MethodDeps(
        equation_name="AssignMassToAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(("m",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    first_stage = _stage(StageKind.PAIR_RATE, (first_method,))
    second_stage = _stage(StageKind.PAIR_RATE, (second_method,))
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(first_stage, second_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
        _gpu_structs={},
        object=SimpleNamespace(
            equation_groups=(
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(AddMass(dest="fluid", sources=["fluid"]),),
                ),
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(
                        AssignMassToAcceleration(dest="fluid", sources=["fluid"]),
                    ),
                ),
            ),
            kernel=SimpleNamespace(dim=1),
        ),
    )
    backend = object.__new__(RecordingBackend)
    backend.helper = helper
    backend.stage_group_by_plan_index = {
        0: (0, -1),
        1: (1, -1),
    }
    backend.stage_launches = []
    backend.traversals = []
    backend.timings = []
    backend.finished = []
    backend.launch_count = 0
    backend.stream = "stream"
    backend.cooperative_outlines = {}
    backend.cooperative_extra_arg_names = {}

    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), ("t", "dt"))

    assert backend.stage_launches == []
    assert resident_launches == [
        (
            "module",
            "fused_cuda_eval_fluid_resident_hbucket_pair_window",
            "context",
            7,
            ("t",),
            "stream",
        )
    ]
    assert backend.traversals == ["resident_hbucket"]
    assert backend.finished == [StageKind.PAIR_RATE, StageKind.PAIR_RATE]
    assert backend.launch_count == 1


def test_generated_stage_backend_launches_hbucket_source_inline_pair_window(
    monkeypatch,
):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    class CubicSpline:
        dim = 1

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_SOURCE_INLINE_PAIR_WINDOWS", "1")
    generated = []
    launched = []

    def fake_generate(
        plan_id,
        stages,
        equations_by_stage,
        precomputes,
        inline_methods,
        old_source_fields,
    ):
        generated.append(
            (
                plan_id,
                stages,
                equations_by_stage,
                precomputes,
                inline_methods,
                old_source_fields,
            )
        )
        return SimpleNamespace(
            name="fused_cuda_eval_fluid_hbucket_source_inline_pair_window",
            source="source",
        )

    monkeypatch.setattr(
        backend_module,
        "generate_hbucket_source_inline_pair_window_outline_from_equations",
        fake_generate,
    )
    monkeypatch.setattr(
        backend_module,
        "_hbucket_source_inline_pair_window_extra_arg_names",
        lambda stages, equations, precomputes, inline_methods, old_source_fields: (
            "prep_pressure0",
        ),
        raising=False,
    )

    def fail_cooperative_launcher(module, name, context, grid_blocks, args, stream):
        raise AssertionError("source-inline pair windows use a full particle grid")

    monkeypatch.setattr(
        backend_module,
        "launch_hbucket_pair_kernel_with_context",
        lambda module, name, context, args: launched.append(
            (module, name, context, args)
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "_launch_cooperative_hbucket_pair_window_kernel",
        fail_cooperative_launcher,
    )

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def _neighbor_context_for_stage(self, evaluator, info):
            return "context"

        def _module_for_stage(self, outline, info, stage):
            self.outline = outline
            self.module_stage = stage
            return "module"

        def _cooperative_module_for_outline(self, outline):
            raise AssertionError("source-inline pair windows do not need grid sync")

        def _cooperative_grid_block_count(self, context):
            raise AssertionError("source-inline pair windows use a full particle grid")

        def _record_pair_traversal_launch(self, traversal):
            self.traversals.append(traversal)

        def _record_stage_timing(self, stage, traversal, timer):
            self.timings.append((stage.kind, traversal, timer))

        def _finish_launched_stage(self, stage):
            self.finished.append(stage.kind)

        def _old_source_snapshot_values(self, info, fields):
            assert fields == ()
            return {}

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="AddMass",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au",)),
        source_reads=frozenset(("m",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(left_stage, right_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
        _gpu_structs={"prep_pressure0": "prep-struct"},
        object=SimpleNamespace(
            equation_groups=(
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(
                        PrepPressure(dest="fluid", sources=None),
                        AddMass(dest="fluid", sources=["fluid"]),
                    ),
                ),
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(PressureAcceleration(dest="fluid", sources=["fluid"]),),
                ),
            ),
            kernel=CubicSpline(),
        ),
    )
    backend = object.__new__(RecordingBackend)
    backend.helper = helper
    backend.stage_group_by_plan_index = {0: (0, -1), 1: (1, -1)}
    backend.cooperative_outlines = {}
    backend.cooperative_extra_arg_names = {}
    backend.traversals = []
    backend.timings = []
    backend.finished = []
    backend.launch_count = 0
    backend.stream = "stream"

    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), (1.0, 0.1))

    assert len(generated) == 1
    assert tuple(method.equation_name for method in generated[0][4]) == (
        "PrepPressure",
    )
    assert generated[0][5] == ()
    assert launched == [
        (
            "module",
            "fused_cuda_eval_fluid_hbucket_source_inline_pair_window",
            "context",
            ("prep-struct",),
        )
    ]
    assert backend.traversals == ["hbucket_source_inline"]
    assert backend.finished == [StageKind.PAIR_RATE, StageKind.PAIR_RATE]
    assert backend.launch_count == 1


def test_generated_stage_backend_splits_blocked_source_inline_pair_window(
    monkeypatch,
):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    class CubicSpline:
        dim = 1

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_SOURCE_INLINE_PAIR_WINDOWS", "1")
    monkeypatch.setenv("PYSPH_FUSED_SOURCE_INLINE_SPLIT_WINDOWS", "1")
    generated = []
    launched = []

    def fake_generate(
        plan_id,
        stages,
        equations_by_stage,
        precomputes,
        inline_methods,
        old_source_fields,
    ):
        generated.append(
            (
                plan_id,
                stages,
                equations_by_stage,
                precomputes,
                inline_methods,
                old_source_fields,
            )
        )
        return SimpleNamespace(
            name="fused_cuda_eval_fluid_hbucket_source_inline_pair_window",
            source="source",
        )

    monkeypatch.setattr(
        backend_module,
        "generate_hbucket_source_inline_pair_window_outline_from_equations",
        fake_generate,
    )
    monkeypatch.setattr(
        backend_module,
        "_hbucket_source_inline_pair_window_extra_arg_names",
        lambda stages, equations, precomputes, inline_methods, old_source_fields: (
            "prep_pressure0",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        backend_module,
        "launch_hbucket_pair_kernel_with_context",
        lambda module, name, context, args: launched.append(
            (module, name, context, args)
        ),
    )

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.stage_launches.append(
                (
                    stage.kind,
                    tuple(method.equation_name for method in stage.methods),
                    info["stage_group"],
                    tuple(extra_args),
                )
            )

        def _neighbor_context_for_stage(self, evaluator, info):
            return "context"

        def _module_for_stage(self, outline, info, stage):
            self.outline = outline
            self.module_stage = stage
            return "module"

        def _cooperative_module_for_outline(self, outline):
            raise AssertionError("split source-inline does not need grid sync")

        def _record_pair_traversal_launch(self, traversal):
            self.traversals.append(traversal)

        def _record_stage_timing(self, stage, traversal, timer):
            self.timings.append((stage.kind, traversal, timer))

        def _finish_launched_stage(self, stage):
            self.finished.append(stage.kind)

        def _old_source_snapshot_values(self, info, fields):
            raise AssertionError("split source-inline should not snapshot old source")

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="ReadSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha",)),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(left_stage, right_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
        _gpu_structs={"prep_pressure0": "prep-struct"},
        object=SimpleNamespace(
            equation_groups=(
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(
                        PrepPressure(dest="fluid", sources=None),
                        ReadSourceAcceleration(dest="fluid", sources=["fluid"]),
                    ),
                ),
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(PressureAcceleration(dest="fluid", sources=["fluid"]),),
                ),
            ),
            kernel=CubicSpline(),
        ),
    )
    backend = object.__new__(RecordingBackend)
    backend.helper = helper
    backend.stage_group_by_plan_index = {0: (0, -1), 1: (1, -1)}
    backend.cooperative_outlines = {}
    backend.cooperative_extra_arg_names = {}
    backend.stage_launches = []
    backend.traversals = []
    backend.timings = []
    backend.finished = []
    backend.launch_count = 0
    backend.stream = "stream"

    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), (1.0, 0.1))

    assert backend.stage_launches == [
        (
            StageKind.PAIR_RATE,
            ("ReadSourceAcceleration",),
            (0, -1),
            (1.0, 0.1),
        )
    ]
    assert len(generated) == 1
    assert [stage.kind for stage in generated[0][1]] == [
        StageKind.POINTWISE,
        StageKind.PAIR_RATE,
    ]
    assert tuple(method.equation_name for method in generated[0][4]) == (
        "PrepPressure",
    )
    assert generated[0][5] == ()
    assert launched == [
        (
            "module",
            "fused_cuda_eval_fluid_hbucket_source_inline_pair_window",
            "context",
            ("prep-struct",),
        )
    ]
    assert backend.traversals == ["hbucket_source_inline"]


def test_source_inline_split_requires_explicit_experimental_flag(monkeypatch):
    from pysph.sph.fused_cuda_stage_backend import source_inline_pair_window_status

    monkeypatch.delenv("PYSPH_FUSED_SOURCE_INLINE_SPLIT_WINDOWS", raising=False)

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="ReadSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha",)),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p")),
        source_reads=frozenset(("p",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    plan = CudaStagePlan(
        stages=(
            _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method)),
            _stage(StageKind.PAIR_RATE, (acceleration_method,)),
        ),
        strict=True,
    )

    status = source_inline_pair_window_status(plan, (0, 1))

    assert status["status"] == "blocked_old_source_snapshots_disabled"


def test_generated_stage_backend_launches_old_state_single_pass_pair_window(
    monkeypatch,
):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    class CubicSpline:
        dim = 1

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_OLD_STATE_SINGLE_PASS_WINDOWS", "1")
    generated = []
    launched = []

    def fake_generate(
        plan_id,
        stages,
        equations_by_stage,
        precompute,
        old_state_methods,
        old_state_fields,
    ):
        generated.append(
            (
                plan_id,
                stages,
                equations_by_stage,
                precompute,
                old_state_methods,
                old_state_fields,
            )
        )
        return SimpleNamespace(
            name="fused_cuda_eval_fluid_hbucket_old_state_pair_stage",
            source="source",
        )

    monkeypatch.setattr(
        backend_module,
        "generate_hbucket_old_state_pair_stage_outline_from_equations",
        fake_generate,
        raising=False,
    )
    monkeypatch.setattr(
        backend_module,
        "_hbucket_old_state_pair_stage_extra_arg_names",
        lambda stages, equations, precompute, old_methods, old_fields: (
            "d_old_au",
            "s_old_au",
            "prep_pressure0",
        ),
        raising=False,
    )

    def fake_launch(module, name, context, args):
        launched.append((module, name, context, args))
        return backend_module.PairLaunchConfig(
            traversal="hbucket",
            n=128,
            block_size=128,
            grid_x=1,
        )

    monkeypatch.setattr(
        backend_module,
        "launch_hbucket_pair_kernel_with_context",
        fake_launch,
    )

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.stage_launches.append(
                (
                    stage.kind,
                    tuple(method.equation_name for method in stage.methods),
                    info["stage_group"],
                    tuple(extra_args),
                )
            )

        def _neighbor_context_for_stage(self, evaluator, info):
            return "context"

        def _module_for_stage(self, outline, info, stage):
            self.outline = outline
            return "module"

        def _record_pair_traversal_launch(self, traversal):
            self.traversals.append(traversal)

        def _record_pair_launch_config(self, launch_config):
            self.launch_configs.append(launch_config)

        def _record_stage_timing(self, stage, traversal, timer):
            self.timings.append((stage.kind, traversal, timer))

        def _finish_launched_stage(self, stage):
            self.finished.append(stage.kind)

        def _old_state_snapshot_values(self, info, fields):
            assert fields == ("au",)
            return {"d_old_au": "old-au", "s_old_au": "old-au"}

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_init = MethodDeps(
        equation_name="ReadDestAndSourceAcceleration",
        method_kind=MethodKind.INITIALIZE,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_loop = MethodDeps(
        equation_name="ReadDestAndSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha", "au")),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_post = MethodDeps(
        equation_name="ReadDestAndSourceAcceleration",
        method_kind=MethodKind.POST_LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("beta",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(
        StageKind.PAIR_RATE,
        (prep_method, diagnostic_init, diagnostic_loop, diagnostic_post),
    )
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(left_stage, right_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
        _gpu_structs={"prep_pressure0": "prep-struct"},
        object=SimpleNamespace(
            equation_groups=(
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(
                        PrepPressure(dest="fluid", sources=None),
                        ReadDestAndSourceAcceleration(dest="fluid", sources=["fluid"]),
                    ),
                ),
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(PressureAcceleration(dest="fluid", sources=["fluid"]),),
                ),
            ),
            kernel=CubicSpline(),
        ),
    )
    backend = object.__new__(RecordingBackend)
    backend.helper = helper
    backend.stage_group_by_plan_index = {0: (0, -1), 1: (1, -1)}
    backend.stage_launches = []
    backend.outlines = {}
    backend.cooperative_outlines = {}
    backend.cooperative_extra_arg_names = {}
    backend.traversals = []
    backend.launch_configs = []
    backend.timings = []
    backend.finished = []
    backend.launch_count = 0
    backend.stream = "stream"

    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), (1.0, 0.1))

    assert backend.stage_launches == [
        (
            StageKind.POINTWISE,
            ("PrepPressure",),
            (0, -1),
            (1.0, 0.1),
        )
    ]
    assert len(generated) == 1
    assert tuple(method.method_kind for method in generated[0][1][0].methods) == (
        MethodKind.INITIALIZE,
        MethodKind.LOOP,
        MethodKind.POST_LOOP,
    )
    assert tuple(method.equation_name for method in generated[0][4]) == (
        "ReadDestAndSourceAcceleration",
    )
    assert generated[0][5] == ("au",)
    assert launched == [
        (
            "module",
            "fused_cuda_eval_fluid_hbucket_old_state_pair_stage",
            "context",
            ("old-au", "old-au", "prep-struct"),
        )
    ]
    assert backend.traversals == ["hbucket_old_state"]
    assert backend.launch_configs[0].traversal == "hbucket_old_state"
    assert backend.finished == [StageKind.PAIR_RATE, StageKind.PAIR_RATE]
    assert backend.launch_count == 1


def test_source_inline_split_keeps_pointwise_prelude_after_old_source_prefix():
    init_method = MethodDeps(
        equation_name="InitDiagnostic",
        method_kind=MethodKind.INITIALIZE,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(("scratch",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="ReadSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("scratch",)),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    post_method = MethodDeps(
        equation_name="PostPressureDiagnostic",
        method_kind=MethodKind.POST_LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha", "p")),
        source_reads=frozenset(),
        dest_writes=frozenset(("beta",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(
        StageKind.PAIR_RATE,
        (init_method, prep_method, diagnostic_method, post_method),
    )
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    plan = CudaStagePlan(stages=(left_stage, right_stage), strict=True)

    prefix_stage, inline_stage, returned_right_stage = (
        _split_source_inline_pair_window_stages(plan, (0, 1))
    )

    assert tuple(method.equation_name for method in prefix_stage.methods) == (
        "InitDiagnostic",
        "ReadSourceAcceleration",
    )
    assert inline_stage.kind is StageKind.POINTWISE
    assert tuple(method.equation_name for method in inline_stage.methods) == (
        "PrepPressure",
        "PostPressureDiagnostic",
    )
    assert returned_right_stage is right_stage


def test_generated_stage_backend_allows_source_inline_with_old_source_snapshot(
    monkeypatch,
):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    class CubicSpline:
        dim = 1

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_SOURCE_INLINE_PAIR_WINDOWS", "1")
    monkeypatch.setenv("PYSPH_FUSED_SOURCE_INLINE_OLD_SOURCE_SNAPSHOTS", "1")
    generated = []
    launched = []

    def fake_generate(
        plan_id,
        stages,
        equations_by_stage,
        precomputes,
        inline_methods,
        old_source_fields,
    ):
        generated.append(
            (
                plan_id,
                stages,
                equations_by_stage,
                precomputes,
                inline_methods,
                old_source_fields,
            )
        )
        return SimpleNamespace(
            name="fused_cuda_eval_fluid_hbucket_source_inline_pair_window",
            source="source",
        )

    monkeypatch.setattr(
        backend_module,
        "generate_hbucket_source_inline_pair_window_outline_from_equations",
        fake_generate,
    )
    monkeypatch.setattr(
        backend_module,
        "_hbucket_source_inline_pair_window_extra_arg_names",
        lambda stages, equations, precomputes, inline_methods, old_source_fields: (
            "prep_pressure0",
        ),
        raising=False,
    )

    def fail_cooperative_launcher(module, name, context, grid_blocks, args, stream):
        raise AssertionError("source-inline pair windows use a full particle grid")

    monkeypatch.setattr(
        backend_module,
        "launch_hbucket_pair_kernel_with_context",
        lambda module, name, context, args: launched.append(
            (module, name, context, args)
        ),
    )
    monkeypatch.setattr(
        backend_module,
        "_launch_cooperative_hbucket_pair_window_kernel",
        fail_cooperative_launcher,
    )

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def _neighbor_context_for_stage(self, evaluator, info):
            return "context"

        def _module_for_stage(self, outline, info, stage):
            self.outline = outline
            self.module_stage = stage
            return "module"

        def _cooperative_module_for_outline(self, outline):
            raise AssertionError("source-inline pair windows do not need grid sync")

        def _cooperative_grid_block_count(self, context):
            raise AssertionError("source-inline pair windows use a full particle grid")

        def _record_pair_traversal_launch(self, traversal):
            self.traversals.append(traversal)

        def _record_stage_timing(self, stage, traversal, timer):
            self.timings.append((stage.kind, traversal, timer))

        def _finish_launched_stage(self, stage):
            self.finished.append(stage.kind)

        def _old_source_snapshot_values(self, info, fields):
            assert fields == ("au",)
            return {"s_old_au": "old-au"}

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="ReadSourceAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha",)),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="PressureAcceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p", "rho")),
        source_reads=frozenset(("m", "p", "rho")),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(("DWIJ",)),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    left_stage = _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method))
    right_stage = _stage(StageKind.PAIR_RATE, (acceleration_method,))
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(left_stage, right_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
        _gpu_structs={"prep_pressure0": "prep-struct"},
        object=SimpleNamespace(
            equation_groups=(
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(
                        PrepPressure(dest="fluid", sources=None),
                        ReadSourceAcceleration(dest="fluid", sources=["fluid"]),
                    ),
                ),
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(PressureAcceleration(dest="fluid", sources=["fluid"]),),
                ),
            ),
            kernel=CubicSpline(),
        ),
    )
    backend = object.__new__(RecordingBackend)
    backend.helper = helper
    backend.stage_group_by_plan_index = {0: (0, -1), 1: (1, -1)}
    backend.cooperative_outlines = {}
    backend.cooperative_extra_arg_names = {}
    backend.traversals = []
    backend.timings = []
    backend.finished = []
    backend.launch_count = 0
    backend.stream = "stream"

    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), (1.0, 0.1))

    assert len(generated) == 1
    assert generated[0][5] == ("au",)
    assert launched == [
        (
            "module",
            "fused_cuda_eval_fluid_hbucket_source_inline_pair_window",
            "context",
            ("prep-struct",),
        )
    ]
    assert backend.traversals == ["hbucket_source_inline"]
    assert backend.finished == [StageKind.PAIR_RATE, StageKind.PAIR_RATE]
    assert backend.launch_count == 1


def test_hbucket_source_inline_extra_args_use_old_source_snapshots(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    monkeypatch.setattr(
        backend_module,
        "_hbucket_source_inline_pair_window_extra_arg_names",
        lambda stages, equations, precomputes, inline_methods, old_source_fields: (
            "prep_pressure0",
            "s_old_au",
        ),
        raising=False,
    )

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def _old_source_snapshot_values(self, info, fields):
            assert fields == ("au",)
            assert info == {"src": "fluid-src"}
            return {"s_old_au": "old-au"}

    backend = object.__new__(RecordingBackend)
    backend.helper = SimpleNamespace(_gpu_structs={"prep_pressure0": "prep-struct"})
    backend.cooperative_extra_arg_names = {}
    stages = (
        _stage(StageKind.PAIR_RATE, (_deps("PrepPressure", MethodKind.LOOP),)),
        _stage(StageKind.PAIR_RATE, (_deps("PressureAcceleration", MethodKind.LOOP),)),
    )
    precomputes = (
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
    )

    args = backend._hbucket_source_inline_pair_window_extra_args(
        stages=stages,
        equations_by_stage=((), ()),
        infos=({"src": "fluid-src"}, {"src": "fluid-src"}),
        extra_args=(1.0, 0.1),
        precomputes=precomputes,
        inline_methods=(_deps("PrepPressure", MethodKind.LOOP),),
        old_source_fields=("au",),
    )

    assert args == ("prep-struct", "old-au")


def test_source_inline_pair_window_blockers_report_old_source_read_conflict():
    from pysph.sph.fused_cuda_stage_backend import (
        source_inline_pair_window_blockers,
    )

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="Diagnostic",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha",)),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="Acceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p")),
        source_reads=frozenset(("p",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    plan = CudaStagePlan(
        stages=(
            _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method)),
            _stage(StageKind.PAIR_RATE, (acceleration_method,)),
        ),
        strict=True,
    )

    blockers = source_inline_pair_window_blockers(plan, (0, 1))

    assert blockers == (
        {
            "reason": "remaining_left_source_reads_right_writes",
            "fields": ("au",),
        },
    )


def test_source_inline_pair_window_blockers_report_non_inline_producer():
    from pysph.sph.fused_cuda_stage_backend import (
        source_inline_pair_window_blockers,
    )

    producer = MethodDeps(
        equation_name="Producer",
        method_kind=MethodKind.POST_LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    consumer = MethodDeps(
        equation_name="Consumer",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(),
        source_reads=frozenset(("p",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    plan = CudaStagePlan(
        stages=(
            _stage(StageKind.PAIR_RATE, (producer,)),
            _stage(StageKind.PAIR_RATE, (consumer,)),
        ),
        strict=True,
    )

    blockers = source_inline_pair_window_blockers(plan, (0, 1))

    assert blockers == (
        {
            "reason": "producer_not_source_inline",
            "fields": ("p",),
        },
    )


def test_source_inline_pair_window_status_reports_old_source_snapshot_window_details(
    monkeypatch,
):
    from pysph.sph.fused_cuda_stage_backend import source_inline_pair_window_status

    monkeypatch.setenv("PYSPH_FUSED_SOURCE_INLINE_OLD_SOURCE_SNAPSHOTS", "1")

    prep_method = MethodDeps(
        equation_name="PrepPressure",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=(),
        dest_reads=frozenset(("rho",)),
        source_reads=frozenset(),
        dest_writes=frozenset(("p",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    diagnostic_method = MethodDeps(
        equation_name="Diagnostic",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("alpha",)),
        source_reads=frozenset(("au",)),
        dest_writes=frozenset(("alpha",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    acceleration_method = MethodDeps(
        equation_name="Acceleration",
        method_kind=MethodKind.LOOP,
        dest="fluid",
        sources=("fluid",),
        dest_reads=frozenset(("au", "p")),
        source_reads=frozenset(("p",)),
        dest_writes=frozenset(("au",)),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )
    plan = CudaStagePlan(
        stages=(
            _stage(StageKind.PAIR_RATE, (prep_method, diagnostic_method)),
            _stage(StageKind.PAIR_RATE, (acceleration_method,)),
        ),
        strict=True,
    )

    status = source_inline_pair_window_status(plan, (0, 1))

    assert status == {
        "stage_indices": (0, 1),
        "status": "launchable_with_old_source_snapshots",
        "source_visible_fields": ("p",),
        "inline_methods": ("PrepPressure.loop",),
        "blockers": (
            {
                "reason": "remaining_left_source_reads_right_writes",
                "fields": ("au",),
            },
        ),
    }


def test_generated_stage_backend_caches_resident_hbucket_pair_window_outline(
    monkeypatch,
):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_RESIDENT_GRID_SYNC", "1")
    generated = []

    def fake_generate(plan_id, stages, equations_by_stage, precomputes):
        generated.append((plan_id, stages, equations_by_stage, precomputes))
        return SimpleNamespace(
            name="fused_cuda_eval_fluid_resident_hbucket_pair_window",
            source=f"source-{len(generated)}",
        )

    monkeypatch.setattr(
        backend_module,
        "generate_resident_hbucket_pair_window_outline_from_equations",
        fake_generate,
    )
    monkeypatch.setattr(
        backend_module,
        "_launch_cooperative_hbucket_pair_window_kernel",
        lambda module, name, context, blocks, args, stream: None,
    )
    monkeypatch.setattr(
        backend_module,
        "_resident_hbucket_pair_window_extra_arg_names",
        lambda stages, equations, precomputes: (),
    )

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def _neighbor_context_for_stage(self, evaluator, info):
            return "context"

        def _cooperative_grid_block_count(self, context):
            return 7

        def _cooperative_module_for_outline(self, outline):
            return f"module:{outline.source}"

        def _record_pair_traversal_launch(self, traversal):
            pass

        def _record_stage_timing(self, stage, traversal, timer):
            pass

        def _finish_launched_stage(self, stage):
            pass

    first_stage = _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),))
    second_stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AssignMassToAcceleration", MethodKind.LOOP),),
    )
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(first_stage, second_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1)},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
        object=SimpleNamespace(
            equation_groups=(
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(AddMass(dest="fluid", sources=["fluid"]),),
                ),
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(
                        AssignMassToAcceleration(dest="fluid", sources=["fluid"]),
                    ),
                ),
            ),
            kernel=SimpleNamespace(dim=1),
        ),
    )
    backend = object.__new__(RecordingBackend)
    backend.helper = helper
    backend.stage_group_by_plan_index = {0: (0, -1), 1: (1, -1)}
    backend.cooperative_outlines = {}
    backend.cooperative_extra_arg_names = {}
    backend.launch_count = 0
    backend.stream = "stream"

    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), ("t", "dt"))
    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), ("t", "dt"))

    assert len(generated) == 1


def test_generated_stage_backend_caches_resident_hbucket_pair_window_arg_names(
    monkeypatch,
):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_RESIDENT_GRID_SYNC", "1")
    name_calls = []

    def fake_arg_names(stages, equations_by_stage, precomputes):
        name_calls.append((stages, equations_by_stage, precomputes))
        return ("t",)

    monkeypatch.setattr(
        backend_module,
        "_resident_hbucket_pair_window_extra_arg_names",
        fake_arg_names,
        raising=False,
    )
    monkeypatch.setattr(
        backend_module,
        "_launch_cooperative_hbucket_pair_window_kernel",
        lambda module, name, context, blocks, args, stream: None,
    )
    monkeypatch.setattr(
        backend_module,
        "generate_resident_hbucket_pair_window_outline_from_equations",
        lambda plan_id, stages, equations_by_stage, precomputes: SimpleNamespace(
            name="fused_cuda_eval_fluid_resident_hbucket_pair_window",
            source="source",
        ),
    )

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def _neighbor_context_for_stage(self, evaluator, info):
            return SimpleNamespace(n=8)

        def _cooperative_grid_block_count(self, context):
            return 1

        def _cooperative_module_for_outline(self, outline):
            return "module"

        def _record_pair_traversal_launch(self, traversal):
            pass

        def _record_stage_timing(self, stage, traversal, timer):
            pass

        def _finish_launched_stage(self, stage):
            pass

    first_stage = _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),))
    second_stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("AssignMassToAcceleration", MethodKind.LOOP),),
    )
    helper = SimpleNamespace(
        cuda_stage_plan=CudaStagePlan(
            stages=(first_stage, second_stage),
            strict=True,
        ),
        calls=(
            {"type": "kernel", "stage_group": (0, -1), "dest": SimpleNamespace()},
            {"type": "kernel", "stage_group": (1, -1), "dest": SimpleNamespace()},
        ),
        _gpu_structs={"": "equation-struct"},
        object=SimpleNamespace(
            equation_groups=(
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(AddMass(dest="fluid", sources=["fluid"]),),
                ),
                SimpleNamespace(
                    has_subgroups=False,
                    equations=(
                        AssignMassToAcceleration(dest="fluid", sources=["fluid"]),
                    ),
                ),
            ),
            kernel=SimpleNamespace(dim=1),
        ),
    )
    gpu = SimpleNamespace(
        au=SimpleNamespace(dev="d_au"),
        m=SimpleNamespace(dev="d_m"),
    )
    info0 = {
        "type": "kernel",
        "stage_group": (0, -1),
        "dest": SimpleNamespace(gpu=gpu),
        "src": SimpleNamespace(gpu=gpu),
    }
    info1 = {
        "type": "kernel",
        "stage_group": (1, -1),
        "dest": SimpleNamespace(gpu=gpu),
        "src": SimpleNamespace(gpu=gpu),
    }
    helper.calls = (info0, info1)
    backend = object.__new__(RecordingBackend)
    backend.helper = helper
    backend.stage_group_by_plan_index = {0: (0, -1), 1: (1, -1)}
    backend.cooperative_outlines = {}
    backend.cooperative_extra_arg_names = {}
    backend.launch_count = 0
    backend.stream = "stream"

    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), (1.0, 0.1))
    backend._launch_cooperative_grid_sync_window("evaluator", (0, 1), (2.0, 0.1))

    assert len(name_calls) == 1


def test_resident_hbucket_pair_window_extra_args_match_codegen_order():
    from pysph.sph.fused_cuda_stage_backend import (
        _resident_hbucket_pair_window_extra_args,
    )

    first = AddMass(dest="fluid", sources=["fluid"])
    first.var_name = "add_mass0"
    second = PressureAcceleration(dest="fluid", sources=["fluid"])
    second.var_name = "pressure0"
    first_stage = _stage(
        StageKind.PAIR_RATE,
        (
            MethodDeps(
                equation_name="AddMass",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(("au",)),
                source_reads=frozenset(("m",)),
                dest_writes=frozenset(("au",)),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )
    second_stage = _stage(
        StageKind.PAIR_RATE,
        (
            MethodDeps(
                equation_name="PressureAcceleration",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(("au", "p", "rho")),
                source_reads=frozenset(("p", "rho", "m")),
                dest_writes=frozenset(("au",)),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(("DWIJ", "RHOIJ", "VIJ")),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )
    gpu = SimpleNamespace(
        au=SimpleNamespace(dev="d_au"),
        m=SimpleNamespace(dev="d_m"),
        p=SimpleNamespace(dev="d_p"),
        rho=SimpleNamespace(dev="d_rho"),
        x=SimpleNamespace(dev="d_x"),
        y=SimpleNamespace(dev="d_y"),
        z=SimpleNamespace(dev="d_z"),
        h=SimpleNamespace(dev="d_h"),
        u=SimpleNamespace(dev="d_u"),
        v=SimpleNamespace(dev="d_v"),
        w=SimpleNamespace(dev="d_w"),
    )
    info = {"dest": SimpleNamespace(gpu=gpu), "src": SimpleNamespace(gpu=gpu)}
    helper = SimpleNamespace(
        _gpu_structs={"add_mass0": "add-struct", "pressure0": "pressure-struct"}
    )

    args = _resident_hbucket_pair_window_extra_args(
        helper,
        (first_stage, second_stage),
        ((first,), (second,)),
        (info, info),
        (1.0, 0.1),
        (
            CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
            cubic_spline_pair_precompute_for_symbols(
                np.int32(1), frozenset(("DWIJ", "RHOIJ", "VIJ"))
            ),
        ),
    )

    assert args[:8] == (
        "d_u",
        "d_u",
        "d_v",
        "d_v",
        "d_w",
        "d_w",
        "d_rho",
        "d_rho",
    )
    assert args[8:] == (
        "add-struct",
        "d_au",
        "d_m",
        "pressure-struct",
        "d_p",
        "d_p",
    )


def test_ctypes_kernel_argument_cache_reuses_storage_and_updates_values():
    from pysph.sph.fused_cuda_stage_backend import _CtypesKernelArgumentCache

    cache = _CtypesKernelArgumentCache()

    first_args, first_storage = cache.arguments(
        (np.uintp(11), np.int32(7), np.float32(1.5))
    )
    second_args, second_storage = cache.arguments(
        (np.uintp(22), np.int32(9), np.float32(2.5))
    )

    assert second_args is first_args
    assert second_storage is first_storage
    assert len(second_storage) == 3
    assert second_storage[0].value == 22
    assert second_storage[1].value == 9
    assert second_storage[2].value == np.float32(2.5)


def test_cooperative_grid_block_count_is_capped_by_particle_grid(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module

    monkeypatch.setattr(backend_module, "_resident_grid_block_count", lambda: 170)
    monkeypatch.setattr(backend_module, "_pair_block_size_for_count", lambda n: 128)

    assert backend_module._cooperative_grid_block_count_for_context(13824) == 108
    assert backend_module._cooperative_grid_block_count_for_context(1000000) == 170


def test_resident_grid_block_count_can_use_multiple_blocks_per_sm(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module

    class Device:
        def get_attribute(self, attribute):
            assert attribute == "multiprocessor_count"
            return 170

    class Cuda:
        class Context:
            @staticmethod
            def get_device():
                return Device()

        class device_attribute:
            MULTIPROCESSOR_COUNT = "multiprocessor_count"

    monkeypatch.setitem(__import__("sys").modules, "pycuda.driver", Cuda)
    monkeypatch.setenv("PYSPH_FUSED_RESIDENT_GRID_BLOCKS_PER_SM", "2")

    assert backend_module._resident_grid_block_count() == 340


def test_auto_resident_grid_sync_only_allows_underfilled_particle_grids(
    monkeypatch,
):
    from pysph.sph import fused_cuda_stage_backend as backend_module

    monkeypatch.setenv("PYSPH_FUSED_RESIDENT_GRID_SYNC", "auto")
    monkeypatch.setattr(backend_module, "_resident_grid_block_count", lambda: 170)
    monkeypatch.setattr(backend_module, "_pair_block_size_for_count", lambda n: 128)

    assert backend_module._resident_grid_sync_context_allowed(SimpleNamespace(n=13824))
    assert not backend_module._resident_grid_sync_context_allowed(
        SimpleNamespace(n=71688)
    )


def test_fused_stage_backend_runs_device_convergence_super_stage_and_skips_legacy_calls():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []
            self.controls = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"]))

        def _launch_device_convergence_super_stage(
            self, evaluator, info, extra_args, t, dt
        ):
            self.controls.append((info["type"], t, dt))

    density_stage = _stage(
        StageKind.PAIR_DENSITY,
        (_deps("DensityIteration", MethodKind.LOOP),),
    )
    convergence_stage = _stage(
        StageKind.DEVICE_CONVERGENCE,
        (_deps("DensityIteration", MethodKind.POST_LOOP),),
    )
    final_stage = _stage(
        StageKind.PAIR_DENSITY,
        (_deps("DensityFinalization", MethodKind.LOOP),),
    )
    helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(
            stages=(density_stage, convergence_stage, final_stage)
        ),
        calls=(
            {"type": "start_iteration", "group": SimpleNamespace(max_iterations=3)},
            {
                "type": "kernel",
                "stage_group": (0, -1),
                "stage_method_kind": "initialize",
            },
            {"type": "kernel", "stage_group": (0, -1), "stage_method_kind": "loop"},
            {
                "type": "kernel",
                "stage_group": (0, -1),
                "stage_method_kind": "post_loop",
            },
            {"type": "stop_iteration"},
            {"type": "kernel", "stage_group": (1, -1)},
        ),
    )

    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[0], (), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[1], (), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[2], (), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[3], (), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[4], (), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[5], (), 0.0, 0.1)

    assert backend.launched == [
        (StageKind.PAIR_DENSITY, (1, -1)),
    ]
    assert backend.controls == [
        ("start_iteration", 0.0, 0.1),
    ]


def test_fused_stage_backend_default_device_convergence_super_stage_replays_fixed_iterations():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"], extra_args))

    density_stage = _stage(
        StageKind.PAIR_DENSITY,
        (_deps("DensityIteration", MethodKind.LOOP),),
    )
    convergence_stage = _stage(
        StageKind.DEVICE_CONVERGENCE,
        (_deps("DensityIteration", MethodKind.POST_LOOP),),
    )
    helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(stages=(density_stage, convergence_stage)),
        calls=(
            {
                "type": "start_iteration",
                "group": SimpleNamespace(
                    min_iterations=1,
                    max_iterations=3,
                    update_nnps=False,
                ),
            },
            {"type": "kernel", "stage_group": (0, -1), "stage_method_kind": "loop"},
            {"type": "stop_iteration"},
        ),
    )

    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[2], ("t", "dt"), 0.0, 0.1)

    assert backend.launched == [
        (StageKind.PAIR_DENSITY, (0, -1), ("t", "dt")),
        (StageKind.PAIR_DENSITY, (0, -1), ("t", "dt")),
        (StageKind.PAIR_DENSITY, (0, -1), ("t", "dt")),
    ]


def test_fused_stage_backend_device_convergence_rebuilds_until_converged():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []
            self.rebuilds = []
            self.convergence = iter((False, True))

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"], extra_args))

        def _device_convergence_has_converged(self, info):
            return next(self.convergence)

        def _update_device_convergence_nnps(self, evaluator, info):
            self.rebuilds.append(info["group"])

    density_stage = _stage(
        StageKind.PAIR_DENSITY,
        (_deps("DensityIteration", MethodKind.LOOP),),
    )
    convergence_stage = _stage(
        StageKind.DEVICE_CONVERGENCE,
        (_deps("DensityIteration", MethodKind.POST_LOOP),),
    )
    group = SimpleNamespace(min_iterations=1, max_iterations=3, update_nnps=True)
    helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(stages=(density_stage, convergence_stage)),
        calls=(
            {"type": "start_iteration", "group": group},
            {"type": "kernel", "stage_group": (0, -1), "stage_method_kind": "loop"},
            {"type": "stop_iteration"},
        ),
    )

    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[2], ("t", "dt"), 0.0, 0.1)

    assert backend.launched == [
        (StageKind.PAIR_DENSITY, (0, -1), ("t", "dt")),
        (StageKind.PAIR_DENSITY, (0, -1), ("t", "dt")),
    ]
    assert backend.rebuilds == [group]
    assert backend.device_convergence_iteration_counts == [2]
    assert backend.device_convergence_rebuild_count == 1


def test_fused_stage_backend_uses_convergence_policy_child_stage_indices():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"], extra_args))

    density_stage = _stage(
        StageKind.PAIR_DENSITY,
        (_deps("DensityIteration", MethodKind.LOOP),),
    )
    rate_stage = _stage(
        StageKind.PAIR_RATE,
        (_deps("Rate", MethodKind.LOOP),),
    )
    convergence_stage = replace(
        _stage(
            StageKind.DEVICE_CONVERGENCE,
            (_deps("DensityIteration", MethodKind.POST_LOOP),),
        ),
        convergence_policy=DeviceConvergencePolicy(
            min_iterations=1,
            max_iterations=2,
            update_nnps=False,
            child_stage_indices=(0,),
            equation_names=("DensityIteration",),
            flag_fields=("equation_has_converged",),
        ),
    )
    group = SimpleNamespace(min_iterations=1, max_iterations=2, update_nnps=False)
    helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(
            stages=(density_stage, rate_stage, convergence_stage)
        ),
        calls=(
            {"type": "start_iteration", "group": group},
            {"type": "kernel", "stage_group": (0, -1), "stage_method_kind": "loop"},
            {"type": "kernel", "stage_group": (1, -1), "stage_method_kind": "loop"},
            {"type": "stop_iteration"},
        ),
    )

    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)

    assert backend.launched == [
        (StageKind.PAIR_DENSITY, (0, -1), ("t", "dt")),
        (StageKind.PAIR_DENSITY, (0, -1), ("t", "dt")),
    ]


def test_generated_stage_backend_policy_convergence_uses_flag_without_python_converged():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    class EquationWithHostConverged:
        def __init__(self):
            self._gpu = object()
            self.equation_has_converged = 1
            self.pulls = []
            self.converged_calls = 0

        def _pull(self, *args):
            self.pulls.append(args)

        def converged(self):
            self.converged_calls += 1
            raise AssertionError("policy flag path must not call Python converged")

    convergence_stage = replace(
        _stage(
            StageKind.DEVICE_CONVERGENCE,
            (_deps("EquationWithHostConverged", MethodKind.POST_LOOP),),
        ),
        convergence_policy=DeviceConvergencePolicy(
            min_iterations=1,
            max_iterations=2,
            update_nnps=True,
            child_stage_indices=(0,),
            equation_names=("EquationWithHostConverged",),
            flag_fields=("equation_has_converged",),
        ),
    )
    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(stages=(convergence_stage,))
    )
    backend.device_convergence_flag = None
    backend.device_convergence_uses_particle_flag = False
    backend.device_convergence_host_flag_pull_count = 0
    equation = EquationWithHostConverged()

    has_converged = backend._device_convergence_has_converged(
        {"equations": (equation,)}
    )

    assert has_converged
    assert equation.pulls == [("equation_has_converged",)]
    assert equation.converged_calls == 0
    assert backend.device_convergence_host_flag_pull_count == 1


def test_generated_stage_backend_prefers_backend_owned_device_convergence_flag():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    class ReadingBackend(GeneratedFusedCudaStageBackend):
        def _read_device_convergence_flag(self):
            self.device_flag_reads += 1
            return True

    class EquationWithForbiddenHostConvergence:
        def __init__(self):
            self._gpu = object()

        def _pull(self, *args):
            raise AssertionError("backend-owned device flag must avoid equation pull")

        def converged(self):
            raise AssertionError("backend-owned device flag must avoid converged")

    convergence_stage = replace(
        _stage(
            StageKind.DEVICE_CONVERGENCE,
            (_deps("EquationWithForbiddenHostConvergence", MethodKind.POST_LOOP),),
        ),
        convergence_policy=DeviceConvergencePolicy(
            min_iterations=1,
            max_iterations=2,
            update_nnps=True,
            child_stage_indices=(0,),
            equation_names=("EquationWithForbiddenHostConvergence",),
            flag_fields=("equation_has_converged",),
        ),
    )
    backend = object.__new__(ReadingBackend)
    backend.helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(stages=(convergence_stage,))
    )
    backend.device_convergence_flag = object()
    backend.device_convergence_uses_particle_flag = True
    backend.device_flag_reads = 0
    backend.device_convergence_device_flag_read_count = 0
    backend.device_convergence_host_flag_pull_count = 0
    equation = EquationWithForbiddenHostConvergence()

    has_converged = backend._device_convergence_has_converged(
        {"equations": (equation,)}
    )

    assert has_converged
    assert backend.device_flag_reads == 1
    assert backend.device_convergence_device_flag_read_count == 1
    assert backend.device_convergence_host_flag_pull_count == 0


def test_generated_stage_backend_can_assume_convergence_after_min_iterations(
    monkeypatch,
):
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    class ReadingBackend(GeneratedFusedCudaStageBackend):
        def _read_device_convergence_flag(self):
            self.device_flag_reads += 1
            raise AssertionError("fixed-min convergence must not read device flag")

    class EquationWithForbiddenHostConvergence:
        def __init__(self):
            self._gpu = object()

        def _pull(self, *args):
            raise AssertionError("fixed-min convergence must not pull host flag")

        def converged(self):
            raise AssertionError("fixed-min convergence must not call converged")

    monkeypatch.setenv("PYSPH_FUSED_ASSUME_CONVERGED_AFTER_MIN_ITERATIONS", "1")
    convergence_stage = replace(
        _stage(
            StageKind.DEVICE_CONVERGENCE,
            (_deps("EquationWithForbiddenHostConvergence", MethodKind.POST_LOOP),),
        ),
        convergence_policy=DeviceConvergencePolicy(
            min_iterations=2,
            max_iterations=250,
            update_nnps=True,
            child_stage_indices=(0,),
            equation_names=("EquationWithForbiddenHostConvergence",),
            flag_fields=("equation_has_converged",),
        ),
    )
    backend = object.__new__(ReadingBackend)
    backend.helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(stages=(convergence_stage,))
    )
    backend.device_convergence_flag = object()
    backend.device_convergence_uses_particle_flag = True
    backend.device_flag_reads = 0
    backend.device_convergence_device_flag_read_count = 0
    backend.device_convergence_host_flag_pull_count = 0
    equation = EquationWithForbiddenHostConvergence()

    has_converged = backend._device_convergence_has_converged(
        {"equations": (equation,)}
    )

    assert has_converged
    assert backend.device_flag_reads == 0
    assert backend.device_convergence_device_flag_read_count == 0
    assert backend.device_convergence_host_flag_pull_count == 0


def test_generated_stage_backend_reuses_neighbor_context_until_h_or_position_write():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    class RecordingBackend(GeneratedFusedCudaStageBackend):
        def __init__(self):
            self.neighbor_contexts = {}
            self.builds = 0

        def _build_neighbor_context(self, evaluator, info):
            self.builds += 1
            return f"context-{self.builds}"

    def deps_with_writes(*writes):
        return MethodDeps(
            equation_name="DensityIteration",
            method_kind=MethodKind.LOOP,
            dest="fluid",
            sources=("fluid",),
            dest_reads=frozenset(),
            source_reads=frozenset(),
            dest_writes=frozenset(writes),
            source_writes=frozenset(),
            precomputed_symbols=frozenset(),
            precomputed_writes=frozenset(),
            unsupported_reasons=(),
            dest_reduction_writes=frozenset(),
            dest_max_reduction_writes=frozenset(),
            dest_reduction_reads=frozenset(),
        )

    backend = RecordingBackend()
    dest = object()
    info = {"dest": dest, "src": dest}
    non_writer = _stage(StageKind.PAIR_RATE, (deps_with_writes("au"),))
    h_writer = _stage(StageKind.PAIR_DENSITY, (deps_with_writes("h"),))

    assert backend._neighbor_context_for_stage("evaluator", info) == "context-1"
    assert backend._neighbor_context_for_stage("evaluator", info) == "context-1"
    assert backend.builds == 1

    backend._finish_launched_stage(non_writer)
    assert backend._neighbor_context_for_stage("evaluator", info) == "context-1"
    assert backend.builds == 1

    backend._finish_launched_stage(h_writer)
    assert backend._neighbor_context_for_stage("evaluator", info) == "context-2"
    assert backend.builds == 2


def test_generated_stage_backend_device_convergence_rebuild_skips_legacy_nnps_update():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.neighbor_contexts = {"fluid": object()}
    calls = []
    evaluator = SimpleNamespace(update_nnps=lambda: calls.append("legacy nnps"))

    backend._update_device_convergence_nnps(evaluator, "info")

    assert calls == []
    assert backend.neighbor_contexts == {}


def test_generated_stage_backend_handles_outer_nnps_update():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    backend = object.__new__(GeneratedFusedCudaStageBackend)

    assert backend.handle_outer_update_nnps("integrator", 0)


def test_neighbor_context_uses_domain_cell_size_for_cell_counts(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module

    recorded = {}

    def fake_build_context(
        x,
        y,
        z,
        h,
        n,
        lower,
        upper,
        periodic,
        radius_scale,
        cell_counts,
        stream,
        cluster_size,
        workspace,
        build_cluster_metadata,
    ):
        recorded["context"] = (
            x,
            y,
            z,
            h,
            n,
            lower,
            upper,
            periodic,
            radius_scale,
            cell_counts,
            stream,
            cluster_size,
            workspace,
            build_cluster_metadata,
        )
        return "context"

    monkeypatch.setattr(
        backend_module,
        "build_fused_cuda_context_with_workspace",
        fake_build_context,
    )
    gpu = SimpleNamespace(
        x=SimpleNamespace(dev="x_dev"),
        y=SimpleNamespace(dev="y_dev"),
        z=SimpleNamespace(dev="z_dev"),
        h=SimpleNamespace(dev="h_dev"),
    )
    particle = SimpleNamespace(gpu=gpu, get_number_of_particles=lambda real: 8)
    manager = SimpleNamespace(
        minimum_image_periodic=True,
        periodic_in_x=True,
        periodic_in_y=False,
        periodic_in_z=True,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=2.0,
        zmin=0.0,
        zmax=1.0,
        cell_size=0.5,
    )
    nnps = SimpleNamespace(
        radius_scale=2.0,
        cell_size=10.0,
        xmin=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        xmax=np.array([20.0, 20.0, 20.0], dtype=np.float32),
        domain=SimpleNamespace(manager=manager),
    )
    evaluator = SimpleNamespace(nnps=nnps)

    workspace = object()
    context = backend_module._neighbor_context_for_info(
        evaluator,
        {"dest": particle, "src": particle},
        "stream",
        [],
        workspace,
        False,
    )

    assert context == "context"
    np.testing.assert_array_equal(
        recorded["context"][9], np.array([2, 20, 2], dtype=np.int32)
    )
    assert recorded["context"][12] is workspace
    assert not recorded["context"][13]


def test_hbucket_neighbor_context_uses_configured_bucket_count(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module

    recorded = {}

    def fake_build_hbucket_context(
        x,
        y,
        z,
        h,
        n,
        lower,
        upper,
        periodic,
        radius_scale,
        bucket_count,
        stream,
        workspace,
        h_reduce_scratch,
    ):
        recorded["context"] = (
            x,
            y,
            z,
            h,
            n,
            lower,
            upper,
            periodic,
            radius_scale,
            bucket_count,
            stream,
            workspace,
            h_reduce_scratch,
        )
        return "hbucket-context"

    monkeypatch.setattr(
        backend_module,
        "build_fused_cuda_hbucket_context_with_workspace",
        fake_build_hbucket_context,
    )
    monkeypatch.setenv("PYSPH_FUSED_HBUCKET_BUCKET_COUNT", "8")
    gpu = SimpleNamespace(
        x=SimpleNamespace(dev="x_dev"),
        y=SimpleNamespace(dev="y_dev"),
        z=SimpleNamespace(dev="z_dev"),
        h=SimpleNamespace(dev="h_dev"),
    )
    particle = SimpleNamespace(gpu=gpu, get_number_of_particles=lambda real: 8)
    manager = SimpleNamespace(
        minimum_image_periodic=True,
        periodic_in_x=True,
        periodic_in_y=False,
        periodic_in_z=True,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=2.0,
        zmin=0.0,
        zmax=1.0,
    )
    nnps = SimpleNamespace(
        radius_scale=2.0,
        xmin=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        xmax=np.array([20.0, 20.0, 20.0], dtype=np.float32),
        domain=SimpleNamespace(manager=manager),
    )
    evaluator = SimpleNamespace(nnps=nnps)

    workspace = object()
    h_reduce_scratch = []
    context = backend_module._neighbor_context_for_info(
        evaluator,
        {"dest": particle, "src": particle},
        "stream",
        h_reduce_scratch,
        workspace,
        "hbucket",
    )

    assert context == "hbucket-context"
    assert recorded["context"][9] == 8
    assert recorded["context"][11] is workspace
    assert recorded["context"][12] is h_reduce_scratch


def test_hbucket_neighbor_context_can_use_fixed_hmin_without_reduce(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module

    recorded = {}

    def fail_reduce_builder(
        x,
        y,
        z,
        h,
        n,
        lower,
        upper,
        periodic,
        radius_scale,
        bucket_count,
        stream,
        workspace,
        h_reduce_scratch,
    ):
        assert False

    def fake_fixed_builder(
        x,
        y,
        z,
        h,
        n,
        lower,
        upper,
        periodic,
        radius_scale,
        bucket_count,
        stream,
        workspace,
        fixed_h_min,
    ):
        recorded["context"] = (
            x,
            y,
            z,
            h,
            n,
            lower,
            upper,
            periodic,
            radius_scale,
            bucket_count,
            stream,
            workspace,
            fixed_h_min,
        )
        return "fixed-hbucket-context"

    monkeypatch.setattr(
        backend_module,
        "build_fused_cuda_hbucket_context_with_workspace",
        fail_reduce_builder,
    )
    monkeypatch.setattr(
        backend_module,
        "build_fused_cuda_hbucket_context_with_fixed_hmin",
        fake_fixed_builder,
    )
    monkeypatch.setenv("PYSPH_FUSED_HBUCKET_FIXED_HMIN", "0.03125")
    gpu = SimpleNamespace(
        x=SimpleNamespace(dev="x_dev"),
        y=SimpleNamespace(dev="y_dev"),
        z=SimpleNamespace(dev="z_dev"),
        h=SimpleNamespace(dev="h_dev"),
    )
    particle = SimpleNamespace(gpu=gpu, get_number_of_particles=lambda real: 8)
    manager = SimpleNamespace(
        minimum_image_periodic=True,
        periodic_in_x=True,
        periodic_in_y=False,
        periodic_in_z=True,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=2.0,
        zmin=0.0,
        zmax=1.0,
    )
    nnps = SimpleNamespace(
        radius_scale=2.0,
        xmin=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        xmax=np.array([20.0, 20.0, 20.0], dtype=np.float32),
        domain=SimpleNamespace(manager=manager),
    )
    evaluator = SimpleNamespace(nnps=nnps)

    context = backend_module._neighbor_context_for_info(
        evaluator,
        {"dest": particle, "src": particle},
        "stream",
        [],
        object(),
        "hbucket",
    )

    assert context == "fixed-hbucket-context"
    assert recorded["context"][12] == np.float32(0.03125)


def test_generated_stage_backend_can_cache_first_hbucket_hmin(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    reduced = []
    fixed_hmins = []

    def fake_reduce_min_float(h, n, stream, scratch):
        reduced.append((h, n, stream, scratch))
        return np.float32(0.03125)

    def fake_fixed_builder(
        x,
        y,
        z,
        h,
        n,
        lower,
        upper,
        periodic,
        radius_scale,
        bucket_count,
        stream,
        workspace,
        fixed_h_min,
    ):
        fixed_hmins.append(fixed_h_min)
        return f"fixed-hbucket-context-{len(fixed_hmins)}"

    monkeypatch.setattr(backend_module, "reduce_min_float", fake_reduce_min_float)
    monkeypatch.setattr(
        backend_module,
        "build_fused_cuda_hbucket_context_with_fixed_hmin",
        fake_fixed_builder,
    )
    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_HBUCKET_FIXED_HMIN", "first")

    gpu = SimpleNamespace(
        x=SimpleNamespace(dev="x_dev"),
        y=SimpleNamespace(dev="y_dev"),
        z=SimpleNamespace(dev="z_dev"),
        h=SimpleNamespace(dev="h_dev"),
    )
    particle = SimpleNamespace(gpu=gpu, get_number_of_particles=lambda real: 8)
    manager = SimpleNamespace(
        minimum_image_periodic=True,
        periodic_in_x=True,
        periodic_in_y=False,
        periodic_in_z=True,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=2.0,
        zmin=0.0,
        zmax=1.0,
    )
    nnps = SimpleNamespace(
        radius_scale=2.0,
        xmin=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        xmax=np.array([20.0, 20.0, 20.0], dtype=np.float32),
        domain=SimpleNamespace(manager=manager),
    )
    evaluator = SimpleNamespace(nnps=nnps)
    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.stream = "stream"
    backend.h_reduce_scratch = []
    backend.neighbor_workspaces = {}
    backend.hbucket_first_hmins = {}
    info = {"dest": particle, "src": particle}

    first_context = backend._build_neighbor_context(evaluator, info)
    second_context = backend._build_neighbor_context(evaluator, info)

    assert first_context == "fixed-hbucket-context-1"
    assert second_context == "fixed-hbucket-context-2"
    assert len(reduced) == 1
    assert reduced[0] == ("h_dev", 8, "stream", backend.h_reduce_scratch)
    assert fixed_hmins == [np.float32(0.03125), np.float32(0.03125)]


def test_generated_stage_backend_can_scale_first_hbucket_hmin(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    reduced = []
    fixed_hmins = []

    def fake_reduce_min_float(h, n, stream, scratch):
        reduced.append((h, n, stream, scratch))
        return np.float32(0.03125)

    def fake_fixed_builder(
        x,
        y,
        z,
        h,
        n,
        lower,
        upper,
        periodic,
        radius_scale,
        bucket_count,
        stream,
        workspace,
        fixed_h_min,
    ):
        fixed_hmins.append(fixed_h_min)
        return "fixed-hbucket-context"

    monkeypatch.setattr(backend_module, "reduce_min_float", fake_reduce_min_float)
    monkeypatch.setattr(
        backend_module,
        "build_fused_cuda_hbucket_context_with_fixed_hmin",
        fake_fixed_builder,
    )
    monkeypatch.setenv("PYSPH_FUSED_PAIR_TRAVERSAL", "hbucket")
    monkeypatch.setenv("PYSPH_FUSED_HBUCKET_FIXED_HMIN", "first")
    monkeypatch.setenv("PYSPH_FUSED_HBUCKET_FIXED_HMIN_SCALE", "1.25")

    gpu = SimpleNamespace(
        x=SimpleNamespace(dev="x_dev"),
        y=SimpleNamespace(dev="y_dev"),
        z=SimpleNamespace(dev="z_dev"),
        h=SimpleNamespace(dev="h_dev"),
    )
    particle = SimpleNamespace(gpu=gpu, get_number_of_particles=lambda real: 8)
    manager = SimpleNamespace(
        minimum_image_periodic=True,
        periodic_in_x=True,
        periodic_in_y=False,
        periodic_in_z=True,
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=2.0,
        zmin=0.0,
        zmax=1.0,
    )
    nnps = SimpleNamespace(
        radius_scale=2.0,
        xmin=np.array([10.0, 10.0, 10.0], dtype=np.float32),
        xmax=np.array([20.0, 20.0, 20.0], dtype=np.float32),
        domain=SimpleNamespace(manager=manager),
    )
    evaluator = SimpleNamespace(nnps=nnps)
    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.stream = "stream"
    backend.h_reduce_scratch = []
    backend.neighbor_workspaces = {}
    backend.hbucket_first_hmins = {}
    info = {"dest": particle, "src": particle}

    backend._build_neighbor_context(evaluator, info)
    backend._build_neighbor_context(evaluator, info)

    assert len(reduced) == 1
    assert fixed_hmins == [np.float32(0.03125 * 1.25), np.float32(0.03125 * 1.25)]


def test_fused_stage_backend_fast_math_options_are_opt_in(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module

    monkeypatch.delenv("PYSPH_FUSED_CUDA_FAST_MATH", raising=False)
    monkeypatch.delenv("PYSPH_FUSED_CUDA_NVCC_OPTIONS", raising=False)

    assert backend_module._source_module_options() == ()

    monkeypatch.setenv("PYSPH_FUSED_CUDA_FAST_MATH", "1")

    assert backend_module._source_module_options() == ("--use_fast_math",)


def test_fused_stage_backend_forwards_extra_nvcc_options(monkeypatch):
    from pysph.sph import fused_cuda_stage_backend as backend_module

    monkeypatch.delenv("PYSPH_FUSED_CUDA_FAST_MATH", raising=False)
    monkeypatch.setenv("PYSPH_FUSED_CUDA_NVCC_OPTIONS", "-Xptxas=-v --maxrregcount=64")

    assert backend_module._source_module_options() == (
        "-Xptxas=-v",
        "--maxrregcount=64",
    )

    monkeypatch.setenv("PYSPH_FUSED_CUDA_FAST_MATH", "1")

    assert backend_module._source_module_options() == (
        "--use_fast_math",
        "-Xptxas=-v",
        "--maxrregcount=64",
    )


def test_pair_stage_launch_segments_are_opt_in(monkeypatch):
    add_mass = _deps("AddMass", MethodKind.LOOP)
    assign_mass = _deps("AssignMassToAcceleration", MethodKind.LOOP)
    stage = replace(
        _stage(StageKind.PAIR_RATE, (add_mass, assign_mass)),
        method_segments=((add_mass,), (assign_mass,)),
    )

    assert _launch_segments_for_stage(stage) == (stage,)

    monkeypatch.setenv("PYSPH_FUSED_SPLIT_PAIR_SEGMENTS", "1")
    segments = _launch_segments_for_stage(stage)

    assert len(segments) == 2
    assert segments[0].methods == (add_mass,)
    assert segments[1].methods == (assign_mass,)
    assert segments[0].method_segments == ()
    assert segments[1].method_segments == ()


def test_fused_stage_backend_skips_legacy_groups_covered_by_merged_stage():
    from pysph.sph.fused_cuda_stage_backend import FusedCudaStageBackend

    class RecordingBackend(FusedCudaStageBackend):
        def __init__(self, helper):
            super().__init__(helper)
            self.launched = []

        def _launch_stage(self, evaluator, stage, info, extra_args):
            self.launched.append((stage.kind, info["stage_group"], extra_args))

    merged_stage = _stage(
        StageKind.POINTWISE,
        (
            _deps("DensityRelation", MethodKind.INITIALIZE),
            _deps("EOS", MethodKind.LOOP),
        ),
    )
    merged_stage = replace(merged_stage, legacy_group_count=2)
    helper = SimpleNamespace(
        cuda_stage_plan=SimpleNamespace(stages=(merged_stage,)),
        calls=(
            {
                "type": "kernel",
                "stage_group": (0, -1),
                "stage_method_kind": "initialize",
            },
            {"type": "kernel", "stage_group": (1, -1), "stage_method_kind": "loop"},
        ),
    )

    backend = RecordingBackend(helper)
    backend.begin_compute("evaluator", 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[0], ("t", "dt"), 0.0, 0.1)
    assert backend.handle_call("evaluator", helper.calls[1], ("t", "dt"), 0.0, 0.1)

    assert backend.launched == [
        (StageKind.POINTWISE, (0, -1), ("t", "dt")),
    ]


def test_generated_stage_backend_selects_pair_precompute_for_viscosity_symbols():
    from pysph.base.kernels import CubicSpline
    from pysph.sph.fused_cuda_stage_backend import _precompute_for_stage

    stage = _stage(
        StageKind.PAIR_RATE,
        (
            MethodDeps(
                equation_name="MonaghanArtificialViscosity",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(),
                source_reads=frozenset(),
                dest_writes=frozenset(),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(
                    ("VIJ", "XIJ", "HIJ", "R2IJ", "RHOIJ1", "EPS", "DWIJ")
                ),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )

    precompute = _precompute_for_stage(stage, CubicSpline(dim=1))

    assert "VIJ" in precompute.symbols
    assert "RHOIJ1" in precompute.symbols
    assert "EPS" in precompute.symbols
    assert "DWIJ" in precompute.symbols


def test_generated_stage_backend_selects_minimal_gradient_h_precompute():
    from pysph.base.kernels import CubicSpline
    from pysph.sph.fused_cuda_stage_backend import _precompute_for_stage

    stage = _stage(
        StageKind.PAIR_RATE,
        (
            MethodDeps(
                equation_name="GradientH",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(),
                source_reads=frozenset(),
                dest_writes=frozenset(),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(("WI", "DWI", "GHI", "GHJ", "GHIJ")),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )

    precompute = _precompute_for_stage(stage, CubicSpline(dim=1))

    assert precompute_argument_names(precompute) == ()
    assert "VIJ" not in precompute.symbols
    assert "RHOIJ" not in precompute.symbols
    assert "GHIJ" in precompute.symbols


def test_generated_stage_backend_avoids_unused_pair_gradient_symbols():
    from pysph.base.kernels import QuinticSpline
    from pysph.sph.fused_cuda_stage_backend import _precompute_for_stage

    stage = _stage(
        StageKind.PAIR_RATE,
        (
            MethodDeps(
                equation_name="MagneticStress",
                method_kind=MethodKind.LOOP,
                dest="fluid",
                sources=("fluid",),
                dest_reads=frozenset(),
                source_reads=frozenset(),
                dest_writes=frozenset(),
                source_writes=frozenset(),
                precomputed_symbols=frozenset(("XIJ", "RIJ", "DWI", "DWJ")),
                precomputed_writes=frozenset(),
                unsupported_reasons=(),
                dest_reduction_writes=frozenset(),
                dest_max_reduction_writes=frozenset(),
                dest_reduction_reads=frozenset(),
            ),
        ),
    )

    precompute = _precompute_for_stage(stage, QuinticSpline(dim=3))

    assert "DWI" in precompute.symbols
    assert "DWJ" in precompute.symbols
    assert "DWIJ" not in precompute.symbols
    assert "HIJ" not in precompute.symbols
    assert all("DWIJ" not in line for line in precompute.lines)
    assert all("HIJ" not in line for line in precompute.lines)


def test_direct_pair_loop_outline_can_precompute_wij_for_equation_wrapper():
    equation = SummationDensity(dest="fluid", sources=["fluid"])
    group = CUDAGroup([equation])
    known_types = {
        "d_rho": KnownType("GLOBAL_MEM float*"),
        "s_m": KnownType("GLOBAL_MEM float*"),
    }
    precompute = cubic_spline_wij_precompute(np.int32(1))
    wrapper_source = group.get_equation_wrappers(
        {
            "d_rho": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
            "WIJ": KnownType("float"),
        }
    )
    call = cuda_equation_method_call_from_equation_with_precomputed(
        equation, "loop", known_types, precompute.symbols
    )

    outline = generate_direct_pair_loop_outline_with_equation_calls_and_precompute(
        "plan0",
        _stage(StageKind.PAIR_DENSITY, (_deps("SummationDensity", MethodKind.LOOP),)),
        wrapper_source,
        precompute,
        (call,),
    )

    assert "float WIJ;" in outline.source
    assert "WIJ = fused_codegen_cubic_spline_wij(RIJ, HIJ, 1);" in outline.source
    assert (
        "SummationDensity_loop(summation_density0, dst, d_rho, src, s_m, WIJ);"
        in outline.source
    )
    assert "float WIJ" not in outline.source.split("__global__ void")[1].split(")")[0]


def test_direct_pair_loop_outline_can_precompute_cubic_gradients_for_wrappers():
    equations = (
        AccumulateDWIJ(dest="fluid", sources=["fluid"]),
        AccumulateDWIAndDWJ(dest="fluid", sources=["fluid"]),
    )
    group = CUDAGroup(list(equations))
    known_types = {
        "d_au": KnownType("GLOBAL_MEM float*"),
        "s_m": KnownType("GLOBAL_MEM float*"),
    }
    wrapper_source = group.get_equation_wrappers(
        {
            "d_au": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
            "DWIJ": KnownType("float*"),
            "DWI": KnownType("float*"),
            "DWJ": KnownType("float*"),
        }
    )
    precompute = cubic_spline_gradient_precompute(np.int32(1))
    calls = tuple(
        cuda_equation_method_call_from_equation_with_precomputed(
            equation, "loop", known_types, precompute.symbols
        )
        for equation in equations
    )

    outline = generate_direct_pair_loop_outline_with_equation_calls_and_precompute(
        "plan0",
        _stage(
            StageKind.PAIR_RATE,
            (
                _deps("AccumulateDWIJ", MethodKind.LOOP),
                _deps("AccumulateDWIAndDWJ", MethodKind.LOOP),
            ),
        ),
        wrapper_source,
        precompute,
        calls,
    )

    assert "float DWIJ[3];" in outline.source
    assert "float DWI[3];" in outline.source
    assert "float DWJ[3];" in outline.source
    assert (
        "fused_codegen_cubic_spline_gradient(DWIJ, XIJ, RIJ, HIJ, 1);" in outline.source
    )
    assert (
        "fused_codegen_cubic_spline_gradient(DWI, XIJ, RIJ, h[dst], 1);"
        in outline.source
    )
    assert (
        "fused_codegen_cubic_spline_gradient(DWJ, XIJ, RIJ, h[src], 1);"
        in outline.source
    )
    assert (
        "AccumulateDWIJ_loop(accumulate_dwij0, dst, src, d_au, s_m, DWIJ);"
        in outline.source
    )
    assert (
        "AccumulateDWIAndDWJ_loop(accumulate_dwi_and_dwj0, dst, src, d_au, s_m, DWI, DWJ);"
        in outline.source
    )


def test_direct_pair_stage_outline_supports_standard_viscosity_precompute_symbols():
    equation = MonaghanArtificialViscosity(dest="fluid", sources=["fluid"])
    precompute = cubic_spline_pair_precompute(np.int32(1))

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        _stage(
            StageKind.PAIR_RATE,
            (
                _deps("MonaghanArtificialViscosity", MethodKind.INITIALIZE),
                _deps("MonaghanArtificialViscosity", MethodKind.LOOP),
            ),
        ),
        (equation,),
        precompute,
    )

    assert "float VIJ[3];" in outline.source
    assert "float EPS;" in outline.source
    assert "float RHOIJ;" in outline.source
    assert "float RHOIJ1;" in outline.source
    assert "VIJ[0] = d_u[dst] - s_u[src];" in outline.source
    assert "EPS = 0.01f * HIJ * HIJ;" in outline.source
    assert "RHOIJ = 0.5f * (d_rho[dst] + s_rho[src]);" in outline.source
    assert "RHOIJ1 = 1.0f / RHOIJ;" in outline.source
    assert "GLOBAL_MEM float* d_u" in outline.source
    assert "GLOBAL_MEM float* s_u" in outline.source
    assert "GLOBAL_MEM float* d_rho" in outline.source
    assert "GLOBAL_MEM float* s_rho" in outline.source
    assert (
        "MonaghanArtificialViscosity_loop(monaghan_artificial_viscosity0, dst, src,"
        in outline.source
    )


def test_direct_pair_stage_outline_supports_gradient_h_precompute_symbols():
    equation = AccumulateGradientH(dest="fluid", sources=["fluid"])
    precompute = cubic_spline_pair_precompute(np.int32(1))

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AccumulateGradientH", MethodKind.LOOP),)),
        (equation,),
        precompute,
    )

    assert "WI" in precompute.symbols
    assert "GHI" in precompute.symbols
    assert "GHJ" in precompute.symbols
    assert "GHIJ" in precompute.symbols
    assert "float WI;" in outline.source
    assert "float GHI;" in outline.source
    assert "float GHJ;" in outline.source
    assert "float GHIJ;" in outline.source
    assert "WI = fused_codegen_cubic_spline_wij(RIJ, h[dst], 1);" in outline.source
    assert (
        "GHI = fused_codegen_cubic_spline_gradient_h(RIJ, h[dst], 1);" in outline.source
    )
    assert (
        "GHJ = fused_codegen_cubic_spline_gradient_h(RIJ, h[src], 1);" in outline.source
    )
    assert (
        "GHIJ = fused_codegen_cubic_spline_gradient_h(RIJ, HIJ, 1);" in outline.source
    )
    assert (
        "AccumulateGradientH_loop(accumulate_gradient_h0, dst, src, d_au, s_m, WI, DWI, GHI, GHJ, GHIJ);"
        in outline.source
    )


def test_gradient_h_pair_precompute_does_not_require_unused_arrays():
    equation = AccumulateGradientH(dest="fluid", sources=["fluid"])
    precompute = cubic_spline_pair_precompute_for_symbols(
        np.int32(1), frozenset(("WI", "DWI", "GHI", "GHJ", "GHIJ"))
    )

    outline = generate_direct_pair_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AccumulateGradientH", MethodKind.LOOP),)),
        (equation,),
        precompute,
    )

    assert precompute_argument_names(precompute) == ()
    assert "float VIJ[3];" not in outline.source
    assert "float RHOIJ;" not in outline.source
    assert "GLOBAL_MEM float* d_u" not in outline.source
    assert "GLOBAL_MEM float* d_rho" not in outline.source
    assert "float GHI;" in outline.source
    assert "float GHJ;" in outline.source
    assert "float GHIJ;" in outline.source


def test_cuda_wrapper_call_pair_kernel_matches_periodic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mass = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    h = np.full(3, 0.10, dtype=np.float32)
    au = np.zeros(3, dtype=np.float32)
    stream = cuda.Stream()
    d_au = gpuarray.to_gpu_async(au, stream=stream)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        3,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([5, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = AddMass(dest="fluid", sources=["fluid"])
    group = CUDAGroup([equation])
    known_types = {
        "d_au": KnownType("GLOBAL_MEM float*"),
        "s_m": KnownType("GLOBAL_MEM float*"),
    }
    wrapper_source = group.get_equation_wrappers(known_types)
    call = cuda_equation_method_call_from_equation(equation, "loop", known_types)
    outline = generate_direct_pair_loop_outline_with_equation_calls(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),)),
        wrapper_source,
        (call,),
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_direct_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (np.uintp(0), d_au, gpuarray.to_gpu_async(mass, stream=stream)),
    )
    got = d_au.get_async(stream=stream)
    stream.synchronize()
    expected = np.asarray(
        [
            np.sum(
                mass[
                    brute_force_neighbor_indices(
                        xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
                    )
                ]
            )
            for i in range(3)
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(got, expected, rtol=1.0e-6, atol=1.0e-6)


def test_cuda_source_parallel_pair_add_mass_matches_periodic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.98, 0.5, 0.5],
            [0.02, 0.5, 0.5],
            [0.25, 0.5, 0.5],
            [0.75, 0.5, 0.5],
            [0.50, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    h = np.full(5, 0.08, dtype=np.float32)
    mass = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    stream = cuda.Stream()
    d_au = gpuarray.zeros(5, np.float32)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        5,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([7, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = AddMass(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())
    outline = generate_source_parallel_pair_stage_outline_from_equations(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AddMass", MethodKind.LOOP),)),
        (equation,),
        precompute,
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_source_parallel_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (np.uintp(0), d_au, gpuarray.to_gpu_async(mass, stream=stream)),
    )
    got = d_au.get_async(stream=stream)
    stream.synchronize()
    expected = np.asarray(
        [
            np.sum(
                mass[
                    brute_force_neighbor_indices(
                        xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
                    )
                ]
            )
            for i in range(5)
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(got, expected, rtol=1.0e-6, atol=1.0e-6)


def test_cuda_source_parallel_pair_sum_and_max_match_periodic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.98, 0.5, 0.5],
            [0.02, 0.5, 0.5],
            [0.25, 0.5, 0.5],
            [0.75, 0.5, 0.5],
            [0.50, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    h = np.full(5, 0.08, dtype=np.float32)
    mass = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    stream = cuda.Stream()
    d_au = gpuarray.zeros(5, np.float32)
    d_dt_cfl = gpuarray.zeros(5, np.float32)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        5,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([7, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = AccumulateDWIJAndMaxSignal(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(
        symbols=frozenset(("DWIJ",)),
        helper_source="",
        lines=("float DWIJ[3];", "DWIJ[0] = 2.0f;"),
    )
    outline = generate_source_parallel_pair_stage_outline_from_equations(
        "plan0",
        _stage(
            StageKind.PAIR_RATE,
            (_deps("AccumulateDWIJAndMaxSignal", MethodKind.LOOP),),
        ),
        (equation,),
        precompute,
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_source_parallel_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (
            np.uintp(0),
            d_au,
            d_dt_cfl,
            gpuarray.to_gpu_async(mass, stream=stream),
        ),
    )
    got_au = d_au.get_async(stream=stream)
    got_dt_cfl = d_dt_cfl.get_async(stream=stream)
    stream.synchronize()
    expected_au = np.asarray(
        [
            2.0
            * np.sum(
                mass[
                    brute_force_neighbor_indices(
                        xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
                    )
                ]
            )
            for i in range(5)
        ],
        dtype=np.float32,
    )
    expected_dt_cfl = np.full(5, 2.0, dtype=np.float32)

    np.testing.assert_allclose(got_au, expected_au, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(got_dt_cfl, expected_dt_cfl, rtol=1.0e-6, atol=1.0e-6)


def test_cuda_pair_stage_runs_initialize_loop_and_post_loop_in_one_kernel():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mass = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    h = np.full(3, 0.10, dtype=np.float32)
    stream = cuda.Stream()
    d_u = gpuarray.to_gpu_async(np.full(3, -9.0, dtype=np.float32), stream=stream)
    d_au = gpuarray.to_gpu_async(np.full(3, -7.0, dtype=np.float32), stream=stream)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        3,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([5, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = InitLoopPost(dest="fluid", sources=["fluid"])
    outline = generate_direct_pair_stage_outline_from_equations(
        "stage0",
        _stage(
            StageKind.PAIR_RATE,
            (
                _deps("InitLoopPost", MethodKind.INITIALIZE),
                _deps("InitLoopPost", MethodKind.LOOP),
                _deps("InitLoopPost", MethodKind.POST_LOOP),
            ),
        ),
        (equation,),
        cubic_spline_wij_precompute(np.int32(1)),
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_direct_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (np.uintp(0), d_u, d_au, gpuarray.to_gpu_async(mass, stream=stream)),
    )
    got_u = d_u.get_async(stream=stream)
    got_au = d_au.get_async(stream=stream)
    stream.synchronize()
    expected = np.asarray(
        [
            np.sum(
                mass[
                    brute_force_neighbor_indices(
                        xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
                    )
                ]
            )
            for i in range(3)
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(got_u, expected, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(got_au, expected, rtol=1.0e-6, atol=1.0e-6)


def test_cuda_cubic_gradient_h_kernel_matches_cubic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    from pysph.base.kernels import CubicSpline

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.05, 0.0, 0.0],
            [0.23, 0.0, 0.0],
            [0.90, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mass = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    h = np.array([0.16, 0.19, 0.17], dtype=np.float32)
    stream = cuda.Stream()
    d_au = gpuarray.to_gpu_async(np.zeros(3, dtype=np.float32), stream=stream)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        3,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([2, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = AccumulateGradientH(dest="fluid", sources=["fluid"])
    outline = generate_direct_pair_stage_outline_from_equations(
        "grad_h0",
        _stage(StageKind.PAIR_RATE, (_deps("AccumulateGradientH", MethodKind.LOOP),)),
        (equation,),
        cubic_spline_pair_precompute_for_symbols(
            np.int32(1), frozenset(("WI", "DWI", "GHI", "GHJ", "GHIJ"))
        ),
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_direct_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (np.uintp(0), d_au, gpuarray.to_gpu_async(mass, stream=stream)),
    )
    got = d_au.get_async(stream=stream)
    stream.synchronize()
    kernel = CubicSpline(dim=1)
    expected = []
    for i in range(3):
        value = 0.0
        for j in brute_force_neighbor_indices(
            xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
        ):
            xij = minimum_image_delta(xyz[i], xyz[j], lower, upper, periodic)
            rij = np.linalg.norm(xij)
            grad = [0.0, 0.0, 0.0]
            kernel.gradient(xij=xij.astype(np.float64), rij=rij, h=h[i], grad=grad)
            value += mass[j] * (
                kernel.kernel(xij=xij.astype(np.float64), rij=rij, h=h[i])
                + grad[0]
                + kernel.gradient_h(xij=xij.astype(np.float64), rij=rij, h=h[i])
                + kernel.gradient_h(xij=xij.astype(np.float64), rij=rij, h=h[j])
                + kernel.gradient_h(
                    xij=xij.astype(np.float64), rij=rij, h=0.5 * (h[i] + h[j])
                )
            )
        expected.append(value)

    np.testing.assert_allclose(
        got, np.asarray(expected, dtype=np.float32), rtol=5.0e-6, atol=5.0e-6
    )


def test_cuda_pointwise_wrapper_call_kernel_matches_numpy_copy():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    stream = cuda.Stream()
    au = np.array([1.0, -2.0, 4.5], dtype=np.float32)
    d_u = gpuarray.to_gpu_async(np.zeros_like(au), stream=stream)
    d_au = gpuarray.to_gpu_async(au, stream=stream)
    equation = CopyAcceleration(dest="fluid", sources=None)
    group = CUDAGroup([equation])
    known_types = {
        "d_u": KnownType("GLOBAL_MEM float*"),
        "d_au": KnownType("GLOBAL_MEM float*"),
    }
    wrapper_source = group.get_equation_wrappers(known_types)
    call = cuda_equation_method_call_from_equation(equation, "loop", known_types)
    outline = generate_pointwise_kernel_outline_with_equation_calls(
        "plan0",
        _stage(StageKind.POINTWISE, (_deps("CopyAcceleration", MethodKind.LOOP),)),
        wrapper_source,
        (call,),
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_pointwise_kernel(
        module, outline.name, au.size, stream, (np.uintp(0), d_u, d_au)
    )
    got = d_u.get_async(stream=stream)
    stream.synchronize()

    np.testing.assert_allclose(got, au, rtol=0.0, atol=0.0)


def test_cuda_pointwise_isothermal_eos_with_struct_matches_numpy():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    stream = cuda.Stream()
    rho = np.array([1.0, 1.25, 0.75], dtype=np.float32)
    d_rho = gpuarray.to_gpu_async(rho, stream=stream)
    d_p = gpuarray.to_gpu_async(np.zeros_like(rho), stream=stream)
    equation = IsothermalEOS(dest="fluid", sources=None, rho0=1.0, c0=10.0, p0=0.5)
    group = CUDAGroup([equation])
    known_types = {
        "d_rho": KnownType("GLOBAL_MEM float*"),
        "d_p": KnownType("GLOBAL_MEM float*"),
    }
    wrapper_source = group.get_equation_wrappers(known_types)
    call = cuda_equation_method_call_from_equation(equation, "loop", known_types)
    outline = generate_pointwise_kernel_outline_with_equation_calls(
        "plan0",
        _stage(StageKind.POINTWISE, (_deps("IsothermalEOS", MethodKind.LOOP),)),
        wrapper_source,
        (call,),
    )
    module = SourceModule(outline.source, no_extern_c=True)
    eq_arg = build_cuda_equation_struct_argument(equation, stream)

    launch_pointwise_kernel(
        module, outline.name, rho.size, stream, (eq_arg, d_rho, d_p)
    )
    got = d_p.get_async(stream=stream)
    stream.synchronize()
    expected = np.float32(0.5) + np.float32(100.0) * (rho - np.float32(1.0))

    np.testing.assert_allclose(got, expected, rtol=2.0e-6, atol=2.0e-6)


def test_cuda_generated_density_eos_rate_chain_matches_cubic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    from pysph.base.kernels import CubicSpline

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.05, 0.0, 0.0],
            [0.23, 0.0, 0.0],
            [0.52, 0.0, 0.0],
            [0.90, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mass = np.array([1.0, 1.2, 0.7, 1.4], dtype=np.float32)
    h = np.full(4, 0.18, dtype=np.float32)
    stream = cuda.Stream()
    d_x = gpuarray.to_gpu_async(xyz[:, 0], stream=stream)
    d_y = gpuarray.to_gpu_async(xyz[:, 1], stream=stream)
    d_z = gpuarray.to_gpu_async(xyz[:, 2], stream=stream)
    d_h = gpuarray.to_gpu_async(h, stream=stream)
    d_m = gpuarray.to_gpu_async(mass, stream=stream)
    d_rho = gpuarray.to_gpu_async(np.zeros(4, dtype=np.float32), stream=stream)
    d_p = gpuarray.to_gpu_async(np.zeros(4, dtype=np.float32), stream=stream)
    d_au = gpuarray.to_gpu_async(np.zeros(4, dtype=np.float32), stream=stream)
    context = build_fused_cuda_context_from_device_arrays(
        d_x,
        d_y,
        d_z,
        d_h,
        4,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([2, 1, 1], dtype=np.int32),
        stream,
        2,
    )

    density = SummationDensity(dest="fluid", sources=["fluid"])
    density_outline = generate_direct_pair_stage_outline_from_equations(
        "chain0",
        _stage(StageKind.PAIR_DENSITY, (_deps("SummationDensity", MethodKind.LOOP),)),
        (density,),
        cubic_spline_wij_precompute(np.int32(1)),
    )
    density_module = SourceModule(density_outline.source, no_extern_c=True)
    launch_direct_pair_kernel_with_context(
        density_module,
        density_outline.name,
        context,
        (np.uintp(0), d_rho, d_m),
    )

    eos = IsothermalEOS(dest="fluid", sources=None, rho0=1.0, c0=10.0, p0=0.5)
    eos_outline = generate_pointwise_stage_outline_from_equations(
        "chain0",
        _stage(StageKind.POINTWISE, (_deps("IsothermalEOS", MethodKind.LOOP),)),
        (eos,),
    )
    eos_module = SourceModule(eos_outline.source, no_extern_c=True)
    eos_arg = build_cuda_equation_struct_argument(eos, stream)
    launch_pointwise_kernel(
        eos_module,
        eos_outline.name,
        mass.size,
        stream,
        (eos_arg, d_rho, d_p),
    )

    rate = PressureAcceleration(dest="fluid", sources=["fluid"])
    rate_outline = generate_direct_pair_stage_outline_from_equations(
        "chain0",
        _stage(StageKind.PAIR_RATE, (_deps("PressureAcceleration", MethodKind.LOOP),)),
        (rate,),
        cubic_spline_gradient_precompute(np.int32(1)),
    )
    rate_module = SourceModule(rate_outline.source, no_extern_c=True)
    launch_direct_pair_kernel_with_context(
        rate_module,
        rate_outline.name,
        context,
        (np.uintp(0), d_au, d_p, d_p, d_rho, d_rho, d_m),
    )
    got_rho = d_rho.get_async(stream=stream)
    got_p = d_p.get_async(stream=stream)
    got_au = d_au.get_async(stream=stream)
    stream.synchronize()

    kernel = CubicSpline(dim=1)
    expected_rho = np.zeros(4, dtype=np.float32)
    for i in range(4):
        for j in brute_force_neighbor_indices(
            xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
        ):
            xij = minimum_image_delta(xyz[i], xyz[j], lower, upper, periodic)
            expected_rho[i] += mass[j] * kernel.kernel(
                xij=xij.astype(np.float64),
                rij=np.linalg.norm(xij),
                h=0.5 * (h[i] + h[j]),
            )
    expected_p = np.float32(0.5) + np.float32(100.0) * (expected_rho - np.float32(1.0))
    expected_au = np.zeros(4, dtype=np.float32)
    for i in range(4):
        for j in brute_force_neighbor_indices(
            xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
        ):
            xij = minimum_image_delta(xyz[i], xyz[j], lower, upper, periodic)
            grad = [0.0, 0.0, 0.0]
            kernel.gradient(
                xij=xij.astype(np.float64),
                rij=np.linalg.norm(xij),
                h=0.5 * (h[i] + h[j]),
                grad=grad,
            )
            pressure = expected_p[i] / (expected_rho[i] * expected_rho[i]) + expected_p[
                j
            ] / (expected_rho[j] * expected_rho[j])
            expected_au[i] += -mass[j] * pressure * grad[0]

    np.testing.assert_allclose(got_rho, expected_rho, rtol=3.0e-6, atol=3.0e-6)
    np.testing.assert_allclose(got_p, expected_p, rtol=3.0e-6, atol=3.0e-6)
    np.testing.assert_allclose(got_au, expected_au, rtol=3.0e-5, atol=3.0e-5)


def test_cuda_summation_density_wij_kernel_matches_cubic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    from pysph.base.kernels import CubicSpline

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mass = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    h = np.full(3, 0.10, dtype=np.float32)
    stream = cuda.Stream()
    d_rho = gpuarray.to_gpu_async(np.zeros(3, dtype=np.float32), stream=stream)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        3,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([5, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = SummationDensity(dest="fluid", sources=["fluid"])
    group = CUDAGroup([equation])
    wrapper_source = group.get_equation_wrappers(
        {
            "d_rho": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
            "WIJ": KnownType("float"),
        }
    )
    precompute = cubic_spline_wij_precompute(np.int32(1))
    call = cuda_equation_method_call_from_equation_with_precomputed(
        equation,
        "loop",
        {
            "d_rho": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
        },
        precompute.symbols,
    )
    outline = generate_direct_pair_loop_outline_with_equation_calls_and_precompute(
        "plan0",
        _stage(StageKind.PAIR_DENSITY, (_deps("SummationDensity", MethodKind.LOOP),)),
        wrapper_source,
        precompute,
        (call,),
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_direct_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (np.uintp(0), d_rho, gpuarray.to_gpu_async(mass, stream=stream)),
    )
    got = d_rho.get_async(stream=stream)
    stream.synchronize()
    kernel = CubicSpline(dim=1)
    expected = np.asarray(
        [
            np.sum(
                [
                    mass[j]
                    * kernel.kernel(
                        xij=minimum_image_delta(
                            xyz[i], xyz[j], lower, upper, periodic
                        ).astype(np.float64),
                        rij=np.linalg.norm(
                            minimum_image_delta(xyz[i], xyz[j], lower, upper, periodic)
                        ),
                        h=0.5 * (h[i] + h[j]),
                    )
                    for j in brute_force_neighbor_indices(
                        xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
                    )
                ]
            )
            for i in range(3)
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(got, expected, rtol=2.0e-6, atol=2.0e-6)


def test_cuda_cubic_gradient_kernel_matches_cubic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    from pysph.base.kernels import CubicSpline

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mass = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    h = np.full(3, 0.10, dtype=np.float32)
    stream = cuda.Stream()
    d_au = gpuarray.to_gpu_async(np.zeros(3, dtype=np.float32), stream=stream)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        3,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([5, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = AccumulateDWIJ(dest="fluid", sources=["fluid"])
    group = CUDAGroup([equation])
    wrapper_source = group.get_equation_wrappers(
        {
            "d_au": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
            "DWIJ": KnownType("float*"),
        }
    )
    precompute = cubic_spline_gradient_precompute(np.int32(1))
    call = cuda_equation_method_call_from_equation_with_precomputed(
        equation,
        "loop",
        {
            "d_au": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
        },
        precompute.symbols,
    )
    outline = generate_direct_pair_loop_outline_with_equation_calls_and_precompute(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AccumulateDWIJ", MethodKind.LOOP),)),
        wrapper_source,
        precompute,
        (call,),
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_direct_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (np.uintp(0), d_au, gpuarray.to_gpu_async(mass, stream=stream)),
    )
    got = d_au.get_async(stream=stream)
    stream.synchronize()
    kernel = CubicSpline(dim=1)
    expected = []
    for i in range(3):
        value = 0.0
        for j in brute_force_neighbor_indices(
            xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(i)
        ):
            xij = minimum_image_delta(xyz[i], xyz[j], lower, upper, periodic)
            grad = [0.0, 0.0, 0.0]
            kernel.gradient(
                xij=xij.astype(np.float64),
                rij=np.linalg.norm(xij),
                h=0.5 * (h[i] + h[j]),
                grad=grad,
            )
            value += mass[j] * grad[0]
        expected.append(value)

    np.testing.assert_allclose(
        got, np.asarray(expected, dtype=np.float32), rtol=2.0e-6, atol=2.0e-6
    )


def test_cuda_summation_density_wij_kernel_matches_quintic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    from pysph.base.kernels import QuinticSpline

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mass = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    h = np.full(3, 0.10, dtype=np.float32)
    stream = cuda.Stream()
    d_rho = gpuarray.to_gpu_async(np.zeros(3, dtype=np.float32), stream=stream)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        3,
        lower,
        upper,
        periodic,
        np.float32(3.0),
        np.array([3, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = SummationDensity(dest="fluid", sources=["fluid"])
    group = CUDAGroup([equation])
    wrapper_source = group.get_equation_wrappers(
        {
            "d_rho": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
            "WIJ": KnownType("float"),
        }
    )
    precompute = quintic_spline_wij_precompute(np.int32(1))
    call = cuda_equation_method_call_from_equation_with_precomputed(
        equation,
        "loop",
        {
            "d_rho": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
        },
        precompute.symbols,
    )
    outline = generate_direct_pair_loop_outline_with_equation_calls_and_precompute(
        "plan0",
        _stage(StageKind.PAIR_DENSITY, (_deps("SummationDensity", MethodKind.LOOP),)),
        wrapper_source,
        precompute,
        (call,),
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_direct_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (np.uintp(0), d_rho, gpuarray.to_gpu_async(mass, stream=stream)),
    )
    got = d_rho.get_async(stream=stream)
    stream.synchronize()
    kernel = QuinticSpline(dim=1)
    expected = np.asarray(
        [
            np.sum(
                [
                    mass[j]
                    * kernel.kernel(
                        xij=minimum_image_delta(
                            xyz[i], xyz[j], lower, upper, periodic
                        ).astype(np.float64),
                        rij=np.linalg.norm(
                            minimum_image_delta(xyz[i], xyz[j], lower, upper, periodic)
                        ),
                        h=0.5 * (h[i] + h[j]),
                    )
                    for j in brute_force_neighbor_indices(
                        xyz, h, lower, upper, periodic, np.float32(3.0), np.int32(i)
                    )
                ]
            )
            for i in range(3)
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(got, expected, rtol=2.0e-6, atol=2.0e-6)


def test_cuda_quintic_gradient_kernel_matches_quintic_bruteforce():
    if "PYSPH_TEST_CUDA_FUSED_CODEGEN" not in os.environ:
        pytest.skip("set PYSPH_TEST_CUDA_FUSED_CODEGEN=1 to run CUDA codegen tests")
    pytest.importorskip("pycuda")

    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    from pysph.base.kernels import QuinticSpline

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mass = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    h = np.full(3, 0.10, dtype=np.float32)
    stream = cuda.Stream()
    d_au = gpuarray.to_gpu_async(np.zeros(3, dtype=np.float32), stream=stream)
    context = build_fused_cuda_context_from_device_arrays(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        3,
        lower,
        upper,
        periodic,
        np.float32(3.0),
        np.array([3, 1, 1], dtype=np.int32),
        stream,
        2,
    )
    equation = AccumulateDWIJ(dest="fluid", sources=["fluid"])
    group = CUDAGroup([equation])
    wrapper_source = group.get_equation_wrappers(
        {
            "d_au": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
            "DWIJ": KnownType("float*"),
        }
    )
    precompute = quintic_spline_gradient_precompute(np.int32(1))
    call = cuda_equation_method_call_from_equation_with_precomputed(
        equation,
        "loop",
        {
            "d_au": KnownType("GLOBAL_MEM float*"),
            "s_m": KnownType("GLOBAL_MEM float*"),
        },
        precompute.symbols,
    )
    outline = generate_direct_pair_loop_outline_with_equation_calls_and_precompute(
        "plan0",
        _stage(StageKind.PAIR_RATE, (_deps("AccumulateDWIJ", MethodKind.LOOP),)),
        wrapper_source,
        precompute,
        (call,),
    )
    module = SourceModule(outline.source, no_extern_c=True)

    launch_direct_pair_kernel_with_context(
        module,
        outline.name,
        context,
        (np.uintp(0), d_au, gpuarray.to_gpu_async(mass, stream=stream)),
    )
    got = d_au.get_async(stream=stream)
    stream.synchronize()
    kernel = QuinticSpline(dim=1)
    expected = []
    for i in range(3):
        value = 0.0
        for j in brute_force_neighbor_indices(
            xyz, h, lower, upper, periodic, np.float32(3.0), np.int32(i)
        ):
            xij = minimum_image_delta(xyz[i], xyz[j], lower, upper, periodic)
            grad = [0.0, 0.0, 0.0]
            kernel.gradient(
                xij=xij.astype(np.float64),
                rij=np.linalg.norm(xij),
                h=0.5 * (h[i] + h[j]),
                grad=grad,
            )
            value += mass[j] * grad[0]
        expected.append(value)

    np.testing.assert_allclose(
        got, np.asarray(expected, dtype=np.float32), rtol=3.0e-6, atol=3.0e-6
    )
