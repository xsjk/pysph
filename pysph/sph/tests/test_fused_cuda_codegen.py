"""Tests for the retained fused CUDA codegen paths."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pysph.base.fused_cuda_nnps import (
    FusedCudaNeighborWorkspace,
    build_fused_cuda_cell_context_with_workspace,
    build_fused_cuda_hbucket_context_with_workspace,
)
from pysph.sph.fused_cuda_codegen import (
    CudaPairPrecompute,
    cubic_spline_gradient_precompute,
    fused_kernel_specs,
    generate_fused_kernel_outline,
    generate_hbucket_pair_stage_outline_from_equations,
    generate_pointwise_kernel_outline_with_equation_calls,
    generate_sorted_cell_pair_stage_outline_from_equations,
    hbucket_context_argument_declarations,
    launch_budget_for_specs,
    launch_hbucket_pair_kernel_with_context,
    launch_sorted_cell_pair_kernel_with_context,
    PairLaunchConfig,
    sorted_cell_context_argument_declarations,
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
    AccumulateDWIJ,
    AddMass,
    CopyAcceleration,
)


def require_cuda():
    pytest.importorskip("pycuda")
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except Exception as exc:
        pytest.skip("CUDA is not available: %s" % exc)
    if cuda.Device.count() == 0:
        pytest.skip("CUDA device is not available")
    return cuda


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


class FakeFunction:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeModule:
    def __init__(self):
        self.function = FakeFunction()

    def get_function(self, name):
        self.name = name
        return self.function


class FakeDeviceArray:
    def __init__(self, dtype_name):
        self.dtype = np.dtype(dtype_name)
        self.gpudata = object()
        self.shape = (1,)


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


def test_sorted_cell_pair_outline_uses_sorted_destination_order():
    deps = _sum_reduction_deps("AddMass", "au")
    stage = _stage(StageKind.PAIR_RATE, (deps,))
    equation = AddMass(dest="fluid", sources=["fluid"])

    outline = generate_sorted_cell_pair_stage_outline_from_equations(
        "plan0",
        stage,
        (equation,),
        CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=()),
        None,
    )

    for declaration in sorted_cell_context_argument_declarations():
        assert declaration in outline.source
    assert "int order = blockIdx.x * blockDim.x + threadIdx.x;" in outline.source
    assert "int dst = sorted_ids[order];" in outline.source
    assert "for (int oz = -search_radius_cells;" in outline.source
    assert "cell_starts[cell]" in outline.source
    assert "bucket_h_max_bits" not in outline.source
    assert "AddMass_loop(add_mass0, dst, src, d_au, s_m);" in outline.source


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


def test_sorted_cell_pair_launch_uses_cell_context_stream_and_grid(monkeypatch):
    monkeypatch.setattr(
        "pysph.sph.fused_cuda_codegen._cuda_multiprocessor_count", lambda: 128
    )
    module = FakeModule()
    context = SimpleNamespace(
        n=513,
        x=FakeDeviceArray("float32"),
        y=FakeDeviceArray("float32"),
        z=FakeDeviceArray("float32"),
        h=FakeDeviceArray("float32"),
        lower=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        periodic=np.array([True, False, True], dtype=np.bool_),
        radius_scale=np.float32(2.0),
        search_radius_cells=np.int32(1),
        cell_counts=np.array([4, 5, 6], dtype=np.int32),
        cell_particle_counts=FakeDeviceArray("int32"),
        cell_starts=FakeDeviceArray("int32"),
        sorted_ids=FakeDeviceArray("int32"),
        stream=object(),
    )

    launch_config = launch_sorted_cell_pair_kernel_with_context(
        module, "kernel", context, (np.uintp(0),)
    )

    assert launch_config == PairLaunchConfig(
        traversal="sorted_cell", n=513, block_size=128, grid_x=5
    )
    assert module.name == "kernel"
    args, kwargs = module.function.calls[0]
    assert args[15] == np.int32(1)
    assert args[16:19] == (np.int32(4), np.int32(5), np.int32(6))
    assert kwargs["block"] == (128, 1, 1)
    assert kwargs["grid"] == (5, 1, 1)
    assert kwargs["stream"] is context.stream


def test_pair_traversal_selects_sorted_cell_before_hbucket():
    assert (
        _pair_traversal_for_stage(SimpleNamespace(cell_particle_counts=object()))
        == "sorted_cell"
    )
    assert _pair_traversal_for_stage(SimpleNamespace(bucket_count=2)) == "hbucket"


def test_backend_records_pair_launch_config_counts():
    from pysph.sph.fused_cuda_stage_backend import GeneratedFusedCudaStageBackend

    backend = object.__new__(GeneratedFusedCudaStageBackend)
    backend.pair_launch_config_counts = {}

    backend._record_pair_launch_config(
        PairLaunchConfig(traversal="sorted_cell", n=513, block_size=128, grid_x=5)
    )
    backend._record_pair_launch_config(
        PairLaunchConfig(traversal="sorted_cell", n=513, block_size=128, grid_x=5)
    )
    backend._record_pair_launch_config(
        PairLaunchConfig(traversal="hbucket", n=1000, block_size=256, grid_x=4)
    )

    assert backend.pair_launch_config_counts == {
        ("sorted_cell", 513, 128, 5): 2,
        ("hbucket", 1000, 256, 4): 1,
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


def test_cuda_sorted_cell_add_mass_matches_low_variation_hbucket():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    rng = np.random.default_rng(20260705)
    n = 4096
    xyz = rng.random((n, 3), dtype=np.float32)
    mass = rng.random(n, dtype=np.float32)
    h = np.full(n, 0.06, dtype=np.float32)
    stream = cuda.Stream()
    d_x = gpuarray.to_gpu_async(xyz[:, 0], stream=stream)
    d_y = gpuarray.to_gpu_async(xyz[:, 1], stream=stream)
    d_z = gpuarray.to_gpu_async(xyz[:, 2], stream=stream)
    d_h = gpuarray.to_gpu_async(h, stream=stream)
    d_mass = gpuarray.to_gpu_async(mass, stream=stream)
    d_cell = gpuarray.to_gpu_async(np.zeros(n, dtype=np.float32), stream=stream)
    d_hbucket = gpuarray.to_gpu_async(np.zeros(n, dtype=np.float32), stream=stream)
    cell_context = build_fused_cuda_cell_context_with_workspace(
        d_x,
        d_y,
        d_z,
        d_h,
        n,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.float32(h.max()),
        stream,
        FusedCudaNeighborWorkspace(),
    )
    hbucket_context = build_fused_cuda_hbucket_context_with_workspace(
        d_x,
        d_y,
        d_z,
        d_h,
        n,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        4,
        stream,
        FusedCudaNeighborWorkspace(),
        [],
    )
    deps = _sum_reduction_deps("AddMass", "au")
    stage = _stage(StageKind.PAIR_RATE, (deps,))
    equation = AddMass(dest="fluid", sources=["fluid"])
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())
    cell_outline = generate_sorted_cell_pair_stage_outline_from_equations(
        "cell", stage, (equation,), precompute, None
    )
    hbucket_outline = generate_hbucket_pair_stage_outline_from_equations(
        "hbucket", stage, (equation,), precompute
    )
    cell_module = SourceModule(cell_outline.source, no_extern_c=True)
    hbucket_module = SourceModule(hbucket_outline.source, no_extern_c=True)

    launch_sorted_cell_pair_kernel_with_context(
        cell_module, cell_outline.name, cell_context, (np.uintp(0), d_cell, d_mass)
    )
    launch_hbucket_pair_kernel_with_context(
        hbucket_module,
        hbucket_outline.name,
        hbucket_context,
        (np.uintp(0), d_hbucket, d_mass),
    )
    got = d_cell.get_async(stream=stream)
    expected = d_hbucket.get_async(stream=stream)
    stream.synchronize()

    np.testing.assert_allclose(got, expected, rtol=2.0e-6, atol=2.0e-6)


def test_cuda_sorted_cell_gradient_kernel_matches_low_variation_hbucket():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)
    rng = np.random.default_rng(20260706)
    n = 4096
    xyz = rng.random((n, 3), dtype=np.float32)
    mass = rng.random(n, dtype=np.float32)
    h = np.full(n, 0.04, dtype=np.float32)
    stream = cuda.Stream()
    d_x = gpuarray.to_gpu_async(xyz[:, 0], stream=stream)
    d_y = gpuarray.to_gpu_async(xyz[:, 1], stream=stream)
    d_z = gpuarray.to_gpu_async(xyz[:, 2], stream=stream)
    d_h = gpuarray.to_gpu_async(h, stream=stream)
    d_mass = gpuarray.to_gpu_async(mass, stream=stream)
    d_cell = gpuarray.to_gpu_async(np.zeros(n, dtype=np.float32), stream=stream)
    d_hbucket = gpuarray.to_gpu_async(np.zeros(n, dtype=np.float32), stream=stream)
    cell_context = build_fused_cuda_cell_context_with_workspace(
        d_x,
        d_y,
        d_z,
        d_h,
        n,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.float32(h.max()),
        stream,
        FusedCudaNeighborWorkspace(),
    )
    hbucket_context = build_fused_cuda_hbucket_context_with_workspace(
        d_x,
        d_y,
        d_z,
        d_h,
        n,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        4,
        stream,
        FusedCudaNeighborWorkspace(),
        [],
    )
    deps = _sum_reduction_deps("AccumulateDWIJ", "au")
    deps = replace(deps, precomputed_symbols=frozenset(("DWIJ",)))
    stage = _stage(StageKind.PAIR_RATE, (deps,))
    equation = AccumulateDWIJ(dest="fluid", sources=["fluid"])
    precompute = cubic_spline_gradient_precompute(np.int32(3))
    cell_outline = generate_sorted_cell_pair_stage_outline_from_equations(
        "cell", stage, (equation,), precompute, None
    )
    hbucket_outline = generate_hbucket_pair_stage_outline_from_equations(
        "hbucket", stage, (equation,), precompute
    )
    cell_module = SourceModule(cell_outline.source, no_extern_c=True)
    hbucket_module = SourceModule(hbucket_outline.source, no_extern_c=True)

    launch_sorted_cell_pair_kernel_with_context(
        cell_module, cell_outline.name, cell_context, (np.uintp(0), d_cell, d_mass)
    )
    launch_hbucket_pair_kernel_with_context(
        hbucket_module,
        hbucket_outline.name,
        hbucket_context,
        (np.uintp(0), d_hbucket, d_mass),
    )
    got = d_cell.get_async(stream=stream)
    expected = d_hbucket.get_async(stream=stream)
    stream.synchronize()

    np.testing.assert_allclose(got, expected, rtol=2.0e-4, atol=5.0e-3)


def _call(equation, method_name, known_types, precomputed_symbols):
    from pysph.sph.fused_cuda_codegen import (
        cuda_equation_method_call_from_equation_with_precomputed,
    )

    return cuda_equation_method_call_from_equation_with_precomputed(
        equation, method_name, known_types, precomputed_symbols
    )
