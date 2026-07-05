"""Tests for fused CUDA neighbor metadata."""

import sys
import types

import numpy as np
import pytest

import pysph.base.fused_cuda_nnps as fused_nnps
from pysph.base.fused_cuda_nnps import (
    FusedCudaHBucketNeighborContext,
    FusedCudaNeighborContext,
    FusedCudaNeighborWorkspace,
    brute_force_neighbor_indices,
    build_fused_cuda_cell_context_with_workspace,
    build_fused_cuda_context_with_workspace,
    build_fused_cuda_hbucket_context_with_workspace,
    build_fused_cuda_neighbor_context_with_workspace,
    cell_counts_from_cell_size,
    cell_counts_from_hmax,
    count_hbucket_traversal_work_from_context,
    count_neighbors_from_context,
    count_neighbors_from_hbucket_context,
    create_fused_cuda_convergence_flag,
    minimum_image_delta,
    read_fused_cuda_convergence_flag,
    reduce_max_float,
    reduce_min_float,
    reset_fused_cuda_convergence_flag,
    wrap_periodic_xyz,
)


class FakeDeviceArray:
    def __init__(self, dtype_name):
        self.dtype = np.dtype(dtype_name)
        self.gpudata = object()
        self.shape = (1,)


class FakeSizedDeviceArray:
    def __init__(self, size, dtype):
        self.dtype = np.dtype(dtype)
        self.shape = (size,)
        self.gpudata = object()


class RecordingKernel:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def __call__(self, *args, **kwargs):
        self.calls.append((self.name, args, kwargs))


class RecordingModule:
    def __init__(self, calls):
        self.calls = calls

    def get_function(self, name):
        return RecordingKernel(name, self.calls)


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


def _bounds():
    return (
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([True, True, True], dtype=np.bool_),
    )


def test_cell_counts_from_hmax_uses_radius_scale_and_minimum_one_cell():
    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 2.0, 0.1], dtype=np.float32)

    counts = cell_counts_from_hmax(
        lower,
        upper,
        np.float32(0.2),
        np.float32(2.0),
    )

    np.testing.assert_array_equal(counts, np.array([2, 5, 1], dtype=np.int32))


def test_cell_counts_from_cell_size_uses_minimum_one_cell():
    lower = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([1.0, 2.0, 0.1], dtype=np.float32)

    counts = cell_counts_from_cell_size(lower, upper, np.float32(0.4))

    np.testing.assert_array_equal(counts, np.array([2, 5, 1], dtype=np.int32))


def test_minimum_image_delta_matches_pysph_xij_direction():
    lower, upper, periodic = _bounds()

    delta = minimum_image_delta(
        np.array([0.01, 0.0, 0.0], dtype=np.float32),
        np.array([0.99, 0.0, 0.0], dtype=np.float32),
        lower,
        upper,
        periodic,
    )

    np.testing.assert_allclose(
        delta, np.array([0.02, 0.0, 0.0], dtype=np.float32), atol=1.0e-6
    )


def test_brute_force_neighbor_indices_use_max_h_and_strict_cutoff():
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.19, 0.0, 0.0],
            [0.21, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    h = np.array([0.05, 0.1, 0.1, 0.1], dtype=np.float32)
    lower = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    upper = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    periodic = np.array([True, True, True], dtype=np.bool_)

    neighbors = brute_force_neighbor_indices(
        xyz,
        h,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.int32(0),
    )

    np.testing.assert_array_equal(neighbors, np.array([0, 1], dtype=np.int32))


def test_context_declares_no_csr_and_separate_xyz_device_layout():
    context = FusedCudaNeighborContext(
        n=4,
        x=FakeDeviceArray("float32"),
        y=FakeDeviceArray("float32"),
        z=FakeDeviceArray("float32"),
        h=FakeDeviceArray("float32"),
        lower=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        periodic=np.array([True, True, False], dtype=np.bool_),
        radius_scale=np.float32(2.0),
        search_radius_cells=np.int32(1),
        cell_counts=np.array([4, 4, 1], dtype=np.int32),
        total_cells=16,
        stream=object(),
        sorted_ids=FakeDeviceArray("int32"),
        cell_starts=FakeDeviceArray("int32"),
        cell_particle_counts=FakeDeviceArray("int32"),
        timings_ms=(("build_cells_ms", 0.11),),
    )

    assert not context.materializes_csr
    assert context.coordinate_layout == "separate_xyz"
    assert context.uses_device_metadata
    assert context.cell_count_tuple == (4, 4, 1)
    assert context.search_radius_cells == np.int32(1)
    assert context.timing_total_ms == 0.11


def test_hbucket_context_declares_bucketed_device_metadata():
    context = FusedCudaHBucketNeighborContext(
        n=4,
        x=FakeDeviceArray("float32"),
        y=FakeDeviceArray("float32"),
        z=FakeDeviceArray("float32"),
        h=FakeDeviceArray("float32"),
        lower=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        periodic=np.array([True, True, False], dtype=np.bool_),
        radius_scale=np.float32(2.0),
        cell_counts=np.array([4, 4, 1], dtype=np.int32),
        total_cells=16,
        bucket_count=4,
        cell_width=np.array([0.25, 0.25, 1.0], dtype=np.float32),
        stream=object(),
        bucket_h_max_bits=FakeDeviceArray("uint32"),
        cell_bucket_h_max_bits=FakeDeviceArray("uint32"),
        sorted_ids=FakeDeviceArray("int32"),
        cell_bucket_starts=FakeDeviceArray("int32"),
        cell_bucket_counts=FakeDeviceArray("int32"),
        timings_ms=(("hbucket_build_ms", 0.11),),
    )

    assert not context.materializes_csr
    assert context.coordinate_layout == "separate_xyz"
    assert context.uses_device_metadata
    assert context.cell_count_tuple == (4, 4, 1)
    assert context.timing_total_ms == 0.11


def test_hbucket_builder_uses_device_hmax_bits_without_materialization(monkeypatch):
    calls = []
    scan_calls = []
    pycuda_module = types.ModuleType("pycuda")
    driver_module = types.ModuleType("pycuda.driver")
    driver_module.memcpy_htod_async = lambda *args: None
    pycuda_module.driver = driver_module
    monkeypatch.setitem(sys.modules, "pycuda", pycuda_module)
    monkeypatch.setitem(sys.modules, "pycuda.driver", driver_module)
    monkeypatch.setattr(fused_nnps, "_ensure_cuda_context", lambda: None)
    monkeypatch.setattr(
        fused_nnps,
        "_ensure_gpu_array",
        lambda array, size, dtype: FakeSizedDeviceArray(size, dtype),
    )
    monkeypatch.setattr(fused_nnps, "_module", lambda: RecordingModule(calls))
    monkeypatch.setattr(fused_nnps, "_event_pair", lambda stream: (object(), object()))
    monkeypatch.setattr(fused_nnps, "_finish_event", lambda start, stop, stream: 0.0)
    monkeypatch.setattr(
        fused_nnps,
        "_scan_int32",
        lambda input_array, output_array, stream: scan_calls.append(
            (input_array, output_array, stream)
        ),
    )

    context = fused_nnps._build_fused_cuda_hbucket_context_from_hmin(
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        8,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([True, True, True], dtype=np.bool_),
        np.float32(2.0),
        4,
        object(),
        FusedCudaNeighborWorkspace(),
        np.float32(0.1),
    )

    kernel_names = [name for name, _args, _kwargs in calls]
    assert kernel_names == [
        "fused_reset_hbucket_metadata",
        "fused_compute_hbucket_ids_counts_xyz",
        "fused_scatter_hbucket_sorted_particles",
    ]
    assert len(scan_calls) == 1
    assert context.bucket_count == 4
    assert context.cell_count_tuple == (5, 5, 5)
    assert context.bucket_h_max_bits.dtype == np.uint32
    assert context.cell_bucket_h_max_bits.dtype == np.uint32


def test_neighbor_builder_selects_fixed_cell_context_for_low_h_variation(monkeypatch):
    selected = []

    def fake_cell_context(*args):
        selected.append(("cell", args[9]))
        return "cell-context"

    monkeypatch.setattr(
        fused_nnps,
        "reduce_min_float",
        lambda array, n, stream, scratch: np.float32(0.1),
    )
    monkeypatch.setattr(
        fused_nnps,
        "reduce_max_float",
        lambda array, n, stream, scratch: np.float32(0.15),
    )
    monkeypatch.setattr(
        fused_nnps, "build_fused_cuda_cell_context_with_workspace", fake_cell_context
    )

    context = build_fused_cuda_neighbor_context_with_workspace(
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        8,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([True, True, True], dtype=np.bool_),
        np.float32(2.0),
        4,
        object(),
        FusedCudaNeighborWorkspace(),
        [],
    )

    assert context == "cell-context"
    assert selected == [("cell", np.float32(0.15))]


def test_neighbor_builder_uses_hbucket_context_for_large_h_variation(monkeypatch):
    selected = []

    def fake_hbucket_context(*args):
        selected.append((args[9], args[-1]))
        return "hbucket-context"

    monkeypatch.setattr(
        fused_nnps,
        "reduce_min_float",
        lambda array, n, stream, scratch: np.float32(0.1),
    )
    monkeypatch.setattr(
        fused_nnps,
        "reduce_max_float",
        lambda array, n, stream, scratch: np.float32(0.45),
    )
    monkeypatch.setattr(
        fused_nnps, "_build_fused_cuda_hbucket_context_from_hmin", fake_hbucket_context
    )

    context = build_fused_cuda_neighbor_context_with_workspace(
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        FakeSizedDeviceArray(8, np.float32),
        8,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([True, True, True], dtype=np.bool_),
        np.float32(2.0),
        4,
        object(),
        FusedCudaNeighborWorkspace(),
        [],
    )

    assert context == "hbucket-context"
    assert selected == [(2, np.float32(0.1))]


def test_hbucket_workspace_reuses_reference_h_bounds(monkeypatch):
    reduce_min_calls = []
    reduce_max_calls = []

    monkeypatch.setattr(
        fused_nnps,
        "reduce_min_float",
        lambda array, n, stream, scratch: reduce_min_calls.append(n) or np.float32(0.1),
    )
    monkeypatch.setattr(
        fused_nnps,
        "reduce_max_float",
        lambda array, n, stream, scratch: (
            reduce_max_calls.append(n) or np.float32(0.45)
        ),
    )
    monkeypatch.setattr(
        fused_nnps,
        "_build_fused_cuda_hbucket_context_from_hmin",
        lambda *args: object(),
    )
    workspace = FusedCudaNeighborWorkspace()
    h = FakeSizedDeviceArray(8, np.float32)

    for _ in range(2):
        build_fused_cuda_hbucket_context_with_workspace(
            FakeSizedDeviceArray(8, np.float32),
            FakeSizedDeviceArray(8, np.float32),
            FakeSizedDeviceArray(8, np.float32),
            h,
            8,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([True, True, True], dtype=np.bool_),
            np.float32(2.0),
            4,
            object(),
            workspace,
            [],
        )

    assert reduce_min_calls == [8]
    assert reduce_max_calls == [8]


def test_cuda_event_timing_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PYSPH_PROFILE_CUDA_EVENTS", raising=False)

    assert fused_nnps._event_pair(object()) == (None, None)
    assert fused_nnps._finish_event(None, None, object()) == 0.0


def test_hbucket_work_counter_launches_exact_traversal_kernel(monkeypatch):
    calls = []
    allocated = []
    pycuda_module = types.ModuleType("pycuda")
    gpuarray_module = types.ModuleType("pycuda.gpuarray")

    def fake_empty(shape, dtype):
        array = FakeSizedDeviceArray(shape[0], dtype)
        allocated.append(array)
        return array

    gpuarray_module.empty = fake_empty
    pycuda_module.gpuarray = gpuarray_module
    monkeypatch.setitem(sys.modules, "pycuda", pycuda_module)
    monkeypatch.setitem(sys.modules, "pycuda.gpuarray", gpuarray_module)
    monkeypatch.setattr(fused_nnps, "_ensure_cuda_context", lambda: None)
    monkeypatch.setattr(fused_nnps, "_module", lambda: RecordingModule(calls))
    stream = object()
    context = FusedCudaHBucketNeighborContext(
        n=8,
        x=FakeSizedDeviceArray(8, np.float32),
        y=FakeSizedDeviceArray(8, np.float32),
        z=FakeSizedDeviceArray(8, np.float32),
        h=FakeSizedDeviceArray(8, np.float32),
        lower=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        upper=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        periodic=np.array([True, True, True], dtype=np.bool_),
        radius_scale=np.float32(2.0),
        cell_counts=np.array([5, 5, 5], dtype=np.int32),
        total_cells=125,
        bucket_count=4,
        cell_width=np.array([0.2, 0.2, 0.2], dtype=np.float32),
        stream=stream,
        bucket_h_max_bits=FakeSizedDeviceArray(4, np.uint32),
        cell_bucket_h_max_bits=FakeSizedDeviceArray(500, np.uint32),
        sorted_ids=FakeSizedDeviceArray(8, np.int32),
        cell_bucket_starts=FakeSizedDeviceArray(500, np.int32),
        cell_bucket_counts=FakeSizedDeviceArray(500, np.int32),
        timings_ms=(("hbucket_build_ms", 0.0),),
    )

    cell_visits, candidate_counts, neighbor_counts = (
        count_hbucket_traversal_work_from_context(context)
    )

    assert (cell_visits, candidate_counts, neighbor_counts) == tuple(allocated)
    assert [array.dtype for array in allocated] == [np.int32, np.int32, np.int32]
    assert [array.shape for array in allocated] == [(8,), (8,), (8,)]
    assert [name for name, _args, _kwargs in calls] == [
        "fused_count_hbucket_traversal_work"
    ]
    assert calls[0][2]["grid"] == (1, 1, 1)
    assert calls[0][2]["stream"] is stream


def test_cuda_cell_builder_creates_no_csr_sorted_cell_metadata():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray

    stream = cuda.Stream()
    x = np.array([0.10, 0.90, 0.20, 0.80], dtype=np.float32)
    y = np.array([0.10, 0.10, 0.90, 0.90], dtype=np.float32)
    z = np.zeros(4, dtype=np.float32)
    h = np.full(4, 0.10, dtype=np.float32)
    lower, upper, periodic = _bounds()
    context = build_fused_cuda_cell_context_with_workspace(
        gpuarray.to_gpu_async(x, stream=stream),
        gpuarray.to_gpu_async(y, stream=stream),
        gpuarray.to_gpu_async(z, stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        4,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.float32(h.max()),
        stream,
        FusedCudaNeighborWorkspace(),
    )

    cell_counts = context.cell_particle_counts.get_async(stream=stream)
    sorted_ids = context.sorted_ids.get_async(stream=stream)
    stream.synchronize()

    assert not context.materializes_csr
    assert context.coordinate_layout == "separate_xyz"
    assert context.search_radius_cells == np.int32(1)
    assert context.cell_count_tuple == (5, 5, 5)
    assert int(cell_counts.sum()) == 4
    np.testing.assert_array_equal(np.sort(sorted_ids), np.arange(4, dtype=np.int32))


def test_cuda_context_workspace_reuses_cell_buffers():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray

    lower, upper, periodic = _bounds()
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    h = np.full(3, 0.10, dtype=np.float32)
    stream = cuda.Stream()
    workspace = FusedCudaNeighborWorkspace()
    context = build_fused_cuda_context_with_workspace(
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
        workspace,
    )
    first_sorted_ids = int(context.sorted_ids.gpudata)
    first_cell_counts = int(context.cell_particle_counts.gpudata)
    context = build_fused_cuda_context_with_workspace(
        context.x,
        context.y,
        context.z,
        context.h,
        3,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        np.array([5, 1, 1], dtype=np.int32),
        stream,
        workspace,
    )

    assert int(context.sorted_ids.gpudata) == first_sorted_ids
    assert int(context.cell_particle_counts.gpudata) == first_cell_counts


def test_cuda_context_neighbor_count_matches_periodic_bruteforce():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray

    lower, upper, periodic = _bounds()
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    h = np.full(3, 0.10, dtype=np.float32)
    stream = cuda.Stream()
    context = build_fused_cuda_context_with_workspace(
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
        FusedCudaNeighborWorkspace(),
    )

    gpu_counts = count_neighbors_from_context(context).get_async(stream=stream)
    stream.synchronize()
    expected = [
        brute_force_neighbor_indices(
            xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(index)
        ).size
        for index in range(3)
    ]

    np.testing.assert_array_equal(gpu_counts, np.asarray(expected, dtype=np.int32))


def test_cuda_neighbor_builder_cell_context_matches_hbucket_for_low_h_variation():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray

    lower, upper, periodic = _bounds()
    rng = np.random.default_rng(20260705)
    n = 2048
    xyz = rng.random((n, 3), dtype=np.float32)
    h = (0.016 * rng.uniform(1.0, 1.75, n)).astype(np.float32)
    stream = cuda.Stream()
    d_x = gpuarray.to_gpu_async(xyz[:, 0], stream=stream)
    d_y = gpuarray.to_gpu_async(xyz[:, 1], stream=stream)
    d_z = gpuarray.to_gpu_async(xyz[:, 2], stream=stream)
    d_h = gpuarray.to_gpu_async(h, stream=stream)
    cell_context = build_fused_cuda_neighbor_context_with_workspace(
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

    cell_counts = count_neighbors_from_context(cell_context).get_async(stream=stream)
    hbucket_counts = count_neighbors_from_hbucket_context(hbucket_context).get_async(
        stream=stream
    )
    stream.synchronize()

    assert isinstance(cell_context, FusedCudaNeighborContext)
    np.testing.assert_array_equal(cell_counts, hbucket_counts)


def test_cuda_hbucket_context_neighbor_count_matches_variable_h_bruteforce():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray

    lower, upper, periodic = _bounds()
    xyz = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [0.50, 0.0, 0.0],
            [0.25, 0.25, 0.0],
        ],
        dtype=np.float32,
    )
    h = np.array([0.05, 0.10, 0.20, 0.05], dtype=np.float32)
    stream = cuda.Stream()
    context = build_fused_cuda_hbucket_context_with_workspace(
        gpuarray.to_gpu_async(xyz[:, 0], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 1], stream=stream),
        gpuarray.to_gpu_async(xyz[:, 2], stream=stream),
        gpuarray.to_gpu_async(h, stream=stream),
        4,
        lower,
        upper,
        periodic,
        np.float32(2.0),
        4,
        stream,
        FusedCudaNeighborWorkspace(),
        [],
    )

    gpu_counts = count_neighbors_from_hbucket_context(context).get_async(stream=stream)
    stream.synchronize()
    expected = [
        brute_force_neighbor_indices(
            xyz, h, lower, upper, periodic, np.float32(2.0), np.int32(index)
        ).size
        for index in range(4)
    ]

    assert context.bucket_count == 2
    assert context.cell_count_tuple == (10, 10, 10)
    np.testing.assert_array_equal(gpu_counts, np.asarray(expected, dtype=np.int32))


def test_cuda_convergence_flag_resets_and_reads_from_device():
    cuda = require_cuda()

    stream = cuda.Stream()
    flag = create_fused_cuda_convergence_flag(stream)

    assert read_fused_cuda_convergence_flag(flag, stream)
    cuda.memset_d32_async(int(flag.gpudata), 0, 1, stream)
    assert not read_fused_cuda_convergence_flag(flag, stream)
    reset_fused_cuda_convergence_flag(flag, stream)
    assert read_fused_cuda_convergence_flag(flag, stream)


def test_cuda_reduce_max_float_reads_device_value():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray

    stream = cuda.Stream()
    values = gpuarray.to_gpu_async(
        np.array([0.10, 0.25, 0.15], dtype=np.float32),
        stream=stream,
    )

    maximum = reduce_max_float(values, 3, stream, [])

    assert maximum == np.float32(0.25)


def test_cuda_reduce_min_float_reads_device_value():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray

    stream = cuda.Stream()
    values = gpuarray.to_gpu_async(
        np.array([0.10, 0.25, 0.15], dtype=np.float32),
        stream=stream,
    )

    minimum = reduce_min_float(values, 3, stream, [])

    assert minimum == np.float32(0.10)


def test_cuda_wrap_periodic_xyz_updates_device_coordinates():
    cuda = require_cuda()
    import pycuda.gpuarray as gpuarray

    stream = cuda.Stream()
    x = gpuarray.to_gpu_async(
        np.array([-0.1, 0.2, 1.1], dtype=np.float32), stream=stream
    )
    y = gpuarray.to_gpu_async(
        np.array([0.4, -0.2, 1.3], dtype=np.float32), stream=stream
    )
    z = gpuarray.to_gpu_async(
        np.array([0.5, 0.6, 0.7], dtype=np.float32), stream=stream
    )

    wrap_periodic_xyz(
        x,
        y,
        z,
        3,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([True, True, False], dtype=np.bool_),
        stream,
    )

    x_host = x.get_async(stream=stream)
    y_host = y.get_async(stream=stream)
    z_host = z.get_async(stream=stream)
    stream.synchronize()
    np.testing.assert_allclose(
        x_host, np.array([0.9, 0.2, 0.1], dtype=np.float32), atol=1.0e-6
    )
    np.testing.assert_allclose(
        y_host, np.array([0.4, 0.8, 0.3], dtype=np.float32), atol=1.0e-6
    )
    np.testing.assert_allclose(
        z_host, np.array([0.5, 0.6, 0.7], dtype=np.float32), atol=1.0e-6
    )
