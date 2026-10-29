import numpy as np
import pytest

from compyle.array import Array
from compyle.config import get_config
from pysph.base.gpu_helper_kernels import (
    get_scan,
    neighbor_count_input,
    neighbor_count_output,
)
from pysph.base.octree_gpu_nnps import OctreeGPUNNPS
from pysph.base.utils import get_particle_array


@pytest.mark.parametrize("backend", ["cuda", "opencl"])
def test_scan_preserves_counts_and_returns_total(backend):
    scan = get_scan(neighbor_count_input, neighbor_count_output, np.uint32, backend)
    lengths = Array(np.uint32, backend=backend)
    starts = Array(np.uint32, backend=backend)
    total = Array(np.uint32, n=1, backend=backend)
    rng = np.random.default_rng(617)
    for count, high in ((1, 10), (513, 256), (17, 1), (1025, 256)):
        expected = rng.integers(0, high, count, dtype=np.uint32)
        lengths.set(expected)
        starts.resize(count)
        scan(lengths=lengths, starts=starts, total=total)
        np.testing.assert_array_equal(lengths.get(), expected)
        np.testing.assert_array_equal(starts.get(), np.cumsum(expected) - expected)
        assert total[0] == expected.sum()


@pytest.mark.parametrize("backend", ["cuda", "opencl"])
def test_empty_destination_cache(monkeypatch, backend):
    monkeypatch.setattr(get_config(), "use_double", False)
    particle = get_particle_array(
        name="fluid", x=[0.1, 0.9], h=[0.1, 0.1], backend=backend
    )
    nnps = OctreeGPUNNPS(dim=1, particles=[particle], leaf_size=16, backend=backend)
    nnps.set_context(0, 0)
    cache = nnps.current_cache
    cache.get_neighbors_gpu()
    assert cache._neighbors_gpu.length == 2
    particle.gpu.resize(0)
    cache.update()
    cache.get_neighbors_gpu()
    assert cache._neighbors_gpu.length == 0
    assert cache._start_idx_gpu.length == 0
    assert cache._nbr_lengths_gpu.length == 0
