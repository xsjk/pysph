import numpy as np
import pytest

from cyarray.carray import UIntArray

from compyle.config import get_config
from pysph.base.gpu_nnps import OctreeGPUNNPS
from pysph.base.kernels import CubicSpline
from pysph.base.nnps import DomainManager
from pysph.base.particle_array import get_ghost_tag
from pysph.base.tree.point_tree import PointTree
from pysph.base.utils import get_particle_array
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.sph.equation import Equation, Group
from pysph.sph.sph_compiler import SPHCompiler


class StoreNonSelfXIJ(Equation):
    def initialize(self, d_idx, d_xij_x, d_nbr_count):
        d_xij_x[d_idx] = 0.0
        d_nbr_count[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_xij_x, d_nbr_count, XIJ):
        if d_idx != s_idx:
            d_xij_x[d_idx] = XIJ[0]
            d_nbr_count[d_idx] += 1.0


class CountKernelImages(Equation):
    def initialize(self, d_idx, d_image_count, d_rij_sum):
        d_image_count[d_idx] = 0.0
        d_rij_sum[d_idx] = 0.0

    def loop(self, d_idx, d_image_count, d_rij_sum, WI, RIJ):
        if WI > 0.0:
            d_image_count[d_idx] += 1.0
            d_rij_sum[d_idx] += RIJ


@pytest.fixture
def cuda_config():
    pytest.importorskip("pycuda")
    cfg = get_config()
    original_use_cuda = cfg.use_cuda
    original_use_double = cfg.use_double
    original_use_local_memory = cfg.use_local_memory
    cfg.use_cuda = True
    cfg.use_double = False
    cfg.use_local_memory = False
    yield
    cfg.use_cuda = original_use_cuda
    cfg.use_double = original_use_double
    cfg.use_local_memory = original_use_local_memory


def _periodic_x_domain():
    return DomainManager(
        xmin=0.0,
        xmax=1.0,
        periodic_in_x=True,
        backend="cuda",
        periodic_mode="minimum_image",
    )


def _thin_periodic_z_domain():
    return DomainManager(
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
        zmin=0.0,
        zmax=0.1,
        periodic_in_z=True,
        backend="cuda",
        periodic_mode="minimum_image",
    )


def test_gpu_domain_manager_minimum_image_wraps_without_ghosts(cuda_config):
    pa = get_particle_array(
        name="fluid",
        x=np.array([-0.01, 1.01]),
        y=np.zeros(2),
        z=np.zeros(2),
        h=np.ones(2) * 0.1,
    )
    num_real = pa.num_real_particles
    domain = _periodic_x_domain()

    OctreeGPUNNPS(
        dim=3,
        particles=[pa],
        domain=domain,
        radius_scale=2.0,
        backend="cuda",
    )
    pa.gpu.pull("x", "tag")

    assert domain.manager.__class__.__name__ == "GPUDomainManager"
    assert pa.gpu.get_number_of_particles() == num_real
    assert pa.get_number_of_particles() == num_real
    assert np.count_nonzero(pa.tag == get_ghost_tag()) == 0
    np.testing.assert_allclose(pa.x, [0.99, 0.01], atol=1e-6)


def test_octree_gpu_nnps_minimum_image_finds_periodic_x_neighbors(cuda_config):
    pa = get_particle_array(
        name="fluid",
        x=np.array([0.01, 0.99]),
        y=np.zeros(2),
        z=np.zeros(2),
        h=np.ones(2) * 0.1,
    )
    domain = _periodic_x_domain()
    nnps = OctreeGPUNNPS(
        dim=3,
        particles=[pa],
        domain=domain,
        radius_scale=2.0,
        backend="cuda",
        leaf_size=8,
        use_elementwise=True,
    )

    nbrs = UIntArray()
    nnps.get_nearest_particles(src_index=0, dst_index=0, d_idx=0, nbrs=nbrs)
    neighbors = [nbrs[i] for i in range(nbrs.length)]
    pa.gpu.pull("tag")

    assert 1 in neighbors
    assert pa.gpu.get_number_of_particles() == pa.num_real_particles
    assert np.count_nonzero(pa.tag == get_ghost_tag()) == 0


def test_octree_gpu_nnps_reuses_neighbor_cids_for_same_context(cuda_config, monkeypatch):
    calls = []
    original = PointTree.find_neighbor_cids

    def counted_find_neighbor_cids(self, tree_src):
        calls.append((id(self), id(tree_src)))
        return original(self, tree_src)

    monkeypatch.setattr(PointTree, "find_neighbor_cids", counted_find_neighbor_cids)
    pa = get_particle_array(
        name="fluid",
        x=np.array([0.01, 0.25, 0.5, 0.99]),
        y=np.zeros(4),
        z=np.zeros(4),
        h=np.ones(4) * 0.1,
    )
    nnps = OctreeGPUNNPS(
        dim=3,
        particles=[pa],
        domain=_periodic_x_domain(),
        radius_scale=2.0,
        backend="cuda",
        leaf_size=8,
        use_elementwise=True,
    )

    nnps.set_context(0, 0)
    nnps.set_context(0, 0)
    assert len(calls) == 1

    nnps.update()
    nnps.set_context(0, 0)
    assert len(calls) == 2


def test_cuda_equation_loop_uses_minimum_image_xij_without_ghosts(cuda_config):
    pa = get_particle_array(
        name="fluid",
        x=np.array([0.01, 0.99]),
        y=np.zeros(2),
        z=np.zeros(2),
        h=np.ones(2) * 0.1,
        xij_x=np.zeros(2),
        nbr_count=np.zeros(2),
    )
    domain = _periodic_x_domain()
    kernel = CubicSpline(dim=3)
    equations = [
        Group(equations=[StoreNonSelfXIJ(dest="fluid", sources=["fluid"])])
    ]
    a_eval = AccelerationEval(
        particle_arrays=[pa],
        equations=equations,
        kernel=kernel,
        backend="cuda",
    )
    SPHCompiler(a_eval, integrator=None).compile()
    nnps = OctreeGPUNNPS(
        dim=3,
        particles=[pa],
        domain=domain,
        radius_scale=kernel.radius_scale,
        backend="cuda",
        leaf_size=8,
        use_elementwise=True,
    )
    a_eval.set_nnps(nnps)

    a_eval.compute(0.0, 0.1)
    pa.gpu.pull("xij_x", "nbr_count", "tag")

    assert pa.num_real_particles == 2
    assert domain.manager.periodic_in_x
    assert pa.gpu.get_number_of_particles() == pa.num_real_particles
    assert np.count_nonzero(pa.tag == get_ghost_tag()) == 0
    np.testing.assert_allclose(pa.nbr_count, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(pa.xij_x, [0.02, -0.02], atol=1e-6)


def test_cuda_equation_loop_counts_periodic_self_images_without_ghosts(cuda_config):
    pa = get_particle_array(
        name="fluid",
        x=np.array([0.5]),
        y=np.array([0.5]),
        z=np.array([0.05]),
        h=np.array([0.06]),
        image_count=np.zeros(1),
        rij_sum=np.zeros(1),
    )
    domain = _thin_periodic_z_domain()
    kernel = CubicSpline(dim=3)
    equations = [
        Group(equations=[CountKernelImages(dest="fluid", sources=["fluid"])])
    ]
    a_eval = AccelerationEval(
        particle_arrays=[pa],
        equations=equations,
        kernel=kernel,
        backend="cuda",
    )
    SPHCompiler(a_eval, integrator=None).compile()
    nnps = OctreeGPUNNPS(
        dim=3,
        particles=[pa],
        domain=domain,
        radius_scale=kernel.radius_scale,
        backend="cuda",
        leaf_size=8,
        use_elementwise=True,
    )
    a_eval.set_nnps(nnps)

    a_eval.compute(0.0, 0.1)
    pa.gpu.pull("image_count", "rij_sum", "tag")

    assert pa.gpu.get_number_of_particles() == pa.num_real_particles
    assert np.count_nonzero(pa.tag == get_ghost_tag()) == 0
    np.testing.assert_allclose(pa.image_count, [3.0], atol=1e-6)
    np.testing.assert_allclose(pa.rij_sum, [0.2], atol=1e-6)


def test_cuda_equation_loop_counts_wide_periodic_image_range(cuda_config):
    pa = get_particle_array(
        name="fluid",
        x=np.array([0.5]),
        y=np.array([0.5]),
        z=np.array([0.05]),
        h=np.array([0.11]),
        image_count=np.zeros(1),
        rij_sum=np.zeros(1),
    )
    domain = _thin_periodic_z_domain()
    kernel = CubicSpline(dim=3)
    equations = [
        Group(equations=[CountKernelImages(dest="fluid", sources=["fluid"])])
    ]
    a_eval = AccelerationEval(
        particle_arrays=[pa],
        equations=equations,
        kernel=kernel,
        backend="cuda",
    )
    SPHCompiler(a_eval, integrator=None).compile()
    nnps = OctreeGPUNNPS(
        dim=3,
        particles=[pa],
        domain=domain,
        radius_scale=kernel.radius_scale,
        backend="cuda",
        leaf_size=8,
        use_elementwise=True,
    )
    a_eval.set_nnps(nnps)

    a_eval.compute(0.0, 0.1)
    pa.gpu.pull("image_count", "rij_sum", "tag")

    assert pa.gpu.get_number_of_particles() == pa.num_real_particles
    assert np.count_nonzero(pa.tag == get_ghost_tag()) == 0
    np.testing.assert_allclose(pa.image_count, [5.0], atol=1e-6)
    np.testing.assert_allclose(pa.rij_sum, [0.6], atol=1e-6)


def test_cuda_local_memory_rejects_minimum_image_periodic(cuda_config):
    cfg = get_config()
    cfg.use_local_memory = True
    pa = get_particle_array(
        name="fluid",
        x=np.array([0.01, 0.99]),
        y=np.zeros(2),
        z=np.zeros(2),
        h=np.ones(2) * 0.1,
        xij_x=np.zeros(2),
        nbr_count=np.zeros(2),
    )
    domain = _periodic_x_domain()
    kernel = CubicSpline(dim=3)
    equations = [
        Group(equations=[StoreNonSelfXIJ(dest="fluid", sources=["fluid"])])
    ]
    a_eval = AccelerationEval(
        particle_arrays=[pa],
        equations=equations,
        kernel=kernel,
        backend="cuda",
    )

    with pytest.raises(NotImplementedError):
        SPHCompiler(a_eval, integrator=None).compile()
