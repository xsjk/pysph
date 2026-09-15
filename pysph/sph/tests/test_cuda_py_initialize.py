import numpy as np
import pytest
from compyle.config import get_config
from compyle.low_level import atomic_min

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, Group


class RetryVote(Equation):
    def __init__(self, dest, sources):
        super().__init__(dest, sources)
        self.vote = 0
        self.calls = 0

    def py_initialize(self, dst, t, dt):
        self.calls += 1
        self.vote = 0

    def post_loop(self, d_idx, d_attempt):
        d_attempt[d_idx] += 1
        if d_idx == 0 and d_attempt[d_idx] < 2:
            atomic_min(self.vote, -1)

    def converged(self):
        self._pull("vote")
        return self.vote == 0


@pytest.mark.parametrize("fused", [True, False])
def test_cuda_py_initialize_resets_once_per_iteration(monkeypatch, fused):
    import pycuda.driver as cuda

    from pysph.base.z_order_gpu_nnps import ZOrderGPUNNPS
    from pysph.tools.sph_evaluator import SPHEvaluator

    cuda.init()
    assert cuda.Device.count() > 0
    monkeypatch.setattr(get_config(), "use_fused_cuda", fused, raising=False)
    monkeypatch.setattr(get_config(), "use_cuda", True)
    monkeypatch.setattr(get_config(), "use_double", False)
    x, y, z = np.indices((32, 32, 32), dtype=np.float64).reshape(3, -1)
    particle = get_particle_array(name="fluid", x=x, y=y, z=z, h=0.49, m=1.0)
    particle.add_property("attempt")
    group = Group(
        [RetryVote("fluid", None)], iterate=True, min_iterations=1, max_iterations=4
    )
    evaluator = SPHEvaluator(
        arrays=[particle],
        equations=[group],
        dim=3,
        kernel=CubicSpline(dim=3),
        backend="cuda",
        nnps_factory=ZOrderGPUNNPS,
    )
    evaluator.evaluate(t=0.0, dt=0.1)
    particle.gpu.pull("attempt")
    assert np.all(particle.attempt == 2)
    assert group.equations[0].calls == 2
