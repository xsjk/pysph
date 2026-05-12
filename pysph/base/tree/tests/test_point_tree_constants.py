import numpy as np

from pysph.base.tree.point_tree import PointTree


def test_index_mask_is_not_computed_in_uint8():
    tree = PointTree.__new__(PointTree)
    tree.dim = 3
    tree.max_depth = 15

    mask, rshift = tree.get_index_constants(1)

    assert rshift == np.uint8(39)
    assert mask == np.uint64(7 << 39)
