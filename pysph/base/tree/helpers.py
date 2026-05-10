import numpy as np
import pyopencl as cl
import pyopencl.cltypes
from pytools import memoize
from compyle.array import Array
from pysph.base.gpu_nnps_helper import GPUNNPSHelper, get_elwise_kernel


_cuda_vector_dtypes = None


def _get_cuda_vector_dtypes():
    global _cuda_vector_dtypes
    if _cuda_vector_dtypes is not None:
        return _cuda_vector_dtypes

    try:
        from pycuda.gpuarray import vec
        from pycuda.tools import get_or_register_dtype
        from compyle import types as compyle_types
    except ImportError as exc:
        raise RuntimeError("CUDA vector dtypes require PyCUDA") from exc

    uint8 = np.dtype([
        ("x", np.uint32),
        ("y", np.uint32),
        ("z", np.uint32),
        ("w", np.uint32),
        ("s4", np.uint32),
        ("s5", np.uint32),
        ("s6", np.uint32),
        ("s7", np.uint32),
    ], align=True)
    double3 = np.dtype([
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
    ], align=True)
    get_or_register_dtype("uint8", uint8)
    get_or_register_dtype("double3", double3)
    registered = {
        "uint2": vec.uint2,
        "uint4": vec.uint4,
        "uint8": uint8,
        "float2": vec.float2,
        "float3": vec.float3,
        "double2": vec.double2,
        "double3": double3,
    }
    for ctype, dtype in registered.items():
        compyle_types.NP_C_TYPE_MAP[np.dtype(dtype)] = ctype

    result = {
        'uint': {
            2: registered['uint2'],
            4: registered['uint4'],
            8: registered['uint8'],
        },
        'float': {
            1: np.float32,
            2: registered['float2'],
            3: registered['float3'],
        },
        'double': {
            1: np.float64,
            2: registered['double2'],
            3: registered['double3'],
        },
    }
    _cuda_vector_dtypes = result
    return result


def _make_cuda_vec(dtype, *values):
    ary = np.zeros(1, dtype=dtype)
    for name, value in zip(dtype.names, values):
        ary[name][0] = value
    return ary[0]


_opencl_make_vec = {
    'float': {
        1: np.float32,
        2: cl.cltypes.make_float2,
        3: cl.cltypes.make_float3
    },
    'double': {
        1: np.float64,
        2: cl.cltypes.make_double2,
        3: cl.cltypes.make_double3
    }
}


@memoize
def get_helper(src_file, c_type=None, backend='opencl'):
    # ctx and c_type are the only parameters that
    # change here
    return GPUNNPSHelper(src_file, backend=backend,
                         c_type=c_type)


@memoize
def get_copy_kernel(backend, dtype1, dtype2, varnames):
    arg_list = [('%(data_t1)s *%(v)s1' % dict(data_t1=dtype1, v=v))
                for v in varnames]
    arg_list += [('%(data_t2)s *%(v)s2' % dict(data_t2=dtype2, v=v))
                 for v in varnames]
    args = ', '.join(arg_list)

    operation = '; '.join(('%(v)s2[i] = (%(data_t2)s)%(v)s1[i];' %
                           dict(v=v, data_t2=dtype2))
                          for v in varnames)
    return get_elwise_kernel('copy_particle_array', args, operation,
                             backend=backend)


_opencl_vector_dtypes = {
    'uint': {
        2: cl.cltypes.uint2,
        4: cl.cltypes.uint4,
        8: cl.cltypes.uint8
    },
    'float': {
        1: cl.cltypes.float,
        2: cl.cltypes.float2,
        3: cl.cltypes.float3,
    },
    'double': {
        1: cl.cltypes.double,
        2: cl.cltypes.double2,
        3: cl.cltypes.double3
    }
}


def make_vec(backend, ctype, dim, *values):
    if backend == 'opencl':
        return _opencl_make_vec[ctype][dim](*values)
    if backend == 'cuda':
        if dim == 1:
            return ctype_to_dtype(ctype)(values[0])
        return _make_cuda_vec(get_vector_dtype(ctype, dim, backend), *values)
    raise RuntimeError("Unsupported GPU backend %s" % backend)


def get_vector_dtype(ctype, dim, backend='opencl'):
    try:
        if backend == 'opencl':
            return _opencl_vector_dtypes[ctype][dim]
        if backend == 'cuda':
            dtype = _get_cuda_vector_dtypes()[ctype][dim]
            if dtype is None:
                raise KeyError
            return dtype
    except KeyError:
        raise ValueError("Vector datatype of type %(ctype)s with %(dim)s items"
                         " is not supported" % dict(ctype=ctype, dim=dim))
    raise RuntimeError("Unsupported GPU backend %s" % backend)


def get_char_dtype(backend='opencl'):
    if backend == 'opencl':
        return cl.cltypes.char
    if backend == 'cuda':
        return np.int8
    raise RuntimeError("Unsupported GPU backend %s" % backend)


def set_uint2(array, x, y, backend='opencl'):
    if backend == 'opencl':
        array.dev[0].set(cl.cltypes.make_uint2(x, y))
    else:
        value = np.zeros(1, dtype=get_vector_dtype('uint', 2, backend))
        value['x'][0] = x
        value['y'][0] = y
        array.dev.set(value)


c2d = {
    'half': np.float16,
    'float': np.float32,
    'double': np.float64
}


def ctype_to_dtype(ctype):
    return c2d[ctype]


class GPUParticleArrayWrapper(object):
    def __init__(self, pa_gpu, c_type_src, c_type, varnames, backend):
        self.c_type = c_type
        self.c_type_src = c_type_src
        self.varnames = varnames
        self.backend = backend
        self._allocate_memory(pa_gpu)
        self.sync(pa_gpu)

    def _gpu_copy(self, pa_gpu):
        copy_kernel = get_copy_kernel(self.backend, self.c_type_src,
                                      self.c_type, self.varnames)
        args = [getattr(pa_gpu, v).dev for v in self.varnames]
        args += [getattr(self, v).dev for v in self.varnames]
        copy_kernel(*args)

    def _allocate_memory(self, pa_gpu):
        shape = getattr(pa_gpu, self.varnames[0]).dev.shape[0]
        for v in self.varnames:
            setattr(self, v,
                    Array(ctype_to_dtype(self.c_type),
                          n=shape, backend=self.backend))

    def _gpu_sync(self, pa_gpu):
        v0 = self.varnames[0]

        if getattr(self, v0).dev.shape != getattr(pa_gpu, v0).dev.shape:
            self._allocate_memory(pa_gpu)
        self._gpu_copy(pa_gpu)

    def sync(self, pa_gpu):
        self._gpu_sync(pa_gpu)


class ParticleArrayWrapper(object):
    """A loose wrapper over Particle Array

    Objective is to transparently maintain a copy of
    the original particle array's position properties
    (x, y, z, h)
    """

    def __init__(self, pa, c_type_src, c_type, varnames, backend='opencl'):
        self._pa = pa
        self.backend = backend
        # If data types are different, then make a copy of the
        # underlying data stored on the device
        if c_type_src != c_type:
            self._pa_gpu_is_copy = True
            self._gpu = GPUParticleArrayWrapper(pa.gpu, c_type_src,
                                                c_type, varnames, backend)
        else:
            self._pa_gpu_is_copy = False
            self._gpu = None

    def get_number_of_particles(self):
        return self._pa.get_number_of_particles()

    @property
    def gpu(self):
        if self._pa_gpu_is_copy:
            return self._gpu
        else:
            return self._pa.gpu

    def sync(self):
        if self._pa_gpu_is_copy:
            self._gpu.sync(self._pa.gpu)
