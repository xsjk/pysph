from mako.template import Template
import os
import re

from compyle.profile import profile_kernel


CUDA_UINT8_PREAMBLE = r"""
typedef struct { unsigned int x, y, z, w, s4, s5, s6, s7; } uint8;
__device__ inline uint8 make_uint8(
    unsigned int x, unsigned int y, unsigned int z, unsigned int w,
    unsigned int s4, unsigned int s5, unsigned int s6, unsigned int s7)
{
    return {x, y, z, w, s4, s5, s6, s7};
}
"""


def _cuda_preamble(preamble="", source=""):
    from pycuda._cluda import CLUDA_PREAMBLE

    parts = [CLUDA_PREAMBLE, _cuda_uint8_preamble(source)]
    parts.append(preamble)
    return "\n".join(parts)


def _cuda_uint8_preamble(source):
    if "uint8" in source or "make_uint8" in source:
        return CUDA_UINT8_PREAMBLE
    return ""


def _convert_vector_fields(src):
    return (
        src.replace(".s0", ".x")
        .replace(".s1", ".y")
        .replace(".s2", ".z")
        .replace(".s3", ".w")
    )


def convert_code_for_backend(src, backend):
    if backend == "opencl":
        return src
    elif backend == "cuda":
        result = _convert_vector_fields(src)
        result = result.replace("__global ", "")
        result = result.replace("global ", "")
        result = result.replace("local ", "__shared__ ")
        result = result.replace("constant ", "const ")
        result = re.sub(
            r"(?<!__device__ )\binline\b", "__device__ inline", result
        )
        result = result.replace("barrier(CLK_LOCAL_MEM_FENCE)", "__syncthreads()")
        result = result.replace("PYOPENCL_ELWISE_CONTINUE", "continue")
        for name in ("contains", "contains_search", "intersects", "pass"):
            result = result.replace(
                "char %s(" % name, "__device__ inline char %s(" % name
            )
        result = result.replace("(uint2)(", "make_uint2(")
        result = result.replace("(uint4)(", "make_uint4(")
        result = result.replace("(uint8)(", "make_uint8(")
        result = re.sub(
            r"typedef\s+struct\s*\{.*?\}\s*float1\s*;", "", result, flags=re.S
        )
        result = re.sub(
            r"typedef\s+struct\s*\{.*?\}\s*double1\s*;", "", result, flags=re.S
        )
        return result
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)


def convert_arguments_for_backend(args, backend):
    if backend == "opencl":
        return args
    elif backend == "cuda":
        result = convert_code_for_backend(args, backend)
        result = result.replace("uint *", "unsigned int *")
        result = result.replace("ulong *", "unsigned long *")
        result = result.replace("uint ", "unsigned int ")
        result = result.replace("ulong ", "unsigned long ")
        return result
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)


def get_context(backend='opencl'):
    if backend == "opencl":
        from compyle.opencl import get_context as get_opencl_context

        return get_opencl_context()
    elif backend == "cuda":
        from compyle.cuda import set_context

        set_context()
        from pycuda.autoinit import context

        return context
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)


def get_queue(backend='opencl'):
    if backend == "opencl":
        from compyle.opencl import get_queue as get_opencl_queue

        return get_opencl_queue()
    elif backend == "cuda":
        return None
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)


def get_generic_scan_kernel(backend):
    if backend == "opencl":
        from pyopencl.scan import GenericScanKernel

        return GenericScanKernel
    elif backend == "cuda":
        from compyle.cuda import GenericScanKernel

        return GenericScanKernel
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)


def convert_kernel_kwargs_for_backend(kwargs, backend):
    if backend == "opencl":
        return kwargs
    elif backend == "cuda":
        result = dict(kwargs)
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)

    if "arguments" in result:
        result["arguments"] = convert_arguments_for_backend(
            result["arguments"], backend
        )

    for key in (
        "input_expr",
        "scan_expr",
        "neutral",
        "output_statement",
        "is_segment_start_expr",
        "preamble",
    ):
        if key in result and result[key] is not None:
            result[key] = convert_code_for_backend(result[key], backend)
    return result


def make_scan_kernel(backend, ctx, dtype, **kwargs):
    scan_kernel = get_generic_scan_kernel(backend)
    if backend == "opencl":
        return scan_kernel(ctx, dtype, **kwargs)
    elif backend == "cuda":
        kwargs = convert_kernel_kwargs_for_backend(kwargs, backend)
        source = "\n".join(
            v for v in kwargs.values() if isinstance(v, str)
        )
        kwargs["preamble"] = "\n".join(
            [_cuda_uint8_preamble(source), kwargs.get("preamble", "")]
        )
        return scan_kernel(dtype, **kwargs)
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)


class CUDASimpleKernel(object):
    """CUDA equivalent of compyle.opencl.SimpleKernel used by GPU NNPS."""

    def __init__(self, args, operation, wgs, name="", preamble=""):
        from pycuda.compiler import SourceModule

        source = r"""
        %(preamble)s

        extern "C" __global__ void %(name)s(%(args)s)
        {
            int lid = threadIdx.x;
            long i = blockIdx.x*blockDim.x + threadIdx.x;

            %(body)s
        }
        """ % dict(preamble=preamble, name=name, args=args, body=operation)
        self.module = SourceModule(source)
        self.knl = self.module.get_function(name)

    def __call__(self, *args, **kwargs):
        import pycuda.driver as drv

        kwargs.pop("queue", None)
        kwargs.pop("wait_for", None)
        gs = kwargs.pop("gs", None)
        ls = kwargs.pop("ls", None)
        time_kernel = kwargs.pop("time_kernel", False)
        if kwargs:
            raise TypeError("unknown keyword arguments: '%s'" % ", ".join(kwargs))
        if gs is None or ls is None:
            raise ValueError("gs and ls can not be empty")
        block_size = int(ls[0])
        grid = ((int(gs[0]) + block_size - 1) // block_size, 1)
        block = (block_size, 1, 1)
        if time_kernel:
            start = drv.Event()
            end = drv.Event()
            start.record()
            self.knl(*args, block=block, grid=grid)
            end.record()
            end.synchronize()
            return start.time_till(end) * 1e-3
        return self.knl(*args, block=block, grid=grid)


def get_simple_kernel(kernel_name, args, src, wgs, preamble="", backend="opencl"):
    if backend == "opencl":
        from compyle.opencl import SimpleKernel, get_context

        knl = SimpleKernel(
            get_context(), args, src, wgs, kernel_name, preamble=preamble
        )
    elif backend == "cuda":
        args = convert_arguments_for_backend(args, backend)
        src = convert_code_for_backend(src, backend)
        preamble = convert_code_for_backend(preamble, backend)
        preamble = _cuda_preamble(preamble, "\n".join([args, src, preamble]))
        knl = CUDASimpleKernel(args, src, wgs, kernel_name, preamble=preamble)
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)

    return profile_kernel(knl, kernel_name, backend=backend)


def get_elwise_kernel(kernel_name, args, src, preamble="", backend="opencl"):
    if backend == "opencl":
        from compyle.opencl import get_context
        from pyopencl.elementwise import ElementwiseKernel

        knl = ElementwiseKernel(
            get_context(), args, src,
            kernel_name, preamble=preamble
        )
    elif backend == "cuda":
        from pycuda.elementwise import ElementwiseKernel

        args = convert_arguments_for_backend(args, backend)
        src = convert_code_for_backend(src, backend)
        preamble = convert_code_for_backend(preamble, backend)
        preamble = _cuda_preamble(preamble, "\n".join([args, src, preamble]))
        knl = ElementwiseKernel(args, src, kernel_name, preamble=preamble)
    else:
        raise RuntimeError("Unsupported GPU backend %s" % backend)
    return profile_kernel(knl, kernel_name, backend=backend)


class GPUNNPSHelper(object):
    def __init__(self, tpl_filename, backend=None, use_double=False,
                 c_type=None):
        """

        Parameters
        ----------
        tpl_filename
            filename of source template
        backend
            backend to use for helper
        use_double:
            Use double precision floating point data types
        c_type:
            c_type to use. Overrides use_double
        """

        self.src_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                tpl_filename),
        )
        self.data_t = "double" if use_double else "float"

        if c_type is not None:
            self.data_t = c_type

        helper_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "gpu_helper_functions.mako"),
        )

        helper_preamble = helper_tpl.get_def("get_helpers").render(
            data_t=self.data_t
        )
        preamble = self.src_tpl.get_def("preamble").render(
            data_t=self.data_t
        )
        self.preamble = "\n".join([helper_preamble, preamble])
        self.cache = {}
        self.backend = backend or 'opencl'

    def _get_code(self, kernel_name, **kwargs):
        arguments = self.src_tpl.get_def("%s_args" % kernel_name).render(
            data_t=self.data_t, **kwargs)

        src = self.src_tpl.get_def("%s_src" % kernel_name).render(
            data_t=self.data_t, **kwargs)

        return arguments, src

    def get_kernel(self, kernel_name, **kwargs):
        key = kernel_name, tuple(kwargs.items())
        wgs = kwargs.get('wgs', None)

        if key in self.cache:
            return self.cache[key]
        else:
            args, src = self._get_code(kernel_name, **kwargs)

            if wgs is None:
                knl = get_elwise_kernel(kernel_name, args, src,
                                        preamble=self.preamble,
                                        backend=self.backend)
            else:
                knl = get_simple_kernel(kernel_name, args, src, wgs,
                                        preamble=self.preamble,
                                        backend=self.backend)

            self.cache[key] = knl
            return knl
