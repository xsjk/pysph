'''This helper module orchestrates the generation of OpenCL/CUDA code, compiles
it and makes it available for use.

Overview
~~~~~~~~~

Look first at sph/tests/test_acceleration_eval.py to see the big picture. The
general idea when using AccelerationEval instances is:

- Create the particle arrays.
- Specify any equations and the SPH kernel.
- Construct the AccelerationEval with the particles, equations and kernel.
- Compile this with SPHCompiler and hand in an NNPS.
  - For the GPU all that changes is the backend and the NNPS.

So the difference in the CPU version and GPU version is the choice of the
backend. The AccelerationEval delegates its actual high-performance work to its
`self.c_acceleration_eval` instance. This instance is either compiled with
Cython or OpenCL. With Cython this is actually a compiled extension module
created with Cython and with OpenCL this is the Python class
OpenCLAccelerationEval in this file. This is where the helpers come in.


The AccelerationEvalCythonHelper and AccelerationEvalOpenCLHelper have three
main methods:

- get_code(): returns the code to be compiled.

- compile(code): compile the code and return the compiled module/opencl
  Program.

- setup_compiled_module(module): sets the AccelerationEval's
  c_acceleration_eval to an instance based on the helper.

The helper basically uses mako templates, code generation via simple string
manipulations, and transpilation to generate HPC code automatically from the
given particle arrays, equations, and kernel.

In this module, an OpenCLAccelerationEval is defined which does the work of
calling the compiled opencl kernels. The AccelerationEvalOpenCLHelper generates
the OpenCL kernels. The general idea of how we generate OpenCL kernels is quite
simple.

We transpile pure Python code using `pysph.base.translator` which generates C
from a subset of pure Python.

- We do not support inheritance but convert classes to simple C-structs and
  functions which take the struct as the first argument.
- Python functions are also transpiled.
- Type inference is done using either conventions like s_idx, d_idx, s_x, d_x,
  WIJ etc. or by type hints given using default arguments. Lists are treated as
  raw pointers to the contained type. One can also set certain predefined
  known_types and the code generator will generate suitable code. There are
  plenty of tests illustrating what is supported in
  ``pysph.base.tests.test_translator``.
- One can also use the ``declare`` function to declare any types in the Python
  code.

This is enough to do what we need. We transpile the kernel, all required
equations, and generate suitable kernels. All structs are converted to suitable
GPU types and the data from the Python classes is converted into suitably
aligned numpy dtypes (using cl.tools.match_dtype_to_c_struct). These are
constructed for each class and stored in an _gpu attribute on the Python
object.  When calling kernels these are passed and pushed/pulled from the GPU.

When the user calls AccelerationEval.compute, this in turn calls the
c_acceleration_eval's compute method. For OpenCL, this is provided by the
OpenCLAccelerationEval class below.

While the implementation is a bit complex, the details a bit hairy, the general
idea is very simple.

'''
import ast
from functools import partial
import inspect
import logging
import os
import re
from textwrap import wrap

import numpy as np
from mako.template import Template

from pysph.base.utils import is_overloaded_method
from pysph.base.device_helper import DeviceHelper

from pysph.sph.acceleration_nnps_helper import generate_body, \
    get_kernel_args_list

from compyle.ext_module import get_platform_dir, get_md5
from compyle.config import get_config
from compyle.profile import profile_ctx
from compyle.translator import (
    CStructHelper, CUDAConverter, OpenCLConverter, literal_to_float,
    ocl_detect_type, ocl_detect_pointer_base_type
)

from .equation import get_predefined_types, KnownType
from .acceleration_eval_cython_helper import (
    get_all_array_names, get_known_types_for_arrays
)

logger = logging.getLogger(__name__)

getfullargspec = inspect.getfullargspec


class Double2Float(ast.NodeTransformer):
    def __init__(self, use_double=False):
        super().__init__()
        self._use_double = use_double

    def visit_Num(self, node):
        s = literal_to_float(node.n, self._use_double)
        return ast.Constant(value=s)


def handle_precomp_literals(code, use_double=False):
    tree = ast.parse(code)
    transformed_tree = Double2Float(use_double).visit(tree)
    # The code below is a hack, since things like 0.5f are invalid Python, the
    # Double2Float transformer injects a string which we remove from the
    # unparsed source below.
    src = ast.unparse(transformed_tree).replace("'", "")
    return src


def periodic_kernel_args(data_t):
    return [
        'int periodic_in_x',
        'int periodic_in_y',
        'int periodic_in_z',
        f'{data_t} xtranslate',
        f'{data_t} ytranslate',
        f'{data_t} ztranslate',
    ]


def get_converter(backend):
    if backend == 'opencl':
        return OpenCLConverter
    elif backend == 'cuda':
        return CUDAConverter
    else:
        raise RuntimeError('Invalid backend: %s' % backend)


def get_kernel_definition(kernel, arg_list):
    sig = 'KERNEL void\n{kernel}\n({args})'.format(
        kernel=kernel, args=', '.join(arg_list),
    )
    return '\n'.join(wrap(sig, width=78, subsequent_indent=' ' * 4,
                          break_long_words=False))


def wrap_code(code, indent=' ' * 4):
    return wrap(
        code, width=74, initial_indent=indent,
        subsequent_indent=indent + ' ' * 4, break_long_words=False
    )


def get_helper_code(helpers, transpiler=None, backend=None):
    """This function generates any additional code for the given list of
    helpers.
    """
    result = []
    if transpiler is None:
        transpiler = get_converter(backend)
    doc = '\n// Helpers.\n'
    result.append(doc)
    for helper in helpers:
        result.append(transpiler.parse_function(helper))
    return result


class DummyQueue(object):
    def finish(self):
        pass


def get_context(backend):
    if backend == 'opencl':
        from compyle.opencl import get_context
        return get_context()
    elif backend == 'cuda':
        from compyle.cuda import set_context
        set_context()
        from pycuda.autoinit import context
        return context
    else:
        raise RuntimeError('Unsupported GPU backend %s' % backend)


def get_queue(backend):
    if backend == 'opencl':
        from compyle.opencl import get_queue
        return get_queue()
    elif backend == 'cuda':
        return DummyQueue()
    else:
        raise RuntimeError('Unsupported GPU backend %s' % backend)


def profile_kernel(knl, backend):
    if backend == 'opencl' or backend == 'cuda':
        from compyle.profile import profile_kernel
        return profile_kernel(knl, knl.function_name, backend=backend)
    else:
        raise RuntimeError('Unsupported GPU backend %s' % backend)


class GPUAccelerationEval(object):
    """Does the actual work of performing the evaluation.
    """

    def __init__(self, helper):
        self.helper = helper
        self.particle_arrays = helper.object.particle_arrays
        self.nnps = None
        self._queue = helper._queue
        cfg = get_config()
        self._use_double = cfg.use_double
        self._use_local_memory = cfg.use_local_memory
        self.stage_backend = None
        self.cuda_stage_plan = helper.cuda_stage_plan

    def _get_index(self, dest, attr):
        if isinstance(attr, str):
            return dest.get(attr)[0]
        else:
            return attr

    def _periodic_args(self):
        dtype = np.float64 if self._use_double else np.float32
        manager = self.nnps.domain.manager
        minimum_image_periodic = manager.minimum_image_periodic
        return [
            np.asarray(int(minimum_image_periodic and manager.periodic_in_x),
                       dtype=np.int32),
            np.asarray(int(minimum_image_periodic and manager.periodic_in_y),
                       dtype=np.int32),
            np.asarray(int(minimum_image_periodic and manager.periodic_in_z),
                       dtype=np.int32),
            np.asarray(manager.xtranslate, dtype=dtype),
            np.asarray(manager.ytranslate, dtype=dtype),
            np.asarray(manager.ztranslate, dtype=dtype),
        ]

    def _call_kernel(self, info, extra_args):
        nnps = self.nnps
        call = info.get('method')
        args = list(info.get('args'))
        dest = info['dest']
        start_idx = self._get_index(dest, info['start_idx'])
        stop_idx = info['stop_idx']
        if stop_idx is None:
            n = dest.get_number_of_particles(info.get('real', True))
        else:
            n = self._get_index(dest, stop_idx)
        args[1] = (n - start_idx,)
        args[3:] = [x() for x in args[3:]]
        # Argument for NP_MAX
        extra_args[-2][...] = n - 1
        # Argument for OFFSET_IDX
        extra_args[-1][...] = start_idx

        if info.get('loop'):
            if self._use_local_memory:
                nnps.set_context(info['src_idx'], info['dst_idx'])

                nnps_args, gs_ls = self.nnps.get_kernel_args('float')
                self._queue.finish()
                args[1] = gs_ls[0]
                args[2] = gs_ls[1]

                # No need for the guard variable for the local memory code.
                args = args + extra_args[:-2] + nnps_args

                call(*args)
                self._queue.finish()
            else:
                nnps.set_context(info['src_idx'], info['dst_idx'])
                cache = nnps.current_cache
                cache.get_neighbors_gpu()
                self._queue.finish()
                args = args + [
                    cache._nbr_lengths_gpu.dev.data,
                    cache._start_idx_gpu.dev.data,
                    cache._neighbors_gpu.dev.data
                ] + self._periodic_args() + extra_args
                call(*args)
        else:
            call(*(args + extra_args))
        self._queue.finish()

    def _sync_from_gpu(self, eq):
        ary = eq._gpu.get()
        for i, name in enumerate(ary.dtype.names):
            setattr(eq, name, ary[0][i])

    def _converged(self, equations):
        for eq in equations:
            if not (eq.converged() > 0):
                return False
        return True

    def _sync_before_host(self):
        self._queue.finish()

    def compute(self, t, dt):
        helper = self.helper
        dtype = np.float64 if self._use_double else np.float32
        extra_args = [np.asarray(t, dtype=dtype),
                      np.asarray(dt, dtype=dtype),
                      np.asarray(0, dtype=np.uint32),
                      np.asarray(0, dtype=np.uint32)]
        self._stage_backend_begin_compute(t, dt)
        i = 0
        iter_count = 0
        iter_start = 0
        while i < len(helper.calls):
            info = helper.calls[i]
            if self._stage_backend_handles_call(info, extra_args, t, dt):
                i += 1
                continue
            type = info['type']
            if type == 'method':
                self._sync_before_host()
                method_name = info.get('method')
                method = getattr(self, method_name)
                if method_name == 'do_reduce':
                    _args = info.get('args')
                    method(_args[0], _args[1], t, dt)
                else:
                    method(*info.get('args'))
            elif type == 'py_initialize':
                self._sync_before_host()
                args = info['dest'], t, dt
                for call in info['calls']:
                    call(*args)
            elif type == 'pre_post':
                self._sync_before_host()
                func = info.get('callable')
                func(*info.get('args'))
            elif type == 'kernel':
                if not self._stage_backend_handles_kernel(info, extra_args):
                    self._call_kernel(info, extra_args)
            elif type == 'check_condition':
                self._sync_before_host()
                group = info['group']
                if not group.condition(t, dt):
                    # Condition failed so skip group.
                    i = info['jump']
            elif type == 'start_iteration':
                iter_count = 0
                iter_start = i
            elif type == 'stop_iteration':
                eqs = info['equations']
                group = info['group']
                iter_count += 1
                has_min_iterations = iter_count >= group.min_iterations
                at_max_iterations = iter_count == group.max_iterations
                has_converged = False
                if has_min_iterations and not at_max_iterations:
                    self._sync_before_host()
                    has_converged = self._converged(eqs)
                if not (has_min_iterations and (at_max_iterations or has_converged)):
                    if group.update_nnps:
                        self._sync_before_host()
                        self.update_nnps()
                    i = iter_start
            i += 1
        if self._stage_backend_needs_final_sync(t, dt):
            self._sync_before_host()

    def _stage_backend_handles_kernel(self, info, extra_args):
        if self.stage_backend is None:
            return False
        return self.stage_backend.handle_kernel(self, info, extra_args)

    def _stage_backend_handles_call(self, info, extra_args, t, dt):
        if self.stage_backend is None:
            return False
        if not hasattr(self.stage_backend, "handle_call"):
            return False
        return self.stage_backend.handle_call(self, info, extra_args, t, dt)

    def _stage_backend_begin_compute(self, t, dt):
        if self.stage_backend is None:
            return
        if hasattr(self.stage_backend, "begin_compute"):
            self.stage_backend.begin_compute(self, t, dt)

    def _stage_backend_needs_final_sync(self, t, dt):
        if self.stage_backend is None:
            return True
        if not hasattr(self.stage_backend, "end_compute"):
            return True
        return self.stage_backend.end_compute(self, t, dt)

    def set_nnps(self, nnps):
        self.nnps = nnps

    def update_particle_arrays(self, arrays):
        raise NotImplementedError('GPU backend is incomplete')

    def _stage_backend_handles_internal_update_nnps(self):
        if self.stage_backend is None:
            return False
        if not hasattr(self.stage_backend, "handle_internal_update_nnps"):
            return False
        return self.stage_backend.handle_internal_update_nnps(self)

    def update_nnps(self):
        if self._stage_backend_handles_internal_update_nnps():
            return
        with profile_ctx('Integrator.update_domain'):
            self.nnps.update_domain()
        with profile_ctx('nnps.update'):
            self.nnps.update()

    def do_reduce(self, eqs, dest, t, dt):
        for eq in eqs:
            eq.reduce(dest, t, dt)


class CUDAAccelerationEval(GPUAccelerationEval):
    def _sync_before_host(self):
        pass

    def _call_kernel(self, info, extra_args):
        from pycuda.gpuarray import splay
        nnps = self.nnps
        call = info.get('method')
        args = list(info.get('args'))
        dest = info['dest']
        start_idx = self._get_index(dest, info['start_idx'])
        stop_idx = info['stop_idx']
        if stop_idx is None:
            n = dest.get_number_of_particles(info.get('real', True))
        else:
            n = self._get_index(dest, stop_idx)
        n_iter = int(n - start_idx)

        # args is actually [queue, None, None, actual_meaningful_args]
        # we do not need the first 3 args on CUDA.
        args = [x() for x in args[3:]]

        # Argument for NP_MAX
        extra_args[-2][...] = n - 1
        # Argument for OFFSET_IDX
        extra_args[-1][...] = start_idx

        gs, ls = splay(n_iter)
        gs, ls = int(gs[0]), int(ls[0])
        num_blocks = (n_iter + ls - 1) // ls

        #num_blocks = int((gs + ls - 1) / ls)
        num_tpb = ls

        if info.get('loop'):
            if self._use_local_memory:
                self._assert_local_memory_supported()
                # FIXME: Fix local memory for CUDA
                nnps.set_context(info['src_idx'], info['dst_idx'])

                nnps_args, gs_ls = self.nnps.get_kernel_args('float')
                args[1] = gs_ls[0]
                args[2] = gs_ls[1]

                # No need for the guard variable for the local memory code.
                args = args + extra_args[:-2] + nnps_args

                call(*args)
            else:
                # find block sizes
                nnps.set_context(info['src_idx'], info['dst_idx'])
                cache = nnps.current_cache
                cache.get_neighbors_gpu()
                args = args + [
                    cache._nbr_lengths_gpu.dev,
                    cache._start_idx_gpu.dev,
                    cache._neighbors_gpu.dev
                ] + self._periodic_args() + extra_args
                call(*args, block=(num_tpb, 1, 1), grid=(num_blocks, 1))
        else:
            call(*(args + extra_args),
                 block=(num_tpb, 1, 1),
                 grid=(num_blocks, 1))

    def _assert_local_memory_supported(self):
        manager = self.nnps.domain.manager
        if manager.minimum_image_periodic:
            msg = "CUDA local-memory loops do not support minimum-image periodic domains"
            raise NotImplementedError(msg)


def add_address_space(known_types):
    for v in known_types.values():
        if 'GLOBAL_MEM' not in v.type:
            v.type = 'GLOBAL_MEM ' + v.type


def get_equations_with_converged(group):
    def _get_eqs(g):
        if g.has_subgroups:
            res = []
            for x in g.equations:
                res.extend(_get_eqs(x))
            return res
        else:
            return g.equations

    eqs = [x for x in _get_eqs(group)
           if is_overloaded_method(x.converged)]
    return eqs


def _cuda_supported_convergence_equation_names(groups):
    names = set()
    for group in groups:
        names.update(_cuda_supported_convergence_names_for_group(group))
    return frozenset(names)


def _cuda_supported_convergence_names_for_group(group):
    names = set()
    if group.has_subgroups:
        for subgroup in group.equations:
            names.update(_cuda_supported_convergence_names_for_group(subgroup))
        return names
    for equation in getattr(group, 'equations', ()):
        if hasattr(equation, 'converged') and hasattr(
                equation, 'equation_has_converged'
        ):
            names.add(equation.__class__.__name__)
    return names


def _use_generated_fused_cuda_stage_backend():
    return bool(getattr(get_config(), 'use_fused_cuda', False))


def convert_to_float_if_needed(code):
    use_double = get_config().use_double
    if not use_double:
        code = re.sub(r'\bdouble\b', 'float', code)
    return code


class AccelerationEvalGPUHelper(object):
    def __init__(self, acceleration_eval):
        self.object = acceleration_eval
        self._use_double = get_config().use_double
        self.backend = acceleration_eval.backend
        self.all_array_names = get_all_array_names(
            self.object.particle_arrays
        )
        self.known_types = get_known_types_for_arrays(
            self.all_array_names
        )
        add_address_space(self.known_types)
        predefined = dict(get_predefined_types(
            self.object.all_group.pre_comp
        ))
        self.known_types.update(predefined)
        self.known_types['NBRS'] = KnownType('GLOBAL_MEM unsigned int*')
        self.data = []
        self._array_map = None
        self._array_index = None
        self._equations = {}
        self._cpu_structs = {}
        self._gpu_structs = {}
        self.calls = []
        self.program = None
        self._ctx = get_context(self.backend)
        self._queue = get_queue(self.backend)

    def _setup_arrays_on_device(self):
        pas = self.object.particle_arrays
        array_map = {}
        array_index = {}
        for idx, pa in enumerate(pas):
            if pa.gpu is None:
                pa.set_device_helper(DeviceHelper(pa, backend=self.backend))
            array_map[pa.name] = pa
            array_index[pa.name] = idx

        self._array_map = array_map
        self._array_index = array_index

        self._setup_structs_on_device()

    def _setup_structs_on_device(self):
        if self.backend == 'opencl':
            import pyopencl as cl
            import pyopencl.array  # noqa: F401
            import pyopencl.tools  # noqa: F401

            gpu = self._gpu_structs
            cpu = self._cpu_structs
            for k, v in cpu.items():
                if v is None:
                    gpu[k] = v
                else:
                    g_struct, code = cl.tools.match_dtype_to_c_struct(
                        self._ctx.devices[0], "dummy", v.dtype
                    )
                    g_v = v.astype(g_struct)
                    gpu[k] = cl.array.to_device(self._queue, g_v)
                    if k in self._equations:
                        self._equations[k]._gpu = gpu[k]
        else:
            from pycuda import gpuarray
            from compyle.cuda import match_dtype_to_c_struct

            gpu = self._gpu_structs
            cpu = self._cpu_structs
            for k, v in cpu.items():
                if v is None:
                    gpu[k] = v
                else:
                    g_struct, code = match_dtype_to_c_struct(
                        None, "junk", v.dtype
                    )
                    g_v = v.astype(g_struct)
                    gpu[k] = gpuarray.to_gpu(g_v)
                    if k in self._equations:
                        self._equations[k]._gpu = gpu[k]

    def _get_argument(self, arg, dest, src=None):
        ary_map = self._array_map
        structs = self._gpu_structs

        # This is needed for late binding on the device helper's attributes
        # which may change at each iteration when particles are added/removed.
        if self.backend == 'opencl':
            def _get_array(gpu_helper, attr):
                return getattr(gpu_helper, attr).dev.data
        else:
            def _get_array(gpu_helper, attr):
                return getattr(gpu_helper, attr).dev

        def _get_struct(obj):
            return obj

        if arg.startswith('d_'):
            return partial(_get_array, ary_map[dest].gpu, arg[2:])
        elif arg.startswith('s_'):
            return partial(_get_array, ary_map[src].gpu, arg[2:])
        else:
            if self.backend == 'opencl':
                return partial(_get_struct, structs[arg].data)
            else:
                return partial(_get_struct, structs[arg])

    def _setup_calls(self):
        calls = []
        prg = self.program
        array_index = self._array_index
        # Track the condition and end_group to facilitate group conditions.
        condition_stack = []
        count = 0
        for item in self.data:
            type = item.get('type')
            info = {}
            if type == 'kernel':
                kernel = item.get('kernel')
                method = getattr(prg, kernel)
                method = profile_kernel(method, self.backend)
                dest = item['dest']
                src = item.get('source', dest)
                args = [self._queue, None, None]
                for arg in item['args']:
                    args.append(self._get_argument(arg, dest, src))
                loop = item['loop']
                args.append(self._get_argument('kern', dest, src))
                info = dict(
                    method=method, dest=self._array_map[dest],
                    src=self._array_map[src], args=args,
                    loop=loop, src_idx=array_index[src],
                    dst_idx=array_index[dest],
                    start_idx=item.get('start_idx'),
                    stop_idx=item.get('stop_idx'),
                    stage_group=item.get('stage_group'),
                    stage_method_kind=item.get('stage_method_kind'),
                    type='kernel'
                )
            elif type == 'method':
                info = dict(item)
                if info.get('method') == 'do_reduce':
                    args = info.get('args')
                    grp = args[0]
                    args[0] = [
                        x for x in grp.equations
                        if is_overloaded_method(x.reduce)
                    ]
                    args[1] = self._array_map[args[1]]
            elif type == 'pre_post':
                info = dict(item)
            elif type == 'py_initialize':
                info = dict(item)
                info['dest'] = self._array_map[item.get('dest')]
            elif 'iteration' in type:
                group = item['group']
                equations = get_equations_with_converged(group._orig_group)
                info = dict(type=type, equations=equations, group=group)
            elif type == 'check_condition':
                info = dict(item)
                info['jump'] = None
                condition_stack.append(info)
            elif type == 'end_group':
                group = item['group']
                if group.condition is not None:
                    cond_info = condition_stack.pop()
                    # count is the location of the next call.
                    # We decrement it here as the counter is incremented
                    # in the helper.
                    cond_info['jump'] = count - 1
                info = {}
            else:
                raise RuntimeError('Unknown type %s' % type)
            if info:
                calls.append(info)
                count += 1
        return calls

    ##########################################################################
    # Public interface.
    ##########################################################################
    def get_code(self):
        path = os.path.join(os.path.dirname(__file__),
                            'acceleration_eval_gpu.mako')
        template = Template(filename=path)
        main = template.render(helper=self)
        self._setup_cuda_stage_plan()
        if self.backend == 'opencl':
            from pyopencl._cluda import CLUDA_PREAMBLE
        elif self.backend == 'cuda':
            from pycuda._cluda import CLUDA_PREAMBLE
        cluda = Template(CLUDA_PREAMBLE).render(double_support=self._use_double)
        main = "\n".join([cluda, main])
        return main

    def _setup_cuda_stage_plan(self):
        if self.backend != 'cuda':
            self.cuda_stage_plan = None
            return
        from pysph.sph.fused_cuda_stage_plan import plan_equation_groups

        supported_convergence = _cuda_supported_convergence_equation_names(
            self.object.equation_groups
        )
        self.cuda_stage_plan = plan_equation_groups(
            self.object.equation_groups, False, supported_convergence
        )

    def setup_compiled_module(self, module):
        object = self.object
        self._setup_arrays_on_device()
        self.calls = self._setup_calls()
        if self.backend == 'opencl':
            acceleration_eval = GPUAccelerationEval(self)
        elif self.backend == 'cuda':
            acceleration_eval = CUDAAccelerationEval(self)
            if _use_generated_fused_cuda_stage_backend():
                from pysph.sph.fused_cuda_stage_backend import (
                    GeneratedFusedCudaStageBackend,
                )

                acceleration_eval.stage_backend = GeneratedFusedCudaStageBackend(self)

        object.set_compiled_object(acceleration_eval)

    def compile(self, code):
        if self.backend == 'opencl':
            ext = '.cl'
            backend = 'OpenCL'
        elif self.backend == 'cuda':
            ext = '.cu'
            backend = 'CUDA'
        code = convert_to_float_if_needed(code)
        path = os.path.expanduser(os.path.join(
            '~', '.pysph', 'source', get_platform_dir()
        ))
        if not os.path.exists(path):
            os.makedirs(path)
        md5 = get_md5(code)
        fname = os.path.join(path, 'm_{0}{1}'.format(md5, ext))
        # If the file already exists, we use it, this allows to write our
        # own edited version if we wish to.
        if os.path.exists(fname):
            msg = 'Reading code from {}'.format(fname)
            logger.info(msg)
            print(msg)
            with open(fname, 'r') as fp:
                code = fp.read()
        else:
            with open(fname, 'w') as fp:
                fp.write(code)
            msg = "{backend} code written to {fname}".format(
                backend=backend, fname=fname
            )
            print(msg)
            logger.info(msg)

        if self.backend == 'opencl':
            import pyopencl as cl
            self.program = cl.Program(self._ctx, code).build(
                options=['-w']
            )
        elif self.backend == 'cuda':
            from compyle.cuda import SourceModule
            self.program = SourceModule(code)
        return self.program

    ##########################################################################
    # Mako interface.
    ##########################################################################
    def get_header(self):
        object = self.object
        Converter = get_converter(self.backend)
        transpiler = Converter(known_types=self.known_types)

        headers = []
        helpers = []
        if hasattr(object.kernel, '_get_helpers_'):
            helpers.extend(object.kernel._get_helpers_())
        for equation in object.all_group.equations:
            if hasattr(equation, '_get_helpers_'):
                for helper in equation._get_helpers_():
                    if helper not in helpers:
                        helpers.append(helper)
        headers.extend(get_helper_code(helpers, transpiler, self.backend))
        headers.append(transpiler.parse_instance(object.kernel))
        cls_name = object.kernel.__class__.__name__
        self.known_types['SPH_KERNEL'] = KnownType(
            'GLOBAL_MEM %s*' % cls_name, base_type=cls_name
        )
        headers.append(object.all_group.get_equation_wrappers(
            self.known_types
        ))
        # This is to be done after the above as the equation names are assigned
        # only at this point.
        cpu_structs = self._cpu_structs
        h = CStructHelper(object.kernel)
        cpu_structs['kern'] = h.get_array()
        for eq in object.all_group.equations:
            self._equations[eq.var_name] = eq
            h.parse(eq)
            cpu_structs[eq.var_name] = h.get_array()
        return '\n'.join(headers)

    def _get_arg_base_types(self, args):
        base_types = []
        for arg in args:
            base_types.append(
                ocl_detect_pointer_base_type(arg, self.known_types.get(arg))
            )
        return base_types

    def _get_typed_args(self, args):
        code = []
        for arg in args:
            type = ocl_detect_type(arg, self.known_types.get(arg))
            code.append('{type} {arg}'.format(
                type=type, arg=arg
            ))

        return code

    def _clean_kernel_args(self, args):
        remove = ('d_idx', 's_idx')
        for a in remove:
            if a in args:
                args.remove(a)

    def _get_simple_kernel(self, g_idx, sg_idx, group, dest, all_eqs, kind,
                           source=None):
        assert kind in ('initialize', 'initialize_pair', 'post_loop', 'loop')
        sub_grp = '' if sg_idx == -1 else 's{idx}'.format(idx=sg_idx)
        if source is None:
            kernel = 'g{g_idx}{sub}_{dest}_{kind}'.format(
                g_idx=g_idx, sub=sub_grp, dest=dest, kind=kind
            )
        else:
            kernel = 'g{g_idx}{sg}_{source}_on_{dest}_{kind}'.format(
                g_idx=g_idx, sg=sub_grp, source=source, dest=dest, kind=kind
            )

        sph_k_name = self.object.kernel.__class__.__name__
        code = [
            'int d_idx = GID_0 * LDIM_0 + LID_0 + OFFSET_IDX;',
            '/* Guard for padded threads. */',
            'if (d_idx > NP_MAX) {return;};',
            'GLOBAL_MEM %s* SPH_KERNEL = kern;' % sph_k_name
        ]
        all_args, py_args, _calls = self._get_equation_method_calls(
            all_eqs, kind, indent=''
        )
        code.extend(_calls)

        s_ary, d_ary = all_eqs.get_array_names()
        if source is None:
            # We only need the dest arrays here as these are simple kernels
            # without a loop so there is no "source".
            _args = sorted(d_ary)
        else:
            d_ary.update(s_ary)
            _args = sorted(d_ary)
        py_args.extend(_args)
        all_args.extend(self._get_typed_args(_args))
        all_args.extend(
            ['GLOBAL_MEM {kernel}* kern'.format(kernel=sph_k_name),
             'double t', 'double dt', 'unsigned int NP_MAX',
             'unsigned int OFFSET_IDX']
        )

        body = '\n'.join([' ' * 4 + x for x in code])
        self.data.append(dict(
            kernel=kernel, args=py_args, dest=dest, loop=False,
            real=group.real, start_idx=group.start_idx,
            stop_idx=group.stop_idx, type='kernel',
            stage_group=(g_idx, sg_idx), stage_method_kind=kind
        ))

        sig = get_kernel_definition(kernel, all_args)
        return (
            '{sig}\n{{\n{body}\n}}'.format(
                sig=sig, body=body
            )
        )

    def _get_equation_method_calls(self, eq_group, kind, indent=''):
        all_args = []
        py_args = []
        code = []
        for eq in eq_group.equations:
            method = getattr(eq, kind, None)
            if method is not None:
                cls = eq.__class__.__name__
                arg = 'GLOBAL_MEM {cls}* {name}'.format(
                    cls=cls, name=eq.var_name
                )
                all_args.append(arg)
                py_args.append(eq.var_name)
                call_args = list(getfullargspec(method).args)
                if 'self' in call_args:
                    call_args.remove('self')
                call_args.insert(0, eq.var_name)

                code.extend(
                    wrap_code(
                        '{cls}_{kind}({args});'.format(
                            cls=cls, kind=kind, args=', '.join(call_args)
                        ),
                        indent=indent
                    )
                )

        return all_args, py_args, code

    def _declare_precomp_vars(self, context):
        decl = []
        names = sorted(context.keys())
        for var in names:
            value = context[var]
            if isinstance(value, int):
                declare = 'long '
                decl.append('{declare}{var} = {value};'.format(
                    declare=declare, var=var, value=value
                ))
            elif isinstance(value, float):
                declare = 'double '
                decl.append('{declare}{var} = {value};'.format(
                    declare=declare, var=var,
                    value=literal_to_float(value, self._use_double)
                ))
            elif isinstance(value, (list, tuple)):
                decl.append(
                    'double {var}[{size}];'.format(
                        var=var, size=len(value)
                    )
                )
        return decl

    def _precomp_lines(self, code, periodic_images=False):
        src = code.strip().splitlines()
        lines = []
        for line in src:
            line = handle_precomp_literals(line, self._use_double)
            stripped = line.strip()
            lines.extend(
                self._minimum_image_xij_lines(
                    stripped, periodic_images=periodic_images
                )
            )
        return lines

    def _minimum_image_xij_lines(self, line, periodic_images=False):
        corrections = {
            'XIJ[0] = d_x[d_idx] - s_x[s_idx]': (
                'periodic_in_x', 'xtranslate', '_periodic_ix'
            ),
            'XIJ[1] = d_y[d_idx] - s_y[s_idx]': (
                'periodic_in_y', 'ytranslate', '_periodic_iy'
            ),
            'XIJ[2] = d_z[d_idx] - s_z[s_idx]': (
                'periodic_in_z', 'ztranslate', '_periodic_iz'
            ),
        }
        if line not in corrections:
            return [line + ';']
        periodic_flag, translate, image_shift = corrections[line]
        lhs = line.split(' = ')[0]
        lines = [
            line + ';',
            'if ({flag} && {lhs} > 0.5*{translate}) {{ {lhs} -= {translate}; }}'.format(
                flag=periodic_flag, lhs=lhs, translate=translate,
            ),
            'if ({flag} && {lhs} < -0.5*{translate}) {{ {lhs} += {translate}; }}'.format(
                flag=periodic_flag, lhs=lhs, translate=translate,
            ),
        ]
        if periodic_images:
            lines.append(
                '{lhs} += {shift} * {translate};'.format(
                    lhs=lhs, shift=image_shift, translate=translate,
                )
            )
        return lines

    def _periodic_image_radius_expr(self, d_ary, s_ary):
        if 'd_h' in d_ary and 's_h' in s_ary:
            return '((d_h[d_idx] > s_h[s_idx]) ? d_h[d_idx] : s_h[s_idx])'
        if 'd_h' in d_ary:
            return 'd_h[d_idx]'
        if 's_h' in s_ary:
            return 's_h[s_idx]'
        return None

    def _periodic_image_setup_lines(self, radius_expr):
        data_t = 'double' if self._use_double else 'float'
        floor_fn = (
            'floor' if self.backend == 'opencl' or self._use_double
            else 'floorf'
        )
        half = literal_to_float(0.5, self._use_double)
        return [
            '{data_t} _periodic_radius = kern->radius_scale * ({radius});'.format(
                data_t=data_t, radius=radius_expr
            ),
            '{data_t} _periodic_radius2 = _periodic_radius * _periodic_radius;'.format(
                data_t=data_t
            ),
            'int _periodic_ix_count = periodic_in_x ? (int){floor_fn}(_periodic_radius / xtranslate + {half}) : 0;'.format(
                floor_fn=floor_fn, half=half
            ),
            'int _periodic_iy_count = periodic_in_y ? (int){floor_fn}(_periodic_radius / ytranslate + {half}) : 0;'.format(
                floor_fn=floor_fn, half=half
            ),
            'int _periodic_iz_count = periodic_in_z ? (int){floor_fn}(_periodic_radius / ztranslate + {half}) : 0;'.format(
                floor_fn=floor_fn, half=half
            ),
            'int _periodic_ix_min = -_periodic_ix_count;',
            'int _periodic_ix_max = _periodic_ix_count;',
            'int _periodic_iy_min = -_periodic_iy_count;',
            'int _periodic_iy_max = _periodic_iy_count;',
            'int _periodic_iz_min = -_periodic_iz_count;',
            'int _periodic_iz_max = _periodic_iz_count;',
        ]

    def _precomputed_has_r2ij(self, eq_group):
        for cb in eq_group.precomputed.values():
            for line in cb.code.strip().splitlines():
                if line.strip().startswith('R2IJ = '):
                    return True
        return False

    def _set_kernel(self, code, kernel):
        if kernel is not None:
            name = kernel.__class__.__name__
            kern = '%s_kernel(kern, ' % name
            grad = '%s_gradient(kern, ' % name
            grad_h = '%s_gradient_h(kern, ' % name
            deltap = '%s_get_deltap(kern)' % name

            code = code.replace('DELTAP', deltap).replace('GRADIENT(', grad)
            return code.replace('KERNEL(', kern).replace('GRADH(', grad_h)
        else:
            return code

    def call_post(self, group):
        self.data.append(dict(callable=group.post, type='pre_post', args=()))

    def call_pre(self, group):
        self.data.append(dict(callable=group.pre, type='pre_post', args=()))

    def call_py_initialize(self, all_eq_group, dest):
        calls = []
        for eq in all_eq_group.equations:
            method = eq.py_initialize
            if is_overloaded_method(method):
                calls.append(method)
        if len(calls) > 0:
            self.data.append(
                dict(calls=calls, type='py_initialize', dest=dest)
            )

    def call_reduce(self, all_eq_group, dest):
        self.data.append(dict(method='do_reduce', type='method',
                              args=[all_eq_group, dest]))

    def call_update_nnps(self, group):
        self.data.append(dict(method='update_nnps',
                              type='method', args=[]))

    def get_initialize_kernel(self, g_idx, sg_idx, group, dest, all_eqs):
        return self._get_simple_kernel(
            g_idx, sg_idx, group, dest, all_eqs, kind='initialize'
        )

    def get_initialize_pair_kernel(self, g_idx, sg_idx, group, dest, source,
                                   eq_group):
        return self._get_simple_kernel(
            g_idx, sg_idx, group, dest, eq_group, kind='initialize_pair',
            source=source
        )

    def get_simple_loop_kernel(self, g_idx, sg_idx, group, dest, all_eqs):
        return self._get_simple_kernel(
            g_idx, sg_idx, group, dest, all_eqs, kind='loop'
        )

    def get_post_loop_kernel(self, g_idx, sg_idx, group, dest, all_eqs):
        return self._get_simple_kernel(
            g_idx, sg_idx, group, dest, all_eqs, kind='post_loop'
        )

    def get_loop_kernel(self, g_idx, sg_idx, group, dest, source, eq_group):
        if get_config().use_local_memory:
            if self.backend == 'cuda':
                msg = "CUDA local-memory loops are not supported"
                raise NotImplementedError(msg)
            return self.get_lmem_loop_kernel(g_idx, sg_idx, group,
                                             dest, source, eq_group)
        kind = 'loop'
        sub_grp = '' if sg_idx == -1 else 's{idx}'.format(idx=sg_idx)
        kernel = 'g{g_idx}{sg}_{source}_on_{dest}_loop'.format(
            g_idx=g_idx, sg=sub_grp, source=source, dest=dest
        )
        sph_k_name = self.object.kernel.__class__.__name__
        context = eq_group.context
        all_args, py_args = [], []
        s_ary, d_ary = eq_group.get_array_names()
        periodic_image_radius = self._periodic_image_radius_expr(d_ary, s_ary)
        use_periodic_images = (
            periodic_image_radius is not None
            and self._precomputed_has_r2ij(eq_group)
        )
        code = self._declare_precomp_vars(context)
        code.extend([
            'unsigned int d_idx = GID_0 * LDIM_0 + LID_0 + OFFSET_IDX;',
            '/* Guard for padded threads. */',
            'if (d_idx > NP_MAX) {return;};',
            'unsigned int s_idx, i;',
            'GLOBAL_MEM %s* SPH_KERNEL = kern;' % sph_k_name,
            'unsigned int start = start_idx[d_idx];',
            'GLOBAL_MEM unsigned int* NBRS = &(neighbors[start]);',
            'int N_NBRS = nbr_length[d_idx];',
            'unsigned int end = start + N_NBRS;'
        ])
        if eq_group.has_loop_all():
            _all_args, _py_args, _calls = self._get_equation_method_calls(
                eq_group, kind='loop_all', indent=''
            )
            code.extend(['', '// Calling loop_all of equations.'])
            code.extend(_calls)
            code.append('')
            all_args.extend(_all_args)
            py_args.extend(_py_args)

        if eq_group.has_loop():
            code.append('// Calling loop of equations.')
            code.append('for (i=start; i<end; i++) {')
            code.append('    s_idx = neighbors[i];')
            pre = []
            for p, cb in eq_group.precomputed.items():
                pre.extend(
                    [
                        (' ' * 12 if use_periodic_images else ' ' * 4) + x
                        for x in self._precomp_lines(
                            cb.code, periodic_images=use_periodic_images
                        )
                    ]
                )
            if len(pre) > 0:
                pre.append('')

            _all_args, _py_args, _calls = self._get_equation_method_calls(
                eq_group, kind,
                indent='                ' if use_periodic_images else '    '
            )
            if use_periodic_images:
                code.extend(
                    [
                        '    ' + line
                        for line in self._periodic_image_setup_lines(
                            periodic_image_radius
                        )
                    ]
                )
                code.append(
                    '    for (int _periodic_ix = _periodic_ix_min; _periodic_ix <= _periodic_ix_max; _periodic_ix++) {'
                )
                code.append(
                    '        for (int _periodic_iy = _periodic_iy_min; _periodic_iy <= _periodic_iy_max; _periodic_iy++) {'
                )
                code.append(
                    '            for (int _periodic_iz = _periodic_iz_min; _periodic_iz <= _periodic_iz_max; _periodic_iz++) {'
                )
                code.extend(pre)
                code.append('            if (R2IJ <= _periodic_radius2) {')
                code.extend(_calls)
                code.append('            }')
                code.append('            }')
                code.append('        }')
                code.append('    }')
            else:
                code.extend(pre)
                code.extend(_calls)
            for arg, py_arg in zip(_all_args, _py_args):
                if arg not in all_args:
                    all_args.append(arg)
                    py_args.append(py_arg)
            code.append('}')

        s_ary.update(d_ary)

        _args = sorted(s_ary)
        py_args.extend(_args)
        all_args.extend(self._get_typed_args(_args))

        body = '\n'.join([' ' * 4 + x for x in code])
        body = self._set_kernel(body, self.object.kernel)

        all_args.extend(
            ['GLOBAL_MEM {kernel}* kern'.format(kernel=sph_k_name),
             'GLOBAL_MEM unsigned int *nbr_length',
             'GLOBAL_MEM unsigned int *start_idx',
             'GLOBAL_MEM unsigned int *neighbors',
             *periodic_kernel_args('double' if self._use_double else 'float'),
             'double t', 'double dt', 'unsigned int NP_MAX',
             'unsigned int OFFSET_IDX']
        )

        self.data.append(dict(
            kernel=kernel, args=py_args, dest=dest, source=source, loop=True,
            real=group.real, start_idx=group.start_idx,
            stop_idx=group.stop_idx, type='kernel',
            stage_group=(g_idx, sg_idx), stage_method_kind=kind
        ))

        sig = get_kernel_definition(kernel, all_args)
        return (
            '{sig}\n{{\n{body}\n\n}}\n'.format(
                sig=sig, body=body
            )
        )

    def get_lmem_loop_kernel(self, g_idx, sg_idx, group, dest, source,
                             eq_group):
        kind = 'loop'
        sub_grp = '' if sg_idx == -1 else 's{idx}'.format(idx=sg_idx)
        kernel = 'g{g_idx}{sg}_{source}_on_{dest}_loop'.format(
            g_idx=g_idx, sg=sub_grp, source=source, dest=dest
        )
        sph_k_name = self.object.kernel.__class__.__name__
        context = eq_group.context
        all_args, py_args = [], []
        setup_code = self._declare_precomp_vars(context)
        setup_code.append('GLOBAL_MEM %s* SPH_KERNEL = kern;' % sph_k_name)

        if eq_group.has_loop_all():
            raise NotImplementedError("loop_all not suported with local "
                                      "memory")
        if group.start_idx > 0 or group.stop_idx is not None:
            raise NotImplementedError(
                "start/stop_idx not supported with local memory."
            )

        loop_code = []
        pre = []
        for p, cb in eq_group.precomputed.items():
            pre.extend([' ' * 4 + x for x in self._precomp_lines(cb.code)])
        if len(pre) > 0:
            pre.append('')
        loop_code.extend(pre)

        _all_args, _py_args, _calls = self._get_equation_method_calls(
            eq_group, kind, indent='    '
        )
        loop_code.extend(_calls)
        for arg, py_arg in zip(_all_args, _py_args):
            if arg not in all_args:
                all_args.append(arg)
                py_args.append(py_arg)

        s_ary, d_ary = eq_group.get_array_names()

        source_vars = set(s_ary)
        source_var_types = self._get_arg_base_types(source_vars)

        def modify_var_name(x):
            if x.startswith('s_'):
                return x + '_global'
            else:
                return x

        s_ary.update(d_ary)

        _args = sorted(s_ary)
        py_args.extend(_args)

        _args_modified = [modify_var_name(x) for x in _args]
        all_args.extend(self._get_typed_args(_args_modified))

        setup_body = '\n'.join([' ' * 4 + x for x in setup_code])
        setup_body = self._set_kernel(setup_body, self.object.kernel)

        loop_body = '\n'.join([' ' * 4 + x for x in loop_code])
        loop_body = self._set_kernel(loop_body, self.object.kernel)

        all_args.extend(
            ['GLOBAL_MEM {kernel}* kern'.format(kernel=sph_k_name),
             'double t', 'double dt']
        )
        all_args.extend(get_kernel_args_list())

        self.data.append(dict(
            kernel=kernel, args=py_args, dest=dest, source=source, loop=True,
            real=group.real, start_idx=group.start_idx,
            stop_idx=group.stop_idx, type='kernel',
            stage_group=(g_idx, sg_idx), stage_method_kind=kind
        ))

        body = generate_body(setup=setup_body, loop=loop_body,
                             vars=source_vars, types=source_var_types,
                             wgs=get_config().wgs)

        sig = get_kernel_definition(kernel, all_args)
        return (
            '{sig}\n{{\n{body}\n\n}}\n'.format(
                sig=sig, body=body
            )
        )

    def check_condition(self, group):
        self.data.append(dict(
            type='check_condition', group=group
        ))

    def end_group(self, group):
        self.data.append(dict(
            type='end_group', group=group
        ))

    def start_iteration(self, group):
        self.data.append(dict(
            type='start_iteration', group=group
        ))

    def stop_iteration(self, group):
        self.data.append(dict(
            type='stop_iteration', group=group,
        ))
