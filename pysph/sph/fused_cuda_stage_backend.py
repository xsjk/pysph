"""Stage-level execution protocol for generic fused CUDA kernels."""

import inspect
import os

import numpy as np

from pysph.sph.fused_cuda_codegen import (
    CudaPairPrecompute,
    _methods_read_fields,
    _methods_read_written_fields,
    _methods_written_fields,
    _pair_segment_methods,
    _snapshot_hoisted_left_post_methods,
    _stage_can_use_pointwise_kernel,
    _unique_equations,
    cubic_spline_pair_precompute_for_symbols,
    gaussian_pair_precompute_for_symbols,
    generate_csr_pair_stage_outline_from_equations,
    generate_snapshot_csr_pair_window_outline_from_equations,
    generate_pointwise_stage_outline_from_equations,
    launch_cuda_kernel,
    precompute_argument_names,
    quintic_spline_pair_precompute_for_symbols,
    snapshot_csr_pair_window_stage,
    super_gaussian_pair_precompute_for_symbols,
    wendland_quintic_c2_1d_pair_precompute_for_symbols,
    wendland_quintic_c4_1d_pair_precompute_for_symbols,
    wendland_quintic_c4_pair_precompute_for_symbols,
    wendland_quintic_c6_1d_pair_precompute_for_symbols,
    wendland_quintic_c6_pair_precompute_for_symbols,
    wendland_quintic_pair_precompute_for_symbols,
)
from pysph.sph.fused_cuda_stage_plan import (
    MethodKind,
    StageKind,
    _equation_method_access,
    _has_overloaded_method,
    _precomputed_source_reads,
)


class FusedCudaStageBackend:
    """Collapse legacy CUDA kernel calls into fused stage launches."""

    def __init__(self, helper):
        self.helper = helper
        self._check_plan_executable(helper)
        (
            self.stage_by_group,
            self.covered_stage_groups,
            self.stage_group_by_plan_index,
        ) = _stage_group_mapping(helper)
        (
            self.snapshot_window_by_group,
            self.snapshot_window_covered_groups,
        ) = _snapshot_window_mapping(helper, self.stage_group_by_plan_index)
        self.has_device_convergence = _has_device_convergence(helper)
        self.launched_groups = set()
        self.device_convergence_skip_groups = set()
        self.device_convergence_active = False
        self.device_convergence_iteration_counts = []
        self.device_convergence_rebuild_count = 0

    def _check_plan_executable(self, helper):
        """Fail loudly instead of silently degrading to the legacy path."""
        plan = helper.cuda_stage_plan
        if any(stage.kind is StageKind.HOST_BOUNDARY for stage in plan.stages):
            raise RuntimeError(
                "The fused CUDA stage backend cannot execute this equation "
                "set: the fused planner lowered one or more groups to a host "
                "boundary (host-side reduce()/convergence iterations are not "
                "supported in fused stages). Use '--cuda' or '--opencl' "
                "instead of '--cuda --fused'."
            )
        constants = {pa.name: set(pa.constants) for pa in helper.object.particle_arrays}
        for equation in helper.object.all_group.equations:
            arrays = {"d_" + name for name in constants[equation.dest]}
            if equation.sources:
                arrays.update(
                    "s_" + name
                    for name in set.intersection(
                        *(constants[source] for source in equation.sources)
                    )
                )
            for kind in MethodKind:
                if not _has_overloaded_method(equation, kind.value):
                    continue
                _, access = _equation_method_access(equation, kind.value)
                invalid = access.helper_arrays.difference(arrays)
                if invalid:
                    raise RuntimeError(
                        "Fused CUDA helper array arguments must be particle "
                        "constants: " + ", ".join(sorted(invalid))
                    )
                if access.dest_writes.intersection(constants[equation.dest]):
                    raise RuntimeError("Fused CUDA cannot write particle constants")
        for stage in plan.stages:
            if stage.kind is StageKind.DEVICE_CONVERGENCE:
                continue
            if stage.dest not in stage.sources:
                continue
            reads = set()
            writes = set()
            for method in stage.methods:
                reads.update(method.source_reads)
                writes.update(method.dest_writes)
            conflicting = reads & writes
            if conflicting:
                raise RuntimeError(
                    "The fused CUDA stage backend cannot execute this equation "
                    "set: stage for '%s' both reads fields (%s) from source "
                    "particles and writes them on the same destination array; "
                    "fused single-kernel execution does not order these "
                    "concurrently, so results are not reproducible and "
                    "iterations may not converge. Use '--cuda' or '--opencl' "
                    "instead of '--cuda --fused'."
                    % (stage.dest, ", ".join(sorted(conflicting)))
                )

    def begin_compute(self, evaluator, t, dt):
        self.launched_groups = set()
        self.device_convergence_skip_groups = set()
        self.device_convergence_active = False

    def handle_call(self, evaluator, info, extra_args, t, dt):
        if info["type"] == "start_iteration":
            if self.has_device_convergence:
                iteration_calls = self._device_convergence_calls(info)
                self.device_convergence_skip_groups = set(
                    call["stage_group"]
                    for call in iteration_calls
                    if call["type"] == "kernel"
                )
                self.device_convergence_active = True
                self._launch_device_convergence_super_stage(
                    evaluator, info, extra_args, iteration_calls, t, dt
                )
                return True
            return False
        if info["type"] == "stop_iteration":
            if self.device_convergence_active:
                self.device_convergence_skip_groups = set()
                self.device_convergence_active = False
                return True
            return False
        if info["type"] == "py_initialize":
            if not self.device_convergence_active:
                self._initialize_equations(evaluator, info, t, dt)
            return True
        if info["type"] != "kernel":
            return False
        stage_group = info["stage_group"]
        if stage_group in self.device_convergence_skip_groups:
            return True
        stage_key = _stage_group_key(info)
        if stage_key in self.snapshot_window_covered_groups:
            return True
        if stage_key in self.covered_stage_groups:
            return True
        if stage_key not in self.stage_by_group:
            return False
        if stage_key in self.launched_groups:
            return True
        if stage_key in self.snapshot_window_by_group:
            self._launch_snapshot_window(
                evaluator, self.snapshot_window_by_group[stage_key], extra_args
            )
        else:
            stage = self.stage_by_group[stage_key]
            self._launch_stage(evaluator, stage, info, extra_args)
        self.launched_groups.add(stage_key)
        return True

    def end_compute(self, evaluator, t, dt):
        return False

    def _launch_stage(self, evaluator, stage, info, extra_args):
        raise NotImplementedError("subclasses must launch the fused CUDA stage")

    def _initialize_equations(self, evaluator, info, t, dt):
        evaluator._call_py_initialize(info, t, dt)

    def _launch_snapshot_window(self, evaluator, stage_indices, extra_args):
        for stage_index in stage_indices:
            info = self._kernel_info_for_plan_stage_index(stage_index)
            stage = self.helper.cuda_stage_plan.stages[stage_index]
            self._launch_stage(evaluator, stage, info, extra_args)

    def _launch_device_convergence_super_stage(
        self, evaluator, info, extra_args, iteration_calls, t, dt
    ):
        group = info["group"]
        for iteration in range(1, group.max_iterations + 1):
            for call in iteration_calls:
                if call["type"] == "py_initialize":
                    self._initialize_equations(evaluator, call, t, dt)
                else:
                    stage = self.stage_by_group[_stage_group_key(call)]
                    self._launch_stage(evaluator, stage, call, extra_args)
            if iteration >= group.min_iterations and (
                iteration == group.max_iterations
                or evaluator._converged(info["equations"])
            ):
                self.device_convergence_iteration_counts.append(iteration)
                return
            if group.update_nnps:
                self.device_convergence_rebuild_count += 1
                evaluator.update_nnps()

    def _device_convergence_calls(self, start_info):
        start_index = None
        for index, call in enumerate(self.helper.calls):
            if call is start_info:
                start_index = index
                break
        assert start_index is not None
        groups = []
        iteration_calls = []
        for call in self.helper.calls[start_index + 1 :]:
            if call["type"] == "stop_iteration":
                return tuple(iteration_calls)
            if call["type"] == "py_initialize":
                iteration_calls.append(call)
            if call["type"] == "kernel":
                stage_key = _stage_group_key(call)
                if stage_key in self.covered_stage_groups:
                    continue
                assert stage_key in self.stage_by_group
                if stage_key not in groups:
                    groups.append(stage_key)
                    iteration_calls.append(call)
        assert False

    def _kernel_info_for_plan_stage_index(self, stage_index):
        assert stage_index in self.stage_group_by_plan_index
        stage_key = self.stage_group_by_plan_index[stage_index]
        for call in self.helper.calls:
            if call["type"] == "kernel" and _stage_group_key(call) == stage_key:
                return call
        assert False


class GeneratedFusedCudaStageBackend(FusedCudaStageBackend):
    """Executable fused CUDA stage backend for the supported strict subset."""

    def __init__(self, helper, block_size):
        super().__init__(helper)
        assert type(block_size) is int
        assert 32 <= block_size <= 1024 and block_size % 32 == 0
        self.block_size = block_size
        # Share the native NNPS and integrator default stream.
        self.stream = None
        self.programs = {}
        self.snapshot_programs = {}
        self.snapshot_arrays = {}
        self.launch_count = 0
        self.traversal_launch_counts = {}
        self.pair_launch_config_counts = {}
        self.stage_timing_ms = {}
        self.stage_timing_counts = {}

    def _launch_snapshot_window(self, evaluator, stage_indices, extra_args):
        infos = tuple(
            self._kernel_info_for_plan_stage_index(index) for index in stage_indices
        )
        pair_args, periodic = self._pair_args(evaluator, infos[0])
        combined_stage, snapshot_fields, program = self._program_for_snapshot_window(
            stage_indices, infos, periodic
        )
        _module, kernel, names = program
        stage_args = tuple(
            _stage_extra_arg_value(self.helper, name, infos[0], extra_args)
            for name in names
        )
        snapshot_args = tuple(
            self._snapshot_field_arg(infos[0], field) for field in snapshot_fields
        )
        timer = _stage_timer(self.stream)
        launch_config = launch_cuda_kernel(
            kernel,
            infos[0]["dest"].get_number_of_particles(True),
            self.stream,
            pair_args + snapshot_args + stage_args,
            "snapshot_csr",
            self.block_size,
        )
        self._record_pair_launch("snapshot_csr")
        self._record_pair_launch_config(launch_config)
        self._record_stage_timing(combined_stage, "snapshot_csr", timer)
        self.launch_count += 1

    def _program_for_snapshot_window(self, stage_indices, infos, periodic):
        key = stage_indices, periodic
        if key in self.snapshot_programs:
            return self.snapshot_programs[key]
        stages = tuple(
            self.helper.cuda_stage_plan.stages[index] for index in stage_indices
        )
        equations_by_stage = tuple(
            _equations_for_stage(self.helper, stage, info["stage_group"])
            for stage, info in zip(stages, infos)
        )
        stage = snapshot_csr_pair_window_stage(stages)
        precompute = _precompute_for_stage(stage, self.helper.object.kernel)
        snapshot_fields = _snapshot_pair_window_fields(stages)
        outline = generate_snapshot_csr_pair_window_outline_from_equations(
            "cuda_eval",
            stages,
            equations_by_stage,
            precompute,
            snapshot_fields,
            self.helper.known_types,
            periodic,
        )
        names = _stage_extra_arg_names(
            stage, _unique_equations(equations_by_stage), precompute
        )
        program = self._compile_program(outline, names)
        self.snapshot_programs[key] = stage, snapshot_fields, program
        return self.snapshot_programs[key]

    def _snapshot_field_arg(self, info, field):
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray

        source = getattr(info["dest"].gpu, field).dev
        key = (id(info["dest"]), field)
        nreal = info["dest"].get_number_of_particles(True)
        if key not in self.snapshot_arrays:
            self.snapshot_arrays[key] = gpuarray.empty(source.shape, source.dtype)
        if self.snapshot_arrays[key].shape[0] < nreal:
            self.snapshot_arrays[key] = gpuarray.empty(source.shape, source.dtype)
        target = self.snapshot_arrays[key]
        cuda.memcpy_dtod_async(
            _device_pointer_int(target),
            _device_pointer_int(source),
            int(nreal) * source.dtype.itemsize,
            self.stream,
        )
        return target

    def _launch_stage(self, evaluator, stage, info, extra_args):
        pair = stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)
        if pair:
            pair_args, periodic = self._pair_args(evaluator, info)
        else:
            pair_args, periodic = (), False
        _module, kernel, names = self._program_for_stage(stage, info, periodic)
        stage_args = tuple(
            _stage_extra_arg_value(self.helper, name, info, extra_args)
            for name in names
        )
        n = info["dest"].get_number_of_particles(True)
        traversal = "csr" if pair else "pointwise"
        arguments = pair_args + stage_args if pair else (np.int32(n), *stage_args)
        timer = _stage_timer(self.stream)
        launch_config = launch_cuda_kernel(
            kernel, n, self.stream, arguments, traversal, self.block_size
        )
        if pair:
            self._record_pair_launch(traversal)
            self._record_pair_launch_config(launch_config)
        self._record_stage_timing(stage, traversal, timer)
        self.launch_count += 1

    def _record_pair_launch(self, traversal):
        if traversal not in self.traversal_launch_counts:
            self.traversal_launch_counts[traversal] = 0
        self.traversal_launch_counts[traversal] += 1

    def _record_pair_launch_config(self, launch_config):
        key = (
            launch_config.traversal,
            launch_config.n,
            launch_config.block_size,
            launch_config.grid_x,
        )
        if key not in self.pair_launch_config_counts:
            self.pair_launch_config_counts[key] = 0
        self.pair_launch_config_counts[key] += 1

    def _record_stage_timing(self, stage, traversal, timer):
        if timer is None:
            return
        elapsed_ms = timer.finish()
        key = _stage_timing_key(stage, traversal)
        if key not in self.stage_timing_ms:
            self.stage_timing_ms[key] = 0.0
            self.stage_timing_counts[key] = 0
        self.stage_timing_ms[key] += elapsed_ms
        self.stage_timing_counts[key] += 1

    def _pair_args(self, evaluator, info):
        dest = info["dest"]
        assert dest is info["src"]
        nnps = evaluator.nnps
        nnps.set_context(info["src_idx"], info["dst_idx"])
        cache = nnps.current_cache
        cache.get_neighbors_gpu()
        periodic_args = evaluator._periodic_args()
        arguments = (
            *(getattr(dest.gpu, name).dev for name in ("x", "y", "z", "h")),
            np.int32(dest.get_number_of_particles(True)),
            cache._nbr_lengths_gpu.dev,
            cache._start_idx_gpu.dev,
            cache._neighbors_gpu.dev,
            *periodic_args,
            np.float32(nnps.radius_scale),
        )
        return arguments, any(periodic_args[:3])

    def _program_for_stage(self, stage, info, periodic):
        key = _stage_group_key(info), periodic
        if key in self.programs:
            return self.programs[key]
        equations = _equations_for_stage(self.helper, stage, info["stage_group"])
        precompute = _precompute_for_stage(stage, self.helper.object.kernel)
        if stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE):
            outline = generate_csr_pair_stage_outline_from_equations(
                "cuda_eval",
                stage,
                equations,
                precompute,
                self.helper.known_types,
                periodic,
            )
        else:
            assert _stage_can_use_pointwise_kernel(stage)
            outline = generate_pointwise_stage_outline_from_equations(
                "cuda_eval", stage, equations, self.helper.known_types
            )
        names = _stage_extra_arg_names(stage, equations, precompute)
        self.programs[key] = self._compile_program(outline, names)
        return self.programs[key]

    def _compile_program(self, outline, names):
        from pycuda.compiler import SourceModule

        module = SourceModule(
            outline.source, no_extern_c=True, options=["--use_fast_math"]
        )
        return module, module.get_function(outline.name), names


def _stage_group_mapping(helper):
    stage_groups = []
    for call in helper.calls:
        if call["type"] == "kernel" and _stage_group_key(call) not in stage_groups:
            stage_groups.append(_stage_group_key(call))
    stages = tuple(
        (index, stage)
        for index, stage in enumerate(helper.cuda_stage_plan.stages)
        if _stage_can_launch_from_kernel_call(stage)
    )
    stage_by_group = {}
    covered_stage_groups = set()
    stage_group_by_plan_index = {}
    group_index = 0
    for plan_index, stage in stages:
        assert group_index < len(stage_groups)
        stage_by_group[stage_groups[group_index]] = stage
        stage_group_by_plan_index[plan_index] = stage_groups[group_index]
        for offset in range(1, stage.legacy_group_count):
            covered_stage_groups.add(stage_groups[group_index + offset])
        group_index += stage.legacy_group_count
    assert group_index == len(stage_groups)
    return stage_by_group, covered_stage_groups, stage_group_by_plan_index


def _stage_group_key(call):
    return call["stage_group"], call["dest"].name


def _snapshot_window_mapping(helper, stage_group_by_plan_index):
    window_by_group = {}
    covered_groups = set()
    for stage_indices in _snapshot_pair_windows(helper.cuda_stage_plan.stages):
        if not all(index in stage_group_by_plan_index for index in stage_indices):
            continue
        first_group = stage_group_by_plan_index[stage_indices[0]]
        window_by_group[first_group] = stage_indices
        for stage_index in stage_indices[1:]:
            covered_groups.add(stage_group_by_plan_index[stage_index])
    return window_by_group, covered_groups


def _snapshot_pair_windows(stages):
    windows = []
    index = 0
    while index < len(stages) - 1:
        left = stages[index]
        right = stages[index + 1]
        if _can_launch_snapshot_pair_window(left, right):
            windows.append((index, index + 1))
            index += 2
        else:
            index += 1
    return tuple(windows)


def _can_launch_snapshot_pair_window(left, right):
    if not (
        left.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)
        and right.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)
        and left.dest == right.dest
        and left.sources == right.sources
        and not _stage_changes_geometry(left)
        and not _stage_changes_geometry(right)
    ):
        return False
    left_pre, left_loop, left_post = _pair_segment_methods(left.methods)
    right_pre, right_loop, _right_post = _pair_segment_methods(right.methods)
    hoisted_left_post, remaining_left_post = _snapshot_hoisted_left_post_methods(
        left_loop, left_post
    )
    right_source_reads = set(_methods_precomputed_source_reads(right.methods))
    for method in right.methods:
        right_source_reads.update(method.source_reads)
    # A snapshot holds old values. Reading newly written source state needs
    # a kernel boundary, even when the writer can be hoisted before the loop.
    if right_source_reads.intersection(
        _methods_written_fields(left_pre + hoisted_left_post)
    ):
        return False
    snapshot_fields = _snapshot_pair_window_fields_from_parts(
        left_loop, right_pre, right_loop
    )
    if _methods_read_written_fields(right_pre, left_loop):
        return False
    if _methods_read_written_fields(right_loop, left_loop + remaining_left_post):
        return False
    if _methods_read_written_fields(remaining_left_post, right_pre + right_loop):
        return False
    if _methods_precomputed_source_reads(left_loop).intersection(snapshot_fields):
        return False
    if snapshot_fields and _method_identities(left_loop).intersection(
        _method_identities(right.methods)
    ):
        return False
    return bool(left_pre or left_loop or hoisted_left_post)


def _has_device_convergence(helper):
    return any(
        stage.kind is StageKind.DEVICE_CONVERGENCE
        for stage in helper.cuda_stage_plan.stages
    )


def _stage_changes_geometry(stage):
    invalidating_fields = frozenset(("x", "y", "z", "h"))
    for method in stage.methods:
        if method.dest_writes.intersection(invalidating_fields):
            return True
        if method.source_writes.intersection(invalidating_fields):
            return True
    return False


def _snapshot_pair_window_fields(stages):
    left_pre, left_loop, _left_post = _pair_segment_methods(stages[0].methods)
    right_pre, right_loop, _right_post = _pair_segment_methods(stages[1].methods)
    assert left_pre or left_loop
    return _snapshot_pair_window_fields_from_parts(left_loop, right_pre, right_loop)


def _snapshot_pair_window_fields_from_parts(left_loop, right_pre, right_loop):
    fields = _methods_read_fields(left_loop).intersection(
        _methods_written_fields(right_pre + right_loop)
    )
    return tuple(sorted(fields))


def _methods_precomputed_source_reads(methods):
    reads = set()
    for method in methods:
        reads.update(_precomputed_source_reads(method.precomputed_symbols))
    return frozenset(reads)


def _method_identities(methods):
    return frozenset((method.equation_name, method.method_kind) for method in methods)


def _stage_can_launch_from_kernel_call(stage):
    if stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE):
        return True
    return _stage_can_use_pointwise_kernel(stage)


class _CudaStageTimer:
    def __init__(self, stream):
        import pycuda.driver as cuda

        self.start = cuda.Event()
        self.stop = cuda.Event()
        self.stream = stream
        self.start.record(stream)

    def finish(self):
        self.stop.record(self.stream)
        self.stop.synchronize()
        return float(self.stop.time_since(self.start))


def _stage_timer(stream):
    if "PYSPH_PROFILE_CUDA_EVENTS" not in os.environ:
        return None
    assert os.environ["PYSPH_PROFILE_CUDA_EVENTS"] == "1"
    return _CudaStageTimer(stream)


def _stage_timing_key(stage, traversal):
    method_names = "+".join(
        f"{method.equation_name}.{method.method_kind.value}" for method in stage.methods
    )
    return f"{stage.kind.value}:{traversal}:{method_names}"


def _equations_for_stage(helper, stage, stage_group):
    equations = [
        equation
        for equation in _equations_for_stage_group(helper, stage_group)
        if equation.dest == stage.dest
    ]
    names = [equation.__class__.__name__ for equation in equations]
    for method in stage.methods:
        if method.equation_name not in names:
            equation = _equation_for_method(
                method.equation_name,
                tuple(
                    equation
                    for equation in _all_equations(helper)
                    if equation.dest == stage.dest
                ),
            )
            equations.append(equation)
            names.append(method.equation_name)
    return tuple(equations)


def _all_equations(helper):
    equations = []
    for group in helper.object.equation_groups:
        equations.extend(_flatten_group_equations(group))
    return tuple(equations)


def _flatten_group_equations(group):
    if not group.has_subgroups:
        return tuple(group.equations)
    equations = []
    for subgroup in group.equations:
        equations.extend(_flatten_group_equations(subgroup))
    return tuple(equations)


def _equations_for_stage_group(helper, stage_group):
    group_index, subgroup_index = stage_group
    group = helper.object.equation_groups[group_index]
    if subgroup_index == -1:
        assert not group.has_subgroups
        return tuple(group.equations)
    subgroup = group.equations[subgroup_index]
    return tuple(subgroup.equations)


def _precompute_for_stage(stage, kernel):
    symbols = set()
    for method in stage.methods:
        symbols.update(method.precomputed_symbols)
    if not symbols:
        return CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())
    dim = np.int32(kernel.dim)
    kernel_name = kernel.__class__.__name__
    if symbols.issubset(
        {
            "HIJ",
            "XIJ",
            "R2IJ",
            "RIJ",
            "WIJ",
            "WI",
            "WJ",
            "DWIJ",
            "DWI",
            "DWJ",
            "GHI",
            "GHJ",
            "GHIJ",
            "VIJ",
            "EPS",
            "RHOIJ",
            "RHOIJ1",
        }
    ):
        if kernel_name == "CubicSpline":
            assert dim in (np.int32(1), np.int32(2), np.int32(3))
            return cubic_spline_pair_precompute_for_symbols(dim, frozenset(symbols))
        if kernel_name == "QuinticSpline":
            assert dim in (np.int32(1), np.int32(2), np.int32(3))
            return quintic_spline_pair_precompute_for_symbols(dim, frozenset(symbols))
        if kernel_name == "WendlandQuinticC2_1D":
            assert dim == np.int32(1)
            return wendland_quintic_c2_1d_pair_precompute_for_symbols(
                dim, frozenset(symbols)
            )
        if kernel_name == "WendlandQuintic":
            assert dim in (np.int32(2), np.int32(3))
            return wendland_quintic_pair_precompute_for_symbols(dim, frozenset(symbols))
        if kernel_name == "WendlandQuinticC4_1D":
            assert dim == np.int32(1)
            return wendland_quintic_c4_1d_pair_precompute_for_symbols(
                dim, frozenset(symbols)
            )
        if kernel_name == "WendlandQuinticC4":
            assert dim in (np.int32(2), np.int32(3))
            return wendland_quintic_c4_pair_precompute_for_symbols(
                dim, frozenset(symbols)
            )
        if kernel_name == "WendlandQuinticC6_1D":
            assert dim == np.int32(1)
            return wendland_quintic_c6_1d_pair_precompute_for_symbols(
                dim, frozenset(symbols)
            )
        if kernel_name == "WendlandQuinticC6":
            assert dim in (np.int32(2), np.int32(3))
            return wendland_quintic_c6_pair_precompute_for_symbols(
                dim, frozenset(symbols)
            )
        if kernel_name == "Gaussian":
            assert dim in (np.int32(1), np.int32(2), np.int32(3))
            return gaussian_pair_precompute_for_symbols(dim, frozenset(symbols))
        if kernel_name == "SuperGaussian":
            assert dim in (np.int32(1), np.int32(2), np.int32(3))
            return super_gaussian_pair_precompute_for_symbols(dim, frozenset(symbols))
        assert False
    assert False


def _stage_extra_arg_names(stage, equations, precompute):
    names = list(precompute_argument_names(precompute))
    for name in _stage_equation_extra_arg_names(stage, equations, precompute):
        _append_once(names, name)
    return tuple(names)


def _stage_equation_extra_arg_names(stage, equations, precompute):
    names = []
    for method in stage.methods:
        equation = _equation_for_method(method.equation_name, equations)
        _append_once(names, equation.var_name)
        args = list(
            inspect.getfullargspec(getattr(equation, method.method_kind.value)).args
        )
        for arg in args:
            if arg in ("self", "d_idx", "s_idx") or arg in precompute.symbols:
                continue
            _append_once(names, arg)
    return tuple(names)


def _append_once(names, name):
    if name not in names:
        names.append(name)


def _stage_extra_arg_value(helper, name, info, extra_args):
    if name in helper._gpu_structs:
        value = helper._gpu_structs[name]
        if value is None:
            return np.uintp(0)
        return value
    if name.startswith("d_"):
        return getattr(info["dest"].gpu, name[2:]).dev
    if name.startswith("s_"):
        return getattr(info["src"].gpu, name[2:]).dev
    if name == "t":
        return extra_args[0]
    if name == "dt":
        return extra_args[1]
    assert False


def _equation_for_method(equation_name, equations):
    matches = [
        equation
        for equation in equations
        if equation.__class__.__name__ == equation_name
    ]
    assert len(matches) == 1
    return matches[0]


def _device_pointer_int(array):
    gpudata = array.gpudata
    if isinstance(gpudata, int):
        return gpudata
    return int(gpudata)
