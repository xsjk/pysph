"""Stage-level execution protocol for generic fused CUDA kernels."""

import inspect
import os
from dataclasses import replace

import numpy as np

from pysph.base.fused_cuda_nnps import (
    FusedCudaNeighborWorkspace,
    build_fused_cuda_neighbor_context_with_workspace,
    wrap_periodic_xyz,
)
from pysph.sph.fused_cuda_codegen import (
    CudaPairPrecompute,
    cubic_spline_pair_precompute_for_symbols,
    gaussian_pair_precompute_for_symbols,
    generate_hbucket_pair_stage_outline_from_equations,
    generate_snapshot_hbucket_pair_window_outline_from_equations,
    PairLaunchConfig,
    generate_pointwise_stage_outline_from_equations,
    launch_hbucket_pair_kernel_with_context,
    launch_pointwise_kernel,
    launch_snapshot_hbucket_pair_window_kernel_with_context,
    precompute_argument_names,
    quintic_spline_pair_precompute_for_symbols,
    snapshot_hbucket_pair_window_stage,
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
)


class FusedCudaStageBackend:
    """Collapse legacy CUDA kernel calls into fused stage launches."""

    def __init__(self, helper):
        self.helper = helper
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

    def begin_compute(self, evaluator, t, dt):
        self.launched_groups = set()
        self.device_convergence_skip_groups = set()
        self.device_convergence_active = False

    def handle_call(self, evaluator, info, extra_args, t, dt):
        if info["type"] == "start_iteration":
            if self.has_device_convergence:
                kernel_infos = self._device_convergence_kernel_infos(info)
                self.device_convergence_skip_groups = set(
                    kernel_info["stage_group"] for kernel_info in kernel_infos
                )
                self.device_convergence_active = True
                self._launch_device_convergence_super_stage(
                    evaluator, info, extra_args, t, dt
                )
                return True
            return False
        if info["type"] == "stop_iteration":
            if self.device_convergence_active:
                self.device_convergence_skip_groups = set()
                self.device_convergence_active = False
                return True
            return False
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

    def handle_outer_update_nnps(self, integrator, index):
        return True

    def handle_reorder_update_nnps(self, solver):
        return True

    def _launch_stage(self, evaluator, stage, info, extra_args):
        raise NotImplementedError("subclasses must launch the fused CUDA stage")

    def _launch_snapshot_window(self, evaluator, stage_indices, extra_args):
        for stage_index in stage_indices:
            info = self._kernel_info_for_plan_stage_index(stage_index)
            stage = self.helper.cuda_stage_plan.stages[stage_index]
            self._launch_stage(evaluator, stage, info, extra_args)

    def _launch_device_convergence_super_stage(
        self, evaluator, info, extra_args, t, dt
    ):
        kernel_infos = self._device_convergence_kernel_infos(info)
        policy = self._device_convergence_policy()
        group = info["group"]
        max_iterations = group.max_iterations
        min_iterations = group.min_iterations
        update_nnps = group.update_nnps
        if policy is not None:
            max_iterations = policy.max_iterations
            min_iterations = policy.min_iterations
            update_nnps = policy.update_nnps
        self._begin_device_convergence_super_stage(kernel_infos)
        try:
            self._run_device_convergence_iterations(
                evaluator,
                info,
                extra_args,
                kernel_infos,
                max_iterations,
                min_iterations,
                update_nnps,
            )
        finally:
            self._end_device_convergence_super_stage()

    def _run_device_convergence_iterations(
        self,
        evaluator,
        info,
        extra_args,
        kernel_infos,
        max_iterations,
        min_iterations,
        update_nnps,
    ):
        for iteration in range(max_iterations):
            self._begin_device_convergence_iteration(info)
            for kernel_info in kernel_infos:
                stage = self.stage_by_group[_stage_group_key(kernel_info)]
                self._launch_stage(evaluator, stage, kernel_info, extra_args)
            iter_count = iteration + 1
            has_min_iterations = iter_count >= min_iterations
            at_max_iterations = iter_count == max_iterations
            has_converged = False
            if has_min_iterations and not at_max_iterations:
                has_converged = self._device_convergence_has_converged(
                    evaluator, info
                )
            if has_min_iterations and (at_max_iterations or has_converged):
                self.device_convergence_iteration_counts.append(iter_count)
                return
            if update_nnps:
                self.device_convergence_rebuild_count += 1
                self._update_device_convergence_nnps(evaluator, info)

    def _device_convergence_has_converged(self, evaluator, info):
        return False

    def _begin_device_convergence_iteration(self, info):
        pass

    def _begin_device_convergence_super_stage(self, kernel_infos):
        pass

    def _end_device_convergence_super_stage(self):
        pass

    def _update_device_convergence_nnps(self, evaluator, info):
        evaluator.update_nnps()

    def _device_convergence_kernel_infos(self, start_info):
        policy = self._device_convergence_policy()
        if policy is not None:
            return tuple(
                self._kernel_info_for_plan_stage_index(index)
                for index in policy.child_stage_indices
            )
        start_index = None
        for index, call in enumerate(self.helper.calls):
            if call is start_info:
                start_index = index
                break
        assert start_index is not None
        groups = []
        kernel_infos = []
        for call in self.helper.calls[start_index + 1 :]:
            if call["type"] == "stop_iteration":
                return tuple(kernel_infos)
            if call["type"] == "kernel":
                stage_key = _stage_group_key(call)
                assert stage_key in self.stage_by_group
                if stage_key not in groups:
                    groups.append(stage_key)
                    kernel_infos.append(call)
        assert False

    def _device_convergence_policy(self):
        for stage in self.helper.cuda_stage_plan.stages:
            if stage.kind is StageKind.DEVICE_CONVERGENCE:
                return stage.convergence_policy
        return None

    def _kernel_info_for_plan_stage_index(self, stage_index):
        assert stage_index in self.stage_group_by_plan_index
        stage_key = self.stage_group_by_plan_index[stage_index]
        for call in self.helper.calls:
            if call["type"] == "kernel" and _stage_group_key(call) == stage_key:
                return call
        assert False


class GeneratedFusedCudaStageBackend(FusedCudaStageBackend):
    """Executable fused CUDA stage backend for the supported strict subset."""

    def __init__(self, helper):
        super().__init__(helper)
        import pycuda.driver as cuda

        self.stream = cuda.Stream()
        self.modules = {}
        self.outlines = {}
        self.neighbor_contexts = {}
        self.neighbor_workspaces = {}
        self.snapshot_arrays = {}
        self.launch_count = 0
        self.h_reduce_scratch = []
        self.traversal_launch_counts = {}
        self.pair_launch_config_counts = {}
        self.stage_timing_ms = {}
        self.stage_timing_counts = {}

    def begin_compute(self, evaluator, t, dt):
        super().begin_compute(evaluator, t, dt)
        self.neighbor_contexts = {}

    def _launch_stage(self, evaluator, stage, info, extra_args):
        self._launch_single_stage(evaluator, stage, info, extra_args)

    def _launch_snapshot_window(self, evaluator, stage_indices, extra_args):
        stages = tuple(
            self.helper.cuda_stage_plan.stages[index] for index in stage_indices
        )
        infos = tuple(
            self._kernel_info_for_plan_stage_index(index) for index in stage_indices
        )
        context = self._neighbor_context_for_stage(evaluator, infos[0])
        equations_by_stage = tuple(
            _equations_for_stage(self.helper, stage, info["stage_group"])
            for stage, info in zip(stages, infos)
        )
        combined_stage = snapshot_hbucket_pair_window_stage(stages)
        precompute = _precompute_for_stage(combined_stage, self.helper.object.kernel)
        snapshot_fields = _snapshot_pair_window_fields(stages)
        outline = self._outline_for_snapshot_hbucket_pair_window(
            stages, equations_by_stage, precompute, snapshot_fields
        )
        module = self._module_for_stage(outline, infos[0], combined_stage)
        combined_equations = _unique_equations_for_window(equations_by_stage)
        stage_args = _stage_extra_args(
            self.helper,
            combined_stage,
            combined_equations,
            infos[0],
            extra_args,
            precompute,
        )
        snapshot_args = tuple(
            self._snapshot_field_arg(infos[0], field) for field in snapshot_fields
        )
        timer = _stage_timer(self.stream)
        launch_config = launch_snapshot_hbucket_pair_window_kernel_with_context(
            module,
            outline.name,
            context,
            snapshot_args,
            stage_args,
        )
        traversal = "snapshot_hbucket"
        self._record_pair_launch(traversal)
        self._record_pair_launch_config(launch_config)
        for stage in stages:
            self._finish_launched_stage(stage)
        self._record_stage_timing(
            _snapshot_hbucket_pair_window_stage(stages), traversal, timer
        )
        self.launch_count += 1

    def _outline_for_snapshot_hbucket_pair_window(
        self, stages, equations_by_stage, precompute, snapshot_fields
    ):
        outline_key = _snapshot_hbucket_pair_window_key(
            stages, precompute, snapshot_fields
        )
        if outline_key in self.outlines:
            return self.outlines[outline_key]
        outline = generate_snapshot_hbucket_pair_window_outline_from_equations(
            "cuda_eval",
            stages,
            equations_by_stage,
            precompute,
            snapshot_fields,
            self.helper.known_types,
        )
        self.outlines[outline_key] = outline
        return outline

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

    def _launch_single_stage(self, evaluator, stage, info, extra_args):
        equations = _equations_for_stage(self.helper, stage, info["stage_group"])
        precompute = _precompute_for_stage(stage, self.helper.object.kernel)
        timer = _stage_timer(self.stream)
        traversal = "pointwise"
        if stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE):
            context = self._neighbor_context_for_stage(evaluator, info)
            traversal = _pair_traversal_for_stage(context)
            outline = self._outline_for_stage(
                stage, equations, precompute, info, traversal
            )
            module = self._module_for_stage(outline, info, stage)
            stage_args = _stage_extra_args(
                self.helper, stage, equations, info, extra_args, precompute
            )
            self._record_pair_launch(traversal)
            launch_config = launch_hbucket_pair_kernel_with_context(
                module, outline.name, context, stage_args
            )
            self._record_pair_launch_config(launch_config)
        elif _stage_can_use_pointwise_kernel(stage):
            outline = self._outline_for_stage(
                stage, equations, precompute, info, traversal
            )
            module = self._module_for_stage(outline, info, stage)
            stage_args = _stage_extra_args(
                self.helper, stage, equations, info, extra_args, precompute
            )
            n = info["dest"].get_number_of_particles(True)
            launch_pointwise_kernel(module, outline.name, n, self.stream, stage_args)
        else:
            assert False
        self._finish_launched_stage(stage)
        self._record_stage_timing(stage, traversal, timer)
        self.launch_count += 1

    def _record_pair_launch(self, traversal):
        if traversal not in self.traversal_launch_counts:
            self.traversal_launch_counts[traversal] = 0
        self.traversal_launch_counts[traversal] += 1

    def _record_pair_launch_config(self, launch_config):
        if launch_config is None:
            return
        assert isinstance(launch_config, PairLaunchConfig)
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

    def _neighbor_context_for_stage(self, evaluator, info):
        key = _neighbor_context_key(info)
        if key not in self.neighbor_contexts:
            self.neighbor_contexts[key] = self._build_neighbor_context(evaluator, info)
        return self.neighbor_contexts[key]

    def _build_neighbor_context(self, evaluator, info):
        key = _neighbor_context_key(info)
        if key not in self.neighbor_workspaces:
            self.neighbor_workspaces[key] = FusedCudaNeighborWorkspace()
        return _neighbor_context_for_info(
            evaluator,
            info,
            self.stream,
            self.h_reduce_scratch,
            self.neighbor_workspaces[key],
        )

    def handle_update_domain(self, integrator):
        manager = integrator.nnps.domain.manager
        if not manager.minimum_image_periodic:
            self.stream.synchronize()
            return False
        lower, upper, periodic = _domain_bounds_and_periodicity_from_manager(manager)
        for particle_array in integrator.acceleration_evals[0].particle_arrays:
            assert particle_array.gpu is not None
            nreal = particle_array.get_number_of_particles(True)
            wrap_periodic_xyz(
                particle_array.gpu.x.dev,
                particle_array.gpu.y.dev,
                particle_array.gpu.z.dev,
                nreal,
                lower,
                upper,
                periodic,
                self.stream,
            )
        return True

    def _finish_launched_stage(self, stage):
        if _stage_invalidates_neighbor_context(stage):
            self.neighbor_contexts = {}
            self._invalidate_neighbor_h_bounds()

    def _device_convergence_has_converged(self, evaluator, info):
        self.stream.synchronize()
        for equation in info["equations"]:
            evaluator._sync_from_gpu(equation)
            if equation.equation_has_converged <= 0:
                return False
        return True

    def _update_device_convergence_nnps(self, evaluator, info):
        self._refresh_ghost_domain(evaluator)
        self.neighbor_contexts = {}

    def handle_internal_update_nnps(self, evaluator):
        self._refresh_ghost_domain(evaluator)
        self.neighbor_contexts = {}
        return True

    def handle_reorder_update_nnps(self, solver):
        self.stream.synchronize()
        self.neighbor_contexts = {}
        self._invalidate_neighbor_h_bounds()
        return False

    def _refresh_ghost_domain(self, evaluator):
        manager = evaluator.nnps.domain.manager
        if not manager.minimum_image_periodic:
            self.stream.synchronize()
            evaluator.nnps.update_domain()
        self._invalidate_neighbor_h_bounds()

    def _invalidate_neighbor_h_bounds(self):
        for workspace in self.neighbor_workspaces.values():
            workspace.invalidate_h_bounds()

    def _outline_for_stage(self, stage, equations, precompute, info, traversal):
        stage_group = info["stage_group"]
        outline_key = (
            stage_group,
            stage.kind.value,
            traversal,
            tuple(
                (method.equation_name, method.method_kind.value)
                for method in stage.methods
            ),
        )
        if outline_key in self.outlines:
            return self.outlines[outline_key]
        if stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE):
            outline = generate_hbucket_pair_stage_outline_from_equations(
                "cuda_eval", stage, equations, precompute, self.helper.known_types
            )
        elif _stage_can_use_pointwise_kernel(stage):
            outline = generate_pointwise_stage_outline_from_equations(
                "cuda_eval", stage, equations, self.helper.known_types
            )
        else:
            assert False
        self.outlines[outline_key] = outline
        return outline

    def _module_for_stage(self, outline, info, stage):
        module_key = (
            info["stage_group"],
            outline.name,
            tuple(
                (method.equation_name, method.method_kind.value)
                for method in stage.methods
            ),
        )
        if module_key in self.modules:
            return self.modules[module_key]
        from pycuda.compiler import SourceModule

        module = SourceModule(
            outline.source, no_extern_c=True, options=list(_source_module_options())
        )
        self.modules[module_key] = module
        return module


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
        and not _stage_invalidates_neighbor_context(left)
        and not _stage_invalidates_neighbor_context(right)
    ):
        return False
    left_pre, left_loop, left_post = _pair_segment_methods_for_backend(left.methods)
    right_pre, right_loop, _right_post = _pair_segment_methods_for_backend(
        right.methods
    )
    hoisted_left_post, remaining_left_post = _snapshot_hoisted_left_post_methods(
        left_loop, left_post
    )
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


def _neighbor_context_key(info):
    return (id(info["dest"]), id(info["src"]))


def _stage_invalidates_neighbor_context(stage):
    invalidating_fields = frozenset(("x", "y", "z", "h"))
    for method in stage.methods:
        if method.dest_writes.intersection(invalidating_fields):
            return True
        if method.source_writes.intersection(invalidating_fields):
            return True
    return False


def _pair_segment_methods_for_backend(methods):
    pair_loop_indices = tuple(
        index for index, method in enumerate(methods) if _is_pair_loop_method(method)
    )
    assert pair_loop_indices
    first_pair_loop_index = pair_loop_indices[0]
    pre_methods = []
    loop_methods = []
    post_methods = []
    for index, method in enumerate(methods):
        if _is_pair_loop_method(method):
            loop_methods.append(method)
        elif _is_source_free_method(method):
            if index < first_pair_loop_index:
                pre_methods.append(method)
            else:
                post_methods.append(method)
        elif _is_pair_pre_loop_method(method):
            pre_methods.append(method)
        elif _is_pair_post_loop_method(method):
            post_methods.append(method)
        else:
            assert False
    return tuple(pre_methods), tuple(loop_methods), tuple(post_methods)


def _snapshot_hoisted_left_post_methods(left_loop, left_post):
    left_loop_writes = _methods_written_fields(left_loop)
    hoisted = []
    remaining = []
    for method in left_post:
        if _is_source_free_method(method) and not _method_read_fields(
            method
        ).intersection(left_loop_writes):
            hoisted.append(method)
        else:
            remaining.append(method)
    return tuple(hoisted), tuple(remaining)


def _snapshot_pair_window_fields(stages):
    left_pre, left_loop, _left_post = _pair_segment_methods_for_backend(
        stages[0].methods
    )
    right_pre, right_loop, _right_post = _pair_segment_methods_for_backend(
        stages[1].methods
    )
    assert left_pre or left_loop
    return _snapshot_pair_window_fields_from_parts(left_loop, right_pre, right_loop)


def _snapshot_pair_window_fields_from_parts(left_loop, right_pre, right_loop):
    fields = _methods_read_fields(left_loop).intersection(
        _methods_written_fields(right_pre + right_loop)
    )
    return tuple(sorted(fields))


def _methods_read_written_fields(readers, writers):
    return bool(
        _methods_read_fields(readers).intersection(_methods_written_fields(writers))
    )


def _methods_read_fields(methods):
    fields = set()
    for method in methods:
        fields.update(_method_read_fields(method))
    return frozenset(fields)


def _method_read_fields(method):
    fields = set(method.dest_reads.difference(method.dest_reduction_reads))
    fields.update(method.source_reads)
    fields.update(_precomputed_source_reads(method.precomputed_symbols))
    return frozenset(fields)


def _methods_written_fields(methods):
    fields = set()
    for method in methods:
        fields.update(method.dest_writes)
        fields.update(method.source_writes)
        fields.update(method.dest_reduction_writes)
        fields.update(method.dest_max_reduction_writes)
    return frozenset(fields)


def _methods_precomputed_source_reads(methods):
    reads = set()
    for method in methods:
        reads.update(_precomputed_source_reads(method.precomputed_symbols))
    return frozenset(reads)


def _method_identities(methods):
    return frozenset((method.equation_name, method.method_kind) for method in methods)


def _is_pair_pre_loop_method(method):
    return bool(method.sources) and method.method_kind in (
        MethodKind.INITIALIZE,
        MethodKind.LOOP_ALL,
    )


def _is_pair_loop_method(method):
    return bool(method.sources) and method.method_kind is MethodKind.LOOP


def _is_pair_post_loop_method(method):
    return (
        bool(method.sources) and method.method_kind is MethodKind.POST_LOOP
    ) or not bool(method.sources)


def _is_source_free_method(method):
    return not bool(method.sources)


def _precomputed_source_reads(symbols):
    reads = set()
    if "VIJ" in symbols:
        reads.update(("u", "v", "w"))
    if "RHOIJ" in symbols or "RHOIJ1" in symbols:
        reads.add("rho")
    if symbols.intersection(frozenset(("XIJ", "R2IJ", "RIJ"))):
        reads.update(("x", "y", "z"))
    if symbols.intersection(
        frozenset(
            ("HIJ", "WIJ", "WI", "WJ", "DWIJ", "DWI", "DWJ", "GHI", "GHJ", "GHIJ")
        )
    ):
        reads.add("h")
    return frozenset(reads)


def _stage_can_launch_from_kernel_call(stage):
    if stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE):
        return True
    return _stage_can_use_pointwise_kernel(stage)


def _stage_can_use_pointwise_kernel(stage):
    if stage.kind is StageKind.POINTWISE:
        return True
    if stage.kind is StageKind.REDUCTION:
        return all(method.method_kind.value != "reduce" for method in stage.methods)
    return False


def _pair_traversal_for_stage(context):
    assert hasattr(context, "cell_bucket_counts")
    return "hbucket"


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


def _source_module_options():
    return ("--use_fast_math",)


def _hbucket_bucket_count():
    return 1


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


def _stage_extra_args(helper, stage, equations, info, extra_args, precompute):
    names = _stage_extra_arg_names(stage, equations, precompute)
    return tuple(
        _stage_extra_arg_value(helper, name, info, extra_args) for name in names
    )


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


def _neighbor_context_for_info(evaluator, info, stream, h_reduce_scratch, workspace):
    dest = info["dest"]
    src = info["src"]
    assert dest is src
    nnps = evaluator.nnps
    lower, upper, periodic = _neighbor_context_bounds_and_periodicity(nnps)
    radius_scale = np.float32(nnps.radius_scale)
    nreal = dest.get_number_of_particles(True)
    source_count = dest.get_number_of_particles()
    context = build_fused_cuda_neighbor_context_with_workspace(
        dest.gpu.x.dev,
        dest.gpu.y.dev,
        dest.gpu.z.dev,
        dest.gpu.h.dev,
        source_count,
        lower,
        upper,
        periodic,
        radius_scale,
        _hbucket_bucket_count(),
        stream,
        workspace,
        h_reduce_scratch,
    )
    return replace(context, destination_count=nreal)


def _neighbor_context_bounds_and_periodicity(nnps):
    domain_manager = nnps.domain.manager
    nnps_lower = np.asarray(nnps.xmin, dtype=np.float32)
    nnps_upper = np.asarray(nnps.xmax, dtype=np.float32)
    domain_lower, domain_upper, periodic = _domain_bounds_and_periodicity_from_manager(
        domain_manager
    )
    lower = np.where(periodic, domain_lower, nnps_lower).astype(np.float32)
    upper = np.where(periodic, domain_upper, nnps_upper).astype(np.float32)
    return lower, upper, periodic


def _domain_bounds_and_periodicity_from_manager(domain_manager):
    use_minimum_image = bool(domain_manager.minimum_image_periodic)
    periodic = np.array(
        [
            use_minimum_image and domain_manager.periodic_in_x,
            use_minimum_image and domain_manager.periodic_in_y,
            use_minimum_image and domain_manager.periodic_in_z,
        ],
        dtype=np.bool_,
    )
    lower = np.array(
        [domain_manager.xmin, domain_manager.ymin, domain_manager.zmin],
        dtype=np.float32,
    )
    upper = np.array(
        [domain_manager.xmax, domain_manager.ymax, domain_manager.zmax],
        dtype=np.float32,
    )
    return lower, upper, periodic


def _snapshot_hbucket_pair_window_stage(stages):
    reason = "; ".join(stage.reason for stage in stages)
    stage = snapshot_hbucket_pair_window_stage(stages)
    return replace(stage, reason=f"snapshot hbucket pair window: {reason}")


def _snapshot_hbucket_pair_window_key(stages, precompute, snapshot_fields):
    return (
        tuple(
            tuple(
                (method.equation_name, method.method_kind.value)
                for method in stage.methods
            )
            for stage in stages
        ),
        tuple(sorted(precompute.symbols)),
        snapshot_fields,
    )


def _unique_equations_for_window(equations_by_stage):
    equations = []
    names = []
    for stage_equations in equations_by_stage:
        for equation in stage_equations:
            name = equation.__class__.__name__
            if name not in names:
                names.append(name)
                equations.append(equation)
    return tuple(equations)


def _device_pointer_int(array):
    gpudata = array.gpudata
    if isinstance(gpudata, int):
        return gpudata
    return int(gpudata)
