"""Stage-level execution protocol for generic fused CUDA kernels."""

import ctypes
import ctypes.util
import inspect
import os
from dataclasses import replace

import numpy as np

from pysph.base.fused_cuda_nnps import (
    FusedCudaNeighborWorkspace,
    build_fused_cuda_neighbor_context_with_workspace,
    create_fused_cuda_convergence_flag,
    read_fused_cuda_convergence_flag,
    reset_fused_cuda_convergence_flag,
    wrap_periodic_xyz,
)
from pysph.sph.fused_cuda_codegen import (
    CudaPairPrecompute,
    cubic_spline_pair_precompute_for_symbols,
    generate_hbucket_pair_stage_outline_from_equations,
    generate_hbucket_pair_stage_outline_from_equations_with_convergence_flag,
    generate_resident_hbucket_pair_window_outline_from_equations,
    PairLaunchConfig,
    generate_pointwise_stage_outline_from_equations,
    launch_hbucket_pair_kernel_with_context,
    launch_pointwise_kernel,
    precompute_argument_names,
    quintic_spline_pair_precompute_for_symbols,
)
from pysph.sph.fused_cuda_stage_plan import (
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
            self.resident_window_by_group,
            self.resident_window_covered_groups,
        ) = _resident_window_mapping(helper, self.stage_group_by_plan_index)
        self.has_device_convergence = _has_device_convergence(helper)
        self.launched_groups = set()
        self.device_convergence_skip_groups = set()
        self.device_convergence_active = False
        self.device_convergence_iteration_counts = []
        self.device_convergence_rebuild_count = 0
        self.device_convergence_host_flag_pull_count = 0
        self.device_convergence_device_flag_read_count = 0

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
        if stage_group in self.resident_window_covered_groups:
            return True
        if stage_group in self.covered_stage_groups:
            return True
        if stage_group not in self.stage_by_group:
            return False
        if stage_group in self.launched_groups:
            return True
        if stage_group in self.resident_window_by_group:
            self._launch_resident_window(
                evaluator, self.resident_window_by_group[stage_group], extra_args
            )
        else:
            stage = self.stage_by_group[stage_group]
            self._launch_stage(evaluator, stage, info, extra_args)
        self.launched_groups.add(stage_group)
        return True

    def end_compute(self, evaluator, t, dt):
        return False

    def handle_outer_update_nnps(self, integrator, index):
        return True

    def handle_reorder_update_nnps(self, solver):
        return True

    def _launch_stage(self, evaluator, stage, info, extra_args):
        raise NotImplementedError("subclasses must launch the fused CUDA stage")

    def _launch_resident_window(self, evaluator, stage_indices, extra_args):
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
                stage = self.stage_by_group[kernel_info["stage_group"]]
                self._launch_stage(evaluator, stage, kernel_info, extra_args)
            iter_count = iteration + 1
            has_min_iterations = iter_count >= min_iterations
            at_max_iterations = iter_count == max_iterations
            has_converged = False
            if has_min_iterations and not at_max_iterations:
                has_converged = self._device_convergence_has_converged(info)
            if has_min_iterations and (at_max_iterations or has_converged):
                self.device_convergence_iteration_counts.append(iter_count)
                return
            if update_nnps:
                self.device_convergence_rebuild_count += 1
                self._update_device_convergence_nnps(evaluator, info)

    def _device_convergence_has_converged(self, info):
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
                stage_group = call["stage_group"]
                assert stage_group in self.stage_by_group
                if stage_group not in groups:
                    groups.append(stage_group)
                    kernel_infos.append(call)
        assert False

    def _device_convergence_policy(self):
        for stage in self.helper.cuda_stage_plan.stages:
            if stage.kind is StageKind.DEVICE_CONVERGENCE:
                return stage.convergence_policy
        return None

    def _kernel_info_for_plan_stage_index(self, stage_index):
        assert stage_index in self.stage_group_by_plan_index
        stage_group = self.stage_group_by_plan_index[stage_index]
        for call in self.helper.calls:
            if call["type"] == "kernel" and call["stage_group"] == stage_group:
                return call
        assert False


class GeneratedFusedCudaStageBackend(FusedCudaStageBackend):
    """Executable fused CUDA stage backend for the supported strict subset."""

    def __init__(self, helper):
        super().__init__(helper)
        import pycuda.driver as cuda

        self.stream = cuda.Stream()
        self.modules = {}
        self.cooperative_modules = {}
        self.outlines = {}
        self.cooperative_outlines = {}
        self.cooperative_extra_arg_names = {}
        self.neighbor_contexts = {}
        self.neighbor_workspaces = {}
        self.device_convergence_flag = None
        self.device_convergence_uses_particle_flag = False
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

    def _launch_resident_window(self, evaluator, stage_indices, extra_args):
        stages = tuple(
            self.helper.cuda_stage_plan.stages[index] for index in stage_indices
        )
        infos = tuple(
            self._kernel_info_for_plan_stage_index(index) for index in stage_indices
        )
        equations_by_stage = tuple(
            _equations_for_stage(self.helper, stage, info["stage_group"])
            for stage, info in zip(stages, infos)
        )
        precomputes = tuple(
            _precompute_for_stage(stage, self.helper.object.kernel) for stage in stages
        )
        context = self._neighbor_context_for_stage(evaluator, infos[0])
        if not _resident_window_preserves_particle_grid(context.n):
            super()._launch_resident_window(evaluator, stage_indices, extra_args)
            return
        outline = self._outline_for_resident_hbucket_pair_window(
            stages, equations_by_stage, precomputes
        )
        module = self._cooperative_module_for_outline(outline)
        stage_args = self._resident_hbucket_pair_window_extra_args(
            stages, equations_by_stage, infos, extra_args, precomputes
        )
        timer = _stage_timer(self.stream)
        launch_config = _launch_cooperative_hbucket_pair_window_kernel(
            module,
            outline.name,
            context,
            _cooperative_grid_block_count_for_context(context.n),
            stage_args,
            self.stream,
        )
        self._record_pair_launch("resident_hbucket")
        self._record_pair_launch_config(launch_config)
        for stage in stages:
            self._finish_launched_stage(stage)
        self._record_stage_timing(
            _resident_hbucket_pair_window_stage(stages), "resident_hbucket", timer
        )
        self.launch_count += 1

    def _outline_for_resident_hbucket_pair_window(
        self, stages, equations_by_stage, precomputes
    ):
        outline_key = _resident_hbucket_pair_window_key(stages, precomputes)
        if outline_key in self.cooperative_outlines:
            return self.cooperative_outlines[outline_key]
        outline = generate_resident_hbucket_pair_window_outline_from_equations(
            "cuda_eval", stages, equations_by_stage, precomputes
        )
        self.cooperative_outlines[outline_key] = outline
        return outline

    def _resident_hbucket_pair_window_extra_args(
        self, stages, equations_by_stage, infos, extra_args, precomputes
    ):
        names_key = _resident_hbucket_pair_window_key(stages, precomputes)
        if names_key not in self.cooperative_extra_arg_names:
            self.cooperative_extra_arg_names[names_key] = (
                _resident_hbucket_pair_window_extra_arg_names(
                    stages, equations_by_stage, precomputes
                )
            )
        return tuple(
            _stage_extra_arg_value(self.helper, name, infos[0], extra_args)
            for name in self.cooperative_extra_arg_names[names_key]
        )

    def _launch_single_stage(self, evaluator, stage, info, extra_args):
        equations = _equations_for_stage(self.helper, stage, info["stage_group"])
        precompute = _precompute_for_stage(stage, self.helper.object.kernel)
        convergence_field = self._device_convergence_field_for_stage(stage)
        timer = _stage_timer(self.stream)
        traversal = "pointwise"
        if stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE):
            context = self._neighbor_context_for_stage(evaluator, info)
            traversal = _pair_traversal_for_stage(context)
            outline = self._outline_for_stage(
                stage, equations, precompute, info, convergence_field, traversal
            )
            module = self._module_for_stage(outline, info, stage)
            stage_args = _stage_extra_args(
                self.helper, stage, equations, info, extra_args, precompute
            )
            if convergence_field is not None:
                stage_args = (self.device_convergence_flag,) + stage_args
            self._record_pair_launch(traversal)
            launch_config = launch_hbucket_pair_kernel_with_context(
                module, outline.name, context, stage_args
            )
            self._record_pair_launch_config(launch_config)
        elif _stage_can_use_pointwise_kernel(stage):
            outline = self._outline_for_stage(
                stage, equations, precompute, info, convergence_field, traversal
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

    def _device_convergence_has_converged(self, info):
        if _assume_converged_after_min_iterations():
            return True
        if (
            self.device_convergence_uses_particle_flag
            and self.device_convergence_flag is not None
        ):
            self.device_convergence_device_flag_read_count += 1
            return self._read_device_convergence_flag()
        policy = self._device_convergence_policy()
        if policy is not None and policy.flag_fields:
            return self._device_convergence_flags_are_positive(info, policy)
        for equation in info["equations"]:
            if hasattr(equation, "_gpu"):
                self.device_convergence_host_flag_pull_count += 1
                equation._pull("equation_has_converged")
            if not (equation.converged() > 0):
                return False
        return True

    def _device_convergence_flags_are_positive(self, info, policy):
        for equation in info["equations"]:
            if equation.__class__.__name__ not in policy.equation_names:
                continue
            if hasattr(equation, "_gpu"):
                self.device_convergence_host_flag_pull_count += 1
                equation._pull(*policy.flag_fields)
            for field in policy.flag_fields:
                if not (getattr(equation, field) > 0):
                    return False
        return True

    def _begin_device_convergence_iteration(self, info):
        if not self.device_convergence_uses_particle_flag:
            return
        if self.device_convergence_flag is None:
            self.device_convergence_flag = create_fused_cuda_convergence_flag(
                self.stream
            )
            return
        reset_fused_cuda_convergence_flag(self.device_convergence_flag, self.stream)

    def _read_device_convergence_flag(self):
        return read_fused_cuda_convergence_flag(
            self.device_convergence_flag, self.stream
        )

    def _begin_device_convergence_super_stage(self, kernel_infos):
        stages = tuple(
            self.stage_by_group[kernel_info["stage_group"]]
            for kernel_info in kernel_infos
        )
        self.device_convergence_uses_particle_flag = all(
            _stage_convergence_particle_field(stage) is not None for stage in stages
        )

    def _end_device_convergence_super_stage(self):
        self.device_convergence_uses_particle_flag = False

    def _device_convergence_field_for_stage(self, stage):
        if not self.device_convergence_uses_particle_flag:
            return None
        return _stage_convergence_particle_field(stage)

    def _update_device_convergence_nnps(self, evaluator, info):
        self.neighbor_contexts = {}

    def handle_internal_update_nnps(self, evaluator):
        self.neighbor_contexts = {}
        return True

    def handle_reorder_update_nnps(self, solver):
        self.neighbor_contexts = {}
        return True

    def _outline_for_stage(
        self, stage, equations, precompute, info, convergence_field, traversal
    ):
        stage_group = info["stage_group"]
        outline_key = (
            stage_group,
            stage.kind.value,
            traversal,
            tuple(
                (method.equation_name, method.method_kind.value)
                for method in stage.methods
            ),
            convergence_field,
        )
        if outline_key in self.outlines:
            return self.outlines[outline_key]
        if stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE):
            if convergence_field is None:
                outline = generate_hbucket_pair_stage_outline_from_equations(
                    "cuda_eval", stage, equations, precompute
                )
            else:
                outline = generate_hbucket_pair_stage_outline_from_equations_with_convergence_flag(
                    "cuda_eval", stage, equations, precompute, convergence_field
                )
        elif _stage_can_use_pointwise_kernel(stage):
            outline = generate_pointwise_stage_outline_from_equations(
                "cuda_eval", stage, equations
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
            "fused_convergence_flag" in outline.source,
        )
        if module_key in self.modules:
            return self.modules[module_key]
        from pycuda.compiler import SourceModule

        module = SourceModule(
            outline.source, no_extern_c=True, options=list(_source_module_options())
        )
        self.modules[module_key] = module
        return module

    def _cooperative_module_for_outline(self, outline):
        module_key = (outline.name, outline.source, tuple(_source_module_options()))
        if module_key in self.cooperative_modules:
            return self.cooperative_modules[module_key]
        module = _CtypesCooperativeCudaModule(outline.source, _source_module_options())
        self.cooperative_modules[module_key] = module
        return module


def _stage_group_mapping(helper):
    stage_groups = []
    for call in helper.calls:
        if call["type"] == "kernel" and call["stage_group"] not in stage_groups:
            stage_groups.append(call["stage_group"])
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


def _resident_window_mapping(helper, stage_group_by_plan_index):
    window_by_group = {}
    covered_groups = set()
    for stage_indices in _resident_pair_windows(helper.cuda_stage_plan.stages):
        if not all(index in stage_group_by_plan_index for index in stage_indices):
            continue
        first_group = stage_group_by_plan_index[stage_indices[0]]
        window_by_group[first_group] = stage_indices
        for stage_index in stage_indices[1:]:
            covered_groups.add(stage_group_by_plan_index[stage_index])
    return window_by_group, covered_groups


def _resident_pair_windows(stages):
    windows = []
    index = 0
    while index < len(stages) - 1:
        left = stages[index]
        right = stages[index + 1]
        if _can_launch_resident_pair_window(left, right):
            windows.append((index, index + 1))
            index += 2
        else:
            index += 1
    return tuple(windows)


def _can_launch_resident_pair_window(left, right):
    return (
        left.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)
        and right.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)
        and left.dest == right.dest
        and left.sources == right.sources
        and not _stage_invalidates_neighbor_context(left)
        and bool(_stage_source_reads(right).intersection(_stage_dest_writes(left)))
    )


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


def _stage_dest_writes(stage):
    writes = set()
    for method in stage.methods:
        writes.update(method.dest_writes)
    return frozenset(writes)


def _stage_source_reads(stage):
    reads = set()
    for method in stage.methods:
        reads.update(method.source_reads)
        reads.update(_precomputed_source_reads(method.precomputed_symbols))
    return frozenset(reads)


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


def _stage_convergence_particle_field(stage):
    for method in stage.methods:
        if "converged" in method.dest_writes:
            return "converged"
    return None


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


def _assume_converged_after_min_iterations():
    return True


def _hbucket_bucket_count():
    return 4


def _stage_timing_key(stage, traversal):
    method_names = "+".join(
        f"{method.equation_name}.{method.method_kind.value}" for method in stage.methods
    )
    return f"{stage.kind.value}:{traversal}:{method_names}"


def _equations_for_stage(helper, stage, stage_group):
    equations = list(_equations_for_stage_group(helper, stage_group))
    names = [equation.__class__.__name__ for equation in equations]
    for method in stage.methods:
        if method.equation_name not in names:
            equation = _equation_for_method(
                method.equation_name,
                _all_equations(helper),
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
            return cubic_spline_pair_precompute_for_symbols(dim, frozenset(symbols))
        if kernel_name == "QuinticSpline":
            return quintic_spline_pair_precompute_for_symbols(dim, frozenset(symbols))
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
    return build_fused_cuda_neighbor_context_with_workspace(
        dest.gpu.x.dev,
        dest.gpu.y.dev,
        dest.gpu.z.dev,
        dest.gpu.h.dev,
        nreal,
        lower,
        upper,
        periodic,
        radius_scale,
        _hbucket_bucket_count(),
        stream,
        workspace,
        h_reduce_scratch,
    )


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


def _resident_hbucket_pair_window_extra_arg_names(
    stages, equations_by_stage, precomputes
):
    names = []
    for precompute in precomputes:
        for name in precompute_argument_names(precompute):
            _append_once(names, name)
    for stage, equations, precompute in zip(stages, equations_by_stage, precomputes):
        for name in _stage_equation_extra_arg_names(stage, equations, precompute):
            _append_once(names, name)
    return tuple(names)


def _resident_hbucket_pair_window_stage(stages):
    methods = tuple(method for stage in stages for method in stage.methods)
    method_segments = tuple(stage.methods for stage in stages)
    reason = "; ".join(stage.reason for stage in stages)
    return replace(
        stages[0],
        methods=methods,
        reason=f"resident hbucket pair window: {reason}",
        legacy_group_count=sum(stage.legacy_group_count for stage in stages),
        method_segments=method_segments,
    )


def _resident_hbucket_pair_window_key(stages, precomputes):
    return (
        tuple(
            tuple(
                (method.equation_name, method.method_kind.value)
                for method in stage.methods
            )
            for stage in stages
        ),
        tuple(tuple(sorted(precompute.symbols)) for precompute in precomputes),
    )


def _resident_window_preserves_particle_grid(n):
    block_size = _pair_block_size_for_count(n)
    particle_blocks = (int(n) + block_size - 1) // block_size
    return particle_blocks <= _resident_grid_block_count()


def _cooperative_grid_block_count_for_context(n):
    block_size = _pair_block_size_for_count(n)
    particle_blocks = (int(n) + block_size - 1) // block_size
    return min(_resident_grid_block_count(), particle_blocks)


def _resident_grid_block_count():
    import pycuda.driver as cuda

    device = cuda.Context.get_device()
    return int(device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT))


def _pair_block_size_for_count(n):
    full_block_size = 256
    full_particle_blocks = (int(n) + full_block_size - 1) // full_block_size
    if full_particle_blocks < _resident_grid_block_count():
        return 128
    return full_block_size


class _CtypesCooperativeCudaModule:
    def __init__(self, source, options):
        from pycuda.compiler import compile

        self._driver = _cuda_driver()
        image = compile(source, no_extern_c=True, options=list(options))
        self._image = ctypes.create_string_buffer(image)
        self._module = ctypes.c_void_p()
        result = self._driver.cuModuleLoadData(
            ctypes.byref(self._module),
            ctypes.cast(self._image, ctypes.c_void_p),
        )
        assert result == 0
        self._functions = {}
        self._argument_caches = {}

    def get_function(self, name):
        if name not in self._functions:
            function = ctypes.c_void_p()
            result = self._driver.cuModuleGetFunction(
                ctypes.byref(function),
                self._module,
                name.encode("ascii"),
            )
            assert result == 0
            self._functions[name] = function
        return self._functions[name]

    def kernel_arguments(self, name, values):
        if name not in self._argument_caches:
            self._argument_caches[name] = _CtypesKernelArgumentCache()
        return self._argument_caches[name].arguments(values)


def _cuda_driver():
    driver = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
    driver.cuModuleLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    driver.cuModuleLoadData.restype = ctypes.c_int
    driver.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    driver.cuModuleGetFunction.restype = ctypes.c_int
    driver.cuLaunchCooperativeKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    driver.cuLaunchCooperativeKernel.restype = ctypes.c_int
    return driver


def _launch_cooperative_hbucket_pair_window_kernel(
    module, kernel_name, context, grid_blocks, extra_args, stream
):
    function = module.get_function(kernel_name)
    block_size = _pair_block_size_for_count(context.n)
    values = _cooperative_hbucket_pair_window_kernel_values(context, extra_args)
    kernel_args, argument_storage = module.kernel_arguments(kernel_name, values)
    result = module._driver.cuLaunchCooperativeKernel(
        function,
        ctypes.c_uint(grid_blocks),
        ctypes.c_uint(1),
        ctypes.c_uint(1),
        ctypes.c_uint(block_size),
        ctypes.c_uint(1),
        ctypes.c_uint(1),
        ctypes.c_uint(0),
        ctypes.c_void_p(int(stream.handle)),
        kernel_args,
    )
    assert result == 0
    assert argument_storage
    return PairLaunchConfig(
        traversal="resident_hbucket",
        n=int(context.n),
        block_size=int(block_size),
        grid_x=int(grid_blocks),
    )


def _cooperative_hbucket_pair_window_kernel_values(context, extra_args):
    return (
        _kernel_arg(context.x),
        _kernel_arg(context.y),
        _kernel_arg(context.z),
        _kernel_arg(context.h),
        np.int32(context.n),
        np.float32(context.lower[0]),
        np.float32(context.upper[0]),
        np.float32(context.lower[1]),
        np.float32(context.upper[1]),
        np.float32(context.lower[2]),
        np.float32(context.upper[2]),
        np.int32(context.periodic[0]),
        np.int32(context.periodic[1]),
        np.int32(context.periodic[2]),
        context.radius_scale,
        np.int32(context.cell_counts[0]),
        np.int32(context.cell_counts[1]),
        np.int32(context.cell_counts[2]),
        np.int32(context.total_cells),
        np.int32(context.bucket_count),
        np.float32(context.cell_width[0]),
        np.float32(context.cell_width[1]),
        np.float32(context.cell_width[2]),
        _kernel_arg(context.bucket_h_max_bits),
        _kernel_arg(context.cell_bucket_h_max_bits),
        _kernel_arg(context.cell_bucket_counts),
        _kernel_arg(context.cell_bucket_starts),
        _kernel_arg(context.sorted_ids),
        *tuple(_kernel_arg(arg) for arg in extra_args),
    )


class _CtypesKernelArgumentCache:
    def __init__(self):
        self.args = None
        self.storage = None
        self.value_types = None

    def arguments(self, values):
        value_types = tuple(type(value) for value in values)
        if self.storage is None:
            self.args, self.storage = _ctypes_kernel_arguments(values)
            self.value_types = value_types
            return self.args, self.storage
        assert value_types == self.value_types
        assert len(values) == len(self.storage)
        for storage, value in zip(self.storage, values):
            _update_ctypes_kernel_argument(storage, value)
        return self.args, self.storage


def _ctypes_kernel_arguments(values):
    storage = tuple(_ctypes_kernel_argument(value) for value in values)
    args = (ctypes.c_void_p * len(storage))()
    for index, value in enumerate(storage):
        args[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
    return args, storage


def _ctypes_kernel_argument(value):
    if isinstance(value, np.float32):
        return ctypes.c_float(float(value))
    if isinstance(value, np.int32):
        return ctypes.c_int(int(value))
    if isinstance(value, np.uint32):
        return ctypes.c_uint(int(value))
    if isinstance(value, np.uintp):
        return ctypes.c_void_p(int(value))
    if isinstance(value, float):
        return ctypes.c_float(value)
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    return ctypes.c_void_p(int(value))


def _update_ctypes_kernel_argument(storage, value):
    if isinstance(storage, ctypes.c_float):
        storage.value = float(value)
        return
    if isinstance(storage, ctypes.c_int):
        storage.value = int(value)
        return
    if isinstance(storage, ctypes.c_uint):
        storage.value = int(value)
        return
    if isinstance(storage, ctypes.c_void_p):
        storage.value = int(value)
        return
    assert False


def _kernel_arg(arg):
    if hasattr(arg, "gpudata"):
        gpudata = arg.gpudata
        if isinstance(gpudata, int):
            return np.uintp(gpudata)
        return gpudata
    return arg
