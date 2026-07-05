"""Generic fused CUDA kernel planning helpers."""

import ast
import inspect
import textwrap
from dataclasses import dataclass

import numpy as np

from pysph.sph.equation import CUDAGroup, KnownType
from pysph.sph.fused_cuda_stage_plan import (
    CudaStagePlan,
    MethodDeps,
    MethodKind,
    StageKind,
    StageNode,
)


@dataclass(frozen=True)
class FusedKernelSpec:
    """One CUDA launch planned from a fused stage graph."""

    name: str
    stage: StageNode
    uses_neighbors: bool


@dataclass(frozen=True)
class FusedKernelOutline:
    """Non-executable fused kernel outline used as a codegen boundary."""

    name: str
    source: str


@dataclass(frozen=True)
class CudaInlineMethodBody:
    """CUDA statements lowered from one PySPH equation method."""

    equation_name: str
    method_kind: MethodKind
    argument_declarations: tuple[str, ...]
    lines: tuple[str, ...]


@dataclass(frozen=True)
class CudaEquationMethodCall:
    """One call from a fused CUDA loop into a generated equation wrapper."""

    equation_name: str
    method_kind: MethodKind
    function_name: str
    argument_declarations: tuple[str, ...]
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class CudaPairPrecompute:
    """CUDA precomputed symbols used by generated pair-loop calls."""

    symbols: frozenset[str]
    helper_source: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class FusedLaunchBudget:
    """Launch-count summary for a fused CUDA stage graph."""

    neighbor_build_count: int
    rhs_core_kernel_count: int
    total_launch_count: int


@dataclass(frozen=True)
class LocalReductionField:
    """One per-destination reduction field accumulated inside a pair kernel."""

    field: str
    operation: str


@dataclass(frozen=True)
class PairLaunchConfig:
    """CUDA launch shape used for one generated pair traversal kernel."""

    traversal: str
    n: int
    block_size: int
    grid_x: int


def fused_kernel_specs(
    plan_id: str, plan: CudaStagePlan
) -> tuple[FusedKernelSpec, ...]:
    """Return CUDA kernel specs for a strict fused stage plan."""
    specs = []
    neighbor_build_dests = []
    for stage in plan.stages:
        assert stage.kind is not StageKind.HOST_BOUNDARY
        if _stage_uses_neighbors(stage) and stage.dest not in neighbor_build_dests:
            specs.append(
                FusedKernelSpec(
                    name=f"fused_{plan_id}_{stage.dest}_neighbor_build",
                    stage=StageNode(
                        kind=StageKind.NEIGHBOR_BUILD,
                        dest=stage.dest,
                        sources=stage.sources,
                        methods=(),
                        reason="metadata for fused pair stages",
                        convergence_policy=None,
                    ),
                    uses_neighbors=False,
                )
            )
            neighbor_build_dests.append(stage.dest)
        specs.append(
            FusedKernelSpec(
                name=fused_kernel_name(plan_id, stage),
                stage=stage,
                uses_neighbors=_stage_uses_neighbors(stage),
            )
        )
    return tuple(specs)


def fused_kernel_name(plan_id: str, stage: StageNode) -> str:
    """Return the stable CUDA function name for one fused stage."""
    assert stage.kind is not StageKind.HOST_BOUNDARY
    return f"fused_{plan_id}_{stage.dest}_{stage.kind.value}"


def launch_budget_for_specs(specs: tuple[FusedKernelSpec, ...]) -> FusedLaunchBudget:
    """Return the launch-count budget represented by kernel specs."""
    neighbor_build_count = sum(
        1 for spec in specs if spec.stage.kind is StageKind.NEIGHBOR_BUILD
    )
    total_launch_count = len(specs)
    return FusedLaunchBudget(
        neighbor_build_count=neighbor_build_count,
        rhs_core_kernel_count=total_launch_count - neighbor_build_count,
        total_launch_count=total_launch_count,
    )


def generate_fused_kernel_outline(plan_id: str, stage: StageNode) -> FusedKernelOutline:
    """Return a stable, non-executable fused kernel outline for one stage."""
    if _stage_uses_neighbors(stage):
        return generate_direct_pair_loop_outline(plan_id, stage)
    name = fused_kernel_name(plan_id, stage)
    method_lines = [
        f"    // {method.equation_name}.{method.method_kind.value}"
        for method in stage.methods
    ]
    source = "\n".join(
        (
            f'extern "C" __global__ void {name}(void)',
            "{",
            *method_lines,
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def generate_pointwise_kernel_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    calls: tuple[CudaEquationMethodCall, ...],
) -> FusedKernelOutline:
    """Return a pointwise CUDA outline using generated equation wrappers."""
    assert _stage_can_use_pointwise_kernel(stage)
    name = fused_kernel_name(plan_id, stage)
    method_lines = _direct_pair_equation_call_lines(stage, calls)
    arguments = ("int n",) + _equation_call_arguments(calls)
    fp32_wrapper_source = _force_cuda_source_fp32(wrapper_source)
    source = "\n".join(
        (
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            "    int dst = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (dst >= n) {",
            "        return;",
            "    }",
            *tuple(
                line.replace("                                        ", "    ")
                for line in method_lines
            ),
            "}",
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def generate_direct_pair_loop_outline(
    plan_id: str, stage: StageNode
) -> FusedKernelOutline:
    """Return a destination-owned direct pair-loop CUDA outline."""
    return _generate_direct_pair_loop_outline(plan_id, stage, ())


def generate_direct_pair_loop_outline_with_inline_bodies(
    plan_id: str, stage: StageNode, bodies: tuple[CudaInlineMethodBody, ...]
) -> FusedKernelOutline:
    """Return a direct pair-loop CUDA outline with lowered equation bodies."""
    return _generate_direct_pair_loop_outline(plan_id, stage, bodies)


def generate_direct_pair_loop_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    calls: tuple[CudaEquationMethodCall, ...],
) -> FusedKernelOutline:
    """Return a direct pair-loop CUDA outline using generated equation wrappers."""
    precompute = CudaPairPrecompute(symbols=frozenset(), helper_source="", lines=())
    return _generate_direct_pair_loop_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls, None
    )


def generate_direct_pair_loop_outline_with_equation_calls_and_precompute(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
) -> FusedKernelOutline:
    """Return a direct pair-loop CUDA outline with wrapper calls and precompute."""
    return _generate_direct_pair_loop_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls, None
    )


def generate_direct_pair_loop_outline_with_convergence_flag(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
    convergence_field: str,
) -> FusedKernelOutline:
    """Return a direct pair-loop outline that clears a device convergence flag."""
    return _generate_direct_pair_loop_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls, convergence_field
    )


def generate_direct_pair_stage_outline_from_equations(
    plan_id: str,
    stage: StageNode,
    equations: tuple[object, ...],
    precompute: CudaPairPrecompute,
) -> FusedKernelOutline:
    """Return a direct pair outline generated from PySPH equations."""
    assert _stage_uses_neighbors(stage)
    known_types = _cuda_known_types_for_stage(stage, equations, precompute.symbols)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_stage(
        stage, equations, known_types, precompute.symbols
    )
    return generate_direct_pair_loop_outline_with_equation_calls_and_precompute(
        plan_id, stage, wrapper_source, precompute, calls
    )


def generate_direct_pair_stage_outline_from_equations_with_convergence_flag(
    plan_id: str,
    stage: StageNode,
    equations: tuple[object, ...],
    precompute: CudaPairPrecompute,
    convergence_field: str,
) -> FusedKernelOutline:
    """Return a direct pair outline that clears a device convergence flag."""
    assert _stage_uses_neighbors(stage)
    known_types = _cuda_known_types_for_stage(stage, equations, precompute.symbols)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_stage(
        stage, equations, known_types, precompute.symbols
    )
    return generate_direct_pair_loop_outline_with_convergence_flag(
        plan_id, stage, wrapper_source, precompute, calls, convergence_field
    )


def generate_cluster_pair_stage_outline_from_equations(
    plan_id: str,
    stage: StageNode,
    equations: tuple[object, ...],
    precompute: CudaPairPrecompute,
) -> FusedKernelOutline:
    """Return a sorted-cell cluster pair outline generated from PySPH equations."""
    assert _stage_uses_neighbors(stage)
    known_types = _cuda_known_types_for_stage(stage, equations, precompute.symbols)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_stage(
        stage, equations, known_types, precompute.symbols
    )
    return _generate_cluster_pair_loop_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls, None
    )


def generate_cluster_pair_stage_outline_from_equations_with_convergence_flag(
    plan_id: str,
    stage: StageNode,
    equations: tuple[object, ...],
    precompute: CudaPairPrecompute,
    convergence_field: str,
) -> FusedKernelOutline:
    """Return a cluster pair outline that clears a device convergence flag."""
    assert _stage_uses_neighbors(stage)
    known_types = _cuda_known_types_for_stage(stage, equations, precompute.symbols)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_stage(
        stage, equations, known_types, precompute.symbols
    )
    return _generate_cluster_pair_loop_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls, convergence_field
    )


def generate_hbucket_pair_stage_outline_from_equations(
    plan_id: str,
    stage: StageNode,
    equations: tuple[object, ...],
    precompute: CudaPairPrecompute,
) -> FusedKernelOutline:
    """Return an h-bucket pair outline generated from PySPH equations."""
    assert _stage_uses_neighbors(stage)
    known_types = _cuda_known_types_for_stage(stage, equations, precompute.symbols)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_stage(
        stage, equations, known_types, precompute.symbols
    )
    return _generate_hbucket_pair_loop_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls, None
    )


def generate_cell_tile_hbucket_pair_stage_outline_from_equations(
    plan_id: str,
    stage: StageNode,
    equations: tuple[object, ...],
    precompute: CudaPairPrecompute,
) -> FusedKernelOutline:
    """Return a cell-tiled h-bucket pair outline generated from PySPH equations."""
    assert _stage_uses_neighbors(stage)
    known_types = _cuda_known_types_for_stage(stage, equations, precompute.symbols)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_stage(
        stage, equations, known_types, precompute.symbols
    )
    return _generate_cell_tile_hbucket_pair_loop_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls
    )


def generate_resident_hbucket_pair_window_outline_from_equations(
    plan_id: str,
    stages: tuple[StageNode, ...],
    equations_by_stage: tuple[tuple[object, ...], ...],
    precomputes: tuple[CudaPairPrecompute, ...],
) -> FusedKernelOutline:
    """Return a resident h-bucket pair-window outline with device grid barriers."""
    assert len(stages) >= 2
    assert len(stages) == len(equations_by_stage)
    assert len(stages) == len(precomputes)
    dest = stages[0].dest
    sources = stages[0].sources
    for stage in stages:
        assert stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)
        assert stage.dest == dest
        assert stage.sources == sources
    known_types = _cuda_known_types_for_stage_window(
        stages, equations_by_stage, precomputes
    )
    equations = _unique_equations(equations_by_stage)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls_by_stage = tuple(
        _cuda_equation_calls_for_stage(
            stage, equations_for_stage, known_types, precompute.symbols
        )
        for stage, equations_for_stage, precompute in zip(
            stages, equations_by_stage, precomputes
        )
    )
    return _generate_resident_hbucket_pair_window_outline_with_equation_calls(
        plan_id, stages, wrapper_source, precomputes, calls_by_stage
    )


def generate_hbucket_source_inline_pair_window_outline_from_equations(
    plan_id: str,
    stages: tuple[StageNode, ...],
    equations_by_stage: tuple[tuple[object, ...], ...],
    precomputes: tuple[CudaPairPrecompute, ...],
    source_inline_methods: tuple[MethodDeps, ...],
) -> FusedKernelOutline:
    """Return one h-bucket pair-window outline with source-local prep values."""
    assert len(stages) == 2
    assert len(equations_by_stage) == 2
    assert len(precomputes) == 2
    assert source_inline_methods
    _assert_hbucket_source_inline_pair_window_stages(stages)
    known_types = _cuda_known_types_for_stage_window(
        stages, equations_by_stage, precomputes
    )
    equations = _unique_equations(equations_by_stage)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    source_inline_fields = _source_inline_fields(source_inline_methods, stages[1])
    wrapper_source = _source_inline_wrapper_source(
        wrapper_source, stages[1].methods, source_inline_fields
    )
    calls_by_stage = tuple(
        _cuda_equation_calls_for_stage(
            stage, equations_for_stage, known_types, precompute.symbols
        )
        for stage, equations_for_stage, precompute in zip(
            stages, equations_by_stage, precomputes
        )
    )
    calls_by_stage = (
        calls_by_stage[0],
        _source_inline_equation_calls(
            calls_by_stage[1], stages[1].methods, source_inline_fields
        ),
    )
    return _generate_hbucket_source_inline_pair_window_outline_with_equation_calls(
        plan_id,
        stages,
        wrapper_source,
        precomputes,
        calls_by_stage,
        source_inline_methods,
        source_inline_fields,
    )


def _assert_hbucket_source_inline_pair_window_stages(
    stages: tuple[StageNode, ...],
) -> None:
    assert len(stages) == 2
    left, right = stages
    if left.kind is StageKind.POINTWISE:
        assert left.dest == right.dest
        assert left.sources == ()
        assert right.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)
        assert right.sources == (left.dest,)
        return
    dest = left.dest
    sources = left.sources
    for stage in stages:
        assert stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)
        assert stage.dest == dest
        assert stage.sources == sources


def hbucket_source_inline_pair_window_argument_names(
    stages: tuple[StageNode, ...],
    equations_by_stage: tuple[tuple[object, ...], ...],
    precomputes: tuple[CudaPairPrecompute, ...],
    source_inline_methods: tuple[MethodDeps, ...],
) -> tuple[str, ...]:
    """Return extra argument names for a source-inline h-bucket window."""
    assert len(stages) == 2
    assert len(equations_by_stage) == 2
    assert len(precomputes) == 2
    _assert_hbucket_source_inline_pair_window_stages(stages)
    known_types = _cuda_known_types_for_stage_window(
        stages, equations_by_stage, precomputes
    )
    equations = _unique_equations(equations_by_stage)
    group = CUDAGroup(list(equations))
    group.get_equation_wrappers(known_types)
    source_inline_fields = _source_inline_fields(source_inline_methods, stages[1])
    calls_by_stage = tuple(
        _cuda_equation_calls_for_stage(
            stage, equations_for_stage, known_types, precompute.symbols
        )
        for stage, equations_for_stage, precompute in zip(
            stages, equations_by_stage, precomputes
        )
    )
    calls_by_stage = (
        calls_by_stage[0],
        _source_inline_equation_calls(
            calls_by_stage[1], stages[1].methods, source_inline_fields
        ),
    )
    source_local_fields = _source_inline_local_fields(source_inline_methods)
    precompute_declarations = tuple(
        declaration
        for declarations in (
            _precompute_argument_declarations(precomputes[0]),
            _precompute_argument_declarations(precomputes[1]),
        )
        for declaration in declarations
    )
    call_arguments = tuple(
        declaration
        for calls in calls_by_stage
        for declaration in _equation_call_arguments(calls)
    )
    source_local_declarations = tuple(
        f"GLOBAL_MEM float* s_{field}"
        for field in source_local_fields
        if f"GLOBAL_MEM float* s_{field}" not in call_arguments
    )
    declarations = _unique_argument_declarations(
        precompute_declarations,
        call_arguments,
        source_local_declarations,
    )
    return tuple(
        _argument_name_from_declaration(declaration) for declaration in declarations
    )


def generate_hbucket_pair_stage_outline_from_equations_with_convergence_flag(
    plan_id: str,
    stage: StageNode,
    equations: tuple[object, ...],
    precompute: CudaPairPrecompute,
    convergence_field: str,
) -> FusedKernelOutline:
    """Return an h-bucket pair outline that clears a device convergence flag."""
    assert _stage_uses_neighbors(stage)
    known_types = _cuda_known_types_for_stage(stage, equations, precompute.symbols)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_stage(
        stage, equations, known_types, precompute.symbols
    )
    return _generate_hbucket_pair_loop_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls, convergence_field
    )


def generate_pointwise_stage_outline_from_equations(
    plan_id: str,
    stage: StageNode,
    equations: tuple[object, ...],
) -> FusedKernelOutline:
    """Return a pointwise outline generated from PySPH equations."""
    assert _stage_can_use_pointwise_kernel(stage)
    known_types = _cuda_known_types_for_stage(stage, equations, frozenset())
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_stage(stage, equations, known_types, frozenset())
    return generate_pointwise_kernel_outline_with_equation_calls(
        plan_id, stage, wrapper_source, calls
    )


def _stage_can_use_pointwise_kernel(stage: StageNode) -> bool:
    if stage.kind is StageKind.POINTWISE:
        return True
    if stage.kind is StageKind.REDUCTION:
        return all(
            method.method_kind is not MethodKind.REDUCE for method in stage.methods
        )
    return False


def _generate_direct_pair_loop_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
    convergence_field: str | None,
) -> FusedKernelOutline:
    assert _stage_uses_neighbors(stage)
    name = fused_kernel_name(plan_id, stage)
    precompute_lines = _direct_pair_precompute_lines(precompute)
    segment_lines = _direct_pair_segment_lines(stage, calls, precompute_lines)
    arguments = (
        fused_context_argument_declarations()
        + _convergence_flag_argument_declarations(convergence_field)
        + _unique_argument_declarations(
            _precompute_argument_declarations(precompute),
            _equation_call_arguments(calls),
        )
    )
    convergence_flag_lines = _convergence_flag_lines(convergence_field)
    fp32_wrapper_source = _force_cuda_source_fp32(wrapper_source)
    source = "\n".join(
        (
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            _DIRECT_PAIR_HELPERS,
            precompute.helper_source,
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            "    int dst = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (dst >= n) {",
            "        return;",
            "    }",
            *segment_lines,
            *convergence_flag_lines,
            "}",
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def _generate_cluster_pair_loop_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
    convergence_field: str | None,
) -> FusedKernelOutline:
    assert _stage_uses_neighbors(stage)
    name = fused_kernel_name(plan_id, stage)
    precompute_lines = _direct_pair_precompute_lines(precompute)
    segment_lines = _cluster_pair_segment_lines(stage, calls, precompute_lines)
    arguments = (
        cluster_context_argument_declarations()
        + _convergence_flag_argument_declarations(convergence_field)
        + _unique_argument_declarations(
            _precompute_argument_declarations(precompute),
            _equation_call_arguments(calls),
        )
    )
    convergence_flag_lines = _convergence_flag_lines(convergence_field)
    fp32_wrapper_source = _force_cuda_source_fp32(wrapper_source)
    source = "\n".join(
        (
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            _DIRECT_PAIR_HELPERS,
            precompute.helper_source,
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            "    int dst_cluster = blockIdx.x;",
            "    if (dst_cluster >= cluster_total) {",
            "        return;",
            "    }",
            "    int lane = threadIdx.x;",
            "    if (lane >= cluster_count[dst_cluster]) {",
            "        return;",
            "    }",
            "    int cell0 = cluster_cell[dst_cluster];",
            "    int base_cz = cell0 / (nx * ny);",
            "    int rem = cell0 - base_cz * nx * ny;",
            "    int base_cy = rem / nx;",
            "    int base_cx = rem - base_cy * nx;",
            "    int dst = sorted_ids[cluster_begin[dst_cluster] + lane];",
            *segment_lines,
            *convergence_flag_lines,
            "}",
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def _generate_hbucket_pair_loop_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
    convergence_field: str | None,
) -> FusedKernelOutline:
    assert _stage_uses_neighbors(stage)
    name = fused_kernel_name(plan_id, stage)
    segment_lines = _hbucket_pair_segment_lines(stage, calls, precompute)
    arguments = (
        hbucket_context_argument_declarations()
        + _convergence_flag_argument_declarations(convergence_field)
        + _unique_argument_declarations(
            _precompute_argument_declarations(precompute),
            _equation_call_arguments(calls),
        )
    )
    convergence_flag_lines = _convergence_flag_lines(convergence_field)
    fp32_wrapper_source = _hbucket_pair_wrapper_source(wrapper_source, stage)
    source = "\n".join(
        (
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            _DIRECT_PAIR_HELPERS,
            precompute.helper_source,
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            "    int dst = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (dst >= n) {",
            "        return;",
            "    }",
            *segment_lines,
            *convergence_flag_lines,
            "}",
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def _generate_cell_tile_hbucket_pair_loop_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
) -> FusedKernelOutline:
    assert _stage_uses_neighbors(stage)
    name = f"{fused_kernel_name(plan_id, stage)}_cell_tile"
    source_fields = _cell_tile_source_fields(stage, precompute, calls)
    segment_lines = _cell_tile_hbucket_pair_segment_lines(
        stage, calls, precompute, source_fields
    )
    arguments = hbucket_context_argument_declarations() + _unique_argument_declarations(
        _precompute_argument_declarations(precompute),
        _equation_call_arguments(calls),
    )
    fp32_wrapper_source = _cell_tile_pair_wrapper_source(
        wrapper_source, stage, source_fields
    )
    source = "\n".join(
        (
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            _DIRECT_PAIR_HELPERS,
            precompute.helper_source,
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            *segment_lines,
            "}",
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def _generate_resident_hbucket_pair_window_outline_with_equation_calls(
    plan_id: str,
    stages: tuple[StageNode, ...],
    wrapper_source: str,
    precomputes: tuple[CudaPairPrecompute, ...],
    calls_by_stage: tuple[tuple[CudaEquationMethodCall, ...], ...],
) -> FusedKernelOutline:
    assert stages
    name = f"fused_{plan_id}_{stages[0].dest}_resident_hbucket_pair_window"
    stage_lines = [
        "    cooperative_groups::grid_group grid = cooperative_groups::this_grid();"
    ]
    for index, stage in enumerate(stages):
        precompute = precomputes[index]
        calls = calls_by_stage[index]
        segment_lines = _hbucket_pair_segment_lines(stage, calls, precompute)
        stage_lines.append(
            "    for (int dst = blockIdx.x * blockDim.x + threadIdx.x; "
            "dst < n; dst += blockDim.x * gridDim.x) {"
        )
        stage_lines.extend(f"    {line}" for line in segment_lines)
        stage_lines.append("    }")
        if index < len(stages) - 1:
            stage_lines.append("    grid.sync();")
    precompute_declarations = tuple(
        declaration
        for precompute in precomputes
        for declaration in _precompute_argument_declarations(precompute)
    )
    call_arguments = tuple(
        declaration
        for calls in calls_by_stage
        for declaration in _equation_call_arguments(calls)
    )
    arguments = hbucket_context_argument_declarations() + _unique_argument_declarations(
        precompute_declarations, call_arguments
    )
    fp32_wrapper_source = _hbucket_pair_window_wrapper_source(wrapper_source, stages)
    source = "\n".join(
        (
            "#include <cooperative_groups.h>",
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            _DIRECT_PAIR_HELPERS,
            *_unique_precompute_helper_sources(precomputes),
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            *stage_lines,
            "}",
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def _generate_hbucket_source_inline_pair_window_outline_with_equation_calls(
    plan_id: str,
    stages: tuple[StageNode, ...],
    wrapper_source: str,
    precomputes: tuple[CudaPairPrecompute, ...],
    calls_by_stage: tuple[tuple[CudaEquationMethodCall, ...], ...],
    source_inline_methods: tuple[MethodDeps, ...],
    source_inline_fields: tuple[str, ...],
) -> FusedKernelOutline:
    assert len(stages) == 2
    name = f"fused_{plan_id}_{stages[0].dest}_hbucket_source_inline_pair_window"
    source_local_fields = _source_inline_local_fields(source_inline_methods)
    source_prep_lines = _source_inline_prep_lines(
        source_inline_methods, calls_by_stage[0], source_local_fields
    )
    first_segment_lines = _hbucket_source_inline_first_segment_lines(
        stages[0], calls_by_stage[0], precomputes[0]
    )
    second_segment_lines = _hbucket_pair_segment_lines_with_source_inline(
        stages[1],
        calls_by_stage[1],
        precomputes[1],
        source_prep_lines,
        source_local_fields,
    )
    precompute_declarations = tuple(
        declaration
        for declarations in (
            _precompute_argument_declarations(precomputes[0]),
            _precompute_argument_declarations(precomputes[1]),
        )
        for declaration in declarations
    )
    call_arguments = tuple(
        declaration
        for calls in calls_by_stage
        for declaration in _equation_call_arguments(calls)
    )
    source_local_declarations = tuple(
        f"GLOBAL_MEM float* s_{field}"
        for field in source_local_fields
        if f"GLOBAL_MEM float* s_{field}" not in call_arguments
    )
    arguments = hbucket_context_argument_declarations() + _unique_argument_declarations(
        precompute_declarations,
        call_arguments,
        source_local_declarations,
    )
    fp32_wrapper_source = _force_cuda_source_fp32(wrapper_source)
    source = "\n".join(
        (
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            _DIRECT_PAIR_HELPERS,
            *_unique_precompute_helper_sources(precomputes),
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            "    int dst = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (dst >= n) {",
            "        return;",
            "    }",
            *first_segment_lines,
            *second_segment_lines,
            "}",
            "}",
        )
    )
    assert source_inline_fields
    return FusedKernelOutline(name=name, source=source)


def cuda_equation_method_call_from_equation(equation, method_name, known_types):
    """Return a fused-loop call into an already generated CUDA equation wrapper."""
    return cuda_equation_method_call_from_equation_with_precomputed(
        equation, method_name, known_types, frozenset()
    )


def cuda_equation_method_call_from_equation_with_precomputed(
    equation, method_name, known_types, precomputed_symbols
):
    """Return a fused-loop equation wrapper call using local precomputed symbols."""
    assert hasattr(equation, method_name)
    assert equation.var_name != ""
    method = getattr(equation, method_name)
    method_kind = MethodKind(method_name)
    call_args = [equation.var_name]
    declarations = [f"GLOBAL_MEM {equation.__class__.__name__}* {equation.var_name}"]
    args = list(inspect.getfullargspec(method).args)
    if "self" in args:
        args.remove("self")
    for arg in args:
        if arg == "d_idx":
            call_args.append("dst")
        elif arg == "s_idx":
            call_args.append("src")
        elif arg in precomputed_symbols:
            call_args.append(arg)
        else:
            call_args.append(arg)
            declarations.append(_typed_argument_declaration(arg, known_types))
    return CudaEquationMethodCall(
        equation_name=equation.__class__.__name__,
        method_kind=method_kind,
        function_name=f"{equation.__class__.__name__}_{method_name}",
        argument_declarations=tuple(declarations),
        arguments=tuple(call_args),
    )


def cubic_spline_wij_precompute(dim: np.int32) -> CudaPairPrecompute:
    """Return FP32 CubicSpline `WIJ` pair precompute code."""
    assert isinstance(dim, np.int32)
    assert dim in (np.int32(1), np.int32(2), np.int32(3))
    return CudaPairPrecompute(
        symbols=frozenset(("HIJ", "XIJ", "R2IJ", "RIJ", "WIJ")),
        helper_source=_CUBIC_SPLINE_WIJ_HELPER,
        lines=(
            "float HIJ;",
            "float XIJ[3];",
            "float R2IJ;",
            "float RIJ;",
            "float WIJ;",
            "HIJ = 0.5f * (h[dst] + h[src]);",
            "XIJ[0] = fused_codegen_minimum_image(x[dst] - x[src], xmax - xmin, periodic_x);",
            "XIJ[1] = fused_codegen_minimum_image(y[dst] - y[src], ymax - ymin, periodic_y);",
            "XIJ[2] = fused_codegen_minimum_image(z[dst] - z[src], zmax - zmin, periodic_z);",
            "R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
            "RIJ = sqrtf(R2IJ);",
            f"WIJ = fused_codegen_cubic_spline_wij(RIJ, HIJ, {int(dim)});",
        ),
    )


def cubic_spline_gradient_precompute(dim: np.int32) -> CudaPairPrecompute:
    """Return FP32 CubicSpline gradient pair precompute code."""
    assert isinstance(dim, np.int32)
    assert dim in (np.int32(1), np.int32(2), np.int32(3))
    return CudaPairPrecompute(
        symbols=frozenset(("HIJ", "XIJ", "R2IJ", "RIJ", "DWIJ", "DWI", "DWJ")),
        helper_source=_CUBIC_SPLINE_GRADIENT_HELPER,
        lines=(
            "float HIJ;",
            "float XIJ[3];",
            "float R2IJ;",
            "float RIJ;",
            "float DWIJ[3];",
            "float DWI[3];",
            "float DWJ[3];",
            "HIJ = 0.5f * (h[dst] + h[src]);",
            "XIJ[0] = fused_codegen_minimum_image(x[dst] - x[src], xmax - xmin, periodic_x);",
            "XIJ[1] = fused_codegen_minimum_image(y[dst] - y[src], ymax - ymin, periodic_y);",
            "XIJ[2] = fused_codegen_minimum_image(z[dst] - z[src], zmax - zmin, periodic_z);",
            "R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
            "RIJ = sqrtf(R2IJ);",
            f"fused_codegen_cubic_spline_gradient(DWIJ, XIJ, RIJ, HIJ, {int(dim)});",
            f"fused_codegen_cubic_spline_gradient(DWI, XIJ, RIJ, h[dst], {int(dim)});",
            f"fused_codegen_cubic_spline_gradient(DWJ, XIJ, RIJ, h[src], {int(dim)});",
        ),
    )


_CUBIC_SPLINE_PAIR_SYMBOL_ORDER = (
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
)


def cubic_spline_pair_precompute_for_symbols(
    dim: np.int32, symbols: frozenset[str]
) -> CudaPairPrecompute:
    """Return FP32 CubicSpline pair precompute code for the requested symbols."""
    assert isinstance(dim, np.int32)
    assert dim in (np.int32(1), np.int32(2), np.int32(3))
    expanded_symbols = _expanded_cubic_spline_pair_symbols(symbols)
    return CudaPairPrecompute(
        symbols=expanded_symbols,
        helper_source=_cubic_spline_pair_helper_source(expanded_symbols),
        lines=_cubic_spline_pair_lines(dim, expanded_symbols),
    )


def cubic_spline_pair_precompute(dim: np.int32) -> CudaPairPrecompute:
    """Return FP32 CubicSpline pair precompute code for common SPH equations."""
    assert isinstance(dim, np.int32)
    assert dim in (np.int32(1), np.int32(2), np.int32(3))
    return cubic_spline_pair_precompute_for_symbols(
        dim, frozenset(_CUBIC_SPLINE_PAIR_SYMBOL_ORDER)
    )


def quintic_spline_wij_precompute(dim: np.int32) -> CudaPairPrecompute:
    """Return FP32 QuinticSpline `WIJ` pair precompute code."""
    assert isinstance(dim, np.int32)
    assert dim in (np.int32(1), np.int32(2), np.int32(3))
    return CudaPairPrecompute(
        symbols=frozenset(("HIJ", "XIJ", "R2IJ", "RIJ", "WIJ")),
        helper_source=_QUINTIC_SPLINE_WIJ_HELPER,
        lines=(
            "float HIJ;",
            "float XIJ[3];",
            "float R2IJ;",
            "float RIJ;",
            "float WIJ;",
            "HIJ = 0.5f * (h[dst] + h[src]);",
            "XIJ[0] = fused_codegen_minimum_image(x[dst] - x[src], xmax - xmin, periodic_x);",
            "XIJ[1] = fused_codegen_minimum_image(y[dst] - y[src], ymax - ymin, periodic_y);",
            "XIJ[2] = fused_codegen_minimum_image(z[dst] - z[src], zmax - zmin, periodic_z);",
            "R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
            "RIJ = sqrtf(R2IJ);",
            f"WIJ = fused_codegen_quintic_spline_wij(RIJ, HIJ, {int(dim)});",
        ),
    )


def quintic_spline_gradient_precompute(dim: np.int32) -> CudaPairPrecompute:
    """Return FP32 QuinticSpline gradient pair precompute code."""
    assert isinstance(dim, np.int32)
    assert dim in (np.int32(1), np.int32(2), np.int32(3))
    return CudaPairPrecompute(
        symbols=frozenset(("HIJ", "XIJ", "R2IJ", "RIJ", "DWIJ", "DWI", "DWJ")),
        helper_source=_QUINTIC_SPLINE_GRADIENT_HELPER,
        lines=(
            "float HIJ;",
            "float XIJ[3];",
            "float R2IJ;",
            "float RIJ;",
            "float DWIJ[3];",
            "float DWI[3];",
            "float DWJ[3];",
            "HIJ = 0.5f * (h[dst] + h[src]);",
            "XIJ[0] = fused_codegen_minimum_image(x[dst] - x[src], xmax - xmin, periodic_x);",
            "XIJ[1] = fused_codegen_minimum_image(y[dst] - y[src], ymax - ymin, periodic_y);",
            "XIJ[2] = fused_codegen_minimum_image(z[dst] - z[src], zmax - zmin, periodic_z);",
            "R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
            "RIJ = sqrtf(R2IJ);",
            f"fused_codegen_quintic_spline_gradient(DWIJ, XIJ, RIJ, HIJ, {int(dim)});",
            f"fused_codegen_quintic_spline_gradient(DWI, XIJ, RIJ, h[dst], {int(dim)});",
            f"fused_codegen_quintic_spline_gradient(DWJ, XIJ, RIJ, h[src], {int(dim)});",
        ),
    )


def quintic_spline_pair_precompute_for_symbols(
    dim: np.int32, symbols: frozenset[str]
) -> CudaPairPrecompute:
    """Return FP32 QuinticSpline pair precompute code for the requested symbols."""
    assert isinstance(dim, np.int32)
    assert dim in (np.int32(1), np.int32(2), np.int32(3))
    expanded_symbols = _expanded_cubic_spline_pair_symbols(symbols)
    return CudaPairPrecompute(
        symbols=expanded_symbols,
        helper_source=_quintic_spline_pair_helper_source(expanded_symbols),
        lines=_quintic_spline_pair_lines(dim, expanded_symbols),
    )


def quintic_spline_pair_precompute(dim: np.int32) -> CudaPairPrecompute:
    """Return FP32 QuinticSpline pair precompute code for common SPH equations."""
    assert isinstance(dim, np.int32)
    assert dim in (np.int32(1), np.int32(2), np.int32(3))
    return quintic_spline_pair_precompute_for_symbols(
        dim, frozenset(_CUBIC_SPLINE_PAIR_SYMBOL_ORDER)
    )


def _expanded_cubic_spline_pair_symbols(symbols: frozenset[str]) -> frozenset[str]:
    assert symbols.issubset(frozenset(_CUBIC_SPLINE_PAIR_SYMBOL_ORDER))
    expanded = set(symbols)
    if expanded.intersection(frozenset(("WIJ", "DWIJ", "GHIJ", "EPS"))):
        expanded.add("HIJ")
    if expanded.intersection(
        frozenset(
            (
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
            )
        )
    ):
        expanded.update(("XIJ", "R2IJ", "RIJ"))
    if "R2IJ" in expanded:
        expanded.add("XIJ")
    if "RHOIJ1" in expanded:
        expanded.add("RHOIJ")
    return frozenset(
        symbol for symbol in _CUBIC_SPLINE_PAIR_SYMBOL_ORDER if symbol in expanded
    )


def _cubic_spline_pair_helper_source(symbols: frozenset[str]) -> str:
    if symbols.intersection(
        frozenset(("WIJ", "WI", "WJ", "DWIJ", "DWI", "DWJ", "GHI", "GHJ", "GHIJ"))
    ):
        return _CUBIC_SPLINE_GRADIENT_HELPER
    return ""


def _quintic_spline_pair_helper_source(symbols: frozenset[str]) -> str:
    if symbols.intersection(
        frozenset(("WIJ", "WI", "WJ", "DWIJ", "DWI", "DWJ", "GHI", "GHJ", "GHIJ"))
    ):
        return _QUINTIC_SPLINE_GRADIENT_HELPER
    return ""


def _cubic_spline_pair_lines(dim: np.int32, symbols: frozenset[str]) -> tuple[str, ...]:
    lines = []
    declaration_lines = {
        "HIJ": "float HIJ;",
        "XIJ": "float XIJ[3];",
        "R2IJ": "float R2IJ;",
        "RIJ": "float RIJ;",
        "WIJ": "float WIJ;",
        "WI": "float WI;",
        "WJ": "float WJ;",
        "DWIJ": "float DWIJ[3];",
        "DWI": "float DWI[3];",
        "DWJ": "float DWJ[3];",
        "GHI": "float GHI;",
        "GHJ": "float GHJ;",
        "GHIJ": "float GHIJ;",
        "VIJ": "float VIJ[3];",
        "EPS": "float EPS;",
        "RHOIJ": "float RHOIJ;",
        "RHOIJ1": "float RHOIJ1;",
    }
    for symbol in _CUBIC_SPLINE_PAIR_SYMBOL_ORDER:
        if symbol in symbols:
            lines.append(declaration_lines[symbol])
    if "HIJ" in symbols:
        lines.append("HIJ = 0.5f * (h[dst] + h[src]);")
    if "XIJ" in symbols:
        lines.extend(
            (
                "XIJ[0] = fused_codegen_minimum_image(x[dst] - x[src], xmax - xmin, periodic_x);",
                "XIJ[1] = fused_codegen_minimum_image(y[dst] - y[src], ymax - ymin, periodic_y);",
                "XIJ[2] = fused_codegen_minimum_image(z[dst] - z[src], zmax - zmin, periodic_z);",
            )
        )
    if "R2IJ" in symbols:
        lines.append("R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];")
    if "RIJ" in symbols:
        lines.append("RIJ = sqrtf(R2IJ);")
    if "WIJ" in symbols:
        lines.append(f"WIJ = fused_codegen_cubic_spline_wij(RIJ, HIJ, {int(dim)});")
    if "WI" in symbols:
        lines.append(f"WI = fused_codegen_cubic_spline_wij(RIJ, h[dst], {int(dim)});")
    if "WJ" in symbols:
        lines.append(f"WJ = fused_codegen_cubic_spline_wij(RIJ, h[src], {int(dim)});")
    if "DWIJ" in symbols:
        lines.append(
            f"fused_codegen_cubic_spline_gradient(DWIJ, XIJ, RIJ, HIJ, {int(dim)});"
        )
    if "DWI" in symbols:
        lines.append(
            f"fused_codegen_cubic_spline_gradient(DWI, XIJ, RIJ, h[dst], {int(dim)});"
        )
    if "DWJ" in symbols:
        lines.append(
            f"fused_codegen_cubic_spline_gradient(DWJ, XIJ, RIJ, h[src], {int(dim)});"
        )
    if "GHI" in symbols:
        lines.append(
            f"GHI = fused_codegen_cubic_spline_gradient_h(RIJ, h[dst], {int(dim)});"
        )
    if "GHJ" in symbols:
        lines.append(
            f"GHJ = fused_codegen_cubic_spline_gradient_h(RIJ, h[src], {int(dim)});"
        )
    if "GHIJ" in symbols:
        lines.append(
            f"GHIJ = fused_codegen_cubic_spline_gradient_h(RIJ, HIJ, {int(dim)});"
        )
    if "VIJ" in symbols:
        lines.extend(
            (
                "VIJ[0] = d_u[dst] - s_u[src];",
                "VIJ[1] = d_v[dst] - s_v[src];",
                "VIJ[2] = d_w[dst] - s_w[src];",
            )
        )
    if "EPS" in symbols:
        lines.append("EPS = 0.01f * HIJ * HIJ;")
    if "RHOIJ" in symbols:
        lines.append("RHOIJ = 0.5f * (d_rho[dst] + s_rho[src]);")
    if "RHOIJ1" in symbols:
        lines.append("RHOIJ1 = 1.0f / RHOIJ;")
    return tuple(lines)


def _quintic_spline_pair_lines(
    dim: np.int32, symbols: frozenset[str]
) -> tuple[str, ...]:
    lines = []
    declaration_lines = {
        "HIJ": "float HIJ;",
        "XIJ": "float XIJ[3];",
        "R2IJ": "float R2IJ;",
        "RIJ": "float RIJ;",
        "WIJ": "float WIJ;",
        "WI": "float WI;",
        "WJ": "float WJ;",
        "DWIJ": "float DWIJ[3];",
        "DWI": "float DWI[3];",
        "DWJ": "float DWJ[3];",
        "GHI": "float GHI;",
        "GHJ": "float GHJ;",
        "GHIJ": "float GHIJ;",
        "VIJ": "float VIJ[3];",
        "EPS": "float EPS;",
        "RHOIJ": "float RHOIJ;",
        "RHOIJ1": "float RHOIJ1;",
    }
    for symbol in _CUBIC_SPLINE_PAIR_SYMBOL_ORDER:
        if symbol in symbols:
            lines.append(declaration_lines[symbol])
    if "HIJ" in symbols:
        lines.append("HIJ = 0.5f * (h[dst] + h[src]);")
    if "XIJ" in symbols:
        lines.extend(
            (
                "XIJ[0] = fused_codegen_minimum_image(x[dst] - x[src], xmax - xmin, periodic_x);",
                "XIJ[1] = fused_codegen_minimum_image(y[dst] - y[src], ymax - ymin, periodic_y);",
                "XIJ[2] = fused_codegen_minimum_image(z[dst] - z[src], zmax - zmin, periodic_z);",
            )
        )
    if "R2IJ" in symbols:
        lines.append("R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];")
    if "RIJ" in symbols:
        lines.append("RIJ = sqrtf(R2IJ);")
    if "WIJ" in symbols:
        lines.append(f"WIJ = fused_codegen_quintic_spline_wij(RIJ, HIJ, {int(dim)});")
    if "WI" in symbols:
        lines.append(f"WI = fused_codegen_quintic_spline_wij(RIJ, h[dst], {int(dim)});")
    if "WJ" in symbols:
        lines.append(f"WJ = fused_codegen_quintic_spline_wij(RIJ, h[src], {int(dim)});")
    if "DWIJ" in symbols:
        lines.append(
            f"fused_codegen_quintic_spline_gradient(DWIJ, XIJ, RIJ, HIJ, {int(dim)});"
        )
    if "DWI" in symbols:
        lines.append(
            f"fused_codegen_quintic_spline_gradient(DWI, XIJ, RIJ, h[dst], {int(dim)});"
        )
    if "DWJ" in symbols:
        lines.append(
            f"fused_codegen_quintic_spline_gradient(DWJ, XIJ, RIJ, h[src], {int(dim)});"
        )
    if "GHI" in symbols:
        lines.append(
            f"GHI = fused_codegen_quintic_spline_gradient_h(RIJ, h[dst], {int(dim)});"
        )
    if "GHJ" in symbols:
        lines.append(
            f"GHJ = fused_codegen_quintic_spline_gradient_h(RIJ, h[src], {int(dim)});"
        )
    if "GHIJ" in symbols:
        lines.append(
            f"GHIJ = fused_codegen_quintic_spline_gradient_h(RIJ, HIJ, {int(dim)});"
        )
    if "VIJ" in symbols:
        lines.extend(
            (
                "VIJ[0] = d_u[dst] - s_u[src];",
                "VIJ[1] = d_v[dst] - s_v[src];",
                "VIJ[2] = d_w[dst] - s_w[src];",
            )
        )
    if "EPS" in symbols:
        lines.append("EPS = 0.01f * HIJ * HIJ;")
    if "RHOIJ" in symbols:
        lines.append("RHOIJ = 0.5f * (d_rho[dst] + s_rho[src]);")
    if "RHOIJ1" in symbols:
        lines.append("RHOIJ1 = 1.0f / RHOIJ;")
    return tuple(lines)


def launch_direct_pair_kernel_with_context(
    module: object,
    kernel_name: str,
    context: object,
    extra_args: tuple[object, ...],
) -> PairLaunchConfig:
    """Launch a generated direct pair kernel with a fused neighbor context."""
    kernel = module.get_function(kernel_name)
    block_size = _pair_block_size_for_count(context.n)
    grid_x = (context.n + block_size - 1) // block_size
    kernel(
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
        np.int32(context.search_radius_cells),
        np.int32(context.cell_counts[0]),
        np.int32(context.cell_counts[1]),
        np.int32(context.cell_counts[2]),
        _kernel_arg(context.cell_particle_counts),
        _kernel_arg(context.cell_starts),
        _kernel_arg(context.sorted_ids),
        *tuple(_kernel_arg(arg) for arg in extra_args),
        block=(block_size, 1, 1),
        grid=(grid_x, 1, 1),
        stream=context.stream,
    )
    return PairLaunchConfig(
        traversal="direct",
        n=int(context.n),
        block_size=int(block_size),
        grid_x=int(grid_x),
    )


def launch_cluster_pair_kernel_with_context(
    module: object,
    kernel_name: str,
    context: object,
    extra_args: tuple[object, ...],
) -> PairLaunchConfig:
    """Launch a generated sorted-cell cluster pair kernel."""
    kernel = module.get_function(kernel_name)
    cluster_size = 64
    kernel(
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
        np.int32(context.search_radius_cells),
        np.int32(context.cell_counts[0]),
        np.int32(context.cell_counts[1]),
        np.int32(context.cell_counts[2]),
        _kernel_arg(context.cell_particle_counts),
        _kernel_arg(context.cell_starts),
        _kernel_arg(context.sorted_ids),
        np.int32(context.cluster_total),
        _kernel_arg(context.cluster_cell),
        _kernel_arg(context.cluster_begin),
        _kernel_arg(context.cluster_count),
        *tuple(_kernel_arg(arg) for arg in extra_args),
        block=(cluster_size, 1, 1),
        grid=(context.cluster_total, 1, 1),
        stream=context.stream,
    )
    return PairLaunchConfig(
        traversal="cluster",
        n=int(context.n),
        block_size=int(cluster_size),
        grid_x=int(context.cluster_total),
    )


def launch_hbucket_pair_kernel_with_context(
    module: object,
    kernel_name: str,
    context: object,
    extra_args: tuple[object, ...],
) -> PairLaunchConfig:
    """Launch a generated h-bucket pair kernel."""
    kernel = module.get_function(kernel_name)
    block_size = _pair_block_size_for_count(context.n)
    grid_x = (context.n + block_size - 1) // block_size
    kernel(
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
        block=(block_size, 1, 1),
        grid=(grid_x, 1, 1),
        stream=context.stream,
    )
    return PairLaunchConfig(
        traversal="hbucket",
        n=int(context.n),
        block_size=int(block_size),
        grid_x=int(grid_x),
    )


def launch_cell_tile_hbucket_pair_kernel_with_context(
    module: object,
    kernel_name: str,
    context: object,
    extra_args: tuple[object, ...],
) -> PairLaunchConfig:
    """Launch a generated cell-tiled h-bucket pair kernel."""
    assert context.bucket_count == 1
    kernel = module.get_function(kernel_name)
    block_size = _cell_tile_hbucket_block_size()
    grid_x = int(context.total_cells)
    kernel(
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
        block=(block_size, 1, 1),
        grid=(grid_x, 1, 1),
        stream=context.stream,
    )
    return PairLaunchConfig(
        traversal="cell_tile_hbucket",
        n=int(context.n),
        block_size=int(block_size),
        grid_x=int(grid_x),
    )


def _pair_block_size_for_count(n: int) -> int:
    full_block_size = 256
    full_particle_blocks = (int(n) + full_block_size - 1) // full_block_size
    if full_particle_blocks < _cuda_multiprocessor_count():
        return 128
    return full_block_size


def _cell_tile_hbucket_block_size() -> int:
    return 128


def _cuda_multiprocessor_count() -> int:
    import pycuda.driver as cuda

    device = cuda.Context.get_device()
    return int(device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT))


def build_cuda_equation_struct_argument(equation, stream: object) -> object:
    """Return a PyCUDA struct argument for a generated equation wrapper."""
    from compyle.cuda import match_dtype_to_c_struct
    from compyle.translator import CStructHelper
    import pycuda.gpuarray as gpuarray

    helper = CStructHelper(equation)
    host_array = helper.get_array()
    assert host_array is not None
    cuda_dtype, _code = match_dtype_to_c_struct(None, "equation", host_array.dtype)
    return gpuarray.to_gpu_async(host_array.astype(cuda_dtype), stream=stream)


def launch_pointwise_kernel(
    module: object,
    kernel_name: str,
    n: int,
    stream: object,
    extra_args: tuple[object, ...],
) -> None:
    """Launch a generated pointwise kernel."""
    assert n > 0
    kernel = module.get_function(kernel_name)
    kernel(
        np.int32(n),
        *tuple(_kernel_arg(arg) for arg in extra_args),
        block=(256, 1, 1),
        grid=((n + 255) // 256, 1, 1),
        stream=stream,
    )


def lower_equation_method_to_cuda(equation, method_name):
    """Lower a simple PySPH equation method body to CUDA statements."""
    assert hasattr(equation, method_name)
    method = getattr(equation, method_name)
    method_kind = MethodKind(method_name)
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    function = _first_function(tree)
    args = inspect.getfullargspec(method).args
    written_arrays = _written_array_names(function)
    declarations = _cuda_method_argument_declarations(args, written_arrays)
    lines = []
    for statement in function.body:
        lines.extend(_lower_statement(statement))
    return CudaInlineMethodBody(
        equation_name=equation.__class__.__name__,
        method_kind=method_kind,
        argument_declarations=declarations,
        lines=tuple(lines),
    )


def _generate_direct_pair_loop_outline(
    plan_id: str, stage: StageNode, bodies: tuple[CudaInlineMethodBody, ...]
) -> FusedKernelOutline:
    """Return a destination-owned direct pair-loop CUDA outline."""
    assert _stage_uses_neighbors(stage)
    name = fused_kernel_name(plan_id, stage)
    method_lines = _direct_pair_method_lines(stage, bodies)
    arguments = fused_context_argument_declarations() + _inline_argument_declarations(
        bodies
    )
    source = "\n".join(
        (
            'extern "C" {',
            _DIRECT_PAIR_HELPERS,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            "    int dst = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (dst >= n) {",
            "        return;",
            "    }",
            "    int base_cx = fused_codegen_clamp_cell(x[dst], xmin, xmax, nx);",
            "    int base_cy = fused_codegen_clamp_cell(y[dst], ymin, ymax, ny);",
            "    int base_cz = fused_codegen_clamp_cell(z[dst], zmin, zmax, nz);",
            "    for (int oz = -search_radius_cells; oz <= search_radius_cells; ++oz) {",
            "        if (!fused_codegen_valid_offset(oz, nz)) {",
            "            continue;",
            "        }",
            "        int cz = 0;",
            "        if (!fused_codegen_neighbor_cell(base_cz, oz, nz, periodic_z, &cz)) {",
            "            continue;",
            "        }",
            "        for (int oy = -search_radius_cells; oy <= search_radius_cells; ++oy) {",
            "            if (!fused_codegen_valid_offset(oy, ny)) {",
            "                continue;",
            "            }",
            "            int cy = 0;",
            "            if (!fused_codegen_neighbor_cell(base_cy, oy, ny, periodic_y, &cy)) {",
            "                continue;",
            "            }",
            "            for (int ox = -search_radius_cells; ox <= search_radius_cells; ++ox) {",
            "                if (!fused_codegen_valid_offset(ox, nx)) {",
            "                    continue;",
            "                }",
            "                int cx = 0;",
            "                if (!fused_codegen_neighbor_cell(base_cx, ox, nx, periodic_x, &cx)) {",
            "                    continue;",
            "                }",
            "                int cell = fused_codegen_linear_cell(cx, cy, cz, nx, ny);",
            "                int begin = cell_starts[cell];",
            "                int end = begin + cell_counts[cell];",
            "                for (int pos = begin; pos < end; ++pos) {",
            "                    int src = sorted_ids[pos];",
            "                    if (fused_codegen_in_support_xyz(",
            "                        dst, src, x, y, z, h, xmin, xmax, ymin, ymax,",
            "                        zmin, zmax, periodic_x, periodic_y, periodic_z,",
            "                        radius_scale",
            "                    )) {",
            *method_lines,
            "                    }",
            "                }",
            "            }",
            "        }",
            "    }",
            "}",
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def _direct_pair_method_lines(
    stage: StageNode, bodies: tuple[CudaInlineMethodBody, ...]
) -> tuple[str, ...]:
    if len(bodies) == 0:
        return tuple(
            f"                                        // {method.equation_name}.{method.method_kind.value}"
            for method in stage.methods
        )
    lines = []
    for method in stage.methods:
        body = _body_for_method(method.equation_name, method.method_kind, bodies)
        for line in body.lines:
            lines.append(f"                                        {line}")
    return tuple(lines)


def _body_for_method(
    equation_name: str,
    method_kind: MethodKind,
    bodies: tuple[CudaInlineMethodBody, ...],
) -> CudaInlineMethodBody:
    matches = [
        body
        for body in bodies
        if body.equation_name == equation_name and body.method_kind is method_kind
    ]
    assert len(matches) == 1
    return matches[0]


def _inline_argument_declarations(
    bodies: tuple[CudaInlineMethodBody, ...],
) -> tuple[str, ...]:
    declarations = []
    for body in bodies:
        for declaration in body.argument_declarations:
            if declaration not in declarations:
                declarations.append(declaration)
    return tuple(declarations)


def _direct_pair_segment_lines(
    stage: StageNode,
    calls: tuple[CudaEquationMethodCall, ...],
    precompute_lines: tuple[str, ...],
) -> tuple[str, ...]:
    lines = []
    for methods in _stage_method_segments(stage):
        pre_methods, loop_methods, post_methods = _pair_segment_methods(methods)
        pre_loop_method_lines = _direct_pair_equation_call_lines(pre_methods, calls)
        loop_method_lines = _direct_pair_equation_call_lines(loop_methods, calls)
        post_loop_method_lines = _direct_pair_equation_call_lines(post_methods, calls)
        lines.extend(
            line.replace("                                        ", "    ")
            for line in pre_loop_method_lines
        )
        lines.extend(
            _direct_pair_neighbor_traversal_lines(precompute_lines, loop_method_lines)
        )
        lines.extend(
            line.replace("                                        ", "    ")
            for line in post_loop_method_lines
        )
    return tuple(lines)


def _cluster_pair_segment_lines(
    stage: StageNode,
    calls: tuple[CudaEquationMethodCall, ...],
    precompute_lines: tuple[str, ...],
) -> tuple[str, ...]:
    lines = []
    for methods in _stage_method_segments(stage):
        pre_methods, loop_methods, post_methods = _pair_segment_methods(methods)
        pre_loop_method_lines = _direct_pair_equation_call_lines(pre_methods, calls)
        loop_method_lines = _direct_pair_equation_call_lines(loop_methods, calls)
        post_loop_method_lines = _direct_pair_equation_call_lines(post_methods, calls)
        lines.extend(
            line.replace("                                        ", "    ")
            for line in pre_loop_method_lines
        )
        lines.extend(
            _cluster_pair_neighbor_traversal_lines(precompute_lines, loop_method_lines)
        )
        lines.extend(
            line.replace("                                        ", "    ")
            for line in post_loop_method_lines
        )
    return tuple(lines)


def _hbucket_pair_segment_lines(
    stage: StageNode,
    calls: tuple[CudaEquationMethodCall, ...],
    precompute: CudaPairPrecompute,
) -> tuple[str, ...]:
    lines = []
    for methods in _stage_method_segments(stage):
        pre_methods, loop_methods, post_methods = _pair_segment_methods(methods)
        reduction_methods = _local_reduction_methods_for_segment(loop_methods)
        reduced_fields = _local_reduction_fields(reduction_methods)
        pre_loop_method_lines = _direct_pair_equation_call_lines(pre_methods, calls)
        if reduction_methods:
            loop_method_lines = _local_reduction_pair_loop_call_lines(
                loop_methods, calls, reduction_methods
            )
        else:
            loop_method_lines = _direct_pair_equation_call_lines(loop_methods, calls)
        post_loop_method_lines = _direct_pair_equation_call_lines(post_methods, calls)
        lines.extend(
            line.replace("                                        ", "    ")
            for line in pre_loop_method_lines
        )
        lines.extend(_local_reduction_initialization_lines(reduced_fields))
        lines.extend(
            _hbucket_pair_neighbor_traversal_lines(
                _hbucket_pair_precompute_lines(precompute),
                precompute,
                (),
                loop_method_lines,
            )
        )
        lines.extend(_local_reduction_commit_lines(reduced_fields))
        lines.extend(
            line.replace("                                        ", "    ")
            for line in post_loop_method_lines
        )
    return tuple(lines)


def _cell_tile_hbucket_pair_segment_lines(
    stage: StageNode,
    calls: tuple[CudaEquationMethodCall, ...],
    precompute: CudaPairPrecompute,
    source_fields: tuple[str, ...],
) -> tuple[str, ...]:
    lines = [
        *_cell_tile_shared_source_declarations(source_fields),
        "    if (bucket_count != 1 || blockIdx.x >= total_cells) {",
        "        return;",
        "    }",
        "    int dest_cell = blockIdx.x;",
        "    int dst_begin = cell_bucket_starts[dest_cell];",
        "    int dst_count = cell_bucket_counts[dest_cell];",
        "    if (dst_count <= 0) {",
        "        return;",
        "    }",
        "    int base_cx = 0;",
        "    int base_cy = 0;",
        "    int base_cz = 0;",
        "    fused_codegen_decode_cell(dest_cell, nx, ny, &base_cx, &base_cy, &base_cz);",
        "    float dest_cell_h = __uint_as_float(cell_bucket_h_max_bits[dest_cell]);",
        "    float bucket_h = __uint_as_float(bucket_h_max_bits[0]);",
        "    if (dest_cell_h <= 0.0f || bucket_h <= 0.0f) {",
        "        return;",
        "    }",
        "    float bucket_support = radius_scale * fmaxf(dest_cell_h, bucket_h);",
        "    int max_x = (int)ceilf(bucket_support / cell_width_x);",
        "    int max_y = (int)ceilf(bucket_support / cell_width_y);",
        "    int max_z = (int)ceilf(bucket_support / cell_width_z);",
        "    int full_x = periodic_x && nx <= 2 * max_x + 1;",
        "    int full_y = periodic_y && ny <= 2 * max_y + 1;",
        "    int full_z = periodic_z && nz <= 2 * max_z + 1;",
        "    int loops_x = full_x ? nx : 2 * max_x + 1;",
        "    int loops_y = full_y ? ny : 2 * max_y + 1;",
        "    int loops_z = full_z ? nz : 2 * max_z + 1;",
        "    for (int dest_local_base = 0; dest_local_base < dst_count; dest_local_base += blockDim.x) {",
        "        int dest_local = dest_local_base + threadIdx.x;",
        "        int fused_active_dst = dest_local < dst_count;",
        "        int dst = 0;",
        "        float dst_x = 0.0f;",
        "        float dst_y = 0.0f;",
        "        float dst_z = 0.0f;",
        "        float dst_h = 0.0f;",
        "        if (fused_active_dst) {",
        "            dst = sorted_ids[dst_begin + dest_local];",
        "            dst_x = x[dst];",
        "            dst_y = y[dst];",
        "            dst_z = z[dst];",
        "            dst_h = h[dst];",
        "        }",
    ]
    for methods in _stage_method_segments(stage):
        pre_methods, loop_methods, post_methods = _pair_segment_methods(methods)
        reduction_methods = _local_reduction_methods_for_segment(loop_methods)
        reduced_fields = _local_reduction_fields(reduction_methods)
        pre_loop_method_lines = _direct_pair_equation_call_lines(pre_methods, calls)
        loop_method_lines = _cell_tile_pair_loop_call_lines(
            loop_methods, calls, reduction_methods, source_fields
        )
        post_loop_method_lines = _direct_pair_equation_call_lines(post_methods, calls)
        lines.append("        if (fused_active_dst) {")
        lines.extend(
            line.replace("                                        ", "            ")
            for line in pre_loop_method_lines
        )
        lines.append("        }")
        lines.extend(
            line.replace("    ", "        ", 1)
            for line in _local_reduction_initialization_lines(reduced_fields)
        )
        lines.extend(
            _cell_tile_hbucket_neighbor_traversal_lines(
                _cell_tile_pair_precompute_lines(precompute, source_fields),
                precompute,
                source_fields,
                loop_method_lines,
            )
        )
        lines.append("        if (fused_active_dst) {")
        lines.extend(
            line.replace("    ", "            ", 1)
            for line in _local_reduction_commit_lines(reduced_fields)
        )
        lines.extend(
            line.replace("                                        ", "            ")
            for line in post_loop_method_lines
        )
        lines.append("        }")
    lines.append("    }")
    return tuple(lines)


def _cell_tile_shared_source_declarations(
    source_fields: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        f"    __shared__ float fused_tile_s_{field}[128];" for field in source_fields
    )


def _cell_tile_hbucket_neighbor_traversal_lines(
    precompute_lines: tuple[str, ...],
    precompute: CudaPairPrecompute,
    source_fields: tuple[str, ...],
    loop_method_lines: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        "        for (int iz = 0; iz < loops_z; ++iz) {",
        "            int cz = iz;",
        "            if (!full_z && !fused_codegen_neighbor_cell(base_cz, iz - max_z, nz, periodic_z, &cz)) {",
        "                continue;",
        "            }",
        "            for (int iy = 0; iy < loops_y; ++iy) {",
        "                int cy = iy;",
        "                if (!full_y && !fused_codegen_neighbor_cell(base_cy, iy - max_y, ny, periodic_y, &cy)) {",
        "                    continue;",
        "                }",
        "                for (int ix = 0; ix < loops_x; ++ix) {",
        "                    int cx = ix;",
        "                    if (!full_x && !fused_codegen_neighbor_cell(base_cx, ix - max_x, nx, periodic_x, &cx)) {",
        "                        continue;",
        "                    }",
        "                    int cell = fused_codegen_linear_cell(cx, cy, cz, nx, ny);",
        "                    int begin = cell_bucket_starts[cell];",
        "                    int count = cell_bucket_counts[cell];",
        "                    for (int tile = 0; tile < count; tile += blockDim.x) {",
        "                        int local = tile + threadIdx.x;",
        "                        if (local < count) {",
        "                            int src_id = sorted_ids[begin + local];",
        *_cell_tile_source_load_lines(source_fields),
        "                        }",
        "                        __syncthreads();",
        "                        int tile_count = count - tile;",
        "                        if (tile_count > blockDim.x) {",
        "                            tile_count = blockDim.x;",
        "                        }",
        "                        if (fused_active_dst) {",
        "                            for (int tile_j = 0; tile_j < tile_count; ++tile_j) {",
        *_cell_tile_pair_support_lines(precompute, "                                "),
        *tuple(
            line.replace(
                "                                        ",
                "                                    ",
            )
            for line in precompute_lines
        ),
        *tuple(
            line.replace(
                "                                        ",
                "                                    ",
            )
            for line in loop_method_lines
        ),
        "                                }",
        "                            }",
        "                        }",
        "                        __syncthreads();",
        "                    }",
        "                }",
        "            }",
        "        }",
    )


def _cell_tile_source_load_lines(source_fields: tuple[str, ...]) -> tuple[str, ...]:
    lines = []
    for field in source_fields:
        source = _cell_tile_source_expression(field)
        lines.append(
            f"                            fused_tile_s_{field}[threadIdx.x] = {source};"
        )
    return tuple(lines)


def _cell_tile_source_expression(field: str) -> str:
    if field == "x":
        return "x[src_id]"
    if field == "y":
        return "y[src_id]"
    if field == "z":
        return "z[src_id]"
    if field == "h":
        return "h[src_id]"
    return f"s_{field}[src_id]"


def _cell_tile_pair_support_lines(
    precompute: CudaPairPrecompute, indent: str
) -> tuple[str, ...]:
    if not _hbucket_pair_reuses_support_distance(precompute):
        return (
            f"{indent}float src_h = fused_tile_s_h[tile_j];",
            f"{indent}float XIJ[3];",
            f"{indent}XIJ[0] = fused_codegen_minimum_image(dst_x - fused_tile_s_x[tile_j], xmax - xmin, periodic_x);",
            f"{indent}XIJ[1] = fused_codegen_minimum_image(dst_y - fused_tile_s_y[tile_j], ymax - ymin, periodic_y);",
            f"{indent}XIJ[2] = fused_codegen_minimum_image(dst_z - fused_tile_s_z[tile_j], zmax - zmin, periodic_z);",
            f"{indent}float R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
            f"{indent}float fused_support = radius_scale * fmaxf(dst_h, src_h);",
            f"{indent}if (R2IJ < fused_support * fused_support) {{",
        )
    return (
        f"{indent}float src_h = fused_tile_s_h[tile_j];",
        f"{indent}float XIJ[3];",
        f"{indent}XIJ[0] = fused_codegen_minimum_image(dst_x - fused_tile_s_x[tile_j], xmax - xmin, periodic_x);",
        f"{indent}XIJ[1] = fused_codegen_minimum_image(dst_y - fused_tile_s_y[tile_j], ymax - ymin, periodic_y);",
        f"{indent}XIJ[2] = fused_codegen_minimum_image(dst_z - fused_tile_s_z[tile_j], zmax - zmin, periodic_z);",
        f"{indent}float R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
        f"{indent}float fused_support = radius_scale * fmaxf(dst_h, src_h);",
        f"{indent}if (R2IJ < fused_support * fused_support) {{",
    )


def _hbucket_source_inline_first_segment_lines(
    stage: StageNode,
    calls: tuple[CudaEquationMethodCall, ...],
    precompute: CudaPairPrecompute,
) -> tuple[str, ...]:
    if stage.kind is StageKind.POINTWISE:
        return tuple(
            line.replace("                                        ", "    ")
            for line in _direct_pair_equation_call_lines(stage, calls)
        )
    return _hbucket_pair_segment_lines(stage, calls, precompute)


def _hbucket_pair_segment_lines_with_source_inline(
    stage: StageNode,
    calls: tuple[CudaEquationMethodCall, ...],
    precompute: CudaPairPrecompute,
    source_inline_lines: tuple[str, ...],
    source_inline_fields: tuple[str, ...],
) -> tuple[str, ...]:
    lines = []
    for methods in _stage_method_segments(stage):
        pre_methods, loop_methods, post_methods = _pair_segment_methods(methods)
        pre_loop_method_lines = _direct_pair_equation_call_lines(pre_methods, calls)
        loop_method_lines = _direct_pair_equation_call_lines(loop_methods, calls)
        post_loop_method_lines = _direct_pair_equation_call_lines(post_methods, calls)
        lines.extend(
            line.replace("                                        ", "    ")
            for line in pre_loop_method_lines
        )
        lines.extend(
            _hbucket_pair_neighbor_traversal_lines(
                _replace_source_inline_reads(
                    _hbucket_pair_precompute_lines(precompute),
                    source_inline_fields,
                ),
                precompute,
                source_inline_lines,
                loop_method_lines,
            )
        )
        lines.extend(
            line.replace("                                        ", "    ")
            for line in post_loop_method_lines
        )
    return tuple(lines)


def _stage_method_segments(stage: StageNode) -> tuple[tuple[object, ...], ...]:
    if stage.method_segments:
        return stage.method_segments
    return (stage.methods,)


def _pair_segment_methods(methods: tuple[object, ...]):
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


def _direct_pair_neighbor_traversal_lines(
    precompute_lines: tuple[str, ...],
    loop_method_lines: tuple[str, ...],
) -> tuple[str, ...]:
    traversal_lines = (
        "    int base_cx = fused_codegen_clamp_cell(x[dst], xmin, xmax, nx);",
        "    int base_cy = fused_codegen_clamp_cell(y[dst], ymin, ymax, ny);",
        "    int base_cz = fused_codegen_clamp_cell(z[dst], zmin, zmax, nz);",
        "    for (int oz = -search_radius_cells; oz <= search_radius_cells; ++oz) {",
        "        if (!fused_codegen_valid_offset(oz, nz)) {",
        "            continue;",
        "        }",
        "        int cz = 0;",
        "        if (!fused_codegen_neighbor_cell(base_cz, oz, nz, periodic_z, &cz)) {",
        "            continue;",
        "        }",
        "        for (int oy = -search_radius_cells; oy <= search_radius_cells; ++oy) {",
        "            if (!fused_codegen_valid_offset(oy, ny)) {",
        "                continue;",
        "            }",
        "            int cy = 0;",
        "            if (!fused_codegen_neighbor_cell(base_cy, oy, ny, periodic_y, &cy)) {",
        "                continue;",
        "            }",
        "            for (int ox = -search_radius_cells; ox <= search_radius_cells; ++ox) {",
        "                if (!fused_codegen_valid_offset(ox, nx)) {",
        "                    continue;",
        "                }",
        "                int cx = 0;",
        "                if (!fused_codegen_neighbor_cell(base_cx, ox, nx, periodic_x, &cx)) {",
        "                    continue;",
        "                }",
        "                int cell = fused_codegen_linear_cell(cx, cy, cz, nx, ny);",
        "                int begin = cell_starts[cell];",
        "                int end = begin + cell_counts[cell];",
        "                for (int pos = begin; pos < end; ++pos) {",
        "                    int src = sorted_ids[pos];",
        "                    if (fused_codegen_in_support_xyz(",
        "                        dst, src, x, y, z, h, xmin, xmax, ymin, ymax,",
        "                        zmin, zmax, periodic_x, periodic_y, periodic_z,",
        "                        radius_scale",
        "                    )) {",
        *precompute_lines,
        *loop_method_lines,
        "                    }",
        "                }",
        "            }",
        "        }",
        "    }",
    )
    return ("    {", *tuple(f"    {line}" for line in traversal_lines), "    }")


def _cluster_pair_neighbor_traversal_lines(
    precompute_lines: tuple[str, ...],
    loop_method_lines: tuple[str, ...],
) -> tuple[str, ...]:
    traversal_lines = (
        "    for (int oz = -search_radius_cells; oz <= search_radius_cells; ++oz) {",
        "        if (!fused_codegen_valid_offset(oz, nz)) {",
        "            continue;",
        "        }",
        "        int cz = 0;",
        "        if (!fused_codegen_neighbor_cell(base_cz, oz, nz, periodic_z, &cz)) {",
        "            continue;",
        "        }",
        "        for (int oy = -search_radius_cells; oy <= search_radius_cells; ++oy) {",
        "            if (!fused_codegen_valid_offset(oy, ny)) {",
        "                continue;",
        "            }",
        "            int cy = 0;",
        "            if (!fused_codegen_neighbor_cell(base_cy, oy, ny, periodic_y, &cy)) {",
        "                continue;",
        "            }",
        "            for (int ox = -search_radius_cells; ox <= search_radius_cells; ++ox) {",
        "                if (!fused_codegen_valid_offset(ox, nx)) {",
        "                    continue;",
        "                }",
        "                int cx = 0;",
        "                if (!fused_codegen_neighbor_cell(base_cx, ox, nx, periodic_x, &cx)) {",
        "                    continue;",
        "                }",
        "                int cell = fused_codegen_linear_cell(cx, cy, cz, nx, ny);",
        "                int begin = cell_starts[cell];",
        "                int end = begin + cell_counts[cell];",
        "                for (int pos = begin; pos < end; ++pos) {",
        "                    int src = sorted_ids[pos];",
        "                    if (fused_codegen_in_support_xyz(",
        "                        dst, src, x, y, z, h, xmin, xmax, ymin, ymax,",
        "                        zmin, zmax, periodic_x, periodic_y, periodic_z,",
        "                        radius_scale",
        "                    )) {",
        *precompute_lines,
        *loop_method_lines,
        "                    }",
        "                }",
        "            }",
        "        }",
        "    }",
    )
    return ("    {", *tuple(f"    {line}" for line in traversal_lines), "    }")


def _hbucket_pair_neighbor_traversal_lines(
    precompute_lines: tuple[str, ...],
    precompute: CudaPairPrecompute,
    source_inline_lines: tuple[str, ...],
    loop_method_lines: tuple[str, ...],
) -> tuple[str, ...]:
    traversal_lines = (
        "    float dst_x = x[dst];",
        "    float dst_y = y[dst];",
        "    float dst_z = z[dst];",
        "    float dst_h = h[dst];",
        "    int base_cx = fused_codegen_clamp_cell(dst_x, xmin, xmax, nx);",
        "    int base_cy = fused_codegen_clamp_cell(dst_y, ymin, ymax, ny);",
        "    int base_cz = fused_codegen_clamp_cell(dst_z, zmin, zmax, nz);",
        "    for (int bucket = 0; bucket < bucket_count; ++bucket) {",
        "        float bucket_h = __uint_as_float(bucket_h_max_bits[bucket]);",
        "        if (bucket_h <= 0.0f) {",
        "            continue;",
        "        }",
        "        float bucket_support = radius_scale * fmaxf(dst_h, bucket_h);",
        "        int max_x = (int)ceilf(bucket_support / cell_width_x);",
        "        int max_y = (int)ceilf(bucket_support / cell_width_y);",
        "        int max_z = (int)ceilf(bucket_support / cell_width_z);",
        "        int full_x = periodic_x && nx <= 2 * max_x + 1;",
        "        int full_y = periodic_y && ny <= 2 * max_y + 1;",
        "        int full_z = periodic_z && nz <= 2 * max_z + 1;",
        "        int loops_x = full_x ? nx : 2 * max_x + 1;",
        "        int loops_y = full_y ? ny : 2 * max_y + 1;",
        "        int loops_z = full_z ? nz : 2 * max_z + 1;",
        "        for (int iz = 0; iz < loops_z; ++iz) {",
        "            int cz = iz;",
        "            if (!full_z) {",
        "                int offset = iz - max_z;",
        "                if (!fused_codegen_neighbor_cell(base_cz, offset, nz, periodic_z, &cz)) {",
        "                    continue;",
        "                }",
        "            }",
        "            for (int iy = 0; iy < loops_y; ++iy) {",
        "                int cy = iy;",
        "                if (!full_y) {",
        "                    int offset = iy - max_y;",
        "                    if (!fused_codegen_neighbor_cell(base_cy, offset, ny, periodic_y, &cy)) {",
        "                        continue;",
        "                    }",
        "                }",
        "                for (int ix = 0; ix < loops_x; ++ix) {",
        "                    int cx = ix;",
        "                    if (!full_x) {",
        "                        int offset = ix - max_x;",
        "                        if (!fused_codegen_neighbor_cell(base_cx, offset, nx, periodic_x, &cx)) {",
        "                            continue;",
        "                        }",
        "                    }",
        "                    int cell = fused_codegen_linear_cell(cx, cy, cz, nx, ny);",
        "                    int flat = bucket * total_cells + cell;",
        "                    float cell_bucket_h = __uint_as_float(cell_bucket_h_max_bits[flat]);",
        "                    if (cell_bucket_h <= 0.0f) {",
        "                        continue;",
        "                    }",
        "                    float cell_support = radius_scale * fmaxf(dst_h, cell_bucket_h);",
        "                    float cell_support2 = cell_support * cell_support;",
        "                    float cell_distance2 = fused_codegen_cell_distance2_to_particle(",
        "                        cell, nx, ny, dst_x, dst_y, dst_z, xmin, xmax,",
        "                        ymin, ymax, zmin, zmax, periodic_x, periodic_y,",
        "                        periodic_z, cell_width_x, cell_width_y, cell_width_z",
        "                    );",
        "                    if (cell_distance2 > cell_support2) {",
        "                        continue;",
        "                    }",
        "                    int begin = cell_bucket_starts[flat];",
        "                    int end = begin + cell_bucket_counts[flat];",
        "                    for (int pos = begin; pos < end; ++pos) {",
        "                        int src = sorted_ids[pos];",
        *_hbucket_pair_support_lines(precompute, "                        "),
        *tuple(
            line.replace(
                "                                        ",
                "                            ",
            )
            for line in source_inline_lines
        ),
        *tuple(
            line.replace(
                "                                        ",
                "                            ",
            )
            for line in precompute_lines
        ),
        *tuple(
            line.replace(
                "                                        ",
                "                            ",
            )
            for line in loop_method_lines
        ),
        "                        }",
        "                    }",
        "                }",
        "            }",
        "        }",
        "    }",
    )
    return ("    {", *tuple(f"    {line}" for line in traversal_lines), "    }")


def _direct_pair_equation_call_lines(
    methods_or_stage: object,
    calls: tuple[CudaEquationMethodCall, ...],
    method_filter=None,
) -> tuple[str, ...]:
    lines = []
    methods = (
        methods_or_stage.methods
        if isinstance(methods_or_stage, StageNode)
        else methods_or_stage
    )
    for method in methods:
        if method_filter is not None and not method_filter(method):
            continue
        call = _equation_call_for_method(
            method.equation_name, method.method_kind, calls
        )
        arguments = ", ".join(call.arguments)
        lines.append(
            f"                                        {call.function_name}({arguments});"
        )
    return tuple(lines)


def _grid_stride_call_lines(
    methods: tuple[object, ...], calls: tuple[CudaEquationMethodCall, ...]
) -> tuple[str, ...]:
    return tuple(
        line.replace("                                        ", "        ")
        for line in _direct_pair_equation_call_lines(methods, calls)
    )


def _local_reduction_pair_loop_call_lines(
    methods: tuple[object, ...],
    calls: tuple[CudaEquationMethodCall, ...],
    reduction_methods: tuple[object, ...],
) -> tuple[str, ...]:
    lines = []
    for method in methods:
        if not _is_pair_loop_method(method):
            continue
        call = _equation_call_for_method(
            method.equation_name, method.method_kind, calls
        )
        reduction = _optional_reduction_method_for_method(
            method.equation_name, method.method_kind, reduction_methods
        )
        arguments = ", ".join(
            _local_reduction_call_argument(argument, reduction)
            for argument in call.arguments
        )
        lines.append(
            f"                                        {call.function_name}({arguments});"
        )
    return tuple(lines)


def _cell_tile_pair_loop_call_lines(
    methods: tuple[object, ...],
    calls: tuple[CudaEquationMethodCall, ...],
    reduction_methods: tuple[object, ...],
    source_fields: tuple[str, ...],
) -> tuple[str, ...]:
    lines = []
    for method in methods:
        if not _is_pair_loop_method(method):
            continue
        call = _equation_call_for_method(
            method.equation_name, method.method_kind, calls
        )
        reduction = _optional_reduction_method_for_method(
            method.equation_name, method.method_kind, reduction_methods
        )
        arguments = ", ".join(
            _cell_tile_call_argument(argument, reduction, source_fields)
            for argument in call.arguments
        )
        lines.append(
            f"                                        {call.function_name}({arguments});"
        )
    return tuple(lines)


def _cell_tile_call_argument(
    argument: str,
    reduction: object | None,
    source_fields: tuple[str, ...],
) -> str:
    if argument == "src":
        return "tile_j"
    if argument.startswith("s_") and argument[2:] in source_fields:
        return f"fused_tile_s_{argument[2:]}"
    return _local_reduction_call_argument(argument, reduction)


def _local_reduction_call_argument(argument: str, reduction: object | None) -> str:
    if reduction is None:
        return argument
    for field in reduction.dest_reduction_writes:
        if argument == f"d_{field}":
            return f"fused_local_d_{field}"
    for field in reduction.dest_max_reduction_writes:
        if argument == f"d_{field}":
            return f"fused_local_d_{field}"
    return argument


def _reduction_method_for_method(
    equation_name: str,
    method_kind: MethodKind,
    reduction_methods: tuple[object, ...],
) -> object:
    matches = [
        method
        for method in reduction_methods
        if method.equation_name == equation_name and method.method_kind is method_kind
    ]
    assert len(matches) == 1
    return matches[0]


def _optional_reduction_method_for_method(
    equation_name: str,
    method_kind: MethodKind,
    reduction_methods: tuple[object, ...],
) -> object | None:
    matches = [
        method
        for method in reduction_methods
        if method.equation_name == equation_name and method.method_kind is method_kind
    ]
    assert len(matches) <= 1
    if not matches:
        return None
    return matches[0]


def _local_reduction_methods_for_stage(stage: StageNode) -> tuple[object, ...]:
    methods = []
    for segment in _stage_method_segments(stage):
        _, loop_methods, _ = _pair_segment_methods(segment)
        for method in _local_reduction_methods_for_segment(loop_methods):
            methods.append(method)
    return tuple(methods)


def _local_reduction_methods_for_stages(
    stages: tuple[StageNode, ...],
) -> tuple[object, ...]:
    methods = []
    for stage in stages:
        for method in _local_reduction_methods_for_stage(stage):
            methods.append(method)
    return tuple(methods)


def _local_reduction_methods_for_segment(
    loop_methods: tuple[object, ...],
) -> tuple[object, ...]:
    reduction_methods = tuple(
        method
        for method in loop_methods
        if method.dest_reduction_writes or method.dest_max_reduction_writes
    )
    if not _has_shared_reduction_write(reduction_methods):
        return ()
    return reduction_methods


def _has_shared_reduction_write(methods: tuple[object, ...]) -> bool:
    fields = []
    for method in methods:
        for field in sorted(method.dest_reduction_writes):
            if field in fields:
                return True
            fields.append(field)
        for field in sorted(method.dest_max_reduction_writes):
            if field in fields:
                return True
            fields.append(field)
    return False


def _is_pair_pre_loop_method(method: object) -> bool:
    return bool(method.sources) and method.method_kind in (
        MethodKind.INITIALIZE,
        MethodKind.LOOP_ALL,
    )


def _is_pair_loop_method(method: object) -> bool:
    return bool(method.sources) and method.method_kind is MethodKind.LOOP


def _is_pair_post_loop_method(method: object) -> bool:
    return (
        bool(method.sources) and method.method_kind is MethodKind.POST_LOOP
    ) or not bool(method.sources)


def _is_source_free_method(method: object) -> bool:
    return not bool(method.sources)


def _direct_pair_precompute_lines(precompute: CudaPairPrecompute) -> tuple[str, ...]:
    return tuple(
        f"                                        {line}" for line in precompute.lines
    )


def _hbucket_pair_precompute_lines(
    precompute: CudaPairPrecompute,
) -> tuple[str, ...]:
    lines = precompute.lines
    if _hbucket_pair_reuses_support_distance(precompute):
        lines = tuple(
            line
            for line in lines
            if not _hbucket_support_distance_precompute_line(line)
        )
        lines = tuple(_hbucket_cached_h_precompute_line(line) for line in lines)
    return tuple(f"                                        {line}" for line in lines)


def _cell_tile_pair_precompute_lines(
    precompute: CudaPairPrecompute,
    source_fields: tuple[str, ...],
) -> tuple[str, ...]:
    lines = _hbucket_pair_precompute_lines(precompute)
    return tuple(_cell_tile_source_read_line(line, source_fields) for line in lines)


def _cell_tile_source_read_line(line: str, source_fields: tuple[str, ...]) -> str:
    rewritten = line
    for field in source_fields:
        rewritten = rewritten.replace(
            f"s_{field}[src]", f"fused_tile_s_{field}[tile_j]"
        )
        rewritten = rewritten.replace(
            f"s_{field}[s_idx]", f"fused_tile_s_{field}[tile_j]"
        )
    return rewritten


def _cell_tile_source_fields(
    stage: StageNode,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
) -> tuple[str, ...]:
    fields = ["x", "y", "z", "h"]
    for method in stage.methods:
        for field in sorted(method.source_reads):
            _append_once(fields, field)
        for field in sorted(_precomputed_source_reads(precompute.symbols)):
            _append_once(fields, field)
    for call in calls:
        for argument in call.arguments:
            if argument.startswith("s_"):
                _append_once(fields, argument[2:])
    return tuple(fields)


def _precomputed_source_reads(symbols: frozenset[str]) -> frozenset[str]:
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


def _hbucket_pair_reuses_support_distance(precompute: CudaPairPrecompute) -> bool:
    return "XIJ" in precompute.symbols


def _hbucket_cached_h_precompute_line(line: str) -> str:
    return line.replace("h[dst]", "dst_h").replace("h[src]", "src_h")


def _hbucket_support_distance_precompute_line(line: str) -> bool:
    return (
        line in ("float XIJ[3];", "float R2IJ;")
        or line.startswith("XIJ[0] = fused_codegen_minimum_image(")
        or line.startswith("XIJ[1] = fused_codegen_minimum_image(")
        or line.startswith("XIJ[2] = fused_codegen_minimum_image(")
        or line == "R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];"
    )


def _hbucket_pair_support_lines(
    precompute: CudaPairPrecompute, indent: str
) -> tuple[str, ...]:
    if not _hbucket_pair_reuses_support_distance(precompute):
        return (
            f"{indent}if (fused_codegen_in_support_xyz_cached(",
            f"{indent}    dst_x, dst_y, dst_z, dst_h, src, x, y, z, h, xmin, xmax, ymin, ymax,",
            f"{indent}    zmin, zmax, periodic_x, periodic_y, periodic_z,",
            f"{indent}    radius_scale",
            f"{indent})) {{",
        )
    return (
        f"{indent}float XIJ[3];",
        f"{indent}XIJ[0] = fused_codegen_minimum_image(dst_x - x[src], xmax - xmin, periodic_x);",
        f"{indent}XIJ[1] = fused_codegen_minimum_image(dst_y - y[src], ymax - ymin, periodic_y);",
        f"{indent}XIJ[2] = fused_codegen_minimum_image(dst_z - z[src], zmax - zmin, periodic_z);",
        f"{indent}float R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
        f"{indent}float src_h = h[src];",
        f"{indent}float fused_support = radius_scale * fmaxf(dst_h, src_h);",
        f"{indent}if (R2IJ < fused_support * fused_support) {{",
    )


def _local_reduction_fields(
    reduction_methods: tuple[object, ...],
) -> tuple[LocalReductionField, ...]:
    fields = []
    for method in reduction_methods:
        for field in sorted(method.dest_reduction_writes):
            item = LocalReductionField(field=field, operation="sum")
            if item not in fields:
                fields.append(item)
        for field in sorted(method.dest_max_reduction_writes):
            item = LocalReductionField(field=field, operation="max")
            if item not in fields:
                fields.append(item)
    return tuple(fields)


def _local_reduction_initialization_lines(
    fields: tuple[LocalReductionField, ...],
) -> tuple[str, ...]:
    lines = []
    for item in fields:
        if item.operation == "sum":
            lines.append(f"    float fused_acc_d_{item.field} = 0.0f;")
        else:
            assert item.operation == "max"
            lines.append(f"    float fused_acc_d_{item.field} = d_{item.field}[dst];")
        lines.append(
            f"    float *fused_local_d_{item.field} = &fused_acc_d_{item.field};"
        )
    return tuple(lines)


def _local_reduction_commit_lines(
    fields: tuple[LocalReductionField, ...],
) -> tuple[str, ...]:
    lines = []
    for item in fields:
        field = item.field
        if item.operation == "sum":
            lines.append(f"    d_{field}[dst] += fused_acc_d_{field};")
        else:
            assert item.operation == "max"
            lines.append(
                f"    d_{field}[dst] = fmaxf(d_{field}[dst], fused_acc_d_{field});"
            )
    return tuple(lines)


def _local_reduction_wrapper_source(
    wrapper_source: str,
    reduction_methods: tuple[object, ...],
) -> str:
    replacements = _local_reduction_wrapper_replacements(reduction_methods)
    lines = []
    pending_fields = ()
    active_fields = ()
    brace_depth = 0
    for line in wrapper_source.splitlines():
        if not active_fields and not pending_fields:
            for function_name, fields in replacements:
                if f"void {function_name}(" in line:
                    pending_fields = fields
                    break
        if pending_fields and "{" in line:
            active_fields = pending_fields
            pending_fields = ()
            brace_depth = 0
        if active_fields:
            for field in active_fields:
                line = line.replace(f"d_{field}[d_idx]", f"d_{field}[0]")
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0:
                active_fields = ()
        lines.append(line)
    return "\n".join(lines)


def _hbucket_pair_wrapper_source(wrapper_source: str, stage: StageNode) -> str:
    return _local_reduction_wrapper_source(
        _force_cuda_source_fp32(wrapper_source),
        _local_reduction_methods_for_stage(stage),
    )


def _cell_tile_pair_wrapper_source(
    wrapper_source: str,
    stage: StageNode,
    source_fields: tuple[str, ...],
) -> str:
    assert source_fields
    return _hbucket_pair_wrapper_source(wrapper_source, stage)


def _hbucket_pair_window_wrapper_source(
    wrapper_source: str, stages: tuple[StageNode, ...]
) -> str:
    return _local_reduction_wrapper_source(
        _force_cuda_source_fp32(wrapper_source),
        _local_reduction_methods_for_stages(stages),
    )


def _local_reduction_wrapper_replacements(
    reduction_methods: tuple[object, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    replacements = []
    for method in reduction_methods:
        fields = []
        for field in sorted(method.dest_reduction_writes):
            _append_once(fields, field)
        for field in sorted(method.dest_max_reduction_writes):
            _append_once(fields, field)
        replacements.append(
            (f"{method.equation_name}_{method.method_kind.value}", tuple(fields))
        )
    return tuple(replacements)


def _source_inline_fields(
    source_inline_methods: tuple[MethodDeps, ...], consumer_stage: StageNode
) -> tuple[str, ...]:
    writes = []
    consumer_reads = _source_reads_for_methods(consumer_stage.methods)
    for method in source_inline_methods:
        for field in sorted(method.dest_writes):
            if field in consumer_reads:
                _append_once(writes, field)
    assert writes
    return tuple(writes)


def _source_inline_local_fields(
    source_inline_methods: tuple[MethodDeps, ...],
) -> tuple[str, ...]:
    fields = []
    for method in source_inline_methods:
        for field in sorted(method.dest_writes):
            _append_once(fields, field)
        for field in sorted(method.dest_reads):
            _append_once(fields, field)
    return tuple(fields)


def _source_reads_for_methods(methods: tuple[MethodDeps, ...]) -> frozenset[str]:
    reads = set()
    for method in methods:
        reads.update(method.source_reads)
    return frozenset(reads)


def _source_inline_wrapper_source(
    wrapper_source: str,
    methods: tuple[MethodDeps, ...],
    fields: tuple[str, ...],
) -> str:
    replacements = _source_inline_wrapper_replacements(methods, fields)
    lines = wrapper_source.splitlines()
    output = []
    index = 0
    while index < len(lines):
        line = lines[index]
        replacement = _source_inline_replacement_for_line(line, replacements)
        if replacement is None:
            output.append(line)
            index += 1
        else:
            function_lines, index = _source_inline_function_lines(
                lines, index, replacement
            )
            output.extend(function_lines)
    return "\n".join(output)


def _source_inline_wrapper_replacements(
    methods: tuple[MethodDeps, ...], fields: tuple[str, ...]
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    replacements = []
    for method in methods:
        active_fields = tuple(field for field in fields if field in method.source_reads)
        if active_fields:
            replacements.append(
                (
                    f"{method.equation_name}_{method.method_kind.value}",
                    active_fields,
                )
            )
    return tuple(replacements)


def _source_inline_replacement_for_line(line, replacements):
    for replacement in replacements:
        function_name, _fields = replacement
        if f"void {function_name}(" in line:
            return replacement
    return None


def _source_inline_function_lines(lines, start, replacement):
    function_name, fields = replacement
    clone = []
    index = start
    brace_depth = 0
    body_started = False
    while index < len(lines):
        line = lines[index].replace(
            f"void {function_name}(",
            f"void {function_name}_source_inline(",
        )
        if body_started:
            line = _source_inline_replace_source_reads_with_values(line, fields)
        clone.append(line)
        if "{" in line and not body_started:
            body_started = True
            for field in fields:
                clone.append(f"    float fused_inline_s_{field}_value = s_{field}[0];")
        brace_depth += line.count("{") - line.count("}")
        index += 1
        if body_started and brace_depth == 0:
            return tuple(clone), index
    assert False


def _source_inline_replace_source_reads_with_values(line, fields):
    return _source_inline_replace_source_reads(line, fields, "_value")


def _source_inline_replace_source_reads_with_slots(line, fields):
    return _source_inline_replace_source_reads(line, fields, "[0]")


def _source_inline_replace_source_reads(line, fields, suffix):
    for field in fields:
        line = line.replace(f"s_{field}[s_idx]", f"fused_inline_s_{field}{suffix}")
        line = line.replace(f"s_{field}[src]", f"fused_inline_s_{field}{suffix}")
    return line


def _replace_source_inline_reads(lines, fields):
    return tuple(
        _source_inline_replace_source_reads_with_slots(line, fields) for line in lines
    )


def _source_inline_equation_calls(
    calls: tuple[CudaEquationMethodCall, ...],
    methods: tuple[MethodDeps, ...],
    fields: tuple[str, ...],
) -> tuple[CudaEquationMethodCall, ...]:
    rewritten = []
    for call, method in zip(calls, methods):
        active_fields = tuple(field for field in fields if field in method.source_reads)
        if not active_fields:
            rewritten.append(call)
        else:
            rewritten.append(
                CudaEquationMethodCall(
                    equation_name=call.equation_name,
                    method_kind=call.method_kind,
                    function_name=f"{call.function_name}_source_inline",
                    argument_declarations=_source_inline_argument_declarations(
                        call.argument_declarations, active_fields
                    ),
                    arguments=_source_inline_call_arguments(
                        call.arguments, active_fields
                    ),
                )
            )
    return tuple(rewritten)


def _source_inline_argument_declarations(
    declarations: tuple[str, ...], fields: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        declaration
        for declaration in declarations
        if _source_inline_declaration_field(declaration) not in fields
    )


def _source_inline_declaration_field(declaration: str) -> str:
    parts = declaration.split()
    name = parts[-1]
    if name.startswith("s_"):
        return name[2:]
    return ""


def _source_inline_call_arguments(
    arguments: tuple[str, ...], fields: tuple[str, ...]
) -> tuple[str, ...]:
    rewritten = []
    for argument in arguments:
        if argument.startswith("s_") and argument[2:] in fields:
            rewritten.append(f"fused_inline_{argument}")
        else:
            rewritten.append(argument)
    return tuple(rewritten)


def _source_inline_prep_lines(
    source_inline_methods: tuple[MethodDeps, ...],
    calls: tuple[CudaEquationMethodCall, ...],
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    lines = []
    read_fields = _source_inline_read_fields(source_inline_methods)
    for field in fields:
        lines.append(f"float fused_inline_s_{field}[1];")
    for field in fields:
        if field in read_fields or not _source_inline_field_is_written(
            field, source_inline_methods
        ):
            lines.append(f"fused_inline_s_{field}[0] = s_{field}[src];")
    for method in source_inline_methods:
        call = _equation_call_for_method(
            method.equation_name, method.method_kind, calls
        )
        lines.append(_source_inline_prep_call_line(call, fields))
    return tuple(lines)


def _source_inline_field_is_written(
    field: str, methods: tuple[MethodDeps, ...]
) -> bool:
    return any(field in method.dest_writes for method in methods)


def _source_inline_read_fields(methods: tuple[MethodDeps, ...]) -> tuple[str, ...]:
    fields = []
    for method in methods:
        for field in sorted(method.dest_reads):
            _append_once(fields, field)
    return tuple(fields)


def _source_inline_prep_call_line(
    call: CudaEquationMethodCall, fields: tuple[str, ...]
) -> str:
    arguments = []
    for argument in call.arguments:
        if argument == "dst":
            arguments.append("0")
        elif argument.startswith("d_") and argument[2:] in fields:
            arguments.append(f"fused_inline_s_{argument[2:]}")
        else:
            arguments.append(argument)
    return f"{call.function_name}({', '.join(arguments)});"


def _equation_call_for_method(
    equation_name: str,
    method_kind: MethodKind,
    calls: tuple[CudaEquationMethodCall, ...],
) -> CudaEquationMethodCall:
    matches = [
        call
        for call in calls
        if call.equation_name == equation_name and call.method_kind is method_kind
    ]
    assert len(matches) == 1
    return matches[0]


def _cuda_equation_calls_for_stage(
    stage: StageNode,
    equations: tuple[object, ...],
    known_types: dict[str, KnownType],
    precomputed_symbols: frozenset[str],
) -> tuple[CudaEquationMethodCall, ...]:
    calls = []
    for method in stage.methods:
        equation = _equation_for_method(method.equation_name, equations)
        calls.append(
            cuda_equation_method_call_from_equation_with_precomputed(
                equation, method.method_kind.value, known_types, precomputed_symbols
            )
        )
    return tuple(calls)


def _cuda_known_types_for_stage(
    stage: StageNode,
    equations: tuple[object, ...],
    precomputed_symbols: frozenset[str],
) -> dict[str, KnownType]:
    known_types = {}
    for method in stage.methods:
        equation = _equation_for_method(method.equation_name, equations)
        args = list(
            inspect.getfullargspec(getattr(equation, method.method_kind.value)).args
        )
        for arg in args:
            if arg in ("self", "d_idx", "s_idx"):
                continue
            if arg in known_types:
                continue
            known_types[arg] = _known_type_for_cuda_arg(arg, precomputed_symbols)
    return known_types


def _cuda_known_types_for_stage_window(
    stages: tuple[StageNode, ...],
    equations_by_stage: tuple[tuple[object, ...], ...],
    precomputes: tuple[CudaPairPrecompute, ...],
) -> dict[str, KnownType]:
    known_types = {}
    for stage, equations, precompute in zip(stages, equations_by_stage, precomputes):
        stage_known_types = _cuda_known_types_for_stage(
            stage, equations, precompute.symbols
        )
        for arg, known_type in stage_known_types.items():
            if arg not in known_types:
                known_types[arg] = known_type
    return known_types


def _unique_equations(
    equations_by_stage: tuple[tuple[object, ...], ...],
) -> tuple[object, ...]:
    equations = []
    names = []
    for stage_equations in equations_by_stage:
        for equation in stage_equations:
            name = equation.__class__.__name__
            if name not in names:
                names.append(name)
                equations.append(equation)
    return tuple(equations)


def _unique_precompute_helper_sources(
    precomputes: tuple[CudaPairPrecompute, ...],
) -> tuple[str, ...]:
    sources = []
    for precompute in precomputes:
        if precompute.helper_source and precompute.helper_source not in sources:
            sources.append(precompute.helper_source)
    return tuple(sources)


def _equation_for_method(equation_name: str, equations: tuple[object, ...]) -> object:
    matches = [
        equation
        for equation in equations
        if equation.__class__.__name__ == equation_name
    ]
    assert len(matches) == 1
    return matches[0]


def _known_type_for_cuda_arg(
    arg: str, precomputed_symbols: frozenset[str]
) -> KnownType:
    if arg in precomputed_symbols:
        return _known_type_for_precomputed_symbol(arg)
    if _is_array_name(arg):
        return KnownType("GLOBAL_MEM float*")
    assert arg in ("t", "dt")
    return KnownType("float")


def _known_type_for_precomputed_symbol(symbol: str) -> KnownType:
    if symbol in ("XIJ", "DWIJ", "DWI", "DWJ", "VIJ"):
        return KnownType("float*")
    return KnownType("float")


def _equation_call_arguments(
    calls: tuple[CudaEquationMethodCall, ...],
) -> tuple[str, ...]:
    declarations = []
    for call in calls:
        for declaration in call.argument_declarations:
            if declaration not in declarations:
                declarations.append(declaration)
    return tuple(declarations)


def precompute_argument_names(precompute: CudaPairPrecompute) -> tuple[str, ...]:
    """Return device array names required by precompute expressions."""
    names = []
    if "VIJ" in precompute.symbols:
        names.extend(("d_u", "s_u", "d_v", "s_v", "d_w", "s_w"))
    if "RHOIJ" in precompute.symbols or "RHOIJ1" in precompute.symbols:
        names.extend(("d_rho", "s_rho"))
    return tuple(dict.fromkeys(names))


def _precompute_argument_declarations(
    precompute: CudaPairPrecompute,
) -> tuple[str, ...]:
    return tuple(
        f"GLOBAL_MEM float* {name}" for name in precompute_argument_names(precompute)
    )


def _unique_argument_declarations(
    *groups: tuple[str, ...],
) -> tuple[str, ...]:
    declarations = []
    for group in groups:
        for declaration in group:
            if declaration not in declarations:
                declarations.append(declaration)
    return tuple(declarations)


def _kernel_argument_name(declaration: str) -> str:
    return declaration.split()[-1]


def _append_once(items, item):
    if item not in items:
        items.append(item)


def _typed_argument_declaration(arg, known_types):
    assert arg in known_types
    known_type = known_types[arg]
    if isinstance(known_type, str):
        type_name = known_type
    else:
        type_name = known_type.type
    return f"{type_name} {arg}"


def _argument_name_from_declaration(declaration: str) -> str:
    return declaration.split()[-1]


def _force_cuda_source_fp32(source: str) -> str:
    """Force generated fused wrapper source to FP32 for this backend path."""
    return source.replace("double", "float")


def _convergence_flag_argument_declarations(
    convergence_field: str | None,
) -> tuple[str, ...]:
    if convergence_field is None:
        return ()
    return ("int *fused_convergence_flag",)


def _convergence_flag_lines(convergence_field: str | None) -> tuple[str, ...]:
    if convergence_field is None:
        return ()
    field_array = f"d_{convergence_field}"
    return (
        f"    if ({field_array}[dst] == 0.0f) {{",
        "        atomicExch(fused_convergence_flag, 0);",
        "    }",
    )


def _kernel_arg(arg):
    if hasattr(arg, "gpudata"):
        gpudata = arg.gpudata
        if isinstance(gpudata, int):
            return np.uintp(gpudata)
        return gpudata
    return arg


def fused_context_argument_declarations() -> tuple[str, ...]:
    """Return the CUDA argument ABI consumed by fused direct pair loops."""
    return (
        "const float *x",
        "const float *y",
        "const float *z",
        "const float *h",
        "int n",
        "float xmin",
        "float xmax",
        "float ymin",
        "float ymax",
        "float zmin",
        "float zmax",
        "int periodic_x",
        "int periodic_y",
        "int periodic_z",
        "float radius_scale",
        "int search_radius_cells",
        "int nx",
        "int ny",
        "int nz",
        "const int *cell_counts",
        "const int *cell_starts",
        "const int *sorted_ids",
    )


def cluster_context_argument_declarations() -> tuple[str, ...]:
    """Return the CUDA argument ABI consumed by fused cluster pair loops."""
    return fused_context_argument_declarations() + (
        "int cluster_total",
        "const int *cluster_cell",
        "const int *cluster_begin",
        "const int *cluster_count",
    )


def hbucket_context_argument_declarations() -> tuple[str, ...]:
    """Return the CUDA argument ABI consumed by fused h-bucket pair loops."""
    return (
        "const float *x",
        "const float *y",
        "const float *z",
        "const float *h",
        "int n",
        "float xmin",
        "float xmax",
        "float ymin",
        "float ymax",
        "float zmin",
        "float zmax",
        "int periodic_x",
        "int periodic_y",
        "int periodic_z",
        "float radius_scale",
        "int nx",
        "int ny",
        "int nz",
        "int total_cells",
        "int bucket_count",
        "float cell_width_x",
        "float cell_width_y",
        "float cell_width_z",
        "const unsigned int *bucket_h_max_bits",
        "const unsigned int *cell_bucket_h_max_bits",
        "const int *cell_bucket_counts",
        "const int *cell_bucket_starts",
        "const int *sorted_ids",
    )


def _argument_block(arguments: tuple[str, ...]) -> str:
    return ",\n".join(f"    {argument}" for argument in arguments)


def _stage_uses_neighbors(stage: StageNode) -> bool:
    return stage.kind in (StageKind.PAIR_DENSITY, StageKind.PAIR_RATE)


def _first_function(tree):
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert functions
    return functions[0]


def _written_array_names(function):
    names = []
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            assert isinstance(node.value, ast.Name)
            name = node.value.id
            assert _is_array_name(name)
            if name not in names:
                names.append(name)
    return tuple(names)


def _cuda_method_argument_declarations(args, written_arrays):
    declarations = []
    ignored = {"self", "d_idx", "s_idx"}
    for arg in args:
        if arg in ignored:
            continue
        assert _is_array_name(arg)
        assert not (arg.startswith("s_") and arg in written_arrays)
        if arg in written_arrays:
            declarations.append(f"float *{arg}")
        else:
            declarations.append(f"const float *{arg}")
    return tuple(declarations)


def _lower_statement(statement):
    if isinstance(statement, ast.Assign):
        assert len(statement.targets) == 1
        target = _lower_subscript(statement.targets[0])
        value = _lower_expr(statement.value)
        return (f"{target} = {value};",)
    if isinstance(statement, ast.AugAssign):
        target = _lower_subscript(statement.target)
        operator = _lower_aug_operator(statement.op)
        value = _lower_expr(statement.value)
        return (f"{target} {operator}= {value};",)
    assert False


def _lower_expr(expression):
    if isinstance(expression, ast.Subscript):
        return _lower_subscript(expression)
    if isinstance(expression, ast.Name):
        return _lower_name(expression.id)
    if isinstance(expression, ast.Constant):
        return _lower_constant(expression.value)
    assert False


def _lower_subscript(expression):
    assert isinstance(expression, ast.Subscript)
    assert isinstance(expression.value, ast.Name)
    name = expression.value.id
    assert _is_array_name(name)
    index = _lower_index(expression.slice)
    return f"{name}[{index}]"


def _lower_index(expression):
    if isinstance(expression, ast.Name):
        return _lower_name(expression.id)
    if isinstance(expression, ast.Constant):
        assert isinstance(expression.value, int)
        return str(expression.value)
    assert False


def _lower_name(name):
    if name == "d_idx":
        return "dst"
    if name == "s_idx":
        return "src"
    assert False


def _lower_constant(value):
    if isinstance(value, float):
        return f"{value:g}f"
    if isinstance(value, int):
        return str(value)
    assert False


def _lower_aug_operator(operator):
    if isinstance(operator, ast.Add):
        return "+"
    if isinstance(operator, ast.Sub):
        return "-"
    if isinstance(operator, ast.Mult):
        return "*"
    if isinstance(operator, ast.Div):
        return "/"
    assert False


def _is_array_name(name):
    return name.startswith("d_") or name.startswith("s_")


_DIRECT_PAIR_HELPERS = r"""
__device__ int fused_codegen_clamp_cell(float value, float lower, float upper, int count)
{
    float span = upper - lower;
    float rel = (value - lower) / span;
    int cell = (int)floorf(rel * (float)count);
    if (cell < 0) {
        cell = 0;
    }
    if (cell >= count) {
        cell = count - 1;
    }
    return cell;
}

__device__ int fused_codegen_linear_cell(int cx, int cy, int cz, int nx, int ny)
{
    return ((cz * ny) + cy) * nx + cx;
}

__device__ void fused_codegen_decode_cell(
    int cell,
    int nx,
    int ny,
    int *cx,
    int *cy,
    int *cz
)
{
    *cx = cell % nx;
    int tmp = cell / nx;
    *cy = tmp % ny;
    *cz = tmp / ny;
}

__device__ bool fused_codegen_valid_offset(int offset, int count)
{
    if (count == 1) {
        return offset == 0;
    }
    if (count == 2) {
        return offset <= 0;
    }
    return true;
}

__device__ int fused_codegen_wrapped_cell(int cell, int count)
{
    if (cell < 0) {
        return cell + count;
    }
    if (cell >= count) {
        return cell - count;
    }
    return cell;
}

__device__ bool fused_codegen_neighbor_cell(
    int base,
    int offset,
    int count,
    int periodic,
    int *out
)
{
    int cell = base + offset;
    if (periodic) {
        *out = fused_codegen_wrapped_cell(cell, count);
        return true;
    }
    if (cell < 0 || cell >= count) {
        return false;
    }
    *out = cell;
    return true;
}

__device__ float fused_codegen_minimum_image(float delta, float length, int periodic)
{
    if (periodic) {
        float half = 0.5f * length;
        if (delta > half) {
            delta -= length;
        }
        if (delta < -half) {
            delta += length;
        }
    }
    return delta;
}

__device__ bool fused_codegen_in_support_xyz_cached(
    float dst_x,
    float dst_y,
    float dst_z,
    float dst_h,
    int src,
    const float *x,
    const float *y,
    const float *z,
    const float *h,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    float zmin,
    float zmax,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float radius_scale
)
{
    float dx = x[src] - dst_x;
    float dy = y[src] - dst_y;
    float dz = z[src] - dst_z;
    dx = fused_codegen_minimum_image(dx, xmax - xmin, periodic_x);
    dy = fused_codegen_minimum_image(dy, ymax - ymin, periodic_y);
    dz = fused_codegen_minimum_image(dz, zmax - zmin, periodic_z);
    float dist2 = dx * dx + dy * dy + dz * dz;
    float support = radius_scale * fmaxf(dst_h, h[src]);
    return dist2 < support * support;
}

__device__ bool fused_codegen_in_support_xyz(
    int dst,
    int src,
    const float *x,
    const float *y,
    const float *z,
    const float *h,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    float zmin,
    float zmax,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float radius_scale
)
{
    return fused_codegen_in_support_xyz_cached(
        x[dst], y[dst], z[dst], h[dst], src, x, y, z, h, xmin, xmax,
        ymin, ymax, zmin, zmax, periodic_x, periodic_y, periodic_z,
        radius_scale
    );
}

__device__ float fused_codegen_axis_cell_distance(
    float point,
    int cell,
    float lower,
    float upper,
    float width,
    int periodic
)
{
    float center = lower + ((float)cell + 0.5f) * width;
    float delta = fused_codegen_minimum_image(center - point, upper - lower, periodic);
    float distance = fabsf(delta) - 0.5f * width;
    if (distance < 0.0f) {
        distance = 0.0f;
    }
    return distance;
}

__device__ float fused_codegen_cell_distance2_to_particle(
    int cell,
    int nx,
    int ny,
    float px,
    float py,
    float pz,
    float xmin,
    float xmax,
    float ymin,
    float ymax,
    float zmin,
    float zmax,
    int periodic_x,
    int periodic_y,
    int periodic_z,
    float cell_width_x,
    float cell_width_y,
    float cell_width_z
)
{
    int cx = cell % nx;
    int tmp = cell / nx;
    int cy = tmp % ny;
    int cz = tmp / ny;
    float dx = fused_codegen_axis_cell_distance(
        px, cx, xmin, xmax, cell_width_x, periodic_x
    );
    float dy = fused_codegen_axis_cell_distance(
        py, cy, ymin, ymax, cell_width_y, periodic_y
    );
    float dz = fused_codegen_axis_cell_distance(
        pz, cz, zmin, zmax, cell_width_z, periodic_z
    );
    return dx * dx + dy * dy + dz * dz;
}
"""


_CUBIC_SPLINE_WIJ_HELPER = r"""
__device__ float fused_codegen_cubic_spline_wij(float rij, float h, int dim)
{
    float h1 = 1.0f / h;
    float q = rij * h1;
    float fac = 0.0f;
    if (dim == 1) {
        fac = 0.6666666666666666f * h1;
    }
    else if (dim == 2) {
        fac = 0.4547284088339867f * h1 * h1;
    }
    else if (dim == 3) {
        fac = 0.3183098861837907f * h1 * h1 * h1;
    }
    float tmp2 = 2.0f - q;
    float val = 0.0f;
    if (q > 2.0f) {
        val = 0.0f;
    }
    else if (q > 1.0f) {
        val = 0.25f * tmp2 * tmp2 * tmp2;
    }
    else {
        val = 1.0f - 1.5f * q * q * (1.0f - 0.5f * q);
    }
    return val * fac;
}
"""


_CUBIC_SPLINE_GRADIENT_HELPER = (
    _CUBIC_SPLINE_WIJ_HELPER
    + r"""
__device__ float fused_codegen_cubic_spline_dwdq(float rij, float h, int dim)
{
    float h1 = 1.0f / h;
    float q = rij * h1;
    float fac = 0.0f;
    if (dim == 1) {
        fac = 0.6666666666666666f * h1;
    }
    else if (dim == 2) {
        fac = 0.4547284088339867f * h1 * h1;
    }
    else if (dim == 3) {
        fac = 0.3183098861837907f * h1 * h1 * h1;
    }
    float tmp2 = 2.0f - q;
    float val = 0.0f;
    if (rij > 1.0e-12f) {
        if (q > 2.0f) {
            val = 0.0f;
        }
        else if (q > 1.0f) {
            val = -0.75f * tmp2 * tmp2;
        }
        else {
            val = -3.0f * q * (1.0f - 0.75f * q);
        }
    }
    else {
        val = 0.0f;
    }
    return val * fac;
}

__device__ void fused_codegen_cubic_spline_gradient(
    float *grad,
    const float *xij,
    float rij,
    float h,
    int dim
)
{
    float h1 = 1.0f / h;
    float tmp = 0.0f;
    if (rij > 1.0e-12f) {
        float wdash = fused_codegen_cubic_spline_dwdq(rij, h, dim);
        tmp = wdash * h1 / rij;
    }
    grad[0] = tmp * xij[0];
    grad[1] = tmp * xij[1];
    grad[2] = tmp * xij[2];
}

__device__ float fused_codegen_cubic_spline_gradient_h(float rij, float h, int dim)
{
    float h1 = 1.0f / h;
    float q = rij * h1;
    float fac = 0.0f;
    if (dim == 1) {
        fac = 0.6666666666666666f * h1;
    }
    else if (dim == 2) {
        fac = 0.4547284088339867f * h1 * h1;
    }
    else if (dim == 3) {
        fac = 0.3183098861837907f * h1 * h1 * h1;
    }
    float tmp2 = 2.0f - q;
    float w = 0.0f;
    float dw = 0.0f;
    if (q > 2.0f) {
        w = 0.0f;
        dw = 0.0f;
    }
    else if (q > 1.0f) {
        w = 0.25f * tmp2 * tmp2 * tmp2;
        dw = -0.75f * tmp2 * tmp2;
    }
    else {
        w = 1.0f - 1.5f * q * q * (1.0f - 0.5f * q);
        dw = -3.0f * q * (1.0f - 0.75f * q);
    }
    return -fac * h1 * (dw * q + w * (float)dim);
}
"""
)


_QUINTIC_SPLINE_WIJ_HELPER = r"""
__device__ float fused_codegen_quintic_spline_wij(float rij, float h, int dim)
{
    float h1 = 1.0f / h;
    float q = rij * h1;
    float fac = 0.0f;
    if (dim == 1) {
        fac = 0.008333333333333333f * h1;
    }
    else if (dim == 2) {
        fac = 0.004660928796724434f * h1 * h1;
    }
    else if (dim == 3) {
        fac = 0.0026525823848649226f * h1 * h1 * h1;
    }
    float tmp3 = 3.0f - q;
    float tmp2 = 2.0f - q;
    float tmp1 = 1.0f - q;
    float val = 0.0f;
    if (q > 3.0f) {
        val = 0.0f;
    }
    else if (q > 2.0f) {
        val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
    }
    else if (q > 1.0f) {
        val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
        val -= 6.0f * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
    }
    else {
        val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
        val -= 6.0f * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
        val += 15.0f * tmp1 * tmp1 * tmp1 * tmp1 * tmp1;
    }
    return val * fac;
}
"""


_QUINTIC_SPLINE_GRADIENT_HELPER = (
    _QUINTIC_SPLINE_WIJ_HELPER
    + r"""
__device__ float fused_codegen_quintic_spline_dwdq(float rij, float h, int dim)
{
    float h1 = 1.0f / h;
    float q = rij * h1;
    float fac = 0.0f;
    if (dim == 1) {
        fac = 0.008333333333333333f * h1;
    }
    else if (dim == 2) {
        fac = 0.004660928796724434f * h1 * h1;
    }
    else if (dim == 3) {
        fac = 0.0026525823848649226f * h1 * h1 * h1;
    }
    float tmp3 = 3.0f - q;
    float tmp2 = 2.0f - q;
    float tmp1 = 1.0f - q;
    float val = 0.0f;
    if (rij > 1.0e-12f) {
        if (q > 3.0f) {
            val = 0.0f;
        }
        else if (q > 2.0f) {
            val = -5.0f * tmp3 * tmp3 * tmp3 * tmp3;
        }
        else if (q > 1.0f) {
            val = -5.0f * tmp3 * tmp3 * tmp3 * tmp3;
            val += 30.0f * tmp2 * tmp2 * tmp2 * tmp2;
        }
        else {
            val = -5.0f * tmp3 * tmp3 * tmp3 * tmp3;
            val += 30.0f * tmp2 * tmp2 * tmp2 * tmp2;
            val -= 75.0f * tmp1 * tmp1 * tmp1 * tmp1;
        }
    }
    else {
        val = 0.0f;
    }
    return val * fac;
}

__device__ void fused_codegen_quintic_spline_gradient(
    float *grad,
    const float *xij,
    float rij,
    float h,
    int dim
)
{
    float h1 = 1.0f / h;
    float tmp = 0.0f;
    if (rij > 1.0e-12f) {
        float wdash = fused_codegen_quintic_spline_dwdq(rij, h, dim);
        tmp = wdash * h1 / rij;
    }
    grad[0] = tmp * xij[0];
    grad[1] = tmp * xij[1];
    grad[2] = tmp * xij[2];
}

__device__ float fused_codegen_quintic_spline_gradient_h(float rij, float h, int dim)
{
    float h1 = 1.0f / h;
    float q = rij * h1;
    float fac = 0.0f;
    if (dim == 1) {
        fac = 0.008333333333333333f * h1;
    }
    else if (dim == 2) {
        fac = 0.004660928796724434f * h1 * h1;
    }
    else if (dim == 3) {
        fac = 0.0026525823848649226f * h1 * h1 * h1;
    }
    float tmp3 = 3.0f - q;
    float tmp2 = 2.0f - q;
    float tmp1 = 1.0f - q;
    float w = 0.0f;
    float dw = 0.0f;
    if (q > 3.0f) {
        w = 0.0f;
        dw = 0.0f;
    }
    else if (q > 2.0f) {
        w = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
        dw = -5.0f * tmp3 * tmp3 * tmp3 * tmp3;
    }
    else if (q > 1.0f) {
        w = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
        w -= 6.0f * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
        dw = -5.0f * tmp3 * tmp3 * tmp3 * tmp3;
        dw += 30.0f * tmp2 * tmp2 * tmp2 * tmp2;
    }
    else {
        w = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
        w -= 6.0f * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
        w += 15.0f * tmp1 * tmp1 * tmp1 * tmp1 * tmp1;
        dw = -5.0f * tmp3 * tmp3 * tmp3 * tmp3;
        dw += 30.0f * tmp2 * tmp2 * tmp2 * tmp2;
        dw -= 75.0f * tmp1 * tmp1 * tmp1 * tmp1;
    }
    return -fac * h1 * (dw * q + w * (float)dim);
}
"""
)


_FUSED_CUDA_COMPYLE_PREAMBLE = r"""
#define GLOBAL_MEM
#define LOCAL_MEM __shared__
#define WITHIN_KERNEL __device__ inline
#define KERNEL extern "C" __global__ void
#define abs fabsf
#define max(x, y) fmaxf((float)(x), (float)(y))
#define min(x, y) fminf((float)(x), (float)(y))
"""
