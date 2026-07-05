"""Planner for a generic fused CUDA SPH stage graph.

This module is intentionally planner-only. It extracts equation read/write
dependencies and reports the stage graph that a fused CUDA backend would need
to execute without changing the existing GPU evaluator.
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass, replace
from enum import Enum

from pysph.base.utils import is_overloaded_method


class MethodKind(Enum):
    """PySPH equation method kinds understood by the fused CUDA planner."""

    INITIALIZE = "initialize"
    INITIALIZE_PAIR = "initialize_pair"
    LOOP = "loop"
    LOOP_ALL = "loop_all"
    POST_LOOP = "post_loop"
    REDUCE = "reduce"


class StageKind(Enum):
    """Fused CUDA stage kinds."""

    NEIGHBOR_BUILD = "neighbor_build"
    INITIALIZE = "initialize"
    PAIR_DENSITY = "pair_density"
    POINTWISE = "pointwise"
    PAIR_RATE = "pair_rate"
    REDUCTION = "reduction"
    DEVICE_CONVERGENCE = "device_convergence"
    HOST_BOUNDARY = "host_boundary"


class StrictPlanError(RuntimeError):
    """Raised when a strict fused plan contains a host boundary."""


@dataclass(frozen=True)
class MethodDeps:
    """Read/write dependencies for one equation method."""

    equation_name: str
    method_kind: MethodKind
    dest: str
    sources: tuple[str, ...]
    dest_reads: frozenset[str]
    source_reads: frozenset[str]
    dest_writes: frozenset[str]
    source_writes: frozenset[str]
    precomputed_symbols: frozenset[str]
    precomputed_writes: frozenset[str]
    unsupported_reasons: tuple[str, ...]
    dest_reduction_writes: frozenset[str]
    dest_max_reduction_writes: frozenset[str]
    dest_reduction_reads: frozenset[str]


@dataclass(frozen=True)
class DeviceConvergencePolicy:
    """Device convergence contract for one iterative fused stage."""

    min_iterations: int
    max_iterations: int
    update_nnps: bool
    child_stage_indices: tuple[int, ...]
    equation_names: tuple[str, ...]
    flag_fields: tuple[str, ...]


@dataclass(frozen=True)
class StageNode:
    """One planned fused CUDA stage."""

    kind: StageKind
    dest: str
    sources: tuple[str, ...]
    methods: tuple[MethodDeps, ...]
    reason: str
    convergence_policy: DeviceConvergencePolicy | None
    legacy_group_count: int = 1
    method_segments: tuple[tuple[MethodDeps, ...], ...] = ()

    @property
    def has_host_boundary(self):
        """Return whether this stage requires host-side execution."""
        return self.kind is StageKind.HOST_BOUNDARY


@dataclass(frozen=True)
class CudaStagePlan:
    """Ordered fused CUDA stage plan."""

    stages: tuple[StageNode, ...]
    strict: bool

    @property
    def has_host_boundary(self):
        """Return whether any planned stage is a host boundary."""
        return any(stage.has_host_boundary for stage in self.stages)

    def assert_strict_supported(self):
        """Raise if strict mode cannot execute the plan fully on device."""
        if self.strict and self.has_host_boundary:
            raise StrictPlanError(self.format_text())

    def format_text(self):
        """Return a compact human-readable plan."""
        lines = []
        for index, stage in enumerate(self.stages):
            methods = ", ".join(
                f"{deps.equation_name}.{deps.method_kind.value}"
                for deps in stage.methods
            )
            policy = _format_convergence_policy(stage.convergence_policy)
            lines.append(
                f"{index}: {stage.kind.value} dest={stage.dest} sources={','.join(stage.sources)} reason={stage.reason} methods={methods}{policy}"
            )
        return "\n".join(lines)


def _format_convergence_policy(policy):
    if policy is None:
        return ""
    children = ",".join(str(index) for index in policy.child_stage_indices)
    flags = ",".join(policy.flag_fields)
    return f" policy=min={policy.min_iterations} max={policy.max_iterations} update_nnps={policy.update_nnps} children={children} flags={flags}"


def analyze_equation_method(equation, method_name):
    """Return dependency metadata for one equation method."""
    assert _has_overloaded_method(equation, method_name)
    method = getattr(equation, method_name)
    method_kind = MethodKind(method_name)
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    function = _first_function(tree)
    args = inspect.getfullargspec(method).args
    precomputed_symbols = _precomputed_symbols_from_args(args)
    access = _AccessCollector(frozenset(precomputed_symbols))
    access.visit_function_body(function)
    sources = equation.sources if equation.sources is not None else ()
    unsupported = tuple(sorted(access.unsupported_reasons))
    pair_reduction = _PairReductionCollector()
    pair_reduction.visit_function_body(function)
    reduction_unsupported = _pair_reduction_unsupported_reasons(
        method_kind, sources, pair_reduction
    )
    dest_reduction_writes = frozenset()
    dest_max_reduction_writes = frozenset()
    dest_reduction_reads = frozenset()
    if not reduction_unsupported:
        dest_reduction_writes = frozenset(pair_reduction.dest_reduction_writes)
        dest_max_reduction_writes = frozenset(pair_reduction.dest_max_reduction_writes)
        dest_reduction_reads = frozenset(
            pair_reduction.dest_reduction_reads.difference(
                pair_reduction.dest_non_reduction_reads
            )
        )
    return MethodDeps(
        equation_name=equation.__class__.__name__,
        method_kind=method_kind,
        dest=equation.dest,
        sources=tuple(sources),
        dest_reads=frozenset(access.dest_reads),
        source_reads=frozenset(access.source_reads),
        dest_writes=frozenset(access.dest_writes),
        source_writes=frozenset(access.source_writes),
        precomputed_symbols=frozenset(precomputed_symbols),
        precomputed_writes=frozenset(access.precomputed_writes),
        unsupported_reasons=unsupported,
        dest_reduction_writes=dest_reduction_writes,
        dest_max_reduction_writes=dest_max_reduction_writes,
        dest_reduction_reads=dest_reduction_reads,
    )


def method_deps_for_equation(equation):
    """Return dependencies for all supported methods on an equation."""
    methods = []
    for method_kind in MethodKind:
        method_name = method_kind.value
        if _has_overloaded_method(equation, method_name):
            methods.append(analyze_equation_method(equation, method_name))
    return tuple(methods)


def _has_overloaded_method(obj, method_name):
    return is_overloaded_method(getattr(obj, method_name))


def plan_equation_groups(groups, strict, supported_convergence):
    """Plan a conservative fused CUDA stage graph for PySPH groups."""
    stages = []
    for group in groups:
        stages.extend(_plan_group(group, supported_convergence, len(stages)))
    plan = CudaStagePlan(stages=_merge_adjacent_stages(tuple(stages)), strict=strict)
    plan.assert_strict_supported()
    return plan


def _merge_adjacent_stages(stages):
    merged = []
    for index, stage in enumerate(stages):
        future_stages = stages[index + 1 :]
        if merged and _can_merge_adjacent_stages(merged[-1], stage, future_stages):
            merged[-1] = _merged_stage(merged[-1], stage, future_stages)
        else:
            merged.append(stage)
    return tuple(merged)


def _merged_stage(left, right, future_stages):
    method_segments = ()
    if _can_merge_pair_rate_stages(left, right):
        method_segments = _stage_method_segments(left) + _stage_method_segments(right)
    elif _can_merge_pair_rate_tail(left, right) and left.method_segments:
        method_segments = left.method_segments[:-1] + (
            left.method_segments[-1] + right.methods,
        )
    method_segments = _coalesced_pair_method_segments(method_segments)
    if _can_merge_pair_rate_head(left, right, future_stages):
        return replace(
            right,
            methods=left.methods + right.methods,
            reason=f"{left.reason}; {right.reason}",
            legacy_group_count=left.legacy_group_count + right.legacy_group_count,
            method_segments=method_segments,
        )
    return replace(
        left,
        methods=left.methods + right.methods,
        reason=f"{left.reason}; {right.reason}",
        legacy_group_count=left.legacy_group_count + right.legacy_group_count,
        method_segments=method_segments,
    )


def _can_merge_adjacent_stages(left, right, future_stages):
    return (
        _can_merge_pointwise_stages(left, right)
        or _can_merge_pair_rate_tail(left, right)
        or _can_merge_pair_rate_head(left, right, future_stages)
        or _can_merge_pair_rate_stages(left, right)
    )


def _can_merge_pointwise_stages(left, right):
    return (
        left.kind is StageKind.POINTWISE
        and right.kind is StageKind.POINTWISE
        and left.dest == right.dest
        and left.sources == right.sources
    )


def _can_merge_pair_rate_tail(left, right):
    return (
        left.kind is StageKind.PAIR_RATE
        and right.kind in (StageKind.POINTWISE, StageKind.REDUCTION)
        and left.dest == right.dest
        and right.sources == ()
        and all(method.method_kind is not MethodKind.REDUCE for method in right.methods)
    )


def _can_merge_pair_rate_head(left, right, future_stages):
    return (
        left.kind is StageKind.POINTWISE
        and right.kind is StageKind.PAIR_RATE
        and left.dest == right.dest
        and left.sources == ()
        and not _stage_invalidates_neighbor_metadata(left)
        and not _stage_source_reads(right).intersection(_stage_dest_writes(left))
    )


def _can_merge_pair_rate_stages(left, right):
    return (
        left.kind is StageKind.PAIR_RATE
        and right.kind is StageKind.PAIR_RATE
        and left.dest == right.dest
        and left.sources == right.sources
        and not _stage_invalidates_neighbor_metadata(left)
        and not _stage_invalidates_neighbor_metadata(right)
        and not _stage_source_reads(right).intersection(_stage_dest_writes(left))
    )


def _coalesced_pair_method_segments(method_segments):
    if not method_segments:
        return ()
    if not _coalesce_pair_segments():
        return method_segments
    coalesced = []
    for segment in method_segments:
        if coalesced and _can_coalesce_pair_method_segments(coalesced[-1], segment):
            coalesced[-1] = _coalesced_pair_method_segment(coalesced[-1], segment)
        else:
            coalesced.append(segment)
    return tuple(coalesced)


def _coalesced_pair_method_segment(left, right):
    left_pre, left_loop, left_post = _pair_segment_methods_for_plan(left)
    right_pre, right_loop, right_post = _pair_segment_methods_for_plan(right)
    return left_pre + right_pre + left_loop + right_loop + left_post + right_post


def _can_coalesce_pair_method_segments(left, right):
    left_pre, left_loop, left_post = _pair_segment_methods_for_plan(left)
    right_pre, right_loop, right_post = _pair_segment_methods_for_plan(right)
    return (
        not _methods_conflict(right_pre, left_loop + left_post)
        and not _methods_conflict(left_loop, right_loop)
        and not _methods_conflict(left_post, right_loop)
    )


def _pair_segment_methods_for_plan(methods):
    pair_loop_indices = tuple(
        index
        for index, method in enumerate(methods)
        if _is_pair_loop_method_dep(method)
    )
    assert pair_loop_indices
    first_pair_loop_index = pair_loop_indices[0]
    pre_methods = []
    loop_methods = []
    post_methods = []
    for index, method in enumerate(methods):
        if _is_pair_loop_method_dep(method):
            loop_methods.append(method)
        elif _is_source_free_method_dep(method):
            if index < first_pair_loop_index:
                pre_methods.append(method)
            else:
                post_methods.append(method)
        elif _is_pair_pre_loop_method_dep(method):
            pre_methods.append(method)
        elif _is_pair_post_loop_method_dep(method):
            post_methods.append(method)
        else:
            assert False
    return tuple(pre_methods), tuple(loop_methods), tuple(post_methods)


def _methods_conflict(left_methods, right_methods):
    left_reads = _methods_non_reduction_reads_with_precompute(left_methods)
    right_reads = _methods_non_reduction_reads_with_precompute(right_methods)
    left_writes = _methods_writes(left_methods)
    right_writes = _methods_writes(right_methods)
    return bool(
        left_reads.intersection(right_writes)
        or right_reads.intersection(left_writes)
        or _methods_precomputed_write_conflicts(left_methods, right_methods)
        or _methods_write_conflicts(left_methods, right_methods)
    )


def _methods_non_reduction_reads_with_precompute(methods):
    reads = set()
    for method in methods:
        reads.update(method.dest_reads.difference(method.dest_reduction_reads))
        reads.update(method.source_reads)
        reads.update(_precomputed_source_reads(method.precomputed_symbols))
    return frozenset(reads)


def _methods_reads_with_precompute(methods):
    reads = set()
    for method in methods:
        reads.update(method.dest_reads)
        reads.update(method.source_reads)
        reads.update(_precomputed_source_reads(method.precomputed_symbols))
    return frozenset(reads)


def _methods_writes(methods):
    writes = set()
    for method in methods:
        writes.update(method.dest_writes)
        writes.update(method.source_writes)
    return frozenset(writes)


def _methods_precomputed_symbols(methods):
    symbols = set()
    for method in methods:
        symbols.update(method.precomputed_symbols)
    return frozenset(symbols)


def _methods_precomputed_writes(methods):
    symbols = set()
    for method in methods:
        symbols.update(method.precomputed_writes)
    return frozenset(symbols)


def _methods_precomputed_write_conflicts(left_methods, right_methods):
    left_writes = _methods_precomputed_writes(left_methods)
    right_writes = _methods_precomputed_writes(right_methods)
    left_symbols = _methods_precomputed_symbols(left_methods)
    right_symbols = _methods_precomputed_symbols(right_methods)
    return (
        left_writes.intersection(right_symbols)
        .union(right_writes.intersection(left_symbols))
        .union(left_writes.intersection(right_writes))
    )


def _methods_write_conflicts(left_methods, right_methods):
    left_writes = _methods_writes(left_methods)
    right_writes = _methods_writes(right_methods)
    if not _local_reduction_accumulators():
        return left_writes.intersection(right_writes)
    left_non_reduction_writes = _methods_non_reduction_writes(left_methods)
    right_non_reduction_writes = _methods_non_reduction_writes(right_methods)
    left_additive_writes = _methods_additive_reduction_writes(left_methods)
    right_additive_writes = _methods_additive_reduction_writes(right_methods)
    left_max_writes = _methods_max_reduction_writes(left_methods)
    right_max_writes = _methods_max_reduction_writes(right_methods)
    same_operator_reductions = left_additive_writes.intersection(
        right_additive_writes
    ).union(left_max_writes.intersection(right_max_writes))
    overlapping_writes = left_writes.intersection(right_writes)
    return (
        overlapping_writes.difference(same_operator_reductions)
        .union(left_non_reduction_writes.intersection(right_writes))
        .union(right_non_reduction_writes.intersection(left_writes))
    )


def _methods_non_reduction_writes(methods):
    writes = set()
    for method in methods:
        reduction_writes = method.dest_reduction_writes.union(
            method.dest_max_reduction_writes
        )
        writes.update(method.dest_writes.difference(reduction_writes))
        writes.update(method.source_writes)
    return frozenset(writes)


def _methods_additive_reduction_writes(methods):
    writes = set()
    for method in methods:
        writes.update(method.dest_reduction_writes)
    return frozenset(writes)


def _methods_max_reduction_writes(methods):
    writes = set()
    for method in methods:
        writes.update(method.dest_max_reduction_writes)
    return frozenset(writes)


def _is_pair_pre_loop_method_dep(method):
    return bool(method.sources) and method.method_kind in (
        MethodKind.INITIALIZE,
        MethodKind.LOOP_ALL,
    )


def _is_pair_loop_method_dep(method):
    return bool(method.sources) and method.method_kind is MethodKind.LOOP


def _is_pair_post_loop_method_dep(method):
    return (
        bool(method.sources) and method.method_kind is MethodKind.POST_LOOP
    ) or not bool(method.sources)


def _is_source_free_method_dep(method):
    return not bool(method.sources)


def _stage_invalidates_neighbor_metadata(stage):
    invalidating = frozenset(("x", "y", "z", "h"))
    return bool(_stage_dest_writes(stage).intersection(invalidating))


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


def _coalesce_pair_segments():
    return True


def _local_reduction_accumulators():
    return True


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


def _stage_method_segments(stage):
    if stage.method_segments:
        return stage.method_segments
    return (stage.methods,)


def _plan_group(group, supported_convergence, stage_start_index):
    stages = []
    if group.condition is not None:
        stages.append(_host_boundary(group, "dynamic condition"))
        return stages
    if group.pre is not None:
        stages.append(_host_boundary(group, "group pre callback"))
        return stages
    if group.post is not None:
        stages.append(_host_boundary(group, "group post callback"))
        return stages
    if group.has_subgroups:
        for subgroup in group.equations:
            stages.extend(
                _plan_group(
                    subgroup, supported_convergence, stage_start_index + len(stages)
                )
            )
        return stages

    methods = []
    unsupported = []
    for equation in group.equations:
        if _has_overloaded_method(equation, "py_initialize"):
            unsupported.append("py_initialize")
        deps = method_deps_for_equation(equation)
        methods.extend(deps)
        for item in deps:
            unsupported.extend(item.unsupported_reasons)
            if item.method_kind is MethodKind.REDUCE:
                unsupported.append("host reduce")
    if unsupported:
        stages.append(
            _host_boundary_with_methods(
                group, tuple(methods), ", ".join(sorted(set(unsupported)))
            )
        )
        return stages

    stage_kind = _stage_kind_for_methods(tuple(methods))
    child_stage_index = stage_start_index + len(stages)
    stages.append(
        StageNode(
            kind=stage_kind,
            dest=_group_dest(group),
            sources=_group_sources(group),
            methods=tuple(methods),
            reason="group methods",
            convergence_policy=None,
        )
    )
    if group.iterate:
        if _group_has_supported_convergence(group, supported_convergence):
            stages.append(
                StageNode(
                    kind=StageKind.DEVICE_CONVERGENCE,
                    dest=_group_dest(group),
                    sources=_group_sources(group),
                    methods=tuple(methods),
                    reason="supported device convergence",
                    convergence_policy=_device_convergence_policy(
                        group, supported_convergence, child_stage_index
                    ),
                )
            )
        else:
            stages.append(
                _host_boundary_with_methods(
                    group, tuple(methods), "unsupported iterative convergence"
                )
            )
    return stages


def _stage_kind_for_methods(methods):
    writes = set()
    has_source_loop = False
    only_reduce = True
    for deps in methods:
        writes.update(deps.dest_writes)
        if deps.sources and deps.method_kind in (
            MethodKind.LOOP,
            MethodKind.LOOP_ALL,
            MethodKind.INITIALIZE_PAIR,
        ):
            has_source_loop = True
        if deps.method_kind is not MethodKind.REDUCE:
            only_reduce = False
    if only_reduce:
        return StageKind.REDUCTION
    if has_source_loop and writes.intersection(
        {"rho", "rho_sum", "arho", "grhox", "grhoy", "grhoz", "dwdh", "omega"}
    ):
        return StageKind.PAIR_DENSITY
    if has_source_loop:
        return StageKind.PAIR_RATE
    if writes.intersection({"dt_adapt", "dt_cfl"}):
        return StageKind.REDUCTION
    return StageKind.POINTWISE


def _group_has_supported_convergence(group, supported_convergence):
    names = {equation.__class__.__name__ for equation in group.equations}
    return bool(names.intersection(set(supported_convergence)))


def _device_convergence_policy(group, supported_convergence, child_stage_index):
    supported = set(supported_convergence)
    equation_names = tuple(
        equation.__class__.__name__
        for equation in group.equations
        if equation.__class__.__name__ in supported
    )
    assert equation_names
    return DeviceConvergencePolicy(
        min_iterations=int(group.min_iterations),
        max_iterations=int(group.max_iterations),
        update_nnps=bool(group.update_nnps),
        child_stage_indices=(child_stage_index,),
        equation_names=equation_names,
        flag_fields=("equation_has_converged",),
    )


def _group_dest(group):
    equation = _first_equation(group)
    return equation.dest


def _group_sources(group):
    sources = []
    for equation in _flatten_equations(group):
        if equation.sources is not None:
            sources.extend(equation.sources)
    return tuple(dict.fromkeys(sources))


def _first_equation(group):
    equations = _flatten_equations(group)
    assert equations
    return equations[0]


def _flatten_equations(group):
    equations = []
    for equation in group.equations:
        if hasattr(equation, "has_subgroups"):
            equations.extend(_flatten_equations(equation))
        else:
            equations.append(equation)
    return equations


def _host_boundary(group, reason):
    return _host_boundary_with_methods(group, (), reason)


def _host_boundary_with_methods(group, methods, reason):
    return StageNode(
        kind=StageKind.HOST_BOUNDARY,
        dest=_group_dest(group),
        sources=_group_sources(group),
        methods=methods,
        reason=reason,
        convergence_policy=None,
    )


def _first_function(tree):
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert functions
    return functions[0]


def _precomputed_symbols_from_args(args):
    ignored = {"self", "d_idx", "s_idx", "t", "dt"}
    symbols = []
    for arg in args:
        if arg in ignored:
            continue
        if arg.startswith("d_") or arg.startswith("s_"):
            continue
        symbols.append(arg)
    return tuple(symbols)


def _field_from_array_name(name):
    return name[2:]


class _AccessCollector(ast.NodeVisitor):
    """Collect destination/source array field accesses from a method AST."""

    def __init__(self, precomputed_symbols):
        self.dest_reads = set()
        self.source_reads = set()
        self.dest_writes = set()
        self.source_writes = set()
        self.precomputed_symbols = precomputed_symbols
        self.precomputed_writes = set()
        self.unsupported_reasons = []

    def visit_function_body(self, function):
        for statement in function.body:
            self.visit(statement)

    def visit_Assign(self, node):
        for target in node.targets:
            self._record_target(target, True)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self._record_target(node.target, True)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        self._record_target(node.target, True)
        self._record_target(node.target, False)
        self.visit(node.value)

    def visit_Call(self, node):
        for arg in node.args:
            if isinstance(arg, ast.Name) and _is_array_name(arg.id):
                self.unsupported_reasons.append(f"bare array argument {arg.id}")
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Subscript(self, node):
        self._record_subscript(node, isinstance(node.ctx, ast.Store))
        self.generic_visit(node)

    def _record_target(self, node, is_write):
        if isinstance(node, ast.Subscript):
            self._record_subscript(node, is_write)
            self.visit(node.slice)
            return
        self.visit(node)

    def _record_subscript(self, node, is_write):
        if not isinstance(node.value, ast.Name):
            return
        name = node.value.id
        if name in self.precomputed_symbols and is_write:
            self.precomputed_writes.add(name)
        if name.startswith("d_"):
            field = _field_from_array_name(name)
            if is_write:
                self.dest_writes.add(field)
            else:
                self.dest_reads.add(field)
        if name.startswith("s_"):
            field = _field_from_array_name(name)
            if is_write:
                self.source_writes.add(field)
                self.unsupported_reasons.append(f"source write {field}")
            else:
                self.source_reads.add(field)


def _is_array_name(name):
    return name.startswith("d_") or name.startswith("s_")


class _PairReductionCollector(ast.NodeVisitor):
    """Collect the subset of pair-loop writes safe for threaded reduction."""

    def __init__(self):
        self.dest_reduction_writes = set()
        self.dest_max_reduction_writes = set()
        self.dest_reduction_reads = set()
        self.dest_non_reduction_reads = set()
        self.unsupported_reasons = []
        self._ignored_dest_reads = []

    def visit_function_body(self, function):
        for statement in function.body:
            self.visit(statement)

    def visit_Assign(self, node):
        ignored_reads = self._ignored_dest_reads_for_assignment(
            node.targets, node.value
        )
        for target in node.targets:
            self._record_assignment_target(target, node.value)
        self._ignored_dest_reads.extend(ignored_reads)
        self.visit(node.value)
        for _ in ignored_reads:
            self._ignored_dest_reads.pop()

    def visit_AnnAssign(self, node):
        ignored_reads = ()
        if node.value is not None:
            ignored_reads = self._ignored_dest_reads_for_assignment(
                (node.target,), node.value
            )
        self._record_assignment_target(node.target, node.value)
        if node.value is not None:
            self._ignored_dest_reads.extend(ignored_reads)
            self.visit(node.value)
            for _ in ignored_reads:
                self._ignored_dest_reads.pop()

    def visit_AugAssign(self, node):
        self._record_augassign_target(node.target, node.op)
        self.visit(node.value)

    def visit_Subscript(self, node):
        info = _array_target_info(node)
        if info is not None:
            prefix, field, index_name = info
            if (
                prefix == "d"
                and isinstance(node.ctx, ast.Load)
                and (field, index_name) not in self._ignored_dest_reads
            ):
                self.dest_non_reduction_reads.add(field)
        self.generic_visit(node)

    def _ignored_dest_reads_for_assignment(self, targets, value):
        ignored_reads = []
        for target in targets:
            info = _array_target_info(target)
            if info is None:
                continue
            prefix, field, index_name = info
            if prefix == "d" and _is_dest_max_reduction_value(field, index_name, value):
                ignored_reads.append((field, index_name))
        return tuple(ignored_reads)

    def _record_assignment_target(self, target, value):
        info = _array_target_info(target)
        if info is None:
            self.visit(target)
            return
        prefix, field, index_name = info
        if prefix == "d":
            if _is_dest_max_reduction_value(field, index_name, value):
                self.dest_max_reduction_writes.add(field)
                self.dest_reduction_reads.add(field)
                return
            self.unsupported_reasons.append(f"destination assignment {field}")
        if prefix == "s":
            self.unsupported_reasons.append(f"source write {field}")

    def _record_augassign_target(self, target, operator):
        info = _array_target_info(target)
        if info is None:
            self.visit(target)
            return
        prefix, field, index_name = info
        if prefix == "s":
            self.unsupported_reasons.append(f"source write {field}")
            return
        if (
            prefix == "d"
            and index_name == "d_idx"
            and isinstance(operator, (ast.Add, ast.Sub))
        ):
            self.dest_reduction_writes.add(field)
            self.dest_reduction_reads.add(field)
            return
        self.unsupported_reasons.append(f"destination assignment {field}")


def _pair_reduction_unsupported_reasons(method_kind, sources, collector):
    unsupported = list(collector.unsupported_reasons)
    if method_kind is not MethodKind.LOOP:
        unsupported.append(f"{method_kind.value} method")
    if not sources:
        unsupported.append("source-free method")
    stateful_reduction_reads = collector.dest_non_reduction_reads.intersection(
        collector.dest_reduction_reads
    )
    for field in sorted(stateful_reduction_reads):
        unsupported.append(f"stateful destination reduction read {field}")
    return unsupported


def _array_target_info(target):
    if not isinstance(target, ast.Subscript):
        return None
    if not isinstance(target.value, ast.Name):
        return None
    name = target.value.id
    if not _is_array_name(name):
        return None
    prefix = name[0]
    field = _field_from_array_name(name)
    index_name = _subscript_index_name(target.slice)
    return prefix, field, index_name


def _subscript_index_name(expression):
    if isinstance(expression, ast.Name):
        return expression.id
    return ""


def _is_dest_max_reduction_value(field, index_name, value):
    if index_name != "d_idx":
        return False
    if not isinstance(value, ast.Call):
        return False
    if not isinstance(value.func, ast.Name):
        return False
    if value.func.id != "max":
        return False
    return any(_is_dest_field_subscript(argument, field) for argument in value.args)


def _is_dest_field_subscript(expression, field):
    info = _array_target_info(expression)
    if info is None:
        return False
    prefix, current_field, index_name = info
    return prefix == "d" and current_field == field and index_name == "d_idx"
