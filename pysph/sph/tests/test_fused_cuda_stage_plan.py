from pytest import raises

from pysph.sph.equation import Equation, Group
from pysph.sph.fused_cuda_stage_plan import (
    MethodDeps,
    MethodKind,
    StageNode,
    StageKind,
    StrictPlanError,
    analyze_pair_reduction_stage,
    analyze_pair_reduction_method,
    analyze_equation_method,
    cooperative_grid_sync_windows,
    cooperative_grid_sync_launch_budget,
    plan_equation_groups,
    resident_rhs_windows,
    source_visible_inline_precompute_windows,
    stage_dependency_barriers,
)


class AssignEquation(Equation):
    def loop(self, d_idx, d_u, d_au):
        d_u[d_idx] = d_au[d_idx]


class AccumulateEquation(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m):
        d_au[d_idx] += s_m[s_idx]


class SubtractAccumulateEquation(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m):
        d_au[d_idx] -= s_m[s_idx]


class OverwritePairEquation(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m):
        d_au[d_idx] = s_m[s_idx]


class SourceWriteEquation(Equation):
    def loop(self, s_idx, s_m):
        s_m[s_idx] = 0.0


class DensityEquation(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx] * WIJ


class EosEquation(Equation):
    def loop(self, d_idx, d_rho, d_p, d_cs):
        d_p[d_idx] = d_rho[d_idx]
        d_cs[d_idx] = 1.0


class RateEquation(Equation):
    def initialize(self, d_idx, d_au):
        d_au[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_p, s_p, DWIJ):
        d_au[d_idx] += (d_p[d_idx] + s_p[s_idx]) * DWIJ[0]


class DestinationPressureRateEquation(Equation):
    def initialize(self, d_idx, d_au):
        d_au[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_p, s_m, DWIJ):
        d_au[d_idx] += d_p[d_idx] * s_m[s_idx] * DWIJ[0]


class SourceIndependentPairEquation(Equation):
    def loop(self, d_idx, s_idx, d_av, s_m):
        d_av[d_idx] += s_m[s_idx]


class PostProcessedPairEquation(Equation):
    def initialize(self, d_idx, d_tmp):
        d_tmp[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_tmp, s_m):
        d_tmp[d_idx] += s_m[s_idx]

    def post_loop(self, d_idx, d_tmp, d_diag):
        d_diag[d_idx] = d_tmp[d_idx]


class DestinationDiagnosticPairEquation(Equation):
    def loop(self, d_idx, s_idx, d_diag, d_av, s_m):
        d_av[d_idx] += d_diag[d_idx] * s_m[s_idx]


class LaterPostReadsEarlierPostPairEquation(Equation):
    def loop(self, d_idx, s_idx, d_av, s_m):
        d_av[d_idx] += s_m[s_idx]

    def post_loop(self, d_idx, d_diag, d_beta):
        d_beta[d_idx] = d_diag[d_idx]


class SourceVisibleWriteEquation(Equation):
    def loop(self, d_idx, s_idx, d_u, s_m):
        d_u[d_idx] += s_m[s_idx]


class SourceVisibleReadEquation(Equation):
    def loop(self, d_idx, s_idx, d_av, s_u):
        d_av[d_idx] += s_u[s_idx]


class SourcePressureReadEquation(Equation):
    def loop(self, d_idx, s_idx, d_av, s_p):
        d_av[d_idx] += s_p[s_idx]


class PrecomputedVelocityReadEquation(Equation):
    def loop(self, d_idx, s_idx, d_av, VIJ):
        d_av[d_idx] += VIJ[0]


class MutatesPrecomputedXijEquation(Equation):
    def loop(self, d_idx, s_idx, d_au, XIJ, RIJ):
        XIJ[0] /= RIJ
        d_au[d_idx] += XIJ[0]


class ReadsPrecomputedXijEquation(Equation):
    def loop(self, d_idx, s_idx, d_av, XIJ):
        d_av[d_idx] += XIJ[0]


class TimeStepEquation(Equation):
    def initialize(self, d_idx, d_dt_adapt, d_au):
        d_dt_adapt[d_idx] = abs(d_au[d_idx])


class SmoothingLengthUpdateEquation(Equation):
    def loop(self, d_idx, d_h, d_rho):
        d_h[d_idx] = d_rho[d_idx]


class PairRateWithTimeStepCandidate(Equation):
    def initialize(self, d_idx, d_au, d_dt_cfl):
        d_au[d_idx] = 0.0
        d_dt_cfl[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_dt_cfl, s_m, DWIJ):
        d_au[d_idx] += s_m[s_idx] * DWIJ[0]
        d_dt_cfl[d_idx] = max(d_dt_cfl[d_idx], abs(DWIJ[0]))


class HostInitEquation(Equation):
    def py_initialize(self, dst, t, dt):
        dst.x[:] = 0.0

    def initialize(self, d_idx, d_u):
        d_u[d_idx] = 0.0


class HostReduceEquation(Equation):
    def initialize(self, d_idx, d_u):
        d_u[d_idx] = 0.0

    def reduce(self, dst, t, dt):
        self.value = dst.u[0]


class SyntheticDensityIteration(DensityEquation):
    def converged(self):
        return 1


def _method_deps(equation_name, method_kind, sources):
    return MethodDeps(
        equation_name=equation_name,
        method_kind=method_kind,
        dest="fluid",
        sources=sources,
        dest_reads=frozenset(),
        source_reads=frozenset(),
        dest_writes=frozenset(),
        source_writes=frozenset(),
        precomputed_symbols=frozenset(),
        precomputed_writes=frozenset(),
        unsupported_reasons=(),
        dest_reduction_writes=frozenset(),
        dest_max_reduction_writes=frozenset(),
        dest_reduction_reads=frozenset(),
    )


def _stage(kind, methods):
    return StageNode(
        kind=kind,
        dest="fluid",
        sources=("fluid",),
        methods=methods,
        reason="test",
        convergence_policy=None,
    )


def test_ast_dependency_extraction_tracks_assignment_reads_and_writes():
    equation = AssignEquation(dest="fluid", sources=["fluid"])

    deps = analyze_equation_method(equation, "loop")

    assert deps.method_kind is MethodKind.LOOP
    assert deps.dest_reads == frozenset({"au"})
    assert deps.dest_writes == frozenset({"u"})
    assert deps.source_reads == frozenset()
    assert deps.source_writes == frozenset()


def test_ast_dependency_extraction_tracks_augassign_source_reads():
    equation = AccumulateEquation(dest="fluid", sources=["fluid"])

    deps = analyze_equation_method(equation, "loop")

    assert deps.dest_reads == frozenset({"au"})
    assert deps.dest_writes == frozenset({"au"})
    assert deps.dest_reduction_writes == frozenset({"au"})
    assert deps.dest_max_reduction_writes == frozenset()
    assert deps.source_reads == frozenset({"m"})
    assert deps.source_writes == frozenset()


def test_ast_dependency_extraction_tracks_precomputed_writes():
    equation = MutatesPrecomputedXijEquation(dest="fluid", sources=["fluid"])

    deps = analyze_equation_method(equation, "loop")

    assert deps.precomputed_symbols == frozenset(("XIJ", "RIJ"))
    assert deps.precomputed_writes == frozenset(("XIJ",))


def test_ast_dependency_extraction_rejects_source_writes():
    equation = SourceWriteEquation(dest="fluid", sources=["fluid"])

    deps = analyze_equation_method(equation, "loop")

    assert deps.source_writes == frozenset({"m"})
    assert deps.unsupported_reasons == ("source write m",)


def test_pair_reduction_analysis_accepts_destination_additive_loop():
    equation = AccumulateEquation(dest="fluid", sources=["fluid"])

    analysis = analyze_pair_reduction_method(equation, "loop")

    assert analysis.supported
    assert analysis.dest_reduction_writes == frozenset({"au"})
    assert analysis.dest_max_reduction_writes == frozenset()
    assert analysis.unsupported_reasons == ()


def test_pair_reduction_analysis_accepts_destination_subtractive_loop():
    equation = SubtractAccumulateEquation(dest="fluid", sources=["fluid"])

    analysis = analyze_pair_reduction_method(equation, "loop")

    assert analysis.supported
    assert analysis.dest_reduction_writes == frozenset({"au"})
    assert analysis.dest_max_reduction_writes == frozenset()
    assert analysis.unsupported_reasons == ()


def test_pair_reduction_analysis_accepts_destination_max_loop():
    equation = PairRateWithTimeStepCandidate(dest="fluid", sources=["fluid"])

    analysis = analyze_pair_reduction_method(equation, "loop")

    assert analysis.supported
    assert analysis.dest_reduction_writes == frozenset({"au"})
    assert analysis.dest_max_reduction_writes == frozenset({"dt_cfl"})
    assert analysis.unsupported_reasons == ()


def test_pair_reduction_analysis_rejects_non_reduction_destination_write():
    equation = OverwritePairEquation(dest="fluid", sources=["fluid"])

    analysis = analyze_pair_reduction_method(equation, "loop")

    assert not analysis.supported
    assert analysis.dest_reduction_writes == frozenset()
    assert analysis.dest_max_reduction_writes == frozenset()
    assert analysis.unsupported_reasons == ("destination assignment au",)


def test_pair_reduction_analysis_rejects_source_writes():
    equation = SourceWriteEquation(dest="fluid", sources=["fluid"])

    analysis = analyze_pair_reduction_method(equation, "loop")

    assert not analysis.supported
    assert analysis.dest_reduction_writes == frozenset()
    assert analysis.dest_max_reduction_writes == frozenset()
    assert analysis.unsupported_reasons == ("source write m",)


def test_pair_reduction_analysis_rejects_source_free_methods():
    equation = AssignEquation(dest="fluid", sources=None)

    analysis = analyze_pair_reduction_method(equation, "loop")

    assert not analysis.supported
    assert analysis.dest_reduction_writes == frozenset()
    assert analysis.dest_max_reduction_writes == frozenset()
    assert analysis.unsupported_reasons == (
        "destination assignment u",
        "source-free method",
    )


def test_pair_reduction_stage_accepts_additive_pair_loop():
    equation = AccumulateEquation(dest="fluid", sources=["fluid"])
    stage = _stage(
        StageKind.PAIR_RATE,
        (_method_deps("AccumulateEquation", MethodKind.LOOP, ("fluid",)),),
    )

    analysis = analyze_pair_reduction_stage(stage, (equation,))

    assert analysis.supported
    assert [method.equation_name for method in analysis.methods] == [
        "AccumulateEquation"
    ]
    assert analysis.unsupported_reasons == ()


def test_pair_reduction_stage_accepts_mixed_sum_and_max_pair_loop():
    equation = PairRateWithTimeStepCandidate(dest="fluid", sources=["fluid"])
    stage = _stage(
        StageKind.PAIR_RATE,
        (_method_deps("PairRateWithTimeStepCandidate", MethodKind.LOOP, ("fluid",)),),
    )

    analysis = analyze_pair_reduction_stage(stage, (equation,))

    assert analysis.supported
    assert analysis.unsupported_reasons == ()
    assert analysis.methods[0].dest_reduction_writes == frozenset({"au"})
    assert analysis.methods[0].dest_max_reduction_writes == frozenset({"dt_cfl"})


def test_pair_reduction_stage_rejects_non_pair_stages():
    equation = EosEquation(dest="fluid", sources=None)
    stage = _stage(
        StageKind.POINTWISE,
        (_method_deps("EosEquation", MethodKind.LOOP, ()),),
    )

    analysis = analyze_pair_reduction_stage(stage, (equation,))

    assert not analysis.supported
    assert analysis.unsupported_reasons == ("non-pair stage",)


def test_stage_planner_emits_common_explicit_sph_shape():
    groups = [
        Group([DensityEquation(dest="fluid", sources=["fluid"])]),
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
        Group([TimeStepEquation(dest="fluid", sources=None)]),
    ]

    plan = plan_equation_groups(groups, False, ())

    assert [stage.kind for stage in plan.stages] == [
        StageKind.PAIR_DENSITY,
        StageKind.POINTWISE,
        StageKind.PAIR_RATE,
    ]
    assert plan.stages[-1].legacy_group_count == 2
    assert not plan.has_host_boundary


def test_resident_rhs_window_summarizes_common_explicit_sph_plan():
    groups = [
        Group([DensityEquation(dest="fluid", sources=["fluid"])]),
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
        Group([TimeStepEquation(dest="fluid", sources=None)]),
    ]
    plan = plan_equation_groups(groups, True, ())

    windows = resident_rhs_windows(plan)

    assert len(windows) == 1
    assert windows[0].dest == "fluid"
    assert windows[0].stage_indices == (0, 1, 2)
    assert windows[0].stage_kinds == (
        StageKind.PAIR_DENSITY,
        StageKind.POINTWISE,
        StageKind.PAIR_RATE,
    )
    assert windows[0].neighbor_build_count == 1
    assert windows[0].rhs_core_kernel_count == 3
    assert windows[0].planned_launch_count == 4
    assert not windows[0].materializes_csr
    assert windows[0].uses_device_metadata


def test_resident_rhs_window_rebuilds_after_h_update_before_next_pair_stage():
    groups = [
        Group([DensityEquation(dest="fluid", sources=["fluid"])]),
        Group([SmoothingLengthUpdateEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
    ]
    plan = plan_equation_groups(groups, True, ())

    windows = resident_rhs_windows(plan)

    assert [window.stage_indices for window in windows] == [(0, 1), (2,)]
    assert [window.neighbor_build_count for window in windows] == [1, 1]
    assert [window.planned_launch_count for window in windows] == [3, 2]


def test_stage_dependency_barriers_report_neighbor_rebuild_after_h_update():
    groups = [
        Group([DensityEquation(dest="fluid", sources=["fluid"])]),
        Group([SmoothingLengthUpdateEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
    ]
    plan = plan_equation_groups(groups, True, ())

    barriers = stage_dependency_barriers(plan)

    assert [(barrier.left_index, barrier.right_index) for barrier in barriers] == [
        (1, 2),
        (1, 2),
    ]
    assert [barrier.reason for barrier in barriers] == [
        "neighbor_metadata_rebuild",
        "source_visible_dependency",
    ]
    assert [barrier.fields for barrier in barriers] == [("h",), ("h",)]


def test_stage_dependency_barriers_report_source_visible_dependency():
    groups = [
        Group([SourceVisibleWriteEquation(dest="fluid", sources=["fluid"])]),
        Group([SourceVisibleReadEquation(dest="fluid", sources=["fluid"])]),
    ]
    plan = plan_equation_groups(groups, True, ())

    barriers = stage_dependency_barriers(plan)

    assert [(barrier.left_index, barrier.right_index) for barrier in barriers] == [
        (0, 1)
    ]
    assert barriers[0].reason == "source_visible_dependency"
    assert barriers[0].fields == ("u",)


def test_stage_dependency_barriers_do_not_report_destination_only_dependency():
    groups = [
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([DestinationPressureRateEquation(dest="fluid", sources=["fluid"])]),
    ]
    plan = plan_equation_groups(groups, True, ())

    assert stage_dependency_barriers(plan) == ()


def test_cooperative_grid_sync_window_fuses_source_visible_pair_dependency():
    groups = [
        Group([SourceVisibleWriteEquation(dest="fluid", sources=["fluid"])]),
        Group([SourceVisibleReadEquation(dest="fluid", sources=["fluid"])]),
    ]
    plan = plan_equation_groups(groups, True, ())

    windows = cooperative_grid_sync_windows(plan)

    assert len(windows) == 1
    assert windows[0].stage_indices == (0, 1)
    assert windows[0].stage_kinds == (StageKind.PAIR_RATE, StageKind.PAIR_RATE)
    assert windows[0].sync_count == 1
    assert windows[0].ordinary_rhs_core_kernel_count == 2
    assert windows[0].cooperative_rhs_core_kernel_count == 1
    assert windows[0].barrier_reasons == ("source_visible_dependency",)
    assert windows[0].barrier_fields == ("u",)


def test_cooperative_grid_sync_window_rejects_neighbor_rebuild_boundary():
    groups = [
        Group([DensityEquation(dest="fluid", sources=["fluid"])]),
        Group([SmoothingLengthUpdateEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
    ]
    plan = plan_equation_groups(groups, True, ())

    assert cooperative_grid_sync_windows(plan) == ()


def test_cooperative_grid_sync_budget_reports_core_kernel_savings():
    groups = [
        Group([DensityEquation(dest="fluid", sources=["fluid"])]),
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([SourceVisibleWriteEquation(dest="fluid", sources=["fluid"])]),
        Group([SourceVisibleReadEquation(dest="fluid", sources=["fluid"])]),
    ]
    plan = plan_equation_groups(groups, True, ())

    budget = cooperative_grid_sync_launch_budget(plan)

    assert budget.ordinary_rhs_core_kernel_count == 3
    assert budget.cooperative_rhs_core_kernel_count == 2
    assert budget.core_kernel_savings == 1
    assert budget.cooperative_planned_launch_count == 3


def test_stage_planner_keeps_pair_loop_with_timestep_candidate_as_pair_stage():
    group = Group([PairRateWithTimeStepCandidate(dest="fluid", sources=["fluid"])])

    plan = plan_equation_groups([group], False, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]


def test_stage_planner_reports_dynamic_condition_as_host_boundary():
    group = Group(
        [DensityEquation(dest="fluid", sources=["fluid"])], condition=lambda t, dt: True
    )

    plan = plan_equation_groups([group], False, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.HOST_BOUNDARY]
    assert plan.has_host_boundary
    assert "dynamic condition" in plan.stages[0].reason


def test_stage_planner_strict_mode_rejects_host_boundary():
    group = Group(
        [DensityEquation(dest="fluid", sources=["fluid"])], condition=lambda t, dt: True
    )

    with raises(StrictPlanError):
        plan_equation_groups([group], True, ())


def test_stage_planner_reports_py_initialize_as_host_boundary():
    group = Group([HostInitEquation(dest="fluid", sources=None)])

    plan = plan_equation_groups([group], False, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.HOST_BOUNDARY]
    assert "py_initialize" in plan.stages[0].reason


def test_stage_planner_reports_host_reduce_as_host_boundary():
    group = Group([HostReduceEquation(dest="fluid", sources=None)])

    plan = plan_equation_groups([group], False, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.HOST_BOUNDARY]
    assert "host reduce" in plan.stages[0].reason


def test_supported_density_iteration_lowers_to_device_convergence():
    group = Group(
        [SyntheticDensityIteration(dest="fluid", sources=["fluid"])],
        min_iterations=2,
        max_iterations=7,
        update_nnps=True,
        iterate=True,
    )

    plan = plan_equation_groups([group], True, ("SyntheticDensityIteration",))

    assert [stage.kind for stage in plan.stages] == [
        StageKind.PAIR_DENSITY,
        StageKind.DEVICE_CONVERGENCE,
    ]
    assert not plan.has_host_boundary
    policy = plan.stages[1].convergence_policy
    assert policy.min_iterations == 2
    assert policy.max_iterations == 7
    assert policy.update_nnps
    assert policy.child_stage_indices == (0,)
    assert policy.equation_names == ("SyntheticDensityIteration",)
    assert policy.flag_fields == ("equation_has_converged",)


def test_device_convergence_format_reports_policy():
    group = Group(
        [SyntheticDensityIteration(dest="fluid", sources=["fluid"])],
        min_iterations=2,
        max_iterations=7,
        update_nnps=True,
        iterate=True,
    )

    plan = plan_equation_groups([group], True, ("SyntheticDensityIteration",))

    assert (
        "policy=min=2 max=7 update_nnps=True children=0 flags=equation_has_converged"
        in plan.format_text()
    )


def test_stage_planner_merges_adjacent_pointwise_groups_for_same_destination():
    groups = [
        Group([AssignEquation(dest="fluid", sources=None)]),
        Group([EosEquation(dest="fluid", sources=None)]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.POINTWISE]
    assert plan.stages[0].legacy_group_count == 2
    assert [method.equation_name for method in plan.stages[0].methods] == [
        "AssignEquation",
        "EosEquation",
    ]


def test_stage_planner_merges_pair_rate_with_source_free_pointwise_tail():
    groups = [
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
        Group([EosEquation(dest="fluid", sources=None)]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert plan.stages[0].legacy_group_count == 2
    assert [
        (method.equation_name, method.method_kind) for method in plan.stages[0].methods
    ] == [
        ("RateEquation", MethodKind.INITIALIZE),
        ("RateEquation", MethodKind.LOOP),
        ("EosEquation", MethodKind.LOOP),
    ]


def test_stage_planner_merges_pair_rate_with_source_free_reduction_tail():
    groups = [
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
        Group([TimeStepEquation(dest="fluid", sources=None)]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert plan.stages[0].legacy_group_count == 2
    assert [
        (method.equation_name, method.method_kind) for method in plan.stages[0].methods
    ] == [
        ("RateEquation", MethodKind.INITIALIZE),
        ("RateEquation", MethodKind.LOOP),
        ("TimeStepEquation", MethodKind.INITIALIZE),
    ]


def test_stage_planner_merges_source_free_pointwise_head_when_pair_reads_dest_only():
    groups = [
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([DestinationPressureRateEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert plan.stages[0].legacy_group_count == 2
    assert [
        (method.equation_name, method.method_kind) for method in plan.stages[0].methods
    ] == [
        ("EosEquation", MethodKind.LOOP),
        ("DestinationPressureRateEquation", MethodKind.INITIALIZE),
        ("DestinationPressureRateEquation", MethodKind.LOOP),
    ]


def test_stage_planner_keeps_pointwise_head_when_pair_reads_written_source_field():
    groups = [
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [
        StageKind.POINTWISE,
        StageKind.PAIR_RATE,
    ]


def test_source_visible_inline_precompute_window_accepts_pointwise_self_source():
    groups = [
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())
    windows = source_visible_inline_precompute_windows(plan)

    assert len(windows) == 1
    assert windows[0].dest == "fluid"
    assert windows[0].stage_indices == (0, 1)
    assert windows[0].fields == ("p",)
    assert [method.equation_name for method in windows[0].producer_methods] == [
        "EosEquation",
    ]
    assert [method.equation_name for method in windows[0].consumer_methods] == [
        "RateEquation",
        "RateEquation",
    ]


def test_source_visible_inline_precompute_window_rejects_neighbor_metadata_write():
    groups = [
        Group([SmoothingLengthUpdateEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert source_visible_inline_precompute_windows(plan) == ()


def test_source_visible_inline_precompute_window_rejects_non_self_source():
    groups = [
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([RateEquation(dest="fluid", sources=["solid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert source_visible_inline_precompute_windows(plan) == ()


def test_stage_planner_merges_adjacent_pair_rates_without_source_visible_dependency():
    groups = [
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
        Group([SourceIndependentPairEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert plan.stages[0].legacy_group_count == 2
    assert len(plan.stages[0].method_segments) == 2


def test_stage_planner_coalesces_independent_pair_segments_when_enabled(monkeypatch):
    monkeypatch.setenv("PYSPH_FUSED_COALESCE_PAIR_SEGMENTS", "1")
    groups = [
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
        Group([SourceIndependentPairEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert len(plan.stages[0].method_segments) == 1
    assert [method.equation_name for method in plan.stages[0].method_segments[0]] == [
        "RateEquation",
        "RateEquation",
        "SourceIndependentPairEquation",
    ]


def test_stage_planner_keeps_segments_when_later_pair_reads_prior_post_loop(
    monkeypatch,
):
    monkeypatch.setenv("PYSPH_FUSED_COALESCE_PAIR_SEGMENTS", "1")
    groups = [
        Group([PostProcessedPairEquation(dest="fluid", sources=["fluid"])]),
        Group([DestinationDiagnosticPairEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert len(plan.stages[0].method_segments) == 2


def test_stage_planner_coalesces_ordered_post_loop_dependency(monkeypatch):
    monkeypatch.setenv("PYSPH_FUSED_COALESCE_PAIR_SEGMENTS", "1")
    groups = [
        Group([PostProcessedPairEquation(dest="fluid", sources=["fluid"])]),
        Group([LaterPostReadsEarlierPostPairEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert len(plan.stages[0].method_segments) == 1


def test_stage_planner_hoists_right_pre_methods_when_coalescing_pair_segments(
    monkeypatch,
):
    monkeypatch.setenv("PYSPH_FUSED_COALESCE_PAIR_SEGMENTS", "1")
    groups = [
        Group([PostProcessedPairEquation(dest="fluid", sources=["fluid"])]),
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert [
        (method.equation_name, method.method_kind)
        for method in plan.stages[0].method_segments[0]
    ] == [
        ("PostProcessedPairEquation", MethodKind.INITIALIZE),
        ("RateEquation", MethodKind.INITIALIZE),
        ("PostProcessedPairEquation", MethodKind.LOOP),
        ("RateEquation", MethodKind.LOOP),
        ("PostProcessedPairEquation", MethodKind.POST_LOOP),
    ]


def test_stage_planner_keeps_segments_for_shared_additive_accumulator_writes(
    monkeypatch,
):
    monkeypatch.setenv("PYSPH_FUSED_COALESCE_PAIR_SEGMENTS", "1")
    groups = [
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
        Group([AccumulateEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert len(plan.stages[0].method_segments) == 2


def test_stage_planner_coalesces_shared_additive_accumulators_with_local_reductions(
    monkeypatch,
):
    monkeypatch.setenv("PYSPH_FUSED_COALESCE_PAIR_SEGMENTS", "1")
    monkeypatch.setenv("PYSPH_FUSED_LOCAL_REDUCTION_ACCUMULATORS", "1")
    groups = [
        Group([RateEquation(dest="fluid", sources=["fluid"])]),
        Group([AccumulateEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert len(plan.stages[0].method_segments) == 1


def test_stage_planner_keeps_segments_when_pair_loop_mutates_precompute(
    monkeypatch,
):
    monkeypatch.setenv("PYSPH_FUSED_COALESCE_PAIR_SEGMENTS", "1")
    monkeypatch.setenv("PYSPH_FUSED_LOCAL_REDUCTION_ACCUMULATORS", "1")
    groups = [
        Group([MutatesPrecomputedXijEquation(dest="fluid", sources=["fluid"])]),
        Group([ReadsPrecomputedXijEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert len(plan.stages[0].method_segments) == 2


def test_stage_planner_keeps_segments_for_non_reduction_shared_write(
    monkeypatch,
):
    monkeypatch.setenv("PYSPH_FUSED_COALESCE_PAIR_SEGMENTS", "1")
    groups = [
        Group([OverwritePairEquation(dest="fluid", sources=["fluid"])]),
        Group([AccumulateEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [StageKind.PAIR_RATE]
    assert len(plan.stages[0].method_segments) == 2


def test_stage_planner_keeps_future_source_visible_pointwise_head_out_of_pair_stage(
    monkeypatch,
):
    monkeypatch.setenv("PYSPH_FUSED_HOIST_SOURCE_VISIBLE_PAIR_WINDOWS", "1")
    groups = [
        Group([EosEquation(dest="fluid", sources=None)]),
        Group([SourceIndependentPairEquation(dest="fluid", sources=["fluid"])]),
        Group([SourcePressureReadEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [
        StageKind.POINTWISE,
        StageKind.PAIR_RATE,
    ]
    assert [
        (method.equation_name, method.method_kind) for method in plan.stages[0].methods
    ] == [
        ("EosEquation", MethodKind.LOOP),
    ]
    assert [
        (method.equation_name, method.method_kind) for method in plan.stages[1].methods
    ] == [
        ("SourceIndependentPairEquation", MethodKind.LOOP),
        ("SourcePressureReadEquation", MethodKind.LOOP),
    ]
    assert len(plan.stages[1].method_segments) == 2


def test_stage_planner_keeps_adjacent_pair_rates_when_next_reads_written_source_field():
    groups = [
        Group([SourceVisibleWriteEquation(dest="fluid", sources=["fluid"])]),
        Group([SourceVisibleReadEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [
        StageKind.PAIR_RATE,
        StageKind.PAIR_RATE,
    ]


def test_stage_planner_keeps_adjacent_pair_rates_when_next_precompute_reads_written_source_field():
    groups = [
        Group([SourceVisibleWriteEquation(dest="fluid", sources=["fluid"])]),
        Group([PrecomputedVelocityReadEquation(dest="fluid", sources=["fluid"])]),
    ]

    plan = plan_equation_groups(groups, True, ())

    assert [stage.kind for stage in plan.stages] == [
        StageKind.PAIR_RATE,
        StageKind.PAIR_RATE,
    ]
