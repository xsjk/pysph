from types import SimpleNamespace

from pysph.sph.fused_cuda_stage_backend import (
    _equations_for_stage,
    _stage_group_mapping,
)
from pysph.sph.fused_cuda_stage_plan import (
    CudaStagePlan,
    MethodKind,
    StageKind,
    StageNode,
    analyze_equation_method,
)
from pysph.sph.tests.fused_cuda_codegen_equations import AddMass


def test_equations_for_stage_selects_matching_destination():
    fluid = AddMass(dest="fluid", sources=["fluid", "solid"])
    solid = AddMass(dest="solid", sources=["fluid", "solid"])
    group = SimpleNamespace(has_subgroups=False, equations=[fluid, solid])
    helper = SimpleNamespace(object=SimpleNamespace(equation_groups=[group]))
    method = analyze_equation_method(fluid, MethodKind.LOOP.value)
    stage = StageNode(
        kind=StageKind.PAIR_RATE,
        dest="fluid",
        sources=("fluid", "solid"),
        methods=(method,),
        reason="test",
        convergence_policy=None,
    )

    assert _equations_for_stage(helper, stage, (0, -1)) == (fluid,)


def test_stage_group_mapping_distinguishes_destinations():
    fluid_stage = StageNode(
        kind=StageKind.PAIR_RATE,
        dest="fluid",
        sources=("fluid", "solid"),
        methods=(),
        reason="test",
        convergence_policy=None,
    )
    solid_stage = StageNode(
        kind=StageKind.PAIR_RATE,
        dest="solid",
        sources=("fluid", "solid"),
        methods=(),
        reason="test",
        convergence_policy=None,
    )
    helper = SimpleNamespace(
        calls=[
            {
                "type": "kernel",
                "stage_group": (0, -1),
                "dest": SimpleNamespace(name="fluid"),
            },
            {
                "type": "kernel",
                "stage_group": (0, -1),
                "dest": SimpleNamespace(name="solid"),
            },
        ],
        cuda_stage_plan=CudaStagePlan(stages=(fluid_stage, solid_stage), strict=False),
    )

    stage_by_group, covered, group_by_index = _stage_group_mapping(helper)
    assert stage_by_group == {
        ((0, -1), "fluid"): fluid_stage,
        ((0, -1), "solid"): solid_stage,
    }
    assert covered == set()
    assert group_by_index == {
        0: ((0, -1), "fluid"),
        1: ((0, -1), "solid"),
    }
