"""Generic fused CUDA kernel planning helpers."""

import inspect
import re
from dataclasses import dataclass

import numpy as np

from pysph.sph.equation import CUDAGroup, KnownType
from pysph.sph.fused_cuda_stage_plan import (
    MethodKind,
    StageKind,
    StageNode,
)


@dataclass(frozen=True)
class FusedKernelOutline:
    """Non-executable fused kernel outline used as a codegen boundary."""

    name: str
    source: str


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


def fused_kernel_name(plan_id: str, stage: StageNode) -> str:
    """Return the stable CUDA function name for one fused stage."""
    assert stage.kind is not StageKind.HOST_BOUNDARY
    return f"fused_{plan_id}_{stage.dest}_{stage.kind.value}"


def generate_pointwise_kernel_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    calls: tuple[CudaEquationMethodCall, ...],
) -> FusedKernelOutline:
    """Return a pointwise CUDA outline using generated equation wrappers."""
    assert _stage_can_use_pointwise_kernel(stage)
    name = fused_kernel_name(plan_id, stage)
    method_lines = _equation_call_lines(stage, calls)
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
        plan_id, stage, wrapper_source, precompute, calls
    )


def generate_snapshot_hbucket_pair_window_outline_from_equations(
    plan_id: str,
    stages: tuple[StageNode, ...],
    equations_by_stage: tuple[tuple[object, ...], ...],
    precompute: CudaPairPrecompute,
    snapshot_fields: tuple[str, ...],
) -> FusedKernelOutline:
    """Return one h-bucket traversal for two dependency-compatible pair stages."""
    assert len(stages) == 2
    assert len(equations_by_stage) == 2
    stage = snapshot_hbucket_pair_window_stage(stages)
    equations = _unique_equations(equations_by_stage)
    known_types = _cuda_known_types_for_stage(stage, equations, precompute.symbols)
    group = CUDAGroup(list(equations))
    wrapper_source = group.get_equation_wrappers(known_types)
    calls = _cuda_equation_calls_for_snapshot_window(
        stage,
        stages[0],
        equations,
        known_types,
        precompute.symbols,
        snapshot_fields,
    )
    return _generate_snapshot_hbucket_pair_window_outline_with_equation_calls(
        plan_id, stage, wrapper_source, precompute, calls, snapshot_fields
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


def _generate_hbucket_pair_loop_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
) -> FusedKernelOutline:
    assert _stage_uses_neighbors(stage)
    name = fused_kernel_name(plan_id, stage)
    segment_lines = _hbucket_pair_segment_lines(stage, calls, precompute)
    arguments = (
        hbucket_context_argument_declarations()
        + _unique_argument_declarations(
            _precompute_argument_declarations(precompute),
            _equation_call_arguments(calls),
        )
    )
    fp32_wrapper_source = _hbucket_pair_wrapper_source(wrapper_source, stage)
    source = "\n".join(
        (
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            _PAIR_TRAVERSAL_HELPERS,
            precompute.helper_source,
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            "    int fused_dst_linear = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (fused_dst_linear >= n) {",
            "        return;",
            "    }",
            (
                "    int dst = n == source_count ? sorted_ids[fused_dst_linear] "
                ": fused_dst_linear;"
            ),
            *segment_lines,
            "}",
            "}",
        )
    )
    return FusedKernelOutline(name=name, source=source)


def _generate_snapshot_hbucket_pair_window_outline_with_equation_calls(
    plan_id: str,
    stage: StageNode,
    wrapper_source: str,
    precompute: CudaPairPrecompute,
    calls: tuple[CudaEquationMethodCall, ...],
    snapshot_fields: tuple[str, ...],
) -> FusedKernelOutline:
    assert _stage_uses_neighbors(stage)
    name = f"fused_{plan_id}_{stage.dest}_snapshot_hbucket_pair_window"
    segment_lines = _hbucket_pair_segment_lines(stage, calls, precompute)
    arguments = (
        hbucket_context_argument_declarations()
        + _snapshot_argument_declarations(snapshot_fields)
        + _unique_argument_declarations(
            _precompute_argument_declarations(precompute),
            _equation_call_arguments(calls),
        )
    )
    fp32_wrapper_source = _hbucket_pair_wrapper_source(wrapper_source, stage)
    source = "\n".join(
        (
            'extern "C" {',
            _FUSED_CUDA_COMPYLE_PREAMBLE,
            _PAIR_TRAVERSAL_HELPERS,
            precompute.helper_source,
            fp32_wrapper_source,
            f"__global__ void {name}(",
            _argument_block(arguments),
            ")",
            "{",
            "    int fused_dst_linear = blockIdx.x * blockDim.x + threadIdx.x;",
            "    if (fused_dst_linear >= n) {",
            "        return;",
            "    }",
            (
                "    int dst = n == source_count ? sorted_ids[fused_dst_linear] "
                ": fused_dst_linear;"
            ),
            *segment_lines,
            "}",
            "}",
        )
    )
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


def launch_hbucket_pair_kernel_with_context(
    module: object,
    kernel_name: str,
    context: object,
    extra_args: tuple[object, ...],
) -> PairLaunchConfig:
    """Launch a generated h-bucket pair kernel."""
    return _launch_hbucket_pair_kernel_with_context(
        module, kernel_name, context, extra_args, "hbucket"
    )


def launch_snapshot_hbucket_pair_window_kernel_with_context(
    module: object,
    kernel_name: str,
    context: object,
    snapshot_args: tuple[object, ...],
    extra_args: tuple[object, ...],
) -> PairLaunchConfig:
    """Launch a generated snapshot h-bucket pair-window kernel."""
    return _launch_hbucket_pair_kernel_with_context(
        module,
        kernel_name,
        context,
        snapshot_args + extra_args,
        "snapshot_hbucket",
    )


def _launch_hbucket_pair_kernel_with_context(
    module: object,
    kernel_name: str,
    context: object,
    extra_args: tuple[object, ...],
    traversal: str,
) -> PairLaunchConfig:
    kernel = module.get_function(kernel_name)
    block_size = _pair_block_size_for_count(context.destination_count)
    grid_x = (context.destination_count + block_size - 1) // block_size
    kernel(
        _kernel_arg(context.x),
        _kernel_arg(context.y),
        _kernel_arg(context.z),
        _kernel_arg(context.h),
        np.int32(context.destination_count),
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
        traversal=traversal,
        n=int(context.destination_count),
        block_size=int(block_size),
        grid_x=int(grid_x),
    )


def _pair_block_size_for_count(n: int) -> int:
    full_block_size = 128
    full_particle_blocks = (int(n) + full_block_size - 1) // full_block_size
    if full_particle_blocks < _cuda_multiprocessor_count():
        return 128
    return full_block_size


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
        pre_loop_method_lines = _equation_call_lines(pre_methods, calls)
        if reduction_methods:
            loop_method_lines = _local_reduction_pair_loop_call_lines(
                loop_methods, calls, reduction_methods
            )
        else:
            loop_method_lines = _equation_call_lines(loop_methods, calls)
        post_loop_method_lines = _equation_call_lines(post_methods, calls)
        lines.extend(
            line.replace("                                        ", "    ")
            for line in pre_loop_method_lines
        )
        lines.extend(_local_reduction_initialization_lines(reduced_fields))
        lines.extend(
            _hbucket_pair_neighbor_traversal_lines(
                _hbucket_pair_precompute_lines(precompute),
                precompute,
                loop_method_lines,
            )
        )
        lines.extend(_local_reduction_commit_lines(reduced_fields))
        lines.extend(
            line.replace("                                        ", "    ")
            for line in post_loop_method_lines
        )
    return tuple(lines)


def snapshot_hbucket_pair_window_stage(stages: tuple[StageNode, ...]) -> StageNode:
    """Return the synthetic single-traversal stage for a snapshot pair window."""
    assert len(stages) == 2
    left = stages[0]
    right = stages[1]
    left_pre, left_loop, left_post = _pair_segment_methods(left.methods)
    right_pre, right_loop, right_post = _pair_segment_methods(right.methods)
    hoisted_left_post, remaining_left_post = _snapshot_hoisted_left_post_methods(
        left_loop, left_post
    )
    assert not _methods_read_written_fields(right_pre, left_loop)
    assert not _methods_read_written_fields(right_loop, left_loop + remaining_left_post)
    assert not _methods_read_written_fields(remaining_left_post, right_pre + right_loop)
    methods = (
        left_pre
        + hoisted_left_post
        + right_pre
        + left_loop
        + right_loop
        + remaining_left_post
        + right_post
    )
    return StageNode(
        kind=right.kind,
        dest=right.dest,
        sources=right.sources,
        methods=methods,
        reason=f"snapshot hbucket pair window: {left.reason}; {right.reason}",
        convergence_policy=None,
        legacy_group_count=left.legacy_group_count + right.legacy_group_count,
        method_segments=(methods,),
    )


def _snapshot_hoisted_left_post_methods(
    left_loop: tuple[object, ...], left_post: tuple[object, ...]
) -> tuple[tuple[object, ...], tuple[object, ...]]:
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


def _hbucket_pair_neighbor_traversal_lines(
    precompute_lines: tuple[str, ...],
    precompute: CudaPairPrecompute,
    loop_method_lines: tuple[str, ...],
) -> tuple[str, ...]:
    traversal_lines = (
        "    float dst_x = x[dst];",
        "    float dst_y = y[dst];",
        "    float dst_z = z[dst];",
        "    float dst_h = h[dst];",
        "    float fused_length_x = xmax - xmin;",
        "    float fused_length_y = ymax - ymin;",
        "    float fused_length_z = zmax - zmin;",
        "    int base_cx = fused_codegen_clamp_cell(dst_x, xmin, xmax, nx);",
        "    int base_cy = fused_codegen_clamp_cell(dst_y, ymin, ymax, ny);",
        "    int base_cz = fused_codegen_clamp_cell(dst_z, zmin, zmax, nz);",
        "    for (int bucket = 0; bucket < bucket_count; ++bucket) {",
        "        float bucket_h = __uint_as_float(bucket_h_max_bits[bucket]);",
        "        if (bucket_h <= 0.0f) {",
        "            continue;",
        "        }",
        "        float bucket_support = radius_scale * fmaxf(dst_h, bucket_h);",
        "        int fused_ix_count = periodic_x ? (int)floorf(bucket_support / fused_length_x + 0.5f) : 0;",
        "        int fused_iy_count = periodic_y ? (int)floorf(bucket_support / fused_length_y + 0.5f) : 0;",
        "        int fused_iz_count = periodic_z ? (int)floorf(bucket_support / fused_length_z + 0.5f) : 0;",
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
        *_hbucket_pair_lines(
            precompute_lines,
            loop_method_lines,
            precompute,
            "                        ",
        ),
        "                    }",
        "                }",
        "            }",
        "        }",
        "    }",
    )
    return ("    {", *tuple(f"    {line}" for line in traversal_lines), "    }")


def _equation_call_lines(
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
        for line in _equation_call_lines(methods, calls)
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


def _local_reduction_methods_for_segment(
    loop_methods: tuple[object, ...],
) -> tuple[object, ...]:
    return tuple(
        method
        for method in loop_methods
        if method.dest_reduction_writes or method.dest_max_reduction_writes
    )


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


def _methods_read_written_fields(
    readers: tuple[object, ...], writers: tuple[object, ...]
) -> bool:
    return bool(
        _methods_read_fields(readers).intersection(_methods_written_fields(writers))
    )


def _methods_read_fields(methods: tuple[object, ...]) -> frozenset[str]:
    fields = set()
    for method in methods:
        fields.update(_method_read_fields(method))
    return frozenset(fields)


def _method_read_fields(method: object) -> frozenset[str]:
    fields = set(method.dest_reads.difference(method.dest_reduction_reads))
    fields.update(method.source_reads)
    fields.update(_precomputed_source_reads(method.precomputed_symbols))
    return frozenset(fields)


def _methods_written_fields(methods: tuple[object, ...]) -> frozenset[str]:
    fields = set()
    for method in methods:
        fields.update(method.dest_writes)
        fields.update(method.source_writes)
        fields.update(method.dest_reduction_writes)
        fields.update(method.dest_max_reduction_writes)
    return frozenset(fields)


def _method_identity(method: object) -> tuple[str, MethodKind]:
    return method.equation_name, method.method_kind


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
        f"{indent}float src_h = h[src];",
        f"{indent}float fused_support = radius_scale * fmaxf(dst_h, src_h);",
        f"{indent}float fused_support2 = fused_support * fused_support;",
        f"{indent}float fused_xij_x = fused_codegen_minimum_image(dst_x - x[src], fused_length_x, periodic_x);",
        f"{indent}float fused_xij_y = fused_codegen_minimum_image(dst_y - y[src], fused_length_y, periodic_y);",
        f"{indent}float fused_xij_z = fused_codegen_minimum_image(dst_z - z[src], fused_length_z, periodic_z);",
        f"{indent}if (fused_ix_count == 0 && fused_iy_count == 0 && fused_iz_count == 0) {{",
        f"{indent}    float XIJ[3];",
        f"{indent}    XIJ[0] = fused_xij_x;",
        f"{indent}    XIJ[1] = fused_xij_y;",
        f"{indent}    XIJ[2] = fused_xij_z;",
        f"{indent}    float R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
        f"{indent}    if (R2IJ <= fused_support2) {{",
    )


def _hbucket_pair_image_support_lines(indent: str) -> tuple[str, ...]:
    return (
        f"{indent}    }}",
        f"{indent}}} else {{",
        f"{indent}for (int fused_ix = -fused_ix_count; fused_ix <= fused_ix_count; ++fused_ix) {{",
        f"{indent}    for (int fused_iy = -fused_iy_count; fused_iy <= fused_iy_count; ++fused_iy) {{",
        f"{indent}        for (int fused_iz = -fused_iz_count; fused_iz <= fused_iz_count; ++fused_iz) {{",
        f"{indent}            float XIJ[3];",
        f"{indent}            XIJ[0] = fused_xij_x + fused_ix * fused_length_x;",
        f"{indent}            XIJ[1] = fused_xij_y + fused_iy * fused_length_y;",
        f"{indent}            XIJ[2] = fused_xij_z + fused_iz * fused_length_z;",
        f"{indent}            float R2IJ = XIJ[0] * XIJ[0] + XIJ[1] * XIJ[1] + XIJ[2] * XIJ[2];",
        f"{indent}            if (R2IJ <= fused_support2) {{",
    )


def _hbucket_pair_lines(
    precompute_lines: tuple[str, ...],
    loop_method_lines: tuple[str, ...],
    precompute: CudaPairPrecompute,
    indent: str,
) -> tuple[str, ...]:
    body_lines = (
        *_hbucket_pair_body_lines(precompute_lines, precompute),
        *_hbucket_pair_body_lines(loop_method_lines, precompute),
    )
    lines = (*_hbucket_pair_support_lines(precompute, indent), *body_lines)
    if _hbucket_pair_reuses_support_distance(precompute):
        lines = (*lines, *_hbucket_pair_image_support_lines(indent), *body_lines)
    return (*lines, *_hbucket_pair_support_end_lines(precompute, indent))


def _hbucket_pair_body_lines(
    lines: tuple[str, ...], precompute: CudaPairPrecompute
) -> tuple[str, ...]:
    if _hbucket_pair_reuses_support_distance(precompute):
        return lines
    return tuple(
        line.replace(
            "                                        ",
            "                            ",
        )
        for line in lines
    )


def _hbucket_pair_support_end_lines(
    precompute: CudaPairPrecompute, indent: str
) -> tuple[str, ...]:
    if not _hbucket_pair_reuses_support_distance(precompute):
        return (f"{indent}}}",)
    return (
        f"{indent}            }}",
        f"{indent}        }}",
        f"{indent}    }}",
        f"{indent}}}",
        f"{indent}}}",
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


def _cuda_equation_calls_for_snapshot_window(
    stage: StageNode,
    left_stage: StageNode,
    equations: tuple[object, ...],
    known_types: dict[str, KnownType],
    precomputed_symbols: frozenset[str],
    snapshot_fields: tuple[str, ...],
) -> tuple[CudaEquationMethodCall, ...]:
    left_loop_methods = _pair_segment_methods(left_stage.methods)[1]
    calls = []
    for method in stage.methods:
        equation = _equation_for_method(method.equation_name, equations)
        call = cuda_equation_method_call_from_equation_with_precomputed(
            equation, method.method_kind.value, known_types, precomputed_symbols
        )
        if _method_identity(method) in tuple(
            _method_identity(item) for item in left_loop_methods
        ):
            call = _snapshot_call_for_fields(call, snapshot_fields)
        calls.append(call)
    return tuple(calls)


def _snapshot_call_for_fields(
    call: CudaEquationMethodCall, snapshot_fields: tuple[str, ...]
) -> CudaEquationMethodCall:
    arguments = tuple(
        _snapshot_argument(argument, snapshot_fields) for argument in call.arguments
    )
    return CudaEquationMethodCall(
        equation_name=call.equation_name,
        method_kind=call.method_kind,
        function_name=call.function_name,
        argument_declarations=call.argument_declarations,
        arguments=arguments,
    )


def _snapshot_argument(argument: str, snapshot_fields: tuple[str, ...]) -> str:
    for field in snapshot_fields:
        if argument in (f"d_{field}", f"s_{field}"):
            return f"fused_snapshot_{field}"
    return argument


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


def _snapshot_argument_declarations(
    snapshot_fields: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        f"GLOBAL_MEM float* fused_snapshot_{field}" for field in snapshot_fields
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
    fp32_source = source.replace("double", "float")
    fp32_source = re.sub(
        r"pow\(\(([^,\n]+)\), \(1\.0f / self->dim\)\)",
        r"fused_codegen_pow_inv_dim(\1, self->dim)",
        fp32_source,
    )
    return re.sub(
        r"pow\(\(([^,\n]+)\), self->dim\)",
        r"fused_codegen_pow_dim(\1, self->dim)",
        fp32_source,
    )


def _kernel_arg(arg):
    if hasattr(arg, "gpudata"):
        gpudata = arg.gpudata
        if isinstance(gpudata, int):
            return np.uintp(gpudata)
        return gpudata
    return arg


def hbucket_context_argument_declarations() -> tuple[str, ...]:
    """Return the CUDA argument ABI consumed by fused h-bucket pair loops."""
    return (
        "const float *x",
        "const float *y",
        "const float *z",
        "const float *h",
        "int n",
        "int source_count",
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


def _is_array_name(name):
    return name.startswith("d_") or name.startswith("s_")


_PAIR_TRAVERSAL_HELPERS = r"""
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

WITHIN_KERNEL float fused_codegen_pow_dim(float value, float dim)
{
    if (dim == 1.0f) {
        return value;
    }
    if (dim == 2.0f) {
        return value * value;
    }
    if (dim == 3.0f) {
        return value * value * value;
    }
    return powf(value, dim);
}

WITHIN_KERNEL float fused_codegen_pow_inv_dim(float value, float dim)
{
    if (dim == 1.0f) {
        return value;
    }
    if (dim == 2.0f) {
        return sqrtf(value);
    }
    if (dim == 3.0f) {
        return cbrtf(value);
    }
    return powf(value, 1.0f / dim);
}
"""
