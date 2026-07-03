"""Small equation classes used by fused CUDA codegen tests."""

from pysph.sph.equation import Equation


class AddMass(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m):
        d_au[d_idx] += s_m[s_idx]


class AddScaledMass(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m):
        d_au[d_idx] += 2.0 * s_m[s_idx]


class AssignMassToAcceleration(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m):
        d_au[d_idx] = s_m[s_idx]


class InitLoopPost(Equation):
    def initialize(self, d_idx, d_u, d_au):
        d_u[d_idx] = 0.0
        d_au[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, s_m):
        d_au[d_idx] += s_m[s_idx]

    def post_loop(self, d_idx, d_u, d_au):
        d_u[d_idx] = d_au[d_idx]


class DensityConvergenceFlag(Equation):
    def initialize(self, d_idx, d_rho, d_converged):
        d_rho[d_idx] = 0.0
        d_converged[d_idx] = 1.0

    def loop(self, d_idx, s_idx, d_rho, s_m, WIJ):
        d_rho[d_idx] += s_m[s_idx] * WIJ

    def post_loop(self, d_idx, d_rho, d_converged):
        if d_rho[d_idx] < 0.0:
            d_converged[d_idx] = 0.0


class CopyAcceleration(Equation):
    def loop(self, d_idx, d_u, d_au):
        d_u[d_idx] = d_au[d_idx]


class PrepPressure(Equation):
    def loop(self, d_idx, d_p, d_rho):
        d_p[d_idx] = 2.0 * d_rho[d_idx]


class SetDensity(Equation):
    def loop(self, d_idx, d_rho):
        d_rho[d_idx] = 1.0


class PreserveThenSetDensity(Equation):
    def loop(self, d_idx, d_rho, d_rho_sum):
        d_rho_sum[d_idx] = d_rho[d_idx]
        d_rho[d_idx] = 1.0


class ReadSourceAcceleration(Equation):
    def loop(self, d_idx, s_idx, d_alpha, s_au):
        d_alpha[d_idx] += s_au[s_idx]


class ReadDestAndSourceAcceleration(Equation):
    def initialize(self, d_idx, d_alpha):
        d_alpha[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_alpha, d_au, s_au):
        d_alpha[d_idx] += d_au[d_idx] + s_au[s_idx]

    def post_loop(self, d_idx, d_alpha, d_beta):
        d_beta[d_idx] = d_alpha[d_idx]


class InitializeTimeStepCandidate(Equation):
    def initialize(self, d_idx, d_dt_adapt, d_au):
        d_dt_adapt[d_idx] = d_au[d_idx]


class AccumulateDWIJ(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m, DWIJ):
        d_au[d_idx] += s_m[s_idx] * DWIJ[0]


class AccumulateRhoIJ(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m, RHOIJ):
        d_au[d_idx] += s_m[s_idx] * RHOIJ


class AccumulateDWIJAndMaxSignal(Equation):
    def loop(self, d_idx, s_idx, d_au, d_dt_cfl, s_m, DWIJ):
        d_au[d_idx] += s_m[s_idx] * DWIJ[0]
        d_dt_cfl[d_idx] = max(d_dt_cfl[d_idx], abs(DWIJ[0]))


class AccumulateDWIAndDWJ(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m, DWI, DWJ):
        d_au[d_idx] += s_m[s_idx] * (DWI[0] + DWJ[0])


class AccumulateGradientH(Equation):
    def loop(self, d_idx, s_idx, d_au, s_m, WI, DWI, GHI, GHJ, GHIJ):
        d_au[d_idx] += s_m[s_idx] * (WI + DWI[0] + GHI + GHJ + GHIJ)


class PressureAcceleration(Equation):
    def loop(self, d_idx, s_idx, d_au, d_p, s_p, d_rho, s_rho, s_m, DWIJ):
        d_au[d_idx] += (
            -s_m[s_idx]
            * (
                d_p[d_idx] / (d_rho[d_idx] * d_rho[d_idx])
                + s_p[s_idx] / (s_rho[s_idx] * s_rho[s_idx])
            )
            * DWIJ[0]
        )
