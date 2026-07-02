# Paper-to-code mapping

This document records how the implementation maps to `main.tex`.

## Problem setup

- `SnowHillSystem.step` implements `s_{j+1}=F(s_j,u_j)`.
- `SnowHillSystem.stage_cost` implements `c(s,u)=sqrt(s^T Q s + 1)+u^T R u`.
- `SnowHillSystem.bounds_for_controls` implements `|u| <= 1`.
- `SnowHillSystem.step` uses RK4 with `t_d=0.1`.
- `snow_hill_residual_acceleration` implements the commented Huber formula for
  the hill residual force.

## AC4MPC

- `terminal_actor_rollout_cost` implements Eq. `(V_f)`.
- `finite_horizon_cost` implements Eq. `(mpc_cost)` using that terminal cost.
- `actor_rollout` implements Eq. `(cl_pi)`, i.e. `Phi_hat` and `Psi_hat`.
- `shift_trajectory` implements the control shift `zeta` and state shift `xi`.
- `AC4MPCController.step` implements Algorithm 1.

The code also keeps the shifted/warm-start trajectory as an explicit candidate
when selecting the minimum cost. This preserves the set condition in Eq. `(K_N)`
even if a numerical solver returns a worse local iterate.

## AC4MPC-RTI

- `AC4MPCRTIController.step` implements Algorithm 2 with active and parallel
  MPC instances.
- `reinitialization_period` is `P`.
- `active_sqp_iterations` and `parallel_sqp_iterations` are `M` and `M_p`.
- `evaluate_policy_rollout` controls the paper's AD-specific omission of pure
  policy-rollout evaluation.

`AcadosSnowHillSolver` implements the paper's acados SQP-RTI backend for the
snow hill OCP. It uses the same `solve(state, initial, max_iterations, source)`
interface as `ScipyMultipleShootingSolver`, so the AC4MPC and AC4MPC-RTI
algorithm code is independent of the numerical backend.

The acados formulation uses:

- discrete RK4 dynamics matching `SnowHillSystem.step`;
- nonlinear least-squares stage cost equivalent to
  `sqrt(s^T Q s + 1) + u^T R u`, which enables Gauss-Newton stage-cost
  Hessians;
- external terminal cost containing the actor rollout and critic;
- a small diagonal custom Hessian for the terminal neural-network cost, matching
  the numerical robustness note in `main.tex`;
- `SQP_RTI` with split preparation/feedback phases for each configured RTI
  iteration.

`ScipyMultipleShootingSolver` remains as a slow transparent fallback and for
unit-level algorithm checks.

## ac4eval

- `ac4eval` implements Eq. `(auxlaw)`.
- `alpha=0` gives open-loop forward simulation.
- `alpha=1` fully applies the actor correction term.
- The terminal portion follows Eq. `(V_f)` with exactly `R` actor steps. The
  pseudo-code loop in Algorithm 3 appears inclusive at `N+R`; the equation and
  text define `R` extra steps, so the code follows the equation.

## Neural networks

- `TanhMlp` is a two-hidden-layer `tanh` MLP with 256 units by default, matching
  the experiment description.
- `SB3SACActor` returns the deterministic SAC action.
- `SB3SACQCost` converts SB3 reward Q values into cost Q values because the
  environment trains with reward `-cost`.
- `export_sac_weights.py`, `NumpySACActor`, and `NumpySACQCost` export and use
  the exact SB3 SAC weights without loading PyTorch in the acados process. The
  CasADi terminal actor/critic expressions use these exported weights directly.

## Verified snow hill run

SB3 SAC was trained for `5e4` steps with the paper architecture and the default
exposed cost weights `Q=diag(1,1)`, `R=0.01`.

- actor-only, 200 steps from `[-5,-1]`: total cost `621.6330`;
- AC4MPC-RTI, `N=20`, `R=20`, `P=5`, `alpha=1`, `M=M_p=1`: total cost
  `611.3258`;
- selected sources: actor `155`, active MPC `37`, parallel MPC `8`.

## Under-specified paper values

The following are not specified numerically in `main.tex` and are therefore
configuration/checkpoint inputs:

- snow hill `Q` and `R`;
- SAC checkpoints at `5e4` and `1e5` steps;
- autonomous-driving model details, random road generation, vehicle parameters,
  obstacle sizes, and cost weights.

The autonomous-driving experiment cannot be strictly reproduced from
`main.tex` alone. This project therefore implements the fully specified snow
hill controlled system and keeps the solver/controller interfaces general.
