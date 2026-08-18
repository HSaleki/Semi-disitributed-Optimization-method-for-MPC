# Distributed Active-Set QP Solver

A primal active-set solver for coupled convex QPs, split across one central node and $M$ local
nodes. Coupling is eliminated by a Schur complement assembled from local contributions; the
active set is traced by a homotopy in $\sigma$ that reaches the original problem data exactly at
$\sigma = 0$. MATLAB, no external dependencies in the solver itself.

The bundled example is a vehicle-platoon MPC problem: $M$ followers, triple-integrator dynamics,
box constraints on jerk / velocity / acceleration, and a chain of inter-vehicle spacing
constraints that couples consecutive agents.

## Problem class

$$
\min_{z_1,\dots,z_M} \; \sum_{i=1}^{M} \tfrac12 z_i^\top Q_i z_i + p_i^\top z_i
\qquad\text{s.t.}\qquad
G_i z_i = g_i,\quad H_i z_i \le h_i,\quad \sum_{i=1}^{M} A_i z_i \le b
$$

with $Q_i \succ 0$. Local constraints are private to agent $i$; the last block is the only
coupling and is the only thing that crosses the network.

## Method

**Local KKT solve (null-space).** With $W_i$ the active local set, the active constraint matrix
$\bar G_i = [\,G_i;\, H_i(W_i,:)\,]$ is factored by sequential Householder QR into a range-space
basis $Y_i$ and a null-space basis $Z_i$. The reduced Hessian $Z_i^\top Q_i Z_i$ is factored once
per active-set change by LDL$^\top$, giving the range and null components

$$
s_{Y_i} = T_i^{-\top} r_{gh,i},
\qquad
\bar s_{Z_i} = (Z_i^\top Q_i Z_i)^{-1}\!\left(Z_i^\top r_{p,i} - Z_i^\top Q_i Y_i s_{Y_i}\right),
\qquad
\Delta \bar x_i = Q_i^{\text{orth}} \begin{bmatrix} s_{Y_i} \\ \bar s_{Z_i}\end{bmatrix}.
$$

**Coupling (Schur complement).** Each agent forms, from the same factorization,

$$
M_i = (Z_i^\top Q_i Z_i)^{-1} Z_i^\top A_i(\mathcal{A}_c,:)^\top,
\qquad
S_i = \left(Z_i^\top A_i(\mathcal{A}_c,:)^\top\right)^{\!\top} M_i,
\qquad
\rho_i = A_i(\mathcal{A}_c,:)\,\Delta\bar x_i ,
$$

and ships $(S_i, \rho_i)$ upward. The central node solves

$$
\Big(\textstyle\sum_i S_i\Big)\,\Delta\nu = \sum_i \rho_i - r_b(\mathcal{A}_c)
$$

and broadcasts $\Delta\nu$. Each agent then corrects its own step,
$s_{Z_i} = \bar s_{Z_i} - M_i \Delta\nu$, so the coupled multiplier is the only quantity that
has to be agreed globally. No agent ever sees another agent's $Q_i, p_i, H_i, h_i$ or $z_i$.

**Homotopy.** The right-hand sides are perturbed so that the warm-start point is exactly optimal
at $\sigma = 1$, and the perturbation is retired linearly:

$$
h_\tau = h - \sigma\, d_h, \qquad p_\tau = p - \sigma\, d_p, \qquad b_\tau = b - \sigma\, r_b ,
\qquad \sigma: 1 \to 0 .
$$

$\sigma$ is the primary state variable (not $\tau = 1-\sigma$), so at termination
$b_\tau \equiv b$ bitwise and the returned point solves the original QP, not a nearby one. Each
outer iteration takes the smallest step admitted by the primal ratio test (an inactive row
becoming tight) and the dual ratio test (an active multiplier reaching zero), over all agents and
the coupled set; the limiting agent is the only one that changes its active set.

**Rank deficiency.** If the active set becomes linearly dependent, the leaving row is chosen by a
dual ratio test on the dependency null vector rather than by factorization order, so that
$H^\top\mu + G^\top\lambda$ is left unchanged and $\mu \ge 0$ is preserved — the drop does not
perturb the Lagrangian gradient. The same test governs `dropRows` on the coupled side.

## Layout

```
src/
  decenSolverx0.m       driver: builds the nodes, runs the outer loop, returns the solution
  LocalNodex0.m         per-agent QP: QR/LDL factorization, ratio tests, Schur contribution
  CentralNodex0.m       aggregation, S dnu = rho solve, global ratio test, homotopy stepping
  TransactionLogger.m   singleton counting doubles/integers crossing each interface
examples/
  starter.m             closed-loop platoon MPC, warm start by time shift, diagnostics + plots
tests/
  smokeTest.m           random coupled QP, cold start, KKT residual assertion
```

`src/` has no external dependencies: every helper is a local function inside the file that uses
it, and the solver calls core MATLAB only.

## Usage

```matlab
data.Q = Q;  data.p = p;      % 1 x M cells
data.G = G;  data.g = g;      % local equalities
data.H = H;  data.h = h;      % local inequalities
data.A = A;  data.b = b;      % coupling: sum_i A_i z_i <= b
data.z0 = z0;                 % warm-start primal, 1 x M cell

data.loc_activeset  = loc_activeset;   % 1 x M cell of logical vectors
data.loc_dual       = loc_dual;        % 1 x M cell, [lambda; mu] stacked
data.coup_activeset = coup_activeset;  % logical, length(b)
data.coup_dual      = coup_dual;       % nu, length(b)

solution = decenSolverx0(data);
```

Returned: `solution.z` (primal per agent), `solution.dual_l` (`[lambda; mu]` per agent),
`solution.dual_c` ($\nu$), `solution.active_l`, `solution.active_c`, `solution.iter`,
`solution.res` (per-node KKT residual vectors, coupled residual last) and `solution.log`
(per-iteration active-set size, limiting agent, $\tau$, coupled-set history).

Cold start: pass `z0` feasible for the local constraints, all-false active sets and zero duals.
The homotopy handles the rest; a cold start simply costs more outer iterations.

Run the example, or the test:

```matlab
cd examples; starter
cd tests;    smokeTest
```

## Verification

`examples/starter.m` computes, at every closed-loop step, the relative KKT residual of the
returned point — with no reference solution involved:

$$
\rho = \max\left\{
\frac{\|Qz+p+G^\top\lambda+H^\top\mu+A^\top\nu\|_\infty}{\max(1,\|p\|_\infty)},\;
\frac{\|Gz-g\|_\infty}{\max(1,\|g\|_\infty)},\;
\frac{[\max r]_+}{s_p},\;
\frac{[-\min y]_+}{s_d},\;
\frac{\|y \odot r\|_\infty}{s_p s_d}
\right\}
$$

It additionally compares against `quadprog` on the assembled monolithic QP
($\|z_d - z^\star\|_\infty$), reports the symmetric difference between the identified active set
and the reference one, and counts weakly active rows — where $W^\star$ is not unique and a nonzero
symmetric difference is not an error.

Iteration counts are reported against the warm-start lower bound
$k_{\min} = |W_{\text{warm}} \oplus W^\star|$. Each outer iteration changes at most one row, so
$K \ge k_{\min}$ always; $K - k_{\min}$ is the quantity worth minimizing.

## Communication

`TransactionLogger` counts array elements crossing each interface. Per outer iteration, with
$m$ the number of active coupled rows and $n_c$ the total number of coupled rows:

| direction | payload | doubles |
| --- | --- | --- |
| local $\to$ central | $S_i, \rho_i$ (all agents on an active-set change, otherwise only the limiting agent) | $m^2 + m$ each |
| local $\to$ central | $A_i \Delta x_i$, $\Delta\tau_i$ | $M(n_c + 1)$ |
| central $\to$ local | $\Delta\nu$ | $M m$ |

so the traffic is $\mathcal{O}(M m^2)$ per iteration and the reduced Hessian never leaves the
agent. `examples/starter.m` plots the per-step totals.

## Requirements

MATLAB R2019b or newer (`tiledlayout`, string arrays). The solver in `src/` uses core MATLAB
only. The example additionally needs the Control System Toolbox (`dlqr`) and the Optimization
Toolbox (`quadprog`) — the latter purely as an independent reference; the solver does not call it.

## Known limitations

- **No proven anti-cycling rule.** `n_zero_budget` is a cycle *detector*: it converts a silent
  hang on a zero-step cascade into a raised error. It is not a bound on cascade length.
- **Conditioning with the horizon.** For the triple integrator the reduced Hessian conditioning
  grows roughly as $\kappa(S) \propto N^6$; long horizons need a deviation-coordinate or
  prestabilized formulation rather than tolerance tuning.
- **Degenerate data.** In the example, $D_{\text{ref}} = D_{\text{hard}}$ places the unconstrained
  minimizer exactly on all $MN$ coupling boundaries, making every coupled row weakly active and
  the ratio tests degenerate. Keep $D_{\text{ref}} > D_{\text{hard}}$.
- **Logging gaps.** `TransactionLogger.logData` returns early on non-numeric input, so logical
  payloads (the broadcast active set) are not counted, and the central node logs no `send` for
  `receiveDeltatau`. Reported downlink volume is a lower bound.

## Citation

<!-- Fill in before publishing. -->

```bibtex
@misc{distributed-mpc-qp,
  author = {},
  title  = {Distributed Active-Set QP Solver},
  year   = {2026},
  url    = {}
}
```

## License

See `LICENSE`.
