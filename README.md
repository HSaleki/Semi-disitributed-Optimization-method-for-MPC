# Distributed Active-Set QP Solver for Model Predictive Control

A primal active-set solver for coupled convex QPs, split across one central node and $M$ local
nodes. Coupling is eliminated by a Schur complement assembled from local contributions; the
active set is traced by a homotopy in $\sigma$ that reaches the original problem data exactly at
$\sigma = 0$.

This method is especially suited for MPC problems as we use previous solution after applying the input 
and warm the new iteration of problem ehich results in a quick 1 iteration of QP solving.

The same solver is provided in three languages. The MATLAB implementation is the reference; the
Python and C++ versions are ports of it.

| | location | status | needs |
| --- | --- | --- | --- |
| MATLAB | [`matlab/`](matlab/) | reference | MATLAB (+ Control System & Optimization Toolboxes for the example) |
| Python | [`python/`](python/) | port, work in progress | `numpy`, `scipy`, `matplotlib` |
| C++ | [`cpp/`](cpp/) | port, early work in progress | a C++17 compiler, [Eigen 3](https://eigen.tumblr.com) |

> **The ports are work in progress.** They compile, run, and reproduce the reference behaviour on
> the platoon benchmark and on random coupled QPs, but they have not been checked against the
> MATLAB implementation iterate by iterate, have no test suite, and their interfaces will change.
> For anything you intend to cite, use the MATLAB version.

The bundled example is a vehicle-platoon MPC problem: $M$ followers, triple-integrator dynamics,
box constraints on jerk / velocity / acceleration, and a chain of inter-vehicle spacing
constraints that couples consecutive agents.

## Problem class

$$
\min_{z_1,\dots,z_M} \; \sum_{i=1}^{M} \frac12 z_i^\top Q_i z_i + p_i^\top z_i
\qquad\text{s.t.}\qquad
G_i z_i = g_i,\quad H_i z_i \le h_i,\quad \sum_{i=1}^{M} A_i z_i \le b
$$

with $Q_i \succ 0$. Local constraints are private to agent $i$; the last block is the only
coupling and is the only quantity that crosses the network.

## Method

**Local KKT solve (null-space).** With $W_i$ the active local set, the active constraint matrix
$\bar G_i = [\,G_i;\, H_i(W_i,:)\,]$ is factored by sequential Householder QR into a range-space
basis $Y_i$ and a null-space basis $Z_i$. The reduced Hessian $Z_i^\top Q_i Z_i$ is factored once
per active-set change by LDL$^\top$, giving

$$
s_{Y_i} = T_i^{-\top} r_{gh,i},
\qquad
\bar s_{Z_i} = (Z_i^\top Q_i Z_i)^{-1}\!\left(Z_i^\top r_{p,i} - Z_i^\top Q_i Y_i s_{Y_i}\right),
\qquad
\Delta \bar x_i = Q_i^{\text{orth}} \begin{bmatrix} s_{Y_i} \\
                                                    \bar s_{Z_i}\end{bmatrix}.
$$

**Coupling (Schur complement).** Each agent forms, from the same factorization,

$$
M_i = (Z_i^\top Q_i Z_i)^{-1} Z_i^\top A_i(\mathcal{A}_c,:)^\top,
\qquad
S_i = \left(Z_i^\top A_i(\mathcal{A}_c,:)^\top\right)^{\!\top} M_i,
\qquad
\rho_i = A_i(\mathcal{A}_c,:)\,\Delta\bar x_i ,
$$

and sends $(S_i, \rho_i)$ up. The central node solves

$$
\Big(\textstyle\sum_i S_i\Big)\,\Delta\nu = \sum_i \rho_i - r_b(\mathcal{A}_c)
$$

and broadcasts $\Delta\nu$; each agent corrects its own step $s_{Z_i} = \bar s_{Z_i} - M_i\Delta\nu$.
The coupled multiplier is the only quantity agreed globally. No agent sees another agent's
$Q_i, p_i, H_i, h_i$ or $z_i$.

**Homotopy.** Right-hand sides are perturbed so the warm-start point is exactly optimal at
$\sigma = 1$, then the perturbation is retired linearly:

$$
h_\tau = h - \sigma\, d_h, \qquad p_\tau = p - \sigma\, d_p, \qquad b_\tau = b - \sigma\, r_b ,
\qquad \sigma: 1 \to 0 .
$$

$\sigma$ is the state variable (not $\tau = 1-\sigma$), so at termination $b_\tau \equiv b$ bitwise
and the returned point solves the original QP. Each outer iteration takes the smallest step
admitted by the primal ratio test (an inactive row going tight) and the dual ratio test (an active
multiplier reaching zero), over all agents and the coupled set; the limiting agent is the only one
that changes its active set.

**Rank deficiency.** If the active set becomes dependent, the leaving row is chosen by a dual ratio
test on the dependency null vector rather than by factorization order, so $H^\top\mu + G^\top\lambda$
is unchanged and $\mu \ge 0$ is preserved — the drop does not perturb the Lagrangian gradient.

## Quick start

**MATLAB**
```matlab
cd matlab/examples
starter          % closed-loop platoon, diagnostics and plots
```

**Python**
```bash
cd python
pip install numpy scipy matplotlib
python dmpc.py 25            # runs, prints per-step table, then plots
python dmpc.py --help
```

**C++**
```bash
cd cpp
cmake -B build && cmake --build build     # or: make
./platoon 25 --csv run.csv
python plot_csv.py run.csv
```

Each subfolder has its own README with the details.

## What the example reports

One line per closed-loop step, in all three:

```
t=  0  K= 193  rho=2.572e-15  |A_c|=  1  central= 197406  local=  197406
```

`t` step index, `K` outer iterations, `rho` the relative KKT residual of the returned point, `|A_c|`
active coupled rows, then the number of doubles crossing each interface.

`rho` is the one number that stands alone — it certifies the returned point with no reference solver:

$$
\rho = \max \{
\frac{\|Qz+p+G^\top\lambda+H^\top\mu+A^\top\nu\|_\infty}{\max(1,\|p\|_\infty)},\;
\frac{\|Gz-g\|_\infty}{\max(1,\|g\|_\infty)},\;
\frac{[\max r]_+}{s_p},\;
\frac{[-\min y]_+}{s_d},\;
\frac{\|y \odot r\|_\infty}{s_p s_d}
\}
$$

These are the relative stationarity residual, primal feasibility of the equalities, primal feasibility of the inequalities, dual feasibility, and complementarity

Iteration counts are reported against the warm-start lower bound
$k_{\min} = |W_{\text{warm}} \oplus W^\star|$: each outer iteration changes at most one row, so
$K \ge k_{\min}$ always, and $K - k_{\min}$ is the quantity worth minimizing. On the platoon
benchmark the cold start costs order $10^2$ iterations and every subsequent warm-started step
converges in one.

## Cross-implementation check

The MATLAB, Python and C++ versions agree on iteration counts and communication volume on the
platoon benchmark (25 steps: 193 iterations cold, one per step warm, identical traffic).
They can differ by an iteration or two on steps where the ratio test has a tie, because the
underlying LDL / QR factorizations break ties differently across libraries (work in progress). 

## Solver interface

Pass per-agent data and a warm start; get back the primal, the duals, the active sets and the
iteration count. All $A_i$ must share one row space — row $k$ of every $A_i$ refers to the same
coupled constraint — and $Q_i \succ 0$. A cold start is legal (arbitrary $z_0$, empty active sets,
zero duals); the homotopy absorbs it at the cost of more iterations. See any subfolder README for
the exact call.

## Known limitations

- **Degenerate data.** In the example $D_{\text{ref}} = D_{\text{hard}}$ puts the unconstrained
  minimizer exactly on all $MN$ coupling boundaries, making every coupled row weakly active and the
  ratio tests degenerate. Keep $D_{\text{ref}} > D_{\text{hard}}$.

## Scope

This is a *simulation* of a distributed algorithm, not a distributed deployment. The local and
central nodes are objects in one process; "sending" is a method call, and the logger counts the
array elements that is communicated between nodes. What the code demonstrates is the information structure —
each agent reveals only $(S_i, \rho_i, A_i\Delta x_i, \Delta\tau_i)$, never its cost or private
constraints — and the resulting $\mathcal{O}(Mm^2)$ message volume. 


