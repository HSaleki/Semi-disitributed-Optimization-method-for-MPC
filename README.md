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

>  **Current status:** The Python and C++ ports are still being validated against the MATLAB implementation. They run on the included platoon example. There is currently no automated test suite.


## Example: Vehicle Platoon MPC

The repository includes a closed-loop MPC simulation with one leader and \(M\) follower vehicles.

Each vehicle is modelled as a triple integrator,

$$
x =
\begin{bmatrix}
p & v & a
\end{bmatrix}^{\mathsf T}
$$

with jerk as the control input. Each follower tries to maintain a desired distance from the vehicle ahead while tracking the leader's velocity.

The default example uses:

| Parameter | Value |
|---|---:|
| Followers | 5 |
| Prediction horizon | 20 |
| Sampling time | 1 s |
| Desired velocity | 20 m/s |
| Desired spacing | 35 m |
| Minimum spacing | 30 m |
| Vehicle model | Triple integrator |
| Control input | Jerk |

The problem includes:

- jerk, velocity and acceleration bounds;
- local MPC constraints for each vehicle;
- coupled spacing constraints between consecutive vehicles;
- a closed-loop simulation in which the first MPC input is applied at every step.

The spacing constraints are

$$
p_i(k)-p_{i-1}(k) \leq -D_{\mathrm{hard}},
$$

which creates the coupling between the local QPs.

The previous MPC solution is shifted and used to warm-start the next problem. This is the main reason the solver can reduce the number of active-set iterations substantially after the initial solve.

The example is also used to compare the three implementations. The solver output records the number of active-set iterations, the KKT residual, the number of active coupled constraints and the amount of data exchanged between the local and central nodes.


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

## Example Output

A typical solver output contains one line per closed-loop MPC step:

```text
t=  0  K= 193  rho=2.572e-15  |A_c|=  1  central= 197406  local= 197406
```

where:

- `t` is the MPC step;
- `K` is the number of active-set iterations;
- `rho` is the relative KKT residual;
- `|A_c|` is the number of active coupled constraints;
- `central` and `local` are the numbers of scalar values transferred through the corresponding interfaces.

The exact iteration count can differ between implementations when ratio tests encounter ties.

## KKT Residual

The solver reports a relative KKT residual for the returned solution:

\[
\rho =
\max\left\{
\frac{\|Qz+p+G^\top\lambda+H^\top\mu+A^\top\nu\|_\infty}
{\max(1,\|p\|_\infty)},
\frac{\|Gz-g\|_\infty}{\max(1,\|g\|_\infty)},
\frac{[\max r]_+}{s_p},
\frac{[-\min y]_+}{s_d},
\frac{\|y\odot r\|_\infty}{s_p s_d}
\right\}.
\]

This combines stationarity, equality feasibility, inequality feasibility, dual feasibility and complementarity.

Unlike an iteration count, the residual does not require a reference implementation. It directly measures the quality of the returned QP solution.

## Warm Start

The MPC example uses the previous solution as the initial point for the next QP.

The primal solution is shifted by one prediction step. The terminal part is filled using a clipped LQR control. The local active sets and dual variables are shifted as well.

For the included platoon benchmark, the initial cold solve requires around 193 iterations, while the subsequent warm-started problems converge in one iteration.

This behaviour is important for the intended MPC application: consecutive QPs differ mainly because the measured state and reference move forward, so most of the previous active set remains useful for the next problem.


## Cross-implementation check

The MATLAB, Python and C++ versions agree on iteration counts and communication volume on the
platoon benchmark.
They can differ by an iteration or two on steps where the ratio test has a tie, because the
underlying LDL / QR factorizations break ties differently across libraries (work in progress). 

## Solver Interface

The solver takes the local QP data and a warm start and returns:

- the primal solution;
- local and coupled dual variables;
- local and coupled active sets;
- the number of active-set iterations;
- convergence information.

The coupled matrices \(A_i\) must use the same row ordering: row \(k\) of every \(A_i\) must correspond to the same global coupled constraint.

A cold start is also possible by providing an initial point, empty active sets and zero multipliers. The homotopy procedure can then move the point to the solution, although this normally requires more iterations.

See the README in each implementation directory for the exact interface.

## Communication

The current implementation is intended to represent a semi-distributed architecture.

Local nodes keep their own QP data and perform the main KKT calculations. The central node handles the coupled part of the problem.

For each active coupled constraint set, the local nodes send the quantities required to assemble the Schur complement and the coupling residual. The central node then returns the coupled multiplier update.

The code includes a transaction logger that counts the scalar values exchanged between local and central nodes. This allows the platoon example to report communication volume alongside solver iterations.

The current code does not simulate network delays, packet loss or asynchronous communication.


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


