# Python Port — Work in Progress

> **Status: work in progress.** This is an ongoing Python port of the MATLAB
> implementation. Both implementations are currently under development, and
> the MATLAB version is the current baseline for development and
> comparison rather than a finalized reference implementation.


Single file, `dmpc.py`. Requires `numpy` and `scipy`; `matplotlib` only for the plots.

## Run

```bash
python3 dmpc.py --help          # all options
python3 dmpc.py                 # 10 closed-loop steps, then plots
python3 dmpc.py 50              # 50 steps
python3 dmpc.py 25 -N 30 -M 8   # horizon 30, eight followers
python3 dmpc.py 25 --save       # write platoon.png instead of opening a window
python3 dmpc.py 25 --no-plot --quiet
```

Nothing is interactive: the script never prompts. Every input is either a command
line argument or a default inside `run_platoon`. The positional argument is the
number of closed-loop steps; `-N`, `-M` and `--ts` set the horizon, the number of
followers and the sample time. Initial conditions, bounds, cost weights,
`D_hard` and `D_ref` are hardcoded at the top of `run_platoon` — edit them there,
or import the function and pass your own problem to `solve()`.

## What it prints

One line per closed-loop step:

```
t=  0  K= 193  rho=7.869e-15  |A_c|=  1  central= 197406  local=  197406
```

| field | meaning |
| --- | --- |
| `t` | closed-loop step index, `0 .. n_iter-1` |
| `K` | outer iterations the solver needed at this step |
| `rho` | relative KKT residual of the returned point, computed with no reference solver |
| `\|A_c\|` | coupled rows active at exit |
| `central` / `local` | doubles crossing the central and local interfaces during this step |

`rho` is the max over stationarity, primal feasibility (equality and inequality),
dual feasibility and complementarity, each relatively scaled. It is the only
number here that stands alone: it certifies the returned point without comparing
against anything.

Then a three-line summary: worst `rho`, mean/max iterations, mean traffic.

## Expected behaviour

A cold start costs on the order of a hundred iterations. From roughly `t = 2`
onward the time-shifted warm start should give `K = 1` — the active set is already
correct and the homotopy closes in one step. If `K` stays large after the first
few steps, the warm start is wrong, not the solver.

## Plots

Plotting is on by default; pass `--no-plot` to skip it. Six panels: positions,
velocities, accelerations, inter-vehicle gaps against `D_hard` and `D_ref`, `rho`
per step, and traffic per step with `K(t)` on the right axis.

If no window appears, you are either on a headless machine (use `--save`, which
switches the backend to Agg and writes `platoon.png`) or matplotlib is missing, in
which case the run prints a note and continues.

## Solver API

```python
sol = solve(data)          # data: dict of lists, one entry per agent
sol["status"]              # "optimal" or "max_iters" -- check this first
sol["z"]                   # list of primal solutions, or None if not converged
sol["dual_l"], sol["dual_c"], sol["active_l"], sol["active_c"], sol["iter"]
rho = kkt_residual(data, sol)
```

`data` keys: `Q`, `p`, `G`, `g`, `H`, `h`, `A` (lists, per agent), `b`, `z0`,
`loc_active`, `loc_dual`, `coup_active`, `coup_dual`. All `A_i` must share one row
space — row `k` of every `A_i` refers to the same coupled constraint.

## Known differences from the MATLAB version

Both are deliberate, and both are fixes rather than translations:

1. `solve()` returns a `status` field and sets `z = None` on non-convergence. The
   MATLAB driver returns silently with empty cells, which the caller can mistake
   for a solution.
2. `send_change` logs the actual `dS`/`drho` payload. The MATLAB version logs
   `S_i`/`rho_i` at that call site — same element count, wrong array.

Consequently the communication totals are **not** comparable across the two
implementations: here both ends of every channel are logged, so central and local
totals agree by construction, whereas in MATLAB several channels are logged on one
side only.

## Not yet done

- No iterate-by-iterate comparison against the MATLAB reference.
- No test suite; correctness is currently checked only by `rho` and by a handful
  of random problems in `_random_problem`.
- `LDLSolve` refactors the reduced Hessian from scratch on every active-set
  change; the incremental QR update path from the MATLAB version is not ported.
- Long horizons are untested here. The conditioning growth of the reduced Hessian
  with `N` applies to this implementation exactly as it does to the MATLAB one.
