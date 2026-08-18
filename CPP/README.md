# Distributed active-set QP solver — C++ (early WIP)

> **Very early work in progress.** This is a rough C++ port of a MATLAB solver.
> It compiles and runs and reproduces the expected behaviour on one benchmark, but
> it has not been tested against the MATLAB reference case by case, has no test
> suite, and the interface will change. Do not rely on it or cite numbers from it.

A primal active-set solver for coupled convex QPs

    min  sum_i  1/2 z_i' Q_i z_i + p_i' z_i
    s.t. G_i z_i  = g_i          (local)
         H_i z_i <= h_i          (local)
         sum_i A_i z_i <= b      (coupled)

with the coupling eliminated by a Schur complement and the active set traced by a
homotopy. The bundled example is a vehicle-platoon MPC problem.

## Files

- `dmpc.hpp` — the solver, header-only.
- `platoon.cpp` — the closed-loop platoon example (has `main`).
- `plot_csv.py` — plots the CSV the example writes.
- `CMakeLists.txt`, `Makefile` — two ways to build.

## Dependency

[Eigen 3](https://eigen.tumblr.com), header-only. Ubuntu: `apt install libeigen3-dev`.
macOS: `brew install eigen`. Or unzip Eigen anywhere and pass its path to the
compiler with `-I`.

## Build and run

```bash
cmake -B build && cmake --build build     # -> build/platoon
# or:  make            (edit EIGEN= in the Makefile if Eigen is elsewhere)

./platoon 25                  # 25 closed-loop steps
./platoon 25 -N 30 -M 8       # horizon 30, eight followers
./platoon 25 --csv run.csv    # write a CSV for plotting
```

Building in an IDE (Code::Blocks, etc.): compile `platoon.cpp`, add Eigen's
folder to the include paths, enable `-std=c++17`, and build in **Release** — a
Debug build of Eigen code is huge and slow. On MinGW a Debug build also needs
`-Wa,-mbig-obj` to get past a "file too big" error.

## Output

```
t=  0  K= 193  rho=2.572e-15  |A_c|=  1  central= 197406  local=  197406
```

`t` step index, `K` outer iterations, `rho` relative KKT residual of the returned
point (no reference solver needed), `|A_c|` active coupled rows, then the number
of doubles crossing each interface.

## Plots

The program draws nothing; it writes a CSV and you plot that.

```bash
pip install pandas matplotlib
./platoon 25 --csv run.csv
python plot_csv.py run.csv            # window
python plot_csv.py run.csv out.png    # image file
```

## Changing the problem

The platoon parameters — horizon, follower count, gaps, bounds, cost weights,
initial states — are constants at the top of `main` in `platoon.cpp`. For a
different problem entirely, include `dmpc.hpp`, fill a `dmpc::Problem` with your
own `Q,p,G,g,H,h,A,b` and warm start, and call `dmpc::solve`.

## Status / not done

- Not verified against the MATLAB reference step by step.
- No tests.
- Dense only; the incremental factorization-update path from the MATLAB version
  is not ported.
- Eigen's pivoted `LDLT` means the arithmetic is not bitwise identical to MATLAB,
  so tie-breaking in the ratio tests can differ by an iteration or two.
