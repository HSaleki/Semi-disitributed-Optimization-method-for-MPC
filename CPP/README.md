# Distributed active-set QP solver — C++

C++ version of the distributed active-set QP solver. The current code is a work in
progress and is mainly used to compare the implementation with the MATLAB version.

The solver handles coupled convex QPs of the form

    min  sum_i  1/2 z_i' Q_i z_i + p_i' z_i
    s.t. G_i z_i  = g_i
         H_i z_i <= h_i
         sum_i A_i z_i <= b

The local problems are reduced using a Schur complement and the active set is
followed with a homotopy. The example is a closed-loop vehicle platoon MPC problem.

## Files

- `dmpc.hpp` — solver implementation.
- `platoon.cpp` — platoon MPC example.
- `plot_csv.py` — plots the CSV output.
- `CMakeLists.txt`, `Makefile` — build files.

## Dependency

Eigen 3 is required. On Ubuntu:

```bash
sudo apt install libeigen3-dev
```

On macOS:

```bash
brew install eigen
```

If Eigen is installed somewhere else, set the include path in the Makefile or pass it
to the compiler.

## Build and run

With CMake:

```bash
cmake -B build
cmake --build build
```

Or:

```bash
make
```

Run the example:

```bash
./platoon 25
./platoon 25 -N 30 -M 8
./platoon 25 --csv run.csv
```

The first argument is the number of closed-loop steps. `-N` changes the prediction
horizon and `-M` changes the number of followers.

For Code::Blocks or another IDE, add the Eigen include directory and compile with
C++17. A Release build is recommended when using Eigen. On MinGW, a Debug build may
also need `-Wa,-mbig-obj`.

## Output

A typical line looks like:

```text
t=  0  K= 193  rho=2.572e-15  |A_c|=  1  central= 197406  local=  197406
```

Here `K` is the number of solver iterations, `rho` is the relative KKT residual,
`|A_c|` is the number of active coupled constraints, and `central`/`local` count
the number of doubles sent across the corresponding interfaces.

## Plotting

```bash
pip install pandas matplotlib
./platoon 25 --csv run.csv
python plot_csv.py run.csv
```

To save a figure instead:

```bash
python plot_csv.py run.csv out.png
```

## Changing the example

The platoon parameters are near the top of `main` in `platoon.cpp`. The horizon,
number of followers, gaps, bounds, weights, and initial states can be changed there.

For another QP, include `dmpc.hpp`, fill a `dmpc::Problem` with the problem data and
warm start, then call `dmpc::solve`.

## Current status

- The C++ version has not been checked against the MATLAB implementation step by step.
- There are no automated tests yet.
- The implementation is dense.
- The incremental factorization update used in the MATLAB version is not ported yet.
- Eigen and MATLAB can make different choices in some degenerate ratio tests.
