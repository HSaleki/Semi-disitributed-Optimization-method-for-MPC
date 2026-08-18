"""Plot the CSV written by the C++ platoon example.

    python plot_csv.py run.csv           # opens a window
    python plot_csv.py run.csv out.png   # writes an image instead

The CSV columns are: t, agent, pos, vel, acc, rho, iter, nAc, central, local.
Needs pandas and matplotlib:  pip install pandas matplotlib
"""

import sys

import matplotlib
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "run.csv"
out = sys.argv[2] if len(sys.argv) > 2 else None
if out:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv(path)
agents = sorted(d["agent"].unique())
# one row per time step for the scalar diagnostics (they repeat across agents)
s = d.drop_duplicates("t").sort_values("t")
col = plt.cm.viridis([i / max(len(agents) - 1, 1) * 0.85 for i in range(len(agents))])

fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))

for k, a in enumerate(agents):
    g = d[d["agent"] == a].sort_values("t")
    ax[0, 0].plot(g["t"], g["pos"], color=col[k], label=f"car {a}")
    ax[0, 1].plot(g["t"], g["vel"], color=col[k])
    ax[0, 2].plot(g["t"], g["acc"], color=col[k])
ax[0, 0].set(xlabel="step", ylabel="position [m]", title="positions")
ax[0, 0].legend(fontsize=8)
ax[0, 1].set(xlabel="step", ylabel="velocity [m/s]", title="velocities")
ax[0, 2].set(xlabel="step", ylabel="acceleration [m/s2]", title="accelerations")

# gap to the vehicle ahead: car 0 follows nothing plotted, so show gaps between
# consecutive cars as position differences
pos = d.pivot(index="t", columns="agent", values="pos")
for k in range(1, len(agents)):
    ax[1, 0].plot(pos.index, pos[agents[k - 1]] - pos[agents[k]],
                  color=col[k], label=f"{agents[k-1]}->{agents[k]}")
ax[1, 0].axhline(30, ls="--", c="r", lw=1.2, label="D_hard")
ax[1, 0].axhline(35, ls=":", c="g", lw=1.2, label="D_ref")
ax[1, 0].set(xlabel="step", ylabel="gap [m]", title="inter-vehicle gaps")
ax[1, 0].legend(fontsize=8)

ax[1, 1].semilogy(s["t"], s["rho"].clip(lower=1e-18), "o-", ms=3)
ax[1, 1].set(xlabel="step", ylabel="rho", title="KKT residual")

a1 = ax[1, 2]
a1.plot(s["t"], s["central"], label="central")
a1.plot(s["t"], s["local"], "--", label="local")
a1.set_yscale("log")
a1.set(xlabel="step", ylabel="doubles/step", title="data transferred")
a2 = a1.twinx()
a2.plot(s["t"], s["iter"], ":k", lw=1.2)
a2.set_ylabel("iterations K")
a2.set_yscale("log")
a1.legend(fontsize=8, loc="upper right")

for row in ax:
    for a_ in row:
        a_.grid(alpha=0.3)
fig.tight_layout()

if out:
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
else:
    plt.show()
