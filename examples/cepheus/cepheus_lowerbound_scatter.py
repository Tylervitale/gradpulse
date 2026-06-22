"""The full, UNSELECTED Cepheus scatter: gradpulse's predicted coherence floor vs the
device's measured CZ error, for all 160 active pairs. This is the honest, non-circular
view of the validation -- every pair, no selection on the prediction.

The one-sided claim is visible directly: points on or below the diagonal (floor <=
measured) satisfy the lower bound; the cluster ON the diagonal is the coherence-limited
("saturation") regime. The "0.8-1.25x" band is shaded only to show where the floor
saturates the measurement -- it is NOT a selection used to compute the headline.

Reads examples/cepheus/cepheus_grape_sweep_realdur.json (no AWS). Writes paper/cepheus_scatter.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "cepheus_grape_sweep_realdur.json")))
g = {k: v for k, v in R.items() if not k.startswith("_f_coh") and "grape_ratio" in v}
meas = np.array([v["measured_err"] for v in g.values()])
floor = np.array([v["grape_floor"] for v in g.values()])
ratio = floor / meas
N = len(g)

sat = (ratio >= 0.8) & (ratio <= 1.25)          # saturation (coherence-limited) regime
below = floor <= meas                            # one-sided lower bound holds
print(f"{N} pairs | lower bound floor<=measured: {below.sum()}/{N} ({100*below.mean():.0f}%) "
      f"| within widest error bar (<=1.42x): {(ratio<=1.42).sum()}/{N} | median ratio {np.median(ratio):.2f}")

fig, ax = plt.subplots(figsize=(5.2, 5.0))
lo, hi = 8e-4, 1.5e-1
ax.fill_between([lo, hi], [lo * 0.8, hi * 0.8], [lo * 1.25, hi * 1.25],
                color="0.85", label="saturation band (0.8-1.25×)", zorder=0)
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="floor = measured (lower-bound edge)", zorder=1)
ax.scatter(meas[~sat], floor[~sat], s=18, c="#3b6", alpha=0.7,
           label=f"control/crosstalk-limited ({(~sat).sum()})", zorder=2)
ax.scatter(meas[sat], floor[sat], s=22, c="#c33", alpha=0.85,
           label=f"coherence-limited / saturation ({sat.sum()})", zorder=3)
ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
       xlabel="measured CZ error (device interleaved-RB)",
       ylabel="gradpulse predicted coherence floor")
ax.set_title("All 160 Cepheus pairs (unselected)", fontsize=11)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
ax.set_aspect("equal")
fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "paper", "cepheus_scatter.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
