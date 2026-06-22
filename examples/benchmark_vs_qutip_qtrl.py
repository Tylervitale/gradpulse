"""Head-to-head benchmark: gradpulse's engine vs qutip-qtrl's analytic GRAPE on the
IDENTICAL control problem.

    python examples/benchmark_vs_qutip_qtrl.py

Same drift, same control operators, same target unitary, same time grid, same
optimizer class (L-BFGS) -- so the only difference is the engine (gradpulse's autodiff
gradients vs qutip-qtrl's analytic ones). Reports achieved fidelity, wall-clock, and
forward evaluations for both.

The honest reading: this is a SOUNDNESS check on qutip-qtrl's own home turf -- a tiny
CLOSED-system unitary synthesis, exactly what its analytic GRAPE is purpose-built for.
It is not meant to show gradpulse is faster (it isn't: the analytic gradient is ~2x
quicker on a problem this small). It shows the autodiff optimizer reaches the SAME
optimum to machine precision in the same order-of-magnitude wall-clock even on the
competitor's best case -- sound, not pathologically slow. That ~2x is the fixed price
of autodiff's generality, and it buys what this benchmark DELIBERATELY does not
exercise and qutip-qtrl does not do natively: OPEN-system (Lindblad) optimization,
arbitrary differentiable penalties, and an independent cross-check. If you genuinely
need small closed-system unitary synthesis at raw speed, use qutip-qtrl. Needs the
``[benchmark]`` extra.
"""
from gradpulse import benchmark as bm

for gate in ("cnot", "iswap"):
    bm.run_benchmark(gate, seed=0)
