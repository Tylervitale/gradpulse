"""Validate gradpulse's decoherence MODEL against published superconducting-qubit gates.

gradpulse reports a *simulator* fidelity. The QuTiP and Liouvillian cross-checks prove
that number is computed correctly, but they cannot prove the *model* matches a real
device -- two independent solvers of the same model agree even if the model is wrong.
This script does the missing check: feed gradpulse the published parameters of a real,
characterized two-qubit gate and show its decoherence-limited floor reproduces the
device's published coherence budget.

Data-driven, not hardcoded
--------------------------
Each published gate is a JSON *anchor* in ``examples/anchors/`` carrying both the device
physics and the validation metadata (measured fidelity, T1 limit, gate timing,
provenance, and the ``coherence_limited`` flag). This script just discovers every anchor
and judges it -- adding a device is dropping in a JSON file, with the ground truth living
in curated, cited data rather than in code. The loading/judging logic is the torch-free
``gradpulse.literature`` module (unit-tested in ``tests/test_literature.py``); this file
is only the multi-minute GRAPE driver around it.

What this validates, and what it does not
-----------------------------------------
* Validates: the open-system (T1 / T_phi) model -- the part gradpulse computes exactly.
* Honest framing: a pure T1/T_phi Markovian model is a LOWER BOUND on a measured
  error-per-gate. It omits residual coherent control error, classical and ZZ crosstalk,
  leakage, and non-Markovian noise -- all of which a hardware RB number contains. So the
  floor equals the measured number only when the gate is itself coherence-limited
  (``coherence_limited: true``); otherwise the anchor asks only that the floor be a valid
  lower bound and the residual is reported as un-modelled error.
* Does NOT claim the optimizer finds the *same* pulse the experiment used (pulse shapes
  are non-unique), nor that every error channel is modelled. We do NOT tune a fudge
  factor to force the floor toward a measured number.

Method (per anchor)
-------------------
1. Build three gradpulse profiles differing only in coherence -- none / T1-only / full --
   from the anchor's measured T1/T2 (the latter two derived, see ``anchor_to_profiles``).
2. Run three INDEPENDENT optimizations and read each best_fidelity. Optimizing
   per-assumption gives the best achievable fidelity under each.
3. ``literature.judge`` compares the decoherence-limited floor to the measured fidelity
   under the anchor's ``coherence_limited`` claim and returns PASS/FAIL.

Runtime: a few minutes PER anchor (it runs real GRAPE). Run:
    python examples/validate_against_literature.py
"""
import sys
import warnings
from pathlib import Path

from gradpulse import ParametricCZOptimizer
from gradpulse.literature import (
    analytic_coherence_limit_epg,
    anchor_to_profiles,
    discover_anchors,
    format_report,
    gate_config,
    judge,
    judge_analytic,
    load_anchor,
)

warnings.filterwarnings("ignore")

ANCHOR_DIR = Path(__file__).parent / "anchors"


def _run(prof, gcfg):
    """One independent optimization; returns the best PROCESS fidelity."""
    if gcfg["kind"] != "cz":
        # The judging/loading is gate-agnostic, but this driver only wires the CZ
        # optimizer. Add an iSWAP/target branch here to extend (kept explicit, not silent).
        raise NotImplementedError(
            f"gate kind '{gcfg['kind']}' is not wired into the validation driver yet "
            "(only 'cz'). Extend _run() to target it."
        )
    opt = ParametricCZOptimizer(
        prof, bandwidth_mhz=gcfg["bandwidth_mhz"], use_drag=gcfg["use_drag"],
        drag_order=gcfg["drag_order"], n_channels=gcfg["n_channels"],
        precision=gcfg["precision"],
    )
    return opt.optimize_multi_seed(
        n_slices=gcfg["n_slices"], dt_ns=gcfg["dt_ns"],
        n_seeds=gcfg["n_seeds"], iterations=gcfg["iterations"],
    )["best_fidelity"]


def validate_anchor(path):
    """Load one anchor, run the three optimizations, judge, print the report."""
    anchor = load_anchor(path)
    gcfg = gate_config(anchor)
    prov = anchor["provenance"]
    print(f"\n--- {anchor['name']} ---")
    print(f"    {prov['citation']}")
    if prov.get("doi"):
        print(f"    doi:{prov['doi']}")
    prof_none, prof_t1, prof_full = anchor_to_profiles(anchor)
    # GRAPE-independent analytic floor from the measured T1/T2 and gate duration.
    analytic = analytic_coherence_limit_epg(prof_full, gcfg["n_slices"] * gcfg["dt_ns"])
    if anchor["validation"].get("floor_method", "grape") == "analytic":
        # Device publishes T1/T2/t_g but not per-pair freq/anharm: judge on the closed-form
        # floor (paper-sourced) instead of a GRAPE run with assumed Hamiltonian parameters.
        print("    [analytic-floor anchor: no GRAPE run; floor = paper T1/T2/t_g]")
        verdict = judge_analytic(anchor, analytic)
    else:
        f_coh = _run(prof_none, gcfg)
        f_t1 = _run(prof_t1, gcfg)
        f_full = _run(prof_full, gcfg)
        verdict = judge(anchor, f_coh, f_t1, f_full, analytic_epg=analytic)
    print(format_report(verdict))
    return verdict


def main():
    print("Validating gradpulse's decoherence model against published hardware gates.")
    print("(runs real GRAPE; a few minutes per anchor)\n")
    anchors = discover_anchors(ANCHOR_DIR)
    if not anchors:
        print(f"No anchors found in {ANCHOR_DIR}.")
        return 1
    print(f"Found {len(anchors)} anchor(s) in {ANCHOR_DIR.name}/: "
          f"{', '.join(p.name for p in anchors)}")

    verdicts = [validate_anchor(p) for p in anchors]

    n_pass = sum(v["passed"] for v in verdicts)
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_pass}/{len(verdicts)} anchor(s) passed.")
    for v in verdicts:
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['name']:32s} "
              f"floor/measured = {v['ratio']:.2f}x")
    print("\nThe T1/T_phi floor is a LOWER BOUND on a measured error-per-gate; it equals")
    print("the measured number only for a coherence-limited gate (coherence_limited:true).")
    print("For coherence_limited:false anchors the floor is reported as a valid lower")
    print("bound and the residual as un-modelled (coherent/crosstalk/leakage) error.")
    return 0 if n_pass == len(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
