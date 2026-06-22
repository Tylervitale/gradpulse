"""Regression guard for the IRB experiment-DESIGN logic (examples/run_irb_on_braket.py).

Max RB length is bounded by the MIN of two limits, and the right fit depends on the device:
  * gate error -- a good (0.4%) gate decays slowly, so a no-decoherence model says "go long"
    (and in that idealized, SYMMETRIC-readout regime a fixed 1/d asymptote tightens the fit);
  * T2 / circuit duration -- on REAL hardware idle decoherence floors any sequence longer
    than ~T2/clifford-time, so the useful max is much shorter (Cepheus: ~44, not 128), and
    real asymmetric readout shifts the asymptote off 1/d so the fit must leave it FREE.

The no-T2 tests below use a deterministic single-exp synthetic (the regime where fixed-1/d
helps); the T2-aware behaviour is checked via suggest_lengths(t2_us=...). The full on-device
model is examples/cepheus_irb_resolution_study.py (MC over the real 11520-Clifford group with
T2 + asymmetric readout, validated against the 2026-06-20 Cepheus canary).
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_EX = pathlib.Path(__file__).resolve().parents[1] / "examples" / "cepheus" / "run_irb_on_braket.py"


@pytest.fixture(scope="module")
def irb():
    spec = importlib.util.spec_from_file_location("irb_example", _EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_suggest_lengths_reaches_floor_deeper_for_better_gates(irb):
    """The ladder must reach the asymptote (to pin it for a free fit). A smaller CZ error
    decays slower, so it needs LONGER sequences to floor (gate-limited, no T2 here)."""
    good = irb.suggest_lengths(0.004)      # Cepheus-like
    typical = irb.suggest_lengths(0.01)
    poor = irb.suggest_lengths(0.02)
    assert max(good) > max(typical) > max(poor)   # better gate -> deeper to floor
    assert max(good) >= 150                         # ~297
    assert max(poor) <= 100                         # ~58
    for lad in (good, typical, poor):              # always a doubling ladder from 1
        assert lad[0] == 1 and sorted(lad) == lad


def test_suggest_lengths_t2_sets_the_floor_location(irb):
    """T2 sets WHERE the floor is, it does not forbid reaching it: shorter T2 -> floor at
    smaller m -> shorter ladder. Cepheus (0.4%, 14us) lands at ~128 (the validated optimum),
    NOT capped short."""
    cepheus = irb.suggest_lengths(0.004, t2_us=14.3)
    long_t2 = irb.suggest_lengths(0.004, t2_us=60.0)
    no_t2 = irb.suggest_lengths(0.004)
    assert 96 <= max(cepheus) <= 160                       # ~128, matches the MC optimum
    assert max(long_t2) > max(cepheus)                     # better coherence -> deeper floor
    assert max(no_t2) > max(long_t2)                       # no decoherence -> deepest


def _synth(irb_mod, lengths, alpha, A, B, seeds, shots, rng, interleaved):
    """Per-sequence survivals for an ideal single-exp decay A*alpha^m + B plus
    binomial shot noise -- the fit's input format."""
    out = []
    for m in lengths:
        s = A * alpha ** m + B
        for _ in range(seeds):
            obs = rng.binomial(shots, min(max(s, 0.0), 1.0)) / shots
            out.append({"length": m, "survival": float(obs),
                        "interleaved": interleaved})
    return out


def _spread(irb_mod, lengths, asymptote, seed):
    """Std + mean of the extracted r_CZ over 40 noise realizations (% units)."""
    r_true, a_ref, A, B = 0.004, 0.99, 0.66, 0.25     # 0.66 = SPAM-cut amplitude
    a_int = a_ref * (1.0 - r_true / 0.75)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(40):
        ref = _synth(irb_mod, lengths, a_ref, A, B, 15, 400, rng, False)
        intl = _synth(irb_mod, lengths, a_int, A, B, 15, 400, rng, True)
        rs.append(irb_mod.fit_irb(ref, intl, lengths, n_boot=1,
                                  asymptote=asymptote)["r_cz"])
    return np.array(rs) * 100.0


def test_fixed_asymptote_helps_in_the_no_T2_symmetric_regime(irb):
    """IDEALIZED no-T2, SYMMETRIC-readout regime ONLY (B=0.25 exactly): here a long ladder
    + fixed 1/d asymptote is ~30x tighter than short + free. On real hardware T2 forbids the
    long ladder and asymmetric readout breaks the fixed asymptote -- see the study + the
    --fixed-asymptote caveat; this test just guards that the fixed-asymptote OPTION works."""
    short_free = _spread(irb, [1, 2, 4, 8, 16, 32], None, seed=0)
    long_fixed = _spread(irb, [1, 2, 4, 8, 16, 32, 64, 128], 0.25, seed=0)
    # long+fixed: unbiased and tight (measured mean 0.407%, std 0.025%)
    assert abs(long_fixed.mean() - 0.400) < 0.05      # within 0.05% of truth
    assert long_fixed.std() < 0.10                    # < 0.10% (1-sigma)
    # short+free: far noisier (measured 29x); guard a conservative 5x
    assert short_free.std() > 5.0 * long_fixed.std()


def test_fixed_asymptote_recovers_clean_decay_exactly(irb):
    """With noiseless data the fixed-asymptote fit recovers the injected r_CZ to the
    alpha-grid resolution (sanity that fixing b=1/d introduces no bias)."""
    r_true, a_ref, A, B = 0.004, 0.99, 0.66, 0.25
    a_int = a_ref * (1.0 - r_true / 0.75)
    lengths = [1, 2, 4, 8, 16, 32, 64, 128]
    rng = np.random.default_rng(0)
    ref = _synth(irb, lengths, a_ref, A, B, 4, 10_000_000, rng, False)
    intl = _synth(irb, lengths, a_int, A, B, 4, 10_000_000, rng, True)
    out = irb.fit_irb(ref, intl, lengths, n_boot=1, asymptote=0.25)
    assert abs(out["r_cz"] - r_true) < 2e-4


def test_fit_resolved_flags_clamped_fits(irb):
    """A short-ladder fit whose r_cz is statistically consistent with 0 must read UNRESOLVED
    (this clamp, trusted as a real fidelity, is what wasted the first $33 sweep)."""
    assert irb.fit_resolved(0.012, 0.002) is True        # clear signal: 6 sigma from 0
    assert irb.fit_resolved(0.0, 0.05) is False           # clamped to 0
    assert irb.fit_resolved(1e-6, 0.02) is False          # ~0 with a big error bar
    assert irb.fit_resolved(0.001, 0.004) is False        # within 3 sigma of 0 -> untrustworthy


def test_select_best_peak_uses_survival_not_clamped_rcz(irb):
    """THE regression for the $33 failure: a destroyed peak whose noisy fit clamped r_cz->0
    must NOT be chosen. Max-survival picks the genuinely good peak; the old min-r_cz logic
    would have picked the clamped-bad one."""
    results = [
        {"peak": 0.10, "r_cz": 0.05, "r_cz_std": 0.01, "surv_lmax": 0.62},   # GOOD: high survival
        {"peak": 0.24, "r_cz": 0.00, "r_cz_std": 0.19, "surv_lmax": 0.27},   # BAD: clamped r_cz=0
        {"peak": 0.17, "r_cz": 0.30, "r_cz_std": 0.02, "surv_lmax": 0.31},   # over-rotated
    ]
    best = irb.select_best_peak(results)[0]
    assert best["peak"] == 0.10                            # max-survival -> the good peak
    # the OLD selector (min r_cz) would have crowned the destroyed, clamped peak:
    assert min(results, key=lambda d: d["r_cz"])["peak"] == 0.24
    # and that clamped fit is correctly flagged untrustworthy
    assert irb.fit_resolved(0.0, 0.19) is False


def _physical_flux_pulse(tmp_path):
    """A tiny rest-0 physical-flux activation (won't trip the binding guard)."""
    p = tmp_path / "pulse.npy"
    np.save(p, -0.6 * np.sin(np.pi * np.linspace(0, 1, 32)))   # 0 at both ends, bipolar-signed
    return str(p)


def test_virtualz_sweep_offline_dry_run(irb, tmp_path, monkeypatch, capsys):
    """STAGE 2 virtual-Z cal: the offline dry-run builds every circuit, passes the
    return-to-|00> gate, and serializes BOTH virtual-Z frame shifts -- end-to-end, no spend."""
    # "Offline" = no QPU spend, not no SDK: this dry-run builds a Braket PulseSequence and
    # asserts its serialized OpenPulse carries both shift_phase ops, so it needs the SDK.
    pytest.importorskip("braket.pulse", reason="needs amazon-braket-sdk")
    argv = ["prog", "--cal-virtualz-sweep", "--qubits", "16", "25",
            "--pulse-file", _physical_flux_pulse(tmp_path), "--cz-peak", "0.226",
            "--vz-grid", "0", "1.0", "2.0", "--lengths", "2", "8", "--seeds", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    irb.main()                                  # offline; raises SystemExit if anything is wrong
    out = capsys.readouterr().out
    assert "CAL VIRTUAL-Z SWEEP" in out
    assert "ideal return-to-|00> check: PASS" in out
    assert "shift_phase ops for 2 nonzero phases = 2 (PASS)" in out


def test_virtualz_sweep_requires_peak(irb, tmp_path, monkeypatch):
    """Stage 2 is meaningless without the Stage-1 peak -- it must refuse to run."""
    argv = ["prog", "--cal-virtualz-sweep", "--qubits", "16", "25",
            "--pulse-file", _physical_flux_pulse(tmp_path),
            "--lengths", "2", "8", "--seeds", "1"]      # no --cz-peak / --native-cal-file
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        irb.main()


def test_virtualz_sweep_circuit_count(irb):
    """Cost/count math: ref + len(grid)**2 * intl (JOINT 2-D phi0 x phi1 grid -- the phases are
    coupled, so sequential 1-D gets stuck; rehearsal-validated in cepheus_closed_loop_cal.py)."""
    lengths, seeds = [2, 4, 8], 2
    ref = irb.bb.native_rb_sequences(lengths, seeds, seed=0, interleaved=False)
    intl = irb.bb.native_rb_sequences(lengths, seeds, seed=0, interleaved=True)
    grid = [0.0, 1.0, 2.0, 3.0]
    n_circ = len(ref) + len(grid) ** 2 * len(intl)
    assert n_circ == len(ref) + 16 * len(intl)         # 4x4 = 16 interleaved blocks
