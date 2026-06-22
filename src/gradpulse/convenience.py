"""gradpulse.convenience -- one-call helpers for the most common workflow.

The full control surface lives on the optimizer classes (``ParametricCZOptimizer``
etc.); these wrappers just collapse the standard "make a profile, make an
optimizer, optimize" boilerplate into a single call with sensible defaults, for a
fast first result. Anything beyond the few exposed knobs -> use the classes
directly (see the module docstrings and ``examples/``).
"""
from __future__ import annotations

from typing import Optional


def _optimize_parametric(gate: str, profile=None, *, n_seeds: int = 4,
                         iterations: int = 200, n_slices: int = 150,
                         dt_ns: float = 1.0, bandwidth_mhz: float = 80.0,
                         precision: str = "single", **optimize_kwargs) -> dict:
    """Shared implementation for the parametric-coupler convenience wrappers."""
    from .parametric import ParametricCouplerProfile, ParametricCZOptimizer
    if profile is None:
        profile = ParametricCouplerProfile()          # representative device
    opt = ParametricCZOptimizer(profile, bandwidth_mhz=bandwidth_mhz,
                                target_gate=gate, precision=precision)
    result = opt.optimize_multi_seed(n_seeds=n_seeds, iterations=iterations,
                                     n_slices=n_slices, dt_ns=dt_ns,
                                     **optimize_kwargs)
    # Attach the optimizer so analysis is immediate (it owns error_budget,
    # robustness_sweep, spectator_fidelity, ... which act on best_raw_param).
    result["optimizer"] = opt
    return result


def optimize_cz(profile=None, *, n_seeds: int = 4, iterations: int = 200,
                n_slices: int = 150, dt_ns: float = 1.0, bandwidth_mhz: float = 80.0,
                precision: str = "single", **optimize_kwargs) -> dict:
    """Optimize a CZ on a parametric-coupler pair in one call, with good defaults.

        import gradpulse as gp
        r = gp.optimize_cz()                      # representative device
        print(r["best_fidelity"])                 # process fidelity to CZ
        r["optimizer"].error_budget(r["best_raw_param"])   # analyze it

    ``profile`` defaults to a representative ``ParametricCouplerProfile()``; pass
    your own (e.g. ``ParametricCouplerProfile.from_ibm_backend(...)``) for a real
    device. Extra keyword arguments pass through to ``optimize_multi_seed`` (e.g.
    ``lbfgs_polish=False``, ``robust_amp_jitter=0.05``). Returns that method's
    result dict, plus ``result["optimizer"]`` for follow-up analysis. For the
    cross-resonance or N-qubit architectures, use ``CrossResonanceZXOptimizer`` /
    ``MultiQubitOptimizer`` directly.
    """
    return _optimize_parametric("cz", profile, n_seeds=n_seeds, iterations=iterations,
                                n_slices=n_slices, dt_ns=dt_ns,
                                bandwidth_mhz=bandwidth_mhz, precision=precision,
                                **optimize_kwargs)


def optimize_iswap(profile=None, *, n_seeds: int = 4, iterations: int = 200,
                   n_slices: int = 150, dt_ns: float = 1.0, bandwidth_mhz: float = 80.0,
                   precision: str = "single", **optimize_kwargs) -> dict:
    """Optimize an iSWAP (the parametric coupler's other native gate) in one call.

    Identical to :func:`optimize_cz` but targets iSWAP; see that docstring for the
    arguments and return value.
    """
    return _optimize_parametric("iswap", profile, n_seeds=n_seeds, iterations=iterations,
                                n_slices=n_slices, dt_ns=dt_ns,
                                bandwidth_mhz=bandwidth_mhz, precision=precision,
                                **optimize_kwargs)


def tunable_coupler_cz(freqs_ghz=(4.40, 5.50, 4.60), anharm_mhz=(-200, -150, -200),
                       g_qubit_coupler_mhz=85.0, t1_ns=(3e4, 2e4, 3e4),
                       t2_ns=(2e4, 1.5e4, 2e4), n_levels=3, delta_max_mhz=300.0,
                       precision="double", verbose=True, **optimizer_kwargs):
    """Build a faithful **tunable-coupler** CZ optimizer (coupler evolved explicitly).

    Unlike the dispersive ``ParametricCZOptimizer`` -- which adiabatically
    eliminates the coupler under Schrieffer-Wolff -- this models the real 3-element
    chain **qubit - coupler - qubit** with the coupler as a live transmon whose
    frequency is flux-tuned (the control), exactly how modern tunable-coupler
    devices (e.g. Rigetti Cepheus, Google Sycamore) realize a CZ: the flux pulse
    on the coupler switches the qubit-qubit interaction on through the coupler-
    mediated ``|11>-|02>`` resonance. Built on ``MultiQubitOptimizer`` (so the
    QuTiP cross-check, divergence guards, and diagnostics all apply).

    ``freqs_ghz`` / ``anharm_mhz`` / ``t1_ns`` / ``t2_ns`` are length-3
    ``(q0, coupler, q1)``. ``g_qubit_coupler_mhz`` is the (always-on) q-coupler
    exchange. The default frequencies place q1 ~200 MHz above q0 (the ``|11>-|02>``
    condition with -200 MHz anharmonicity). Each element gets a frequency-control
    channel: the coupler flux activates the gate; the two qubit detunings supply
    the single-qubit Z corrections (the analogue of hardware virtual-Z).

    Returns a configured ``MultiQubitOptimizer``; call ``.optimize(...)`` on it.
    This is a 27-dim open-system model -- heavier than the pair optimizers (seconds
    per iteration); use ``n_seeds``/``iterations`` accordingly. Cross-check a result
    with ``gradpulse.validate.multiqubit_cross_check``.
    """
    from .multiqubit import MultiQubitProfile, MultiQubitOptimizer
    prof = MultiQubitProfile(
        n_qubits=3, freqs_ghz=list(freqs_ghz), anharm_mhz=list(anharm_mhz),
        t1_ns=list(t1_ns), t2_ns=list(t2_ns),
        couplings={(0, 1): g_qubit_coupler_mhz, (1, 2): g_qubit_coupler_mhz},
        n_levels=n_levels,
    )
    return MultiQubitOptimizer(
        prof, target_gate="cz", target_qubits=(0, 2),
        drive_qubits=[], tunable_edges=[], freq_control_qubits=[0, 1, 2],
        delta_max_mhz=delta_max_mhz, open_system=True, precision=precision,
        verbose=verbose, **optimizer_kwargs)


def coupler_in_loop_cz(profile=None, *, coupler_freq_ghz: float = 5.9,
                       coupler_anharm_mhz: float = -250.0, gc_mhz: float = 95.0,
                       coupler_t1_ns: float = 1.5e4, coupler_t2_ns: float = 1.0e4,
                       n_levels: int = 3, n_seeds: int = 2, iterations: int = 150,
                       n_slices: int = 160, dt_ns: float = 1.0, delta_max_mhz: float = 300.0,
                       precision: str = "double", verbose: bool = True,
                       **optimize_kwargs) -> dict:
    """Optimize a CZ with the tunable coupler **explicitly in the loop**, starting
    from a pair (``ParametricCouplerProfile``) -- the opt-in that captures the
    coupler leakage the dispersive pair model eliminates *by construction*.

    The pair ``ParametricCZOptimizer`` adiabatically eliminates the coupler under
    Schrieffer-Wolff, so its only leakage channel is the qubits' own ``|2>`` states;
    the coupler's population is identically zero because the coupler is gone. This
    helper takes the *same two qubits* (frequencies, anharmonicities, T1/T2 are read
    from ``profile``) and re-introduces the coupler as a live transmon between them
    (``coupler_freq_ghz``, ``coupler_anharm_mhz``, bare exchange ``gc_mhz``), then
    optimizes the flux-activated CZ on the explicit 3-element chain. You stay in the
    pair workflow -- hand it the profile you already use -- and get back the number
    the pair model cannot produce: how much population transiently/finally sits in
    the coupler.

    Built on the QuTiP-cross-checked :func:`tunable_coupler_cz` /
    ``MultiQubitOptimizer`` engine (so :func:`gradpulse.validate.multiqubit_cross_check`
    applies verbatim -- the returned ``optimizer`` is passed straight to it). This is
    a 27-dim open-system model: seconds per iteration, so defaults are modest; raise
    ``iterations``/``n_seeds`` for a production pulse.

    Honesty note -- this is **not** a controlled fidelity ablation of the pair model.
    The pair model activates the exchange *parametrically* (flux modulation -> sideband)
    while the explicit coupler here activates it by *DC-tuning the coupler toward the
    ``|11>-|02>`` resonance*; the two realize the same gate by different mechanisms, so a
    raw ``F_explicit - F_pair`` subtraction would conflate elimination error with two
    optimizers finding different optima at different durations. The rigorous,
    apples-to-apples residual of the elimination itself is
    :func:`gradpulse.validate.coupler_elimination_cross_check` (the ``O((gc/Delta)^2)``
    swap-trajectory check). What this function adds on top is the **leakage** the
    eliminated model structurally cannot see, plus ``sw_param = (gc/Delta)^2`` so you
    can read off whether the pair model's elimination is even in its regime of validity
    for your device.

    Returns the underlying ``optimize`` result dict, augmented with::

        coupler_leakage   final population outside {q0,q1 in {0,1}, coupler=0}
                          (the coupler + qubit-|2> leakage; >= the pair model's,
                          and the part attributable to the coupler is invisible to it)
        sw_param          (gc/Delta)^2, the Schrieffer-Wolff small parameter
        J_eff_mhz         (gc^2/2)(1/D0 + 1/D1), the static exchange the coupler mediates
        coupler_freq_ghz  the coupler frequency used
        optimizer         the MultiQubitOptimizer (feed to multiqubit_cross_check)
    """
    import math
    if profile is None:
        from .parametric import ParametricCouplerProfile
        profile = ParametricCouplerProfile()
    f0, f1 = float(profile.freq_ghz_q1), float(profile.freq_ghz_q2)
    a0 = float(profile.anharm_ghz_q1) * 1000.0
    a1 = float(profile.anharm_ghz_q2) * 1000.0
    opt = tunable_coupler_cz(
        freqs_ghz=(f0, coupler_freq_ghz, f1),
        anharm_mhz=(a0, coupler_anharm_mhz, a1),
        g_qubit_coupler_mhz=gc_mhz,
        t1_ns=(profile.t1_ns_q1, coupler_t1_ns, profile.t1_ns_q2),
        t2_ns=(profile.t2_ns_q1, coupler_t2_ns, profile.t2_ns_q2),
        n_levels=n_levels, delta_max_mhz=delta_max_mhz,
        precision=precision, verbose=verbose)
    result = opt.optimize(n_slices=n_slices, dt_ns=dt_ns, iterations=iterations,
                          n_seeds=n_seeds, verbose=verbose, **optimize_kwargs)
    # SW-elimination context: detunings q-coupler and the mediated static exchange.
    d0 = (f0 - coupler_freq_ghz) * 1000.0       # MHz
    d1 = (f1 - coupler_freq_ghz) * 1000.0
    result["coupler_leakage"] = float(result.get("leakage", float("nan")))
    result["sw_param"] = float((gc_mhz / d0) ** 2)
    result["J_eff_mhz"] = float((gc_mhz ** 2 / 2.0) * (1.0 / d0 + 1.0 / d1))
    result["coupler_freq_ghz"] = float(coupler_freq_ghz)
    result["optimizer"] = opt
    return result
