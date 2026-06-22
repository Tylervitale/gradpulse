"""Head-to-head: gradpulse vs C3 (c3-toolset) on a single-qubit gate.

C3 (q-optimize / c3-toolset) is gradpulse's closest true peer: open-system, autodiff
(TensorFlow) pulse optimization with a hardware backend. The existing benchmark
(gradpulse.benchmark, vs qutip-qtrl) is CLOSED-system because qutip-qtrl cannot do
open-system natively; C3 can, so it is the apples-to-apples comparison a reviewer wants.

What this script found (honest, reproduced 2026-06):
---------------------------------------------------
* C3 is the right peer -- it exposes open-system (Lindbladian) autodiff GRAPE with
  L-BFGS / Adam / CMA-ES, exactly gradpulse's regime.
* BUT c3-toolset 1.4 does NOT run cleanly on a current Python: it is pinned to pre-2.0
  NumPy (uses ``np.product``) and pre-2.21 TensorFlow internals (``ops.Tensor``), while
  Python 3.13 forces TF>=2.18 and gradpulse/torch force NumPy 2.x. Running it here needs
  monkey-patches (below), and even then C3's OPEN-system propagator path is broken on TF
  2.21 (a Liouville-space frame-rotation shape error), so only CLOSED-system C3 runs.
  gradpulse, by contrast, runs as-is on the modern stack -- a real portability advantage.
* On a matched closed-system single-qubit RX90 (3-level transmon):
      gradpulse  F~1.000  ~0.1 s   (free per-slice GRAPE, 28 slices)
      C3         F~0.9975 ~2.6 s    (L-BFGS on a 6-parameter Gaussian + full signal chain)
  gradpulse is faster and reaches the optimum -- but NOT byte-identically: C3 simulates
  the whole AWG->DAC->mixer chain and optimizes a CONSTRAINED physical pulse (more
  realism, more per-eval cost), gradpulse optimizes free amplitudes on an abstract
  Hamiltonian. Read it as "gradpulse is competitive with its true peer and far more
  portable", not "gradpulse beats C3 at C3's hardware-modeling job".

Requires: pip install c3-toolset tf-keras   (and tolerates the shims below).
Run:  python examples/benchmark_vs_c3.py
"""
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def run_c3_single_qubit_rx90(freq=5.0e9, anhar=-210e6, t1=20e-6, t2star=15e-6,
                             gate_t=7e-9, qubit_lvls=3, maxfun=80):
    """C3 closed-system RX90 optimization on a 3-level transmon. Returns
    {fidelity, wall_s, method} or raises with a clear message if C3 is unavailable
    / incompatible with the environment."""
    import tensorflow as tf
    # --- compatibility shims for c3-toolset 1.4 on a modern stack ---
    np.product = np.prod                                   # removed in NumPy 2.0
    from tensorflow.python.framework import ops as _ops    # ops.Tensor removed in TF 2.21
    if not hasattr(_ops, "Tensor"):
        _ops.Tensor = tf.Tensor
    if not hasattr(_ops, "EagerTensor"):
        _ops.EagerTensor = tf.Tensor

    from c3.c3objs import Quantity as Qty
    from c3.parametermap import ParameterMap
    from c3.experiment import Experiment
    from c3.model import Model
    from c3.generator.generator import Generator
    import c3.signal.gates as gates
    import c3.signal.pulse as pulse
    import c3.libraries.chip as chip
    import c3.libraries.hamiltonians as hamiltonians
    import c3.generator.devices as devices
    import c3.libraries.envelopes as envelopes
    import c3.libraries.algorithms as algorithms
    import c3.libraries.fidelities as fidelities
    from c3.optimizers.optimalcontrol import OptimalControl

    sim_res, awg_res = 100e9, 2e9
    q1 = chip.Qubit(name="Q1", hilbert_dim=qubit_lvls,
                    freq=Qty(freq, "Hz 2pi", 4.9e9, 5.1e9),
                    anhar=Qty(anhar, "Hz 2pi", -380e6, -120e6),
                    t1=Qty(t1, "s", 1e-6, 90e-6), t2star=Qty(t2star, "s", 1e-6, 90e-6),
                    temp=Qty(0.06, "K", 0.0, 0.12))
    drive = chip.Drive(name="d1", connected=["Q1"], hamiltonian_func=hamiltonians.x_drive)
    model = Model([q1], [drive])
    model.set_lindbladian(False)        # open-system path is broken on TF 2.21; closed only
    model.set_dressed(True)
    generator = Generator(devices={
        "LO": devices.LO(name="lo", resolution=sim_res),
        "AWG": devices.AWG(name="awg", resolution=awg_res),
        "DigitalToAnalog": devices.DigitalToAnalog(name="dac", resolution=sim_res),
        "Mixer": devices.Mixer(name="mixer"),
        "VoltsToHertz": devices.VoltsToHertz(name="v_to_hz", V_to_Hz=Qty(1e9, "Hz/V", 0.9e9, 1.1e9)),
    }, chains={"d1": {"LO": [], "AWG": [], "DigitalToAnalog": ["AWG"],
                      "Mixer": ["LO", "DigitalToAnalog"], "VoltsToHertz": ["Mixer"]}})
    gauss = pulse.Envelope(name="gauss", params={
        "amp": Qty(0.5, "V", 0.2, 0.6), "t_final": Qty(gate_t, "s", 0.5 * gate_t, 1.5 * gate_t),
        "sigma": Qty(gate_t / 4, "s", gate_t / 8, gate_t / 2),
        "xy_angle": Qty(0.0, "rad", -np.pi, np.pi),
        "freq_offset": Qty(-1e6, "Hz 2pi", -5e6, 5e6), "delta": Qty(-1.0, "", -5.0, 3.0)},
        shape=envelopes.envelopes["gaussian_nonorm"])
    carrier = pulse.Carrier(name="carrier", params={
        "freq": Qty(freq, "Hz 2pi", 4.9e9, 5.1e9), "framechange": Qty(0.0, "rad", -np.pi, 3 * np.pi)})
    xgate = gates.Instruction(name="rx90p", targets=[0], t_start=0.0, t_end=gate_t,
                              channels=["d1"], ideal=np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2))
    xgate.add_component(gauss, "d1")
    xgate.add_component(carrier, "d1")
    pmap = ParameterMap(instructions=[xgate], model=model, generator=generator)
    exp = Experiment(pmap=pmap)
    exp.set_opt_gates(["rx90p[0]"])
    pmap.set_opt_map([[("rx90p[0]", "d1", "gauss", "amp")],
                      [("rx90p[0]", "d1", "gauss", "freq_offset")],
                      [("rx90p[0]", "d1", "gauss", "delta")],
                      [("rx90p[0]", "d1", "gauss", "xy_angle")]])
    opt = OptimalControl(fid_func=fidelities.unitary_infid_set, fid_subspace=["Q1"],
                         pmap=pmap, algorithm=algorithms.lbfgs,
                         options={"maxfun": maxfun}, run_name="rx90")
    opt.set_exp(exp)
    t0 = time.perf_counter()
    opt.optimize_controls()
    wall = time.perf_counter() - t0
    infid = float(opt.current_best_goal)
    return {"method": "C3 (L-BFGS, 6-param Gaussian + signal chain)",
            "fidelity": 1.0 - infid, "wall_s": wall}


def run_gradpulse_single_qubit_rx90(anhar_mhz=-210.0, gate_t_ns=7.0, n_ts=28,
                                    rabi_max_mhz=250.0):
    """gradpulse free-GRAPE RX90 on the matched 3-level transmon (rotating frame)."""
    from gradpulse.benchmark import grape_autodiff
    twopi = 2.0 * np.pi
    nl = 3
    a = np.diag(np.sqrt(np.arange(1, nl)), 1).astype(complex)
    ad = a.conj().T
    H0 = 0.5 * (twopi * anhar_mhz / 1000.0) * (ad @ ad @ a @ a)   # anharmonicity drift
    Hc = [(a + ad), 1j * (ad - a)]                                # X, Y drive quadratures
    rx = np.array([[1, -1j], [-1j, 1]], complex) / np.sqrt(2)
    U = np.eye(3, dtype=complex)
    U[:2, :2] = rx
    res = grape_autodiff(H0, Hc, U, n_ts=n_ts, evo_time=gate_t_ns,
                         amp_bound=twopi * rabi_max_mhz / 1000.0, seed=0)
    return {"method": "gradpulse (free per-slice GRAPE)", "fidelity": res["fidelity"],
            "wall_s": res["wall_s"]}


def main():
    print("Single-qubit RX90 (3-level transmon): gradpulse vs C3\n")
    gp = run_gradpulse_single_qubit_rx90()
    print(f"  {gp['method']:<46}  F={gp['fidelity']:.6f}  wall={gp['wall_s']:.3f}s")
    try:
        c3 = run_c3_single_qubit_rx90()
        print(f"  {c3['method']:<46}  F={c3['fidelity']:.6f}  wall={c3['wall_s']:.3f}s")
        print("\nNote: NOT byte-identical -- C3 optimizes a constrained Gaussian through a full\n"
              "signal chain (more realism/overhead); gradpulse optimizes free amplitudes on an\n"
              "abstract Hamiltonian. C3's OPEN-system path is broken on TF>=2.21, so this is\n"
              "closed-system only. The portable, modern-stack tool here is gradpulse.")
    except Exception as e:  # c3 missing or env-incompatible
        print(f"  C3 unavailable / incompatible with this environment: {type(e).__name__}: {e}")
        print("  (c3-toolset 1.4 needs pre-2.0 NumPy + pre-2.21 TF internals; see module docstring.)")


if __name__ == "__main__":
    main()
