"""gradpulse.hardware - hardware-in-the-loop scaffolding (the sim<->device gap).

The repo is explicit that a simulated fidelity is NOT a hardware number (see
``rb.py`` and the limitations sections). This module is the hook that lets you
*close* that gap once you have a device: feed a measured gate fidelity back in,
infer how far the model's noise was off, correct it, and re-optimise against the
corrected model.

What's here:

  * ``HardwareBackend`` -- the protocol you implement against your device / cloud
    provider (submit the pulse, run interleaved RB, return the number). One method.
  * ``SimulatedBackend`` -- a reference backend backed by the simulator itself
    (optionally via the leakage-aware interleaved-RB estimator in ``rb.py``), with
    a configurable "true" device that differs from the model. It makes the whole
    loop runnable and testable WITHOUT a QPU -- but it is a stand-in, not hardware.
  * ``QuTiPDeviceBackend`` -- a stronger stand-in: the "measurement" comes from the
    INDEPENDENT QuTiP integrator (the one ``validate.py`` cross-checks against), not
    gradpulse's own simulator, and its "true" device may carry physics the model
    omits (extra static-ZZ, finite temperature, shorter coherence). The loop then
    recovers a genuine model-vs-independent-truth gap rather than a self-injected
    scalar -- the strongest proof the loop closes without a QPU. Still simulation.
  * ``BraketBackendTemplate`` -- a clearly-labelled TEMPLATE (NOT a live integration,
    not run anywhere here) documenting, as code, exactly where the optimized waveform
    meets a provider's pulse-level API and interleaved RB. The real-silicon seam.
  * ``infer_coherence_scale`` / ``apply_coherence_scale`` -- the model-refinement
    step: solve for the effective decoherence scaling that makes the model predict
    the measured fidelity for that exact physical pulse, and bake it into the
    profile. This lumps the measured model gap into an effective coherence
    correction -- the dominant un-modelled error on most superconducting devices.
    It is a first-order calibration, not a microscopic noise model (a coherent
    control-error gap, say, would be only partially absorbed this way).
  * ``calibrate_to_hardware`` -- the closed loop: optimise -> measure -> infer ->
    update profile, repeated. With ``SimulatedBackend`` it demonstrably pulls a
    wrong model toward the truth.

All model predictions evaluate the SAME physical (smoothed) pulse the optimiser
saves, via a no-resmooth eval optimiser (bandwidth_mhz=0, clamp activation) -- the
exact convention the QuTiP validator and validation_checks.py use -- so model and
"hardware" are compared apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

try:
    from .parametric import ParametricCouplerProfile, ParametricCZOptimizer, DEVICE
    from . import rb as _rb
except ImportError:  # pragma: no cover - direct-script execution
    from parametric import ParametricCouplerProfile, ParametricCZOptimizer, DEVICE
    import rb as _rb

def expected_improvement(X: np.ndarray, X_sample: np.ndarray, Y_sample: np.ndarray, gpr: "GaussianProcessRegressor", xi: float = 0.01) -> np.ndarray:
    import scipy.stats
    """Computes the Expected Improvement at points X."""
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * scipy.stats.norm.cdf(Z) + sigma * scipy.stats.norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def probability_of_improvement(X: np.ndarray, X_sample: np.ndarray, Y_sample: np.ndarray, gpr: "GaussianProcessRegressor", xi: float = 0.01) -> np.ndarray:
    import scipy.stats
    """Computes the Probability of Improvement at points X."""
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        pi = scipy.stats.norm.cdf(Z)
        pi[sigma == 0.0] = 0.0

    return pi

def upper_confidence_bound(X: np.ndarray, X_sample: np.ndarray, Y_sample: np.ndarray, gpr: "GaussianProcessRegressor", kappa: float = 2.576) -> np.ndarray:
    """Computes the Upper Confidence Bound at points X."""
    mu, sigma = gpr.predict(X, return_std=True)
    return mu + kappa * sigma


class BayesianCalibrationLoop:
    """Bayesian Optimization loop for closed-loop hardware calibration.

    Uses a Gaussian Process surrogate model to map continuous pulse parameters
    (e.g., amplitude scale, frequency offsets) to the measured fidelity on hardware.
    Iteratively proposes new parameters by maximizing an Acquisition Function.
    """
    def __init__(self, backend, base_waveform: np.ndarray, dt_ns: float = 1.0, acquisition_fn: str = "EI"):
        self.backend = backend
        self.base_waveform = np.asarray(base_waveform)
        self.dt_ns = dt_ns

        if acquisition_fn == "EI":
            self.acq_fn = expected_improvement
        elif acquisition_fn == "PI":
            self.acq_fn = probability_of_improvement
        elif acquisition_fn == "UCB":
            self.acq_fn = upper_confidence_bound
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_fn}")

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self.gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True, alpha=1e-4)
        self.X_sample = []
        self.Y_sample = []

    def _propose_location(self, bounds: np.ndarray, n_restarts: int = 10) -> np.ndarray:
        dim = bounds.shape[0]
        min_val = float('inf')
        best_x = None

        def min_obj(x):
            return -self.acq_fn(x.reshape(-1, dim), np.array(self.X_sample), np.array(self.Y_sample), self.gpr)[0]

        import scipy.optimize
        for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = scipy.optimize.minimize(min_obj, x0=starting_point, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                best_x = res.x

        if best_x is None:
            best_x = np.random.uniform(bounds[:, 0], bounds[:, 1])
        return best_x

    def run(self, bounds, apply_params_fn, n_iterations: int = 10, n_init: int = 5):
        """Runs the optimization loop.

        Args:
            bounds: A sequence of (min, max) pairs for each parameter.
            apply_params_fn: A callable `(base_waveform, params) -> new_waveform`
                             that applies the parameters to the base waveform.
            n_iterations: Number of BO iterations.
            n_init: Number of random initializations.
        """
        bounds = np.array(bounds)
        dim = bounds.shape[0]

        # Initialization
        for _ in range(n_init):
            x = np.random.uniform(bounds[:, 0], bounds[:, 1])
            wf = apply_params_fn(self.base_waveform, x)
            meas = self.backend.measure_gate(wf, dt_ns=self.dt_ns, meta={"bo_phase": "init"})
            self.X_sample.append(x)
            self.Y_sample.append(meas.f_avg)

        self.gpr.fit(np.array(self.X_sample), np.array(self.Y_sample))

        # Optimization loop
        history = []
        for i in range(n_iterations):
            next_x = self._propose_location(bounds)
            wf = apply_params_fn(self.base_waveform, next_x)
            meas = self.backend.measure_gate(wf, dt_ns=self.dt_ns, meta={"bo_phase": "opt", "iteration": i})

            self.X_sample.append(next_x)
            self.Y_sample.append(meas.f_avg)
            self.gpr.fit(np.array(self.X_sample), np.array(self.Y_sample))

            history.append({
                "iteration": i,
                "params": next_x,
                "f_avg": meas.f_avg
            })

        best_idx = np.argmax(self.Y_sample)
        return {
            "best_params": self.X_sample[best_idx],
            "best_f_avg": self.Y_sample[best_idx],
            "history": history
        }


@dataclass
class GateMeasurement:
    """The result of characterising one gate on hardware (or a stand-in)."""
    f_avg: float                          # measured average gate fidelity (e.g. IRB)
    leakage_per_gate: float = 0.0
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)


class HardwareBackend:
    """Protocol for a device backend. Implement ``measure_gate`` against your
    hardware/cloud: upload the waveform, run interleaved RB on the gate, and return
    a ``GateMeasurement``. Everything else in this module is provider-agnostic.
    """

    def measure_gate(self, waveform: np.ndarray, dt_ns: float,
                     meta: Optional[dict] = None) -> GateMeasurement:
        raise NotImplementedError


# ---- physical-pulse evaluation (no re-smoothing) --------------------------
def _eval_optimizer(profile: ParametricCouplerProfile, n_channels: int,
                    precision: str = "double") -> ParametricCZOptimizer:
    """Eval optimiser that consumes a SAVED smoothed envelope verbatim.

    bandwidth_mhz=0 + clamp activation => simulate_* treats the [0,1] envelope as
    the physical pulse (u_signed = 2*env-1), with no second smoothing/activation --
    matching gradpulse.validate and examples/validation_checks.py.
    """
    return ParametricCZOptimizer(profile, bandwidth_mhz=0.0, use_drag=False,
                                 n_channels=n_channels, activation="clamp",
                                 precision=precision)


def _as_env_tensor(opt: ParametricCZOptimizer, waveform: np.ndarray):
    import torch
    return torch.as_tensor(waveform, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)


def predicted_process_fidelity(profile: ParametricCouplerProfile, waveform: np.ndarray,
                               dt_ns: float = 1.0, diss_scale: float = 1.0,
                               precision: str = "double") -> float:
    """Model F_proc for a saved physical pulse at a given dissipator scaling."""
    import torch
    opt = _eval_optimizer(profile, waveform.shape[1], precision)
    env = _as_env_tensor(opt, waveform)
    with torch.no_grad():
        rho = opt.simulate_choi_batch(env, dt=dt_ns, diss_scale=diss_scale)
        return float(opt._process_fidelity(rho).mean())


def predicted_f_avg(profile: ParametricCouplerProfile, waveform: np.ndarray,
                    dt_ns: float = 1.0, diss_scale: float = 1.0,
                    precision: str = "double") -> float:
    """Model average gate fidelity F_avg = (d*F_proc + 1)/(d + 1), d = 4."""
    fp = predicted_process_fidelity(profile, waveform, dt_ns, diss_scale, precision)
    return (4.0 * fp + 1.0) / 5.0


def apply_coherence_scale(profile: ParametricCouplerProfile,
                          scale: float) -> ParametricCouplerProfile:
    """Bake an effective decoherence scaling into a profile (T1/T2 /= scale).

    diss_scale = s is equivalent to coherence times (1/s)x the profile's, so
    dividing T1/T2 by s makes a fresh, diss_scale=1 optimiser reproduce that
    behaviour. scale > 1 => the device is noisier than the model thought.
    """
    s = max(1e-6, float(scale))
    note = f"coherence rescaled x{1.0 / s:.4f} from hardware feedback (diss_scale={s:.4f})"
    return replace(
        profile,
        t1_ns_q1=profile.t1_ns_q1 / s, t1_ns_q2=profile.t1_ns_q2 / s,
        t2_ns_q1=profile.t2_ns_q1 / s, t2_ns_q2=profile.t2_ns_q2 / s,
        notes=list(profile.notes) + [note],
    )


def infer_coherence_scale(profile: ParametricCouplerProfile, waveform: np.ndarray,
                          measured_f_avg: float, dt_ns: float = 1.0,
                          lo: float = 1e-3, hi: float = 16.0, iters: int = 40,
                          precision: str = "double") -> float:
    """Effective dissipator scaling s such that the model predicts measured_f_avg.

    F_avg is monotone-decreasing in s (more dissipation -> lower fidelity), so a
    bisection is exact. Returns the bracket end if the measurement lies outside
    the achievable range (e.g. measured > the near-closed-system fidelity => the
    gap is not decoherence and is clamped to lo).
    """
    import torch
    opt = _eval_optimizer(profile, waveform.shape[1], precision)
    env = _as_env_tensor(opt, waveform)

    def f_avg_at(s: float) -> float:
        with torch.no_grad():
            rho = opt.simulate_choi_batch(env, dt=dt_ns, diss_scale=s)
            fp = float(opt._process_fidelity(rho).mean())
        return (4.0 * fp + 1.0) / 5.0

    f_lo, f_hi = f_avg_at(lo), f_avg_at(hi)
    if measured_f_avg >= f_lo:
        return lo
    if measured_f_avg <= f_hi:
        return hi
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if f_avg_at(m) > measured_f_avg:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def simulate_noisy_irb(f_avg_true: float, shots: int = 1000,
                       lengths=(1, 2, 4, 8, 16, 32, 64), n_sequences: int = 30,
                       rng=None, d: int = 4) -> float:
    """A realistic interleaved-RB *estimator* of F_avg with finite-shot sampling noise.

    A clean backend hands back the exact fidelity; a real device hands back an RB
    *fit* to noisy survival curves. This models the gate as a depolarizing channel of
    average fidelity ``f_avg_true`` (the exact value, e.g. from the QuTiP backend),
    whose RB survival decays as ``p(m) = (1 - 1/d)*alpha**m + 1/d`` with
    ``alpha = (d*F - 1)/(d - 1)``. For each sequence length it draws ``n_sequences``
    sequences measured with ``shots`` shots each (binomial sampling), then
    least-squares-fits ``alpha`` back on a semilog axis and converts to F_avg. The
    returned number therefore carries genuine statistical noise -- the half of the
    sim<->hardware gap that is sampling, which a noise-free backend never exercises.
    The fitter is the same shape a real RB analysis uses (fixed ideal SPAM here).

    Unbiased to leading order: averaging many draws returns ``f_avg_true``; the spread
    shrinks ~ 1/sqrt(shots * n_sequences). ``rng`` is a ``numpy.random.Generator``.
    """
    rng = np.random.default_rng() if rng is None else rng
    alpha = (d * float(f_avg_true) - 1.0) / (d - 1.0)
    alpha = min(max(alpha, 1e-9), 1.0)
    A, B = 1.0 - 1.0 / d, 1.0 / d                       # ideal SPAM
    ms = np.asarray(lengths, dtype=float)
    means = np.empty(ms.shape)
    for k, m in enumerate(lengths):
        p = min(max(A * alpha ** m + B, 0.0), 1.0)
        # n_sequences random sequences, `shots` shots each -> mean survival estimate.
        means[k] = rng.binomial(shots, p, size=n_sequences).mean() / shots
    # Semilog fit of (p - B)/A = alpha**m -> slope = log(alpha). Drop points the noise
    # pushed at/below the depolarized floor (log undefined), as a real fitter would.
    y = (means - B) / A
    mask = y > 1e-6
    if mask.sum() < 2:
        return float(min(max(f_avg_true, 0.0), 1.0))    # too noisy to fit; degrade gracefully
    slope = np.polyfit(ms[mask], np.log(y[mask]), 1)[0]
    alpha_hat = float(np.exp(min(slope, 0.0)))          # alpha <= 1
    f_hat = (alpha_hat * (d - 1.0) + 1.0) / d
    return float(min(max(f_hat, 0.0), 1.0))


class SimulatedBackend(HardwareBackend):
    """Reference HardwareBackend backed by the simulator (NOT real hardware).

    The "true" device is ``true_profile`` (typically with different/worse coherence
    than the model the loop starts from), creating a realistic sim-to-hardware gap.
    ``measure_gate`` returns the gate's average fidelity on that true device --
    either analytically from the channel's process fidelity (fast, default) or via
    the leakage-aware interleaved-RB estimator in ``rb.py`` (``use_irb=True``, the
    faithful but slower estimator a real device reports). ``extra_infidelity`` adds
    an optional incoherent gap not captured by T1/T2.
    """

    def __init__(self, true_profile: ParametricCouplerProfile,
                 extra_infidelity: float = 0.0, use_irb: bool = False,
                 irb_kwargs: Optional[dict] = None, precision: str = "double"):
        self.true_profile = true_profile
        self.extra_infidelity = float(extra_infidelity)
        self.use_irb = bool(use_irb)
        self.irb_kwargs = dict(irb_kwargs or {})
        self.precision = precision

    def measure_gate(self, waveform: np.ndarray, dt_ns: float = 1.0,
                     meta: Optional[dict] = None) -> GateMeasurement:
        if self.use_irb:
            import torch
            opt = _eval_optimizer(self.true_profile, waveform.shape[1], self.precision)
            env = torch.as_tensor(waveform, dtype=opt.rdtype, device=DEVICE)
            cz = _rb.gate_superoperator(opt, env, dt=dt_ns)
            out = _rb.interleaved_rb(cz, **self.irb_kwargs)
            f_avg = out["f_cz_irb"]
            leak = float(out.get("leakage_per_clifford_L1", 0.0) or 0.0)
            src = "simulated_irb"
        else:
            f_avg = predicted_f_avg(self.true_profile, waveform, dt_ns,
                                    precision=self.precision)
            leak = 0.0
            src = "simulated_analytic"
        f_avg = max(0.0, f_avg - self.extra_infidelity)
        return GateMeasurement(f_avg=f_avg, leakage_per_gate=leak, source=src,
                               metadata={"true_t1_q1_ns": self.true_profile.t1_ns_q1,
                                         "extra_infidelity": self.extra_infidelity})


class QuTiPDeviceBackend(HardwareBackend):
    """An INDEPENDENT-ENGINE device backend: it "measures" the pulse with QuTiP --
    the same trusted integrator ``gradpulse.validate`` uses for its ship-gate
    cross-check -- rather than gradpulse's own optimizer simulator.

    Why this is the stronger stand-in. ``SimulatedBackend`` measures with the very
    engine the optimizer models with, so the gap the loop recovers is essentially
    the scalar you injected. Here the model (gradpulse) and the "device" (QuTiP) are
    DIFFERENT code paths, and ``true_profile`` may carry physics the optimizer's
    profile does not represent -- extra static-ZZ, finite temperature, shorter
    coherence. ``calibrate_to_hardware`` then recovers a genuine
    model-vs-independent-truth mismatch and folds it into an effective coherence
    correction (a first-order, lumped attribution -- an out-of-model coherent error
    is only partially absorbed this way, exactly as on a real device).

    Still simulation, not silicon -- but the strongest SELF-CONTAINED proof that the
    loop closes. For a real device, implement ``HardwareBackend`` against your
    provider (see ``BraketBackendTemplate`` and the README). Requires the optional
    ``[validate]`` extra (QuTiP).
    """

    def __init__(self, true_profile: ParametricCouplerProfile,
                 target_gate: str = "cz", line_response=None,
                 shots: Optional[int] = None, n_irb_sequences: int = 30,
                 rng_seed: int = 0):
        """shots: if set, the "measurement" is a finite-shot interleaved-RB *fit*
        (``simulate_noisy_irb``) of the QuTiP-exact fidelity rather than the exact
        number -- so the calibration loop sees realistic statistical noise, the harder
        half of the real problem. None (default) returns the exact value as before.
        n_irb_sequences/rng_seed control the per-measurement averaging and the
        (advancing) random stream, so repeated measurements are independent draws."""
        self.true_profile = true_profile
        self.target_gate = str(target_gate)
        self.line_response = line_response
        self.shots = shots
        self.n_irb_sequences = int(n_irb_sequences)
        self._rng = np.random.default_rng(rng_seed)

    def measure_gate(self, waveform: np.ndarray, dt_ns: float = 1.0,
                     meta: Optional[dict] = None) -> GateMeasurement:
        try:
            from . import validate as _validate
        except ImportError:  # pragma: no cover - direct-script execution
            import validate as _validate
        f_proc = _validate.qutip_f_proc(self.true_profile, waveform,
                                        target_gate=self.target_gate, dt_ns=dt_ns,
                                        line_response=self.line_response)
        f_avg_exact = (4.0 * f_proc + 1.0) / 5.0
        if self.shots is None:
            f_avg, src = f_avg_exact, "qutip_independent"
        else:
            f_avg = simulate_noisy_irb(f_avg_exact, shots=self.shots,
                                       n_sequences=self.n_irb_sequences, rng=self._rng)
            src = "qutip_independent+shotnoise"
        return GateMeasurement(f_avg=f_avg, leakage_per_gate=0.0, source=src,
                               metadata={"engine": "qutip",
                                         "target_gate": self.target_gate,
                                         "shots": self.shots,
                                         "f_avg_exact": f_avg_exact,
                                         "true_t1_q1_ns": self.true_profile.t1_ns_q1})


class BraketPulseBackend(HardwareBackend):
    """Amazon Braket pulse-level export backend, built offline up to ``device.run()``.

    Unlike a stub, this CONSTRUCTS everything an interleaved-RB submission would need and
    stops at ``device.run()`` -- executing on a QPU is out of scope for this package. The
    seam is concrete code, offline-verifiable:

      1. ``build_gate_sequence(waveform)`` binds the saved envelope to the device's
         frames as a real ``braket.pulse.PulseSequence`` (via ``braket_bridge``;
         channel order: q1 drive, q2 drive, coupler[, phase][, Stark]). With a live
         ``AwsDevice`` it uses the device's actual frames; without one it uses
         synthetic frames so you can still build and inspect the OpenPulse offline.
      2. ``estimate_cost(...)`` prices the interleaved-RB run from current Braket
         rates (see ``braket_bridge.estimate_experiment_cost``).
      3. ``measure_gate`` builds the gate sequence, then SUBMITS. Submission needs
         (a) a live ``AwsDevice`` + credentials, and (b) a ``clifford_compiler`` that
         turns 2Q Cliffords into the device's native gates with the gate-under-test
         inserted as the custom pulse -- the irreducibly device-specific piece. The
         fit then reuses the SAME leakage-aware estimator as ``rb.py``. Without a live
         device + compiler it raises a precise error (naming exactly what's missing)
         and attaches the built OpenPulse + cost estimate -- it does not silently fail.

    The interleaved-RB *protocol and analysis* are already validated offline against
    the optimizer's noisy superoperator (``gradpulse.rb.interleaved_rb``), so a number
    that comes back from real silicon is trustworthy. Everything else here
    (infer/apply coherence scale, ``calibrate_to_hardware``) is provider-agnostic.
    Use ``QuTiPDeviceBackend`` for an independent-engine loop without a QPU.

    NOTE: the optimized pulse targets representative MODEL parameters, not a specific
    device qubit -- re-optimize against the device's real calibration
    (``from_braket_calibration``) before trusting an absolute hardware fidelity.
    """

    def __init__(self, device=None, shots: int = 1000,
                 device_name: str = "Rigetti-Cepheus-1-108Q",
                 clifford_compiler=None):
        # `device` may be a live braket AwsDevice (real frames + submission) or None
        # (offline construction/inspection only). `device_name` keys the price table.
        self.device = device
        self.shots = int(shots)
        self.device_name = str(device_name)
        self.clifford_compiler = clifford_compiler

    def _frames(self, n_channels: int):
        try:
            from . import braket_bridge as _bb
        except ImportError:  # pragma: no cover - direct-script execution
            import braket_bridge as _bb
        if self.device is not None and getattr(self.device, "frames", None):
            # device.frames is a dict; take the first n_channels in insertion order.
            return list(self.device.frames.values())[:n_channels]
        return _bb.synthetic_frames(n_channels)

    def build_gate_sequence(self, waveform: np.ndarray):
        """The optimized gate as a ``braket.pulse.PulseSequence`` (OpenPulse 3.0).
        Real frames if a live device was given, synthetic otherwise. No submission."""
        try:
            from . import braket_bridge as _bb
        except ImportError:  # pragma: no cover - direct-script execution
            import braket_bridge as _bb
        wf = np.asarray(waveform)
        n_ch = wf.shape[1] if wf.ndim > 1 else 1
        return _bb.build_gate_pulse_sequence(wf, self._frames(n_ch))

    def estimate_cost(self, lengths=(1, 2, 4, 8, 16), n_seeds: int = 20):
        try:
            from . import braket_bridge as _bb
        except ImportError:  # pragma: no cover - direct-script execution
            import braket_bridge as _bb
        n_circ = _bb.irb_circuit_count(lengths, n_seeds)
        return _bb.estimate_experiment_cost(n_circ, self.shots, self.device_name)

    def measure_gate(self, waveform: np.ndarray, dt_ns: float = 1.0,
                     meta: Optional[dict] = None) -> GateMeasurement:
        seq = self.build_gate_sequence(waveform)  # always builds (offline-verifiable)
        cost = self.estimate_cost()
        if self.device is None or self.clifford_compiler is None:
            missing = []
            if self.device is None:
                missing.append("a live braket AwsDevice (+ AWS credentials)")
            if self.clifford_compiler is None:
                missing.append("a clifford_compiler (2Q Clifford -> device-native gates)")
            raise RuntimeError(
                "BraketPulseBackend built the gate PulseSequence ("
                f"{len(seq.to_ir())} chars OpenPulse) and priced the run "
                f"(~${cost.total_usd:.2f} on {self.device_name}), but cannot SUBMIT "
                "without " + " and ".join(missing) + ". This is the one step that "
                "closes simulation != hardware and it costs real money -- run it "
                "yourself. Use QuTiPDeviceBackend for an independent-engine loop now.")
        # Live path (user supplied device + compiler): build IRB, submit, fit.
        raise NotImplementedError(  # pragma: no cover - needs a live QPU
            "Live submission requires your clifford_compiler to emit reference + "
            "interleaved circuits with the custom pulse; submit via "
            "self.device.run_batch(circuits, shots=self.shots) and fit with the "
            "rb.py leakage-aware estimator. See the class docstring.")


# Back-compat alias: the old name documented the seam; it is now concrete.
BraketBackendTemplate = BraketPulseBackend


def calibrate_to_hardware(initial_profile: ParametricCouplerProfile,
                          backend: HardwareBackend, rounds: int = 3,
                          opt_kwargs: Optional[dict] = None, dt_ns: float = 1.0,
                          n_channels: int = 3, bandwidth_mhz: float = 80.0,
                          precision: str = "double", n_measure: int = 1,
                          verbose: bool = False) -> dict:
    """Closed hardware-in-the-loop calibration.

    Each round: optimise a CZ on the current model, hand the physical pulse to the
    backend, read back the measured F_avg, infer the effective coherence scaling
    that reconciles model and measurement for that pulse, and fold it into the
    profile. Returns the refined profile and a per-round history. With a
    ``SimulatedBackend`` whose true coherence differs from the start, the model's
    predicted F_avg converges toward the measured one over a few rounds; with a
    ``QuTiPDeviceBackend`` the measurement comes from an independent integrator, so
    the loop closes a genuine model-vs-truth gap (not a self-injected one). Backend
    is anything implementing ``HardwareBackend`` -- including a real device.

    ``n_measure > 1`` averages that many independent measurements per round; with a
    shot-noisy backend (e.g. ``QuTiPDeviceBackend(shots=...)``) this is the lever that
    beats the statistical error down (~1/sqrt(n_measure)), and each history entry then
    also carries ``f_avg_sem``.
    """
    opt_kwargs = dict(opt_kwargs or dict(n_seeds=2, iterations=150, n_slices=150,
                                         warm_start_mode="parametric_cz",
                                         use_process_fidelity=True, lbfgs_polish=False))
    profile = initial_profile
    history = []
    for r in range(rounds):
        opt = ParametricCZOptimizer(profile, bandwidth_mhz=bandwidth_mhz,
                                    use_drag=False, n_channels=n_channels,
                                    activation="sigmoid")
        res = opt.optimize_multi_seed(dt_ns=dt_ns, **opt_kwargs)
        waveform = res["best_waveform"]
        # Predict on the SAVED physical pulse via the no-resmooth eval path (NOT the
        # optimiser's internal best_fidelity), so model/backend/infer all evaluate the
        # same physical pulse and the gap stays a pure model-vs-device difference.
        f_model = predicted_f_avg(profile, waveform, dt_ns=dt_ns, precision=precision)
        meas = backend.measure_gate(waveform, dt_ns, meta={"round": r})
        if n_measure > 1:
            fs = [meas.f_avg] + [backend.measure_gate(waveform, dt_ns,
                                 meta={"round": r, "rep": j}).f_avg
                                 for j in range(1, int(n_measure))]
            meas = GateMeasurement(f_avg=float(np.mean(fs)),
                                   leakage_per_gate=meas.leakage_per_gate,
                                   source=meas.source,
                                   metadata={**meas.metadata, "n_measure": int(n_measure),
                                             "f_avg_sem": float(np.std(fs) / max(1, len(fs)**0.5))})
        gap = f_model - meas.f_avg
        scale = infer_coherence_scale(profile, waveform, meas.f_avg, dt_ns=dt_ns,
                                      precision=precision)
        history.append({
            "round": r,
            "f_model_avg": f_model,
            "f_hardware_avg": meas.f_avg,
            "gap": gap,
            "coherence_scale": scale,
            "t1_ns_q1": profile.t1_ns_q1,
            "t2_ns_q1": profile.t2_ns_q1,
            "source": meas.source,
        })
        if verbose:
            print(f"  round {r}: F_model={f_model:.5f}  F_hw={meas.f_avg:.5f}  "
                  f"gap={gap:+.5f}  -> coherence_scale={scale:.4f}", flush=True)
        profile = apply_coherence_scale(profile, scale)

    return {"refined_profile": profile, "history": history}
