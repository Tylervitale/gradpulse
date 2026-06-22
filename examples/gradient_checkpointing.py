"""Gradient checkpointing: cut autograd memory so the open-system optimizer reaches
larger registers, with the SAME optimum.

    python examples/gradient_checkpointing.py

The open-system Choi stack (4**N operators evolved through every slice) hits a MEMORY
wall before a compute wall, because autograd stores every intermediate. Passing
``checkpoint_segments=S`` keeps only the state at S segment boundaries and recomputes
the interiors in backward -- memory drops ~O(Nt/S) at ~2x forward compute. The result
is identical; only peak memory changes. This is the memory-side lever, complementary
to the compute-side ``state_transfer`` estimator.
"""
import torch

from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer
from gradpulse.parametric import DEVICE

profile = MultiQubitProfile(n_qubits=3, n_levels=2,
                            couplings={(0, 1): 12.0, (1, 2): 12.0})
opt = MultiQubitOptimizer(profile, target_gate="cz", target_qubits=(0, 1),
                          open_system=True)

# Same optimization, with the slice loop checkpointed into 4 segments.
res = opt.optimize(n_slices=60, dt_ns=1.0, iterations=60, n_seeds=1, lr=0.06,
                   checkpoint_segments=4)
print(f"F_proc with checkpointing = {res['best_fidelity']:.5f} "
      f"(same optimum as checkpoint_segments=0, lower peak memory)")

# It is also available on the pair simulators -- value and gradients are unchanged:
pair = MultiQubitOptimizer(profile, target_gate="cz", target_qubits=(0, 1),
                           open_system=True)
print("\nVerify equivalence (value matches between plain and checkpointed):")
kernel = pair._smoother(40, 1.0)
raw = 0.1 * torch.randn(1, 40, pair.n_channels, dtype=pair.rdtype, device=DEVICE)
xs = pair._smooth(torch.sigmoid(raw), kernel)
f_plain = pair._process_fidelity_choi(pair._propagate_choi(xs, 1.0))[0]
f_ckpt = pair._process_fidelity_choi(pair._propagate_choi(xs, 1.0, checkpoint_segments=5))[0]
print(f"  plain={float(f_plain):.8f}  checkpointed={float(f_ckpt):.8f}  "
      f"diff={abs(float(f_plain)-float(f_ckpt)):.1e}")
