import pytest
import torch
from src.gradpulse.parametric import ParametricCZOptimizer
from src.gradpulse.dlp import Rule, SoftRelational

def test_dlp_rule_integration():
    def condition(m):
        # IF leakage > 0.001
        return SoftRelational.greater_than(m["leakage"], 0.001, temperature=0.01)

    def consequence(m):
        # THEN fidelity > 0.99
        return SoftRelational.greater_than(m["fidelity"], 0.99, temperature=0.01)

    rule = Rule(condition_fn=condition, consequence_fn=consequence, weight=2.0)

    # Initialize optimizer with the rule
    opt = ParametricCZOptimizer(dlp_rules=[rule])

    # Run a tiny optimization loop (2 iterations) to make sure the loop runs with the rule evaluating
    res = opt.optimize_multi_seed(n_seeds=1, iterations=2, n_slices=20, lbfgs_polish=True, lbfgs_iters=2)

    assert res is not None
    assert "best_fidelity" in res
