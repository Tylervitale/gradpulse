import torch
import pytest
from src.gradpulse.dlp import SoftLogic, SoftRelational, Rule

def test_soft_not():
    x = torch.tensor([0.0, 0.5, 1.0])
    not_x = SoftLogic.soft_not(x)
    assert torch.allclose(not_x, torch.tensor([1.0, 0.5, 0.0]))

def test_soft_and():
    x = torch.tensor([0.0, 0.5, 1.0])
    y = torch.tensor([1.0, 0.5, 1.0])

    # product
    res = SoftLogic.soft_and(x, y, method="product")
    assert torch.allclose(res, torch.tensor([0.0, 0.25, 1.0]))

    # lukasiewicz
    res = SoftLogic.soft_and(x, y, method="lukasiewicz")
    assert torch.allclose(res, torch.tensor([0.0, 0.0, 1.0]))

    # goedel
    res = SoftLogic.soft_and(x, y, method="goedel")
    assert torch.allclose(res, torch.tensor([0.0, 0.5, 1.0]))

def test_soft_or():
    x = torch.tensor([0.0, 0.5, 1.0])
    y = torch.tensor([0.0, 0.5, 0.0])

    # product
    res = SoftLogic.soft_or(x, y, method="product")
    assert torch.allclose(res, torch.tensor([0.0, 0.75, 1.0]))

    # lukasiewicz
    res = SoftLogic.soft_or(x, y, method="lukasiewicz")
    assert torch.allclose(res, torch.tensor([0.0, 1.0, 1.0]))

    # goedel
    res = SoftLogic.soft_or(x, y, method="goedel")
    assert torch.allclose(res, torch.tensor([0.0, 0.5, 1.0]))

def test_soft_relational():
    # greater_than
    res = SoftRelational.greater_than(torch.tensor(1.5), 1.0, temperature=0.1)
    assert res.item() > 0.99  # very true

    res = SoftRelational.greater_than(torch.tensor(0.5), 1.0, temperature=0.1)
    assert res.item() < 0.01  # very false

    # less_than
    res = SoftRelational.less_than(torch.tensor(0.5), 1.0, temperature=0.1)
    assert res.item() > 0.99  # very true

    # equals
    res = SoftRelational.equals(torch.tensor(1.0), 1.0, temperature=0.1)
    assert torch.isclose(res, torch.tensor(1.0))
    res = SoftRelational.equals(torch.tensor(1.5), 1.0, temperature=0.1)
    assert res.item() < 0.01

def test_rule_gradient_propagation():
    # metrics: bandwidth and leakage
    metrics = {
        "bandwidth": torch.tensor(120.0, requires_grad=True),
        "leakage": torch.tensor(0.05, requires_grad=True)
    }

    # IF leakage > 0.01 THEN bandwidth < 100
    def condition(m):
        return SoftRelational.greater_than(m["leakage"], 0.01, temperature=0.01)

    def consequence(m):
        return SoftRelational.less_than(m["bandwidth"], 100.0, temperature=10.0)

    rule = Rule(condition_fn=condition, consequence_fn=consequence, weight=2.0)
    penalty = rule(metrics)

    # Rule should be violated because leakage (0.05) > 0.01 is True,
    # but bandwidth (120) < 100 is False. Implication is False -> Penalty > 0.
    assert penalty.item() > 1.0

    # Check gradients
    penalty.backward()

    # We should have gradients driving leakage down and bandwidth down
    assert metrics["leakage"].grad is not None
    assert metrics["bandwidth"].grad is not None
    assert metrics["bandwidth"].grad.item() > 0.0  # reducing bandwidth lowers the penalty
