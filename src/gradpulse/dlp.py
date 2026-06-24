import torch

class SoftLogic:
    """
    Differentiable logical programming primitives using t-norms and smooth approximations.
    These operators operate on probabilities or truth values in the range [0, 1].
    """

    @staticmethod
    def soft_not(x: torch.Tensor) -> torch.Tensor:
        """Differentiable NOT operation."""
        return 1.0 - x

    @staticmethod
    def soft_and(x: torch.Tensor, y: torch.Tensor, method: str = "product") -> torch.Tensor:
        """
        Differentiable AND operation.
        Methods:
          - 'product': x * y (Product t-norm)
          - 'lukasiewicz': max(0, x + y - 1)
          - 'goedel': min(x, y)
        """
        if method == "product":
            return x * y
        elif method == "lukasiewicz":
            return torch.relu(x + y - 1.0)
        elif method == "goedel":
            return torch.minimum(x, y)
        else:
            raise ValueError(f"Unknown soft_and method: {method}")

    @staticmethod
    def soft_or(x: torch.Tensor, y: torch.Tensor, method: str = "product") -> torch.Tensor:
        """
        Differentiable OR operation.
        Methods:
          - 'product': x + y - x * y (Probabilistic sum)
          - 'lukasiewicz': min(1, x + y)
          - 'goedel': max(x, y)
        """
        if method == "product":
            return x + y - x * y
        elif method == "lukasiewicz":
            return torch.clamp(x + y, max=1.0)
        elif method == "goedel":
            return torch.maximum(x, y)
        else:
            raise ValueError(f"Unknown soft_or method: {method}")

    @staticmethod
    def implies(x: torch.Tensor, y: torch.Tensor, method: str = "product") -> torch.Tensor:
        """
        Differentiable implication: IF x THEN y
        Equivalent to: soft_or(soft_not(x), y)
        """
        return SoftLogic.soft_or(SoftLogic.soft_not(x), y, method=method)


class SoftRelational:
    """
    Smooth approximations of relational operators to convert continuous metrics
    into truth values in [0, 1].
    """

    @staticmethod
    def greater_than(x: torch.Tensor, threshold: float, temperature: float = 1.0) -> torch.Tensor:
        """
        Returns truth value in [0, 1] indicating if x > threshold.
        Lower temperature makes the transition sharper.
        """
        return torch.sigmoid((x - threshold) / temperature)

    @staticmethod
    def less_than(x: torch.Tensor, threshold: float, temperature: float = 1.0) -> torch.Tensor:
        """
        Returns truth value in [0, 1] indicating if x < threshold.
        Lower temperature makes the transition sharper.
        """
        return torch.sigmoid((threshold - x) / temperature)

    @staticmethod
    def equals(x: torch.Tensor, target: float, temperature: float = 1.0) -> torch.Tensor:
        """
        Returns truth value in [0, 1] indicating if x == target.
        Modeled as a smooth Gaussian-like peak around target.
        """
        return torch.exp(-0.5 * ((x - target) / temperature) ** 2)


class Proposition:
    """
    Base class for differentiable declarative constraints.
    """
    def evaluate(self, metrics: dict) -> torch.Tensor:
        """
        Evaluates the proposition given a dictionary of current metrics
        and returns a penalty scalar in [0, 1]. A higher value implies
        stronger constraint violation.
        """
        raise NotImplementedError

    def __call__(self, metrics: dict) -> torch.Tensor:
        return self.evaluate(metrics)


class Rule(Proposition):
    """
    A concrete rule composed of a condition (premise) and a consequence.
    IF condition THEN consequence.

    If the rule is violated, it returns a penalty > 0.
    """
    def __init__(self, condition_fn, consequence_fn, weight: float = 1.0, method: str = "product"):
        self.condition_fn = condition_fn
        self.consequence_fn = consequence_fn
        self.weight = weight
        self.method = method

    def evaluate(self, metrics: dict) -> torch.Tensor:
        # Evaluate truth values of condition and consequence (should be in [0, 1])
        condition_truth = self.condition_fn(metrics)
        consequence_truth = self.consequence_fn(metrics)

        # Rule implication: IF condition THEN consequence
        # Truth value of implication = soft_or(not(condition), consequence)
        implication_truth = SoftLogic.implies(condition_truth, consequence_truth, method=self.method)

        # The penalty is how much the implication is FALSE
        violation = SoftLogic.soft_not(implication_truth)

        return self.weight * violation
