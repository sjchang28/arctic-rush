"""Categorical value/reward support.

MuZero does not regress value and reward directly. It applies an invertible
squashing transform

    h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x

and then predicts a *distribution* over a fixed integer support in that
transformed space, trained with cross entropy. Two things fall out of this:
the loss is scale-free, so the value term stops dwarfing the policy term the way
plain MSE on raw returns did; and a categorical head can represent uncertainty
about a bimodal return (solve now vs. never) that a single scalar cannot.

The `eps * x` term keeps the transform invertible and its gradient bounded away
from zero for large |x|.
"""

import torch


EPSILON = 0.001


def scalar_transform(x: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    """h(x): squash a value into the space the support is defined over."""

    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x


def inverse_scalar_transform(x: torch.Tensor, epsilon: float = EPSILON) -> torch.Tensor:
    """h^-1(x): the exact inverse of `scalar_transform`."""

    if epsilon == 0:
        return torch.sign(x) * (x ** 2 + 2 * torch.abs(x))

    numerator = torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1
    return torch.sign(x) * (((numerator / (2 * epsilon)) ** 2) - 1)


def scalar_to_support(x: torch.Tensor, support_size: int) -> torch.Tensor:
    """Two-hot encode scalars onto the integer support [-support_size, support_size].

    Args:
        x: any shape.
        support_size: the support spans 2 * support_size + 1 bins.

    Returns:
        Tensor of shape (*x.shape, 2 * support_size + 1).
    """

    x = scalar_transform(x)
    x = torch.clamp(x, -support_size, support_size)

    floor = x.floor()
    upper_weight = x - floor

    support = torch.zeros(*x.shape, 2 * support_size + 1, dtype=torch.float32, device=x.device)

    lower_index = (floor + support_size).long().unsqueeze(-1)
    support.scatter_(-1, lower_index, (1 - upper_weight).unsqueeze(-1))

    # The upper bin only exists when x is not already at the top of the support.
    upper_index = torch.clamp(lower_index + 1, max=2 * support_size)
    support.scatter_add_(-1, upper_index, upper_weight.unsqueeze(-1))

    return support


def support_to_scalar(logits: torch.Tensor, support_size: int) -> torch.Tensor:
    """Expected value of a categorical head, mapped back to real units."""

    probabilities = torch.softmax(logits, dim=-1)
    support = torch.arange(
        -support_size, support_size + 1, dtype=torch.float32, device=logits.device
    )

    expectation = (probabilities * support).sum(dim=-1)
    return inverse_scalar_transform(expectation)
