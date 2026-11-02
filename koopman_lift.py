"""Small EDMD-style lifted-dynamics laboratory.

This is a transparent, scalar, data-driven demonstration. It is not a claim
that a finite polynomial dictionary is an exact Koopman representation.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def lift(x: float, degree: int = 3) -> tuple[float, ...]:
    """Return the monomial dictionary (x, x², ..., x**degree)."""
    if not math.isfinite(x):
        raise ValueError("state must be finite")
    if degree < 1:
        raise ValueError("degree must be >= 1")
    return tuple(x ** power for power in range(1, degree + 1))


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small dense system using pivoted Gaussian elimination."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular normal-equation matrix; increase ridge")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [u - factor * v for u, v in zip(a[row], a[col])]
    return [a[i][-1] for i in range(n)]


def fit_edmd(
    states: Sequence[float],
    next_states: Sequence[float],
    degree: int = 3,
    ridge: float = 1e-6,
) -> tuple[tuple[float, ...], ...]:
    """Fit a regularized lifted linear map K using paired scalar transitions.

    Convention: z_next[j] = sum_i z[i] * K[i][j].
    """
    if len(states) != len(next_states) or len(states) < degree + 1:
        raise ValueError("paired data must contain at least degree + 1 transitions")
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    pairs = [(lift(x, degree), lift(y, degree)) for x, y in zip(states, next_states)]
    gram = [[sum(z[i] * z[j] for z, _ in pairs) for j in range(degree)] for i in range(degree)]
    for i in range(degree):
        gram[i][i] += ridge
    columns = []
    for j in range(degree):
        rhs = [sum(z[i] * zn[j] for z, zn in pairs) for i in range(degree)]
        columns.append(_solve(gram, rhs))
    return tuple(tuple(columns[j][i] for j in range(degree)) for i in range(degree))


def predict_lifted(state: float, operator: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Apply a fitted operator to the lifted state."""
    z = lift(state, len(operator))
    if any(len(row) != len(operator) for row in operator):
        raise ValueError("operator must be square")
    return tuple(sum(z[i] * operator[i][j] for i in range(len(z))) for j in range(len(z)))


def rollout(state: float, operator: Sequence[Sequence[float]], steps: int) -> list[tuple[float, ...]]:
    """Return lifted predictions; the first element is the initial lift."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    trajectory = [lift(state, len(operator))]
    for _ in range(steps):
        z = trajectory[-1]
        trajectory.append(tuple(sum(z[i] * operator[i][j] for i in range(len(z))) for j in range(len(z))))
    return trajectory


def linear_step(z: tuple[float, ...], rate: float = 0.9) -> tuple[float, ...]:
    """Known stable linear map used only for a controlled demo."""
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    return tuple(rate * value for value in z)
