"""Small EDMD-style lifted-dynamics laboratory.

The implementation uses a scalar polynomial observable dictionary. It is an
empirical finite-dimensional approximation, not proof of Koopman invariance.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def lift(x: float, degree: int = 3) -> tuple[float, ...]:
    """Return the dictionary (1, x, x², ..., x**degree)."""
    if not math.isfinite(x):
        raise ValueError("state must be finite")
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    return (1.0,) + tuple(x ** power for power in range(1, degree + 1))


def _validate_operator(operator: Sequence[Sequence[float]]) -> int:
    dimension = len(operator)
    if dimension < 2 or any(len(row) != dimension for row in operator):
        raise ValueError("operator must be a non-empty square matrix for a degree >= 1 lift")
    if not all(math.isfinite(value) for row in operator for value in row):
        raise ValueError("operator values must be finite")
    return dimension


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small dense system using partial-pivot Gaussian elimination."""
    n = len(rhs)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square and match rhs")
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise ValueError("singular normal-equation matrix; increase ridge or data coverage")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [left - factor * right for left, right in zip(a[row], a[col])]
    return [a[index][-1] for index in range(n)]


def fit_edmd(
    states: Sequence[float],
    next_states: Sequence[float],
    degree: int = 3,
    ridge: float = 1e-8,
) -> tuple[tuple[float, ...], ...]:
    """Fit z_next[j] = sum_i z[i] * K[i][j] by ridge-regularized least squares.

    The constant observable is included and is not ridge-penalized.
    """
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    dimension = degree + 1
    if len(states) != len(next_states) or len(states) < dimension:
        raise ValueError("paired data must contain at least degree + 1 transitions")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")

    pairs = [(lift(x, degree), lift(y, degree)) for x, y in zip(states, next_states)]
    gram = [
        [sum(z[i] * z[j] for z, _ in pairs) for j in range(dimension)]
        for i in range(dimension)
    ]
    for index in range(1, dimension):
        gram[index][index] += ridge

    columns = []
    for output_index in range(dimension):
        rhs = [
            sum(z[input_index] * z_next[output_index] for z, z_next in pairs)
            for input_index in range(dimension)
        ]
        columns.append(_solve(gram, rhs))

    return tuple(
        tuple(columns[output_index][input_index] for output_index in range(dimension))
        for input_index in range(dimension)
    )


def predict_lifted(
    state: float,
    operator: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Apply a fitted operator once and return the predicted lifted state."""
    dimension = _validate_operator(operator)
    z = lift(state, dimension - 1)
    return tuple(
        sum(z[i] * operator[i][j] for i in range(dimension))
        for j in range(dimension)
    )


def predict_state(state: float, operator: Sequence[Sequence[float]]) -> float:
    """Return the scalar-state prediction (the x observable is index 1)."""
    return predict_lifted(state, operator)[1]


def rollout(
    state: float,
    operator: Sequence[Sequence[float]],
    steps: int,
) -> list[tuple[float, ...]]:
    """Return lifted predictions; the first element is the initial lift."""
    dimension = _validate_operator(operator)
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    trajectory = [lift(state, dimension - 1)]
    for _ in range(steps):
        trajectory.append(predict_lifted(trajectory[-1][1], operator))
    return trajectory


def linear_step(z: tuple[float, ...], rate: float = 0.9) -> tuple[float, ...]:
    """Known stable linear map used only for a controlled demo."""
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be finite and in [0, 1]")
    return tuple(rate * value for value in z)
