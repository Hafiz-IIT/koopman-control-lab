"""Small EDMD-style lifted-dynamics laboratory.

The implementation uses a scalar polynomial observable dictionary. It is an
empirical finite-dimensional approximation, not proof of Koopman invariance.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def lift(x: float, degree: int = 3) -> tuple[float, ...]:
    """Return the dictionary (1, x, x², ..., x**degree)."""
    x = _finite("state", x)
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    return (1.0,) + tuple(x ** power for power in range(1, degree + 1))


def _validate_operator(operator: Sequence[Sequence[float]]) -> int:
    if not isinstance(operator, Sequence) or isinstance(operator, (str, bytes)):
        raise ValueError("operator must be a square numeric sequence")
    dimension = len(operator)
    if dimension < 2 or any(not isinstance(row, Sequence) or len(row) != dimension for row in operator):
        raise ValueError("operator must be a non-empty square matrix for a degree >= 1 lift")
    for row in operator:
        for value in row:
            _finite("operator value", value)
    return dimension


def _apply_operator(
    lifted_state: Sequence[float],
    operator: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    dimension = _validate_operator(operator)
    if not isinstance(lifted_state, Sequence) or len(lifted_state) != dimension:
        raise ValueError("lifted state dimension must match operator")
    for value in lifted_state:
        _finite("lifted state value", value)
    result = tuple(
        sum(lifted_state[i] * operator[i][j] for i in range(dimension))
        for j in range(dimension)
    )
    for value in result:
        _finite("operator output", value)
    return result


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small dense system using partial-pivot Gaussian elimination."""
    if not isinstance(rhs, Sequence) or not isinstance(matrix, Sequence):
        raise ValueError("matrix and rhs must be sequences")
    n = len(rhs)
    if n < 1 or len(matrix) != n or any(not isinstance(row, Sequence) or len(row) != n for row in matrix):
        raise ValueError("matrix must be square and match rhs")
    for value in rhs:
        _finite("rhs value", value)
    for row in matrix:
        for value in row:
            _finite("matrix value", value)
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
    solution = [a[index][-1] for index in range(n)]
    for value in solution:
        _finite("solution value", value)
    return solution


def fit_edmd(
    states: Sequence[float],
    next_states: Sequence[float],
    degree: int = 3,
    ridge: float = 1e-8,
) -> tuple[tuple[float, ...], ...]:
    """Fit z_next[j] = sum_i z[i] * K[i][j] by ridge-regularized least squares.

    The constant observable is included and is not ridge-penalized.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    if not isinstance(states, Sequence) or not isinstance(next_states, Sequence):
        raise ValueError("states and next_states must be sequences")
    dimension = degree + 1
    if len(states) != len(next_states) or len(states) < dimension:
        raise ValueError("paired data must contain at least degree + 1 transitions")
    for value in list(states) + list(next_states):
        _finite("training state", value)
    ridge = _finite("ridge", ridge)
    if ridge < 0:
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
    return _apply_operator(lift(state, dimension - 1), operator)


def predict_state(state: float, operator: Sequence[Sequence[float]]) -> float:
    """Return the scalar-state prediction (the x observable is index 1)."""
    return predict_lifted(state, operator)[1]


def rollout(
    state: float,
    operator: Sequence[Sequence[float]],
    steps: int,
) -> list[tuple[float, ...]]:
    """Iterate the learned linear operator in lifted space.

    After the first step, predictions are not projected back onto the nonlinear
    image of the lifting map. This is the standard linear lifted rollout.
    """
    dimension = _validate_operator(operator)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    trajectory = [lift(state, dimension - 1)]
    for _ in range(steps):
        trajectory.append(_apply_operator(trajectory[-1], operator))
    return trajectory


def linear_step(z: tuple[float, ...], rate: float = 0.9) -> tuple[float, ...]:
    """Known stable linear map used only for a controlled demo."""
    rate = _finite("rate", rate)
    if not 0 <= rate <= 1:
        raise ValueError("rate must be in [0, 1]")
    if not isinstance(z, Sequence) or not z:
        raise ValueError("z must be a non-empty sequence")
    for value in z:
        _finite("lifted value", value)
    return tuple(rate * value for value in z)
