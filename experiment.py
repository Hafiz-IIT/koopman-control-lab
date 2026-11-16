"""Reproducible scalar EDMD prediction benchmark.

All systems are synthetic and all model settings are fixed before evaluation.
The script writes per-seed results, aggregate statistics, and a JSON manifest.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path

from koopman_lift import fit_edmd, rollout

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEEDS = tuple(range(10))
DEGREES = (1, 2, 3)
RIDGE = 1e-8
TRAIN_N = 600
TEST_N = 300
OOD_N = 300
ROLLOUTS = 100
HORIZON = 20
TRAIN_LOW, TRAIN_HIGH = -1.0, 1.0
OOD_LOW, OOD_HIGH = -1.5, 1.5
T_CRIT_95_DF9 = 2.262157

SYSTEMS = {
    "cubic_contraction": lambda x: 0.85 * x - 0.12 * x ** 3,
    "sinusoidal_contraction": lambda x: 0.75 * x + 0.10 * math.sin(2.0 * x),
    "tanh_contraction": lambda x: 0.70 * x + 0.08 * math.tanh(2.0 * x),
    "quadratic_affine_contraction": lambda x: 0.70 * x + 0.12 * x * x + 0.03,
    "piecewise_regime_shift": lambda x: (
        0.78 * x + 0.06 * math.sin(2.0 * x) + (0.08 if x >= 0.25 else -0.08)
    ),
}

METRICS = (
    "one_step_rmse",
    "one_step_max_abs_error",
    "ood_one_step_rmse",
    "ood_one_step_max_abs_error",
    "rollout_rmse_20",
    "rollout_max_abs_error_20",
)
RAW_FIELDS = ("system", "seed", "method", "degree", *METRICS)


def rmse(predicted: list[float], actual: list[float]) -> float:
    if len(predicted) != len(actual) or not actual:
        raise ValueError("RMSE requires equal, non-empty sequences")
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(actual))


def max_abs_error(predicted: list[float], actual: list[float]) -> float:
    if len(predicted) != len(actual) or not actual:
        raise ValueError("maximum error requires equal, non-empty sequences")
    return max(abs(p - a) for p, a in zip(predicted, actual))


def evaluate_predictions(
    system_name: str,
    function,
    seed: int,
    train_states: list[float],
    train_next: list[float],
    test_states: list[float],
    test_next: list[float],
    ood_states: list[float],
    ood_next: list[float],
    rollout_starts: list[float],
    method: str,
    degree: int,
    operator=None,
) -> dict:
    if operator is None:
        one_step = test_states
        ood_one_step = ood_states
        predicted_rollouts = [
            [initial] * HORIZON for initial in rollout_starts
        ]
    else:
        one_step = [rollout(x, operator, 1)[1][1] for x in test_states]
        ood_one_step = [rollout(x, operator, 1)[1][1] for x in ood_states]
        predicted_rollouts = [
            [row[1] for row in rollout(initial, operator, HORIZON)[1:]]
            for initial in rollout_starts
        ]

    actual_rollouts = []
    for initial in rollout_starts:
        current = initial
        trajectory = []
        for _ in range(HORIZON):
            current = function(current)
            trajectory.append(current)
        actual_rollouts.append(trajectory)

    predicted_flat = [value for trajectory in predicted_rollouts for value in trajectory]
    actual_flat = [value for trajectory in actual_rollouts for value in trajectory]

    return {
        "system": system_name,
        "seed": seed,
        "method": method,
        "degree": degree,
        "one_step_rmse": rmse(one_step, test_next),
        "one_step_max_abs_error": max_abs_error(one_step, test_next),
        "ood_one_step_rmse": rmse(ood_one_step, ood_next),
        "ood_one_step_max_abs_error": max_abs_error(ood_one_step, ood_next),
        "rollout_rmse_20": rmse(predicted_flat, actual_flat),
        "rollout_max_abs_error_20": max_abs_error(predicted_flat, actual_flat),
    }


def run_experiment() -> list[dict]:
    rows = []
    for system_name, function in SYSTEMS.items():
        for seed in SEEDS:
            train_rng = random.Random(seed)
            train_states = [
                train_rng.uniform(TRAIN_LOW, TRAIN_HIGH) for _ in range(TRAIN_N)
            ]
            train_next = [function(x) for x in train_states]

            test_rng = random.Random(seed + 10_000)
            test_states = [
                test_rng.uniform(TRAIN_LOW, TRAIN_HIGH) for _ in range(TEST_N)
            ]
            test_next = [function(x) for x in test_states]

            ood_rng = random.Random(seed + 20_000)
            ood_states = [
                ood_rng.uniform(OOD_LOW, OOD_HIGH) for _ in range(OOD_N)
            ]
            ood_next = [function(x) for x in ood_states]

            rollout_rng = random.Random(seed + 30_000)
            rollout_starts = [
                rollout_rng.uniform(TRAIN_LOW, TRAIN_HIGH) for _ in range(ROLLOUTS)
            ]

            rows.append(evaluate_predictions(
                system_name, function, seed, train_states, train_next,
                test_states, test_next, ood_states, ood_next,
                rollout_starts, "persistence", 0,
            ))

            for degree in DEGREES:
                operator = fit_edmd(train_states, train_next, degree=degree, ridge=RIDGE)
                method = "affine_linear" if degree == 1 else f"edmd_degree_{degree}"
                rows.append(evaluate_predictions(
                    system_name, function, seed, train_states, train_next,
                    test_states, test_next, ood_states, ood_next,
                    rollout_starts, method, degree, operator,
                ))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (row["system"], row["method"], row["degree"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (system, method, degree), group in sorted(grouped.items()):
        result = {"system": system, "method": method, "degree": degree, "n_seeds": len(group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            margin = T_CRIT_95_DF9 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
            result[f"{metric}_ci95_low"] = mean - margin
            result[f"{metric}_ci95_high"] = mean + margin
        summary_rows.append(result)
    return summary_rows


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_rows = run_experiment()
    summary_rows = summarize(raw_rows)

    raw_path = RESULTS / "koopman_scalar_per_seed.csv"
    summary_path = RESULTS / "koopman_scalar_summary.csv"
    manifest_path = RESULTS / "manifest.json"

    write_csv(raw_path, raw_rows, RAW_FIELDS)
    summary_fields = (
        "system", "method", "degree", "n_seeds",
        "one_step_rmse_mean", "one_step_rmse_sd",
        "one_step_rmse_ci95_low", "one_step_rmse_ci95_high",
        "ood_one_step_rmse_mean", "ood_one_step_rmse_sd",
        "ood_one_step_rmse_ci95_low", "ood_one_step_rmse_ci95_high",
        "rollout_rmse_20_mean", "rollout_rmse_20_sd",
        "rollout_rmse_20_ci95_low", "rollout_rmse_20_ci95_high",
    )
    compact_summary = [
        {field: row[field] for field in summary_fields}
        for row in summary_rows
    ]
    write_csv(summary_path, compact_summary, summary_fields)

    manifest = {
        "study": "scalar_edmd_prediction_benchmark",
        "version": "0.1",
        "status": "completed",
        "results_available": True,
        "seed_values": list(SEEDS),
        "systems": list(SYSTEMS.keys()),
        "methods": ["persistence", "affine_linear", "edmd_degree_2", "edmd_degree_3"],
        "configuration": {
            "train_transitions_per_seed": TRAIN_N,
            "test_transitions_per_seed": TEST_N,
            "ood_transitions_per_seed": OOD_N,
            "rollout_trajectories_per_seed": ROLLOUTS,
            "rollout_horizon": HORIZON,
            "training_state_range": [TRAIN_LOW, TRAIN_HIGH],
            "ood_state_range": [OOD_LOW, OOD_HIGH],
            "ridge": RIDGE,
            "ridge_selection": "fixed before test evaluation; no test tuning",
            "confidence_interval": "two-sided 95% t interval across 10 seeds (df=9)",
        },
        "output_files": [raw_path.name, summary_path.name],
        "limitations": [
            "Scalar synthetic systems only; no physical-system validation.",
            "Finite polynomial dictionaries are approximations, not invariant-subspace proofs.",
            "Prediction accuracy does not establish control or MPC utility.",
            "Multi-step lifted states are not projected back onto the nonlinear lifting manifold.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(raw_rows)} per-seed rows and {len(summary_rows)} aggregate rows.")
    print(f"Results: {raw_path.relative_to(ROOT)}, {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
