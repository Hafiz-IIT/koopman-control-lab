# Koopman Control Lab

An inspectable laboratory for fitting finite-dimensional lifted dynamics from scalar transition data. The implementation uses a monomial dictionary with a constant observable and ridge-regularized EDMD-style least squares.

## Research question
When does a finite lifted linear model approximate nonlinear scalar dynamics, and how do dictionary degree, regularization, and data coverage affect prediction error?

## Implemented
- polynomial lifting: (1, x, x², ..., x^d)
- regularized least-squares operator estimation
- explicit operator convention and one-step state prediction
- multi-step lifted rollout
- deterministic synthetic demonstration
- validation for malformed inputs and singular systems

## Run
    python demo.py
    python experiment.py
    python -m unittest discover -s tests -v

## Model convention
The fitted operator K is stored so that z_next[j] = sum_i z[i] * K[i][j]. It is a finite-dimensional approximation learned from paired observations, not proof of an invariant Koopman subspace.

## Scope
Scalar synthetic-data prediction prototype only. It is not a robot controller, validated physical model, stability proof, or state-of-the-art claim. No hardware was tested.

## Results
The reproducible scalar benchmark writes per-seed outputs and an aggregate summary under `results/`. See [EVALUATION.md](EVALUATION.md) for the frozen experiment design.

## Related work
- [Safe RL Action Gate](https://github.com/Hafiz-IIT/safe-rl-action-gate)
- [Residual RL with Linear Priors](https://github.com/Hafiz-IIT/residual-rl-linear-priors)
- [Hybrid LQR + RL Control](https://github.com/Hafiz-IIT/hybrid-lqr-rl-control)
- [Companion manuscript](https://github.com/Hafiz-IIT/koopman-control-paper)
