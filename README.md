# Koopman Control Lab

> **Research prototype:** transparent lifted-state control experiments inspired by Koopman representations.

## Research question
Can a nonlinear state be represented in a lifted feature space where simple linear evolution/control models remain useful?

## Implemented
- transparent polynomial lifting
- deterministic lifted-state representation
- small control experiment suitable for inspection
- reproducible tests

## Quickstart
```bash
python demo.py
python -m unittest discover -s tests -v
```

## Boundary
This is a Koopman-inspired educational/research prototype. It is not a claim of an identified Koopman operator, optimality, or real-world deployment.

Related: [Safe RL Action Gate](https://github.com/Hafiz-IIT/safe-rl-action-gate)
