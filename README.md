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
\n\n## Example and evidence boundary\n\nThe `demo.py` path is the primary runnable example, while `tests/` captures the deterministic implementation contract.\n\n## Results and limitations\n\nThe current artifact demonstrates a transparent polynomial lifting workflow. It does **not** identify or learn a Koopman operator, establish optimality, or validate a physical control system. Results should therefore be interpreted as controlled prototype behavior rather than empirical claims about nonlinear control generally.\n\n## Companion manuscript\n\nSee [Koopman Control — Research Manuscript](https://github.com/Hafiz-IIT/koopman-control-paper) for the manuscript companion.\n