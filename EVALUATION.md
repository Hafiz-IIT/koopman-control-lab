# Evaluation protocol — scalar lifted-dynamics prediction

## Research question
When does a finite-dimensional polynomial EDMD-style model improve one-step and multi-step prediction over persistence and a fitted affine linear baseline on controlled scalar nonlinear systems?

## Systems
Use four smooth maps (cubic contraction, sinusoidal contraction, tanh contraction, and quadratic-affine contraction) plus one piecewise regime-shift map. Publish the exact equations in the experiment source. All in-distribution initial states are sampled uniformly from [-1, 1].

## Protocol
- Ten independent fixed seeds (0–9).
- 600 training transitions, 300 in-distribution test transitions, and 300 out-of-range test transitions per seed and system.
- Out-of-range states are sampled uniformly from [-1.5, 1.5].
- One hundred independent rollout initial states per seed; report 20-step prediction error.
- Compare persistence with EDMD dictionaries of degrees 1, 2, and 3. Degree 1 is the fitted affine linear baseline because the constant observable is included.
- Fix ridge regularization at 1e-8 before evaluation; do not tune on test data.
- Use identical train/test samples across model degrees within each seed.

## Metrics
One-step RMSE and maximum absolute error, out-of-range one-step RMSE, and 20-step rollout RMSE and maximum absolute error. Report per-seed outputs and aggregate mean, sample standard deviation, and 95% t-interval across seeds.

## Scope and limitations
This is a scalar synthetic prediction study. It does not establish an invariant Koopman subspace, control/MPC utility, high-dimensional agent performance, physical-system validity, or safety. Multi-step lifted predictions can leave the image of the lifting map; this is measured as an empirical forecasting error, not interpreted as a valid physical state embedding.

Current status: experiment runner and results are being added; conclusions remain limited to the declared synthetic maps.
