"""Fit and inspect a small EDMD-style model on synthetic scalar transitions."""
from koopman_lift import fit_edmd, predict_lifted, rollout

states = [0.05 * i for i in range(1, 13)]
next_states = [0.9 * x + 0.02 * x * x for x in states]
operator = fit_edmd(states, next_states, degree=2, ridge=1e-5)
print("EDMD-style lifted model (synthetic data; not a physical system)")
print("operator:", operator)
print("one-step prediction at x=0.25:", predict_lifted(0.25, operator))
print("three-step lifted rollout:", rollout(0.25, operator, 3))
