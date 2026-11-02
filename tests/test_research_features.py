import unittest
from koopman_lift import fit_edmd, predict_lifted, rollout, lift

class EDMDTests(unittest.TestCase):
    def test_fit_known_linear_contraction(self):
        xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ys = [0.8*x for x in xs]
        operator = fit_edmd(xs, ys, degree=2, ridge=1e-5)
        predicted = predict_lifted(0.25, operator)
        self.assertAlmostEqual(predicted[0], 0.2, places=3)

    def test_rollout_length(self):
        operator = ((0.9, 0.0), (0.0, 0.81))
        self.assertEqual(len(rollout(0.5, operator, 4)), 5)

    def test_invalid_degree(self):
        with self.assertRaises(ValueError):
            lift(1.0, 0)

if __name__ == "__main__": unittest.main()
