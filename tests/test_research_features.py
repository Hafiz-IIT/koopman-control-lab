import unittest

from koopman_lift import fit_edmd, predict_lifted, predict_state, rollout, lift


class EDMDTests(unittest.TestCase):
    def test_fit_known_affine_contraction(self):
        xs = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]
        ys = [0.8 * x + 0.1 for x in xs]
        operator = fit_edmd(xs, ys, degree=1, ridge=1e-10)
        self.assertAlmostEqual(predict_state(0.25, operator), 0.3, places=7)
        self.assertAlmostEqual(predict_lifted(0.25, operator)[0], 1.0, places=7)

    def test_rollout_length_and_constant_observable(self):
        operator = ((1.0, 0.0), (0.0, 0.9))
        trajectory = rollout(0.5, operator, 4)
        self.assertEqual(len(trajectory), 5)
        self.assertTrue(all(abs(row[0] - 1.0) < 1e-12 for row in trajectory))

    def test_invalid_degree(self):
        with self.assertRaises(ValueError):
            lift(1.0, 0)

    def test_mismatched_pairs(self):
        with self.assertRaises(ValueError):
            fit_edmd([0.1, 0.2], [0.2], degree=1)

    def test_ragged_operator_rejected(self):
        with self.assertRaises(ValueError):
            predict_lifted(0.5, ((1.0, 0.0), (0.1,)))

    def test_nonfinite_training_data_rejected(self):
        with self.assertRaises(ValueError):
            fit_edmd([0.1, float("nan"), 0.3], [0.2, 0.3, 0.4], degree=1)

    def test_boolean_degree_and_fractional_steps_rejected(self):
        with self.assertRaises(ValueError):
            lift(1.0, True)
        with self.assertRaises(ValueError):
            rollout(0.5, ((1.0, 0.0), (0.0, 0.9)), 1.5)

    def test_operator_overflow_rejected(self):
        with self.assertRaises(ValueError):
            predict_lifted(2.0, ((1e308, 0.0), (0.0, 1.0)))

    def test_nonfinite_operator_rejected(self):
        with self.assertRaises(ValueError):
            predict_lifted(0.5, ((1.0, 0.0), (0.0, float("nan"))))

    def test_negative_ridge_rejected(self):
        with self.assertRaises(ValueError):
            fit_edmd([0.1, 0.2, 0.3], [0.2, 0.3, 0.4], degree=1, ridge=-1.0)


if __name__ == "__main__":
    unittest.main()
