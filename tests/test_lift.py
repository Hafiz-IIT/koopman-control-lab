import unittest
from koopman_lift import lift, linear_step

class LiftTests(unittest.TestCase):
    def test_polynomial_lift(self):
        self.assertEqual(lift(2.0),(2.0,4.0,8.0))
    def test_step(self):
        self.assertEqual(linear_step((1.0,2.0,3.0),0.5),(0.5,1.0,1.5))

if __name__=="__main__":
    unittest.main()
