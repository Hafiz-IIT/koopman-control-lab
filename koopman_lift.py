def lift(x: float) -> tuple[float, float, float]:
    """Transparent polynomial lift [x, x^2, x^3]."""
    return (x, x*x, x*x*x)

def linear_step(z: tuple[float,float,float], rate: float = 0.9):
    return tuple(rate*v for v in z)
