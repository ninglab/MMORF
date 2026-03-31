from .base import ValueFn
class Synth(ValueFn):
    """
    A value function for synthesis planning that always returns 1.
    This is a placeholder and should be replaced with a more meaningful implementation.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.coeff = 1.0
        self.intercept = 0.0
        self.children = []

    def __call__(self, v_orig=None, **kwargs):
        return v_orig * self.coeff + self.intercept
    
    def __repr__(self):
        return f"{self.coeff} * Synth() + {self.intercept}"