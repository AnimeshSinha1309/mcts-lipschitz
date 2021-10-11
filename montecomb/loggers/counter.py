class FunctionCallCounter:
    """A class to count the number of calls to a function"""

    def __init__(self):
        """Initializes the counter"""
        self.count = 0

    def __call__(self, *args, **kwargs):
        """Increase the counter"""
        self.count += 1

    def reset(self):
        """Reset the counter back to 0"""
        self.count = 0
