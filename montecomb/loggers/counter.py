class FunctionCallCounter:
    """A class to count the number of calls to a function"""

    def __init__(self):
        """Initializes the counter"""
        self._count = 0
        self._uniques = set()

    def __call__(self, state, value):
        """Increase the counter"""
        self._count += 1
        self._uniques.add(state)

    @property
    def unique_calls(self):
        return len(self._uniques)

    @property
    def total_calls(self):
        return self._count

    def reset(self):
        """Reset the counter back to 0"""
        self._count = 0
