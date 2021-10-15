"""Module housing the local effects dataset"""

import typing as ty
import numpy as np

from .subset_additions import SubsetAdditionDataset


class LocalEffectsDataset(SubsetAdditionDataset):
    """Database modelling local effects as a sum over subsets.

    A dataset where the evaluation is a sum over subsets of the chosen set,
    but only small subsets contribute to the evaluation.
    """

    def __init__(
        self,
        n: int = 10,
        local_radius: int = 4,
        metrics: ty.List = None,
        *args,
        **kwargs
    ):
        """Create a new LocalEffectsDataset

        :param n: Number of actions we can take
        :param local_radius: Maximum size of subset which can contribute to evaluation
        :param metrics: A list of logger objects which keep track of call metrics
        :param args: Arguments to pass to the parent constructor
        :param kwargs: Keyword Arguments to pass to the parent constructor
        """
        super().__init__(n, metrics, *args, **kwargs)
        self.local_radius = local_radius

        def generate_random_small_set() -> int:
            set_size = np.random.randint(1, self.local_radius + 1)
            chosen_bits = np.random.choice(self.n, size=set_size, replace=False)
            return ty.cast(int, np.sum(1 << chosen_bits))

        self.rewards = {
            generate_random_small_set(): np.random.random() * 2 - 1 for _i in range(200)
        }
