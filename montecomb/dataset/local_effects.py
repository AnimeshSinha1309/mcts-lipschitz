import typing
import typing as ty
import numpy as np

from .subset_additions import SubsetAdditionDataset


class LocalEffectsDataset(SubsetAdditionDataset):
    def __init__(self, n: int = 10, local_radius: int = 4, metrics: ty.List = None):
        super().__init__(n, metrics)
        self.local_radius = local_radius

        def generate_random_small_set() -> int:
            set_size = np.random.randint(1, self.local_radius + 1)
            chosen_bits = np.random.choice(self.n, size=set_size, replace=False)
            return typing.cast(int, np.sum(1 << chosen_bits))

        self.rewards = {
            generate_random_small_set(): np.random.random() * 2 - 1 for _i in range(200)
        }
