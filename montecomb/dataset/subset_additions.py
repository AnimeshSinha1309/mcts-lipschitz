import typing as ty
import numpy as np

from .meta_dataset import MetaDataset


class SubsetAdditionDataset(MetaDataset):
    def __init__(
        self, n: int = 10, metrics: ty.List = None, *, features_count: int = 200
    ):
        self.n = n
        self.metrics = metrics if metrics is not None else []
        self.rewards = {
            np.random.randint(2 ** n): np.random.random() * 2 - 1
            for _i in range(features_count)
        }

    def __call__(self, val):
        answer = 0
        for number, reward in self.rewards.items():
            if val & number == number:
                answer += reward
        for metric in self.metrics:
            metric(val, answer)  # Log all the results in
        return answer
