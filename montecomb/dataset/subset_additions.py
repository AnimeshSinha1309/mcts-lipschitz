"""Module for housing the subset addition dataset"""

import typing as ty
import numpy as np

from .meta_dataset import MetaDataset


class SubsetAdditionDataset(MetaDataset):
    """Database modelling generalized evaluation as a sum over subsets.

    A dataset where the evaluation is a sum over subsets of the chosen set,
    and all subsets are equally likely to contribute to the result.
    """

    def __init__(
        self, n: int = 10, metrics: ty.List = None, *, features_count: int = 200
    ) -> None:
        """Create a new SubsetAdditionDataset object.
        This object will support a combinatorial action space of size 2^n

        :param n: The number of actions available in the dataset
        :param metrics: A list of logger objects which keep track of call metrics
        :param features_count: number of subsets which have non-0 evaluation
        """
        self.n = n
        self.metrics = metrics if metrics is not None else []
        self.rewards = {
            np.random.randint(2 ** n): np.random.random() * 2 - 1
            for _i in range(features_count)
        }

    def __call__(self, val: int) -> float:
        """Compute the evaluation for the given value

        :param val: The set to find the evaluation for, represented as int
        :return: The value of the evaluation
        """
        answer = 0
        for number, reward in self.rewards.items():
            if val & number == number:
                answer += reward
        for metric in self.metrics:
            metric(val, answer)  # Log all the results in
        return answer
