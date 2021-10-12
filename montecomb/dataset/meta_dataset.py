import abc

import typing as ty


class MetaDataset(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: int) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: int, action: int) -> ty.Tuple[int, float]:
        raise NotImplementedError
