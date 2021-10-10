import abc


class MetaDataset(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: int) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, state: int, action: int) -> tuple[int, float]:
        raise NotImplementedError
