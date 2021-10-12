import abc


class MetaDataset(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: int) -> float:
        raise NotImplementedError

    @staticmethod
    def step(state: int, action: int) -> int:
        new_state = state | (1 << action)
        return new_state
