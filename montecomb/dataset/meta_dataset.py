import abc


class MetaDataset(abc.ABC):
    @abc.abstractmethod
    def __call__(self, state: int) -> float:
        """Compute the reward for any given state
        :param state: The current set of actions, as an integer
        :return: The evaluation for that state
        """
        raise NotImplementedError

    @staticmethod
    def step(state: int, action: int) -> int:
        """Given a state and action value, get the next state you would reach.
        This is the same as gym's env.step, it's a part of the evaluator because
        all evaluators are responsible for implementing the combinatorial
        structure of the search space.
        :param state: The current action set, as an integer
        :param action: The action being currently taken
        :return: The resulting action set, as an integer
        """
        new_state = state | (1 << action)
        return new_state
