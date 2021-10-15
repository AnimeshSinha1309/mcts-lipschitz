"""Random Agent that just tries out all actions and takes the best one"""

import tqdm.auto as tqdm

from ..dataset.meta_dataset import MetaDataset


class FullSearchAgent:
    """A full-search agent for evaluating the combination of moves"""

    def __init__(
        self,
        num_actions: int,
        evaluator: MetaDataset,
    ) -> None:
        """Initialize a new full searcher object.
        This initializes an agent which will try out all the different
        states to try out and evaluate which states will be the best.
        :param num_actions: Maximum number of actions that can be taken
        :param evaluator: Loss function which takes the state and given evaluation
        """
        self.evaluator = evaluator
        self.num_actions = num_actions
        self.best_state: int = 0
        self.best_reward: float = 0.0

    def search(self, _n_trials: int = 0) -> None:
        """Perform the full search for best value
        :param _n_trials: Dummy variable
        """
        for state in tqdm.trange(2 ** self.num_actions):
            reward = self.evaluator(state)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_state = state

    def act(self) -> int:
        """Searches and returns the best result
        :return: The value of the best state found
        """
        self.search(0)
        return self.best_state
