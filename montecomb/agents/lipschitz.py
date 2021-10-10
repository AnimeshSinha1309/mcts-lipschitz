"""
The idea of implementing MCTS inspired from the Lipschitz bandit problem
"""
import numpy as np

from ..dataset.meta_dataset import MetaDataset


class LipschitzSamplerAgent:
    """A lipschitz sample agent for evaluating the combination of moves
    Uses adaptive discretization for avoiding normal MCTS.
    """

    def __init__(
        self,
        num_actions: int,
        evaluator: 'MetaDataset',
    ):
        """Initialize a new lipschitz sampler object.
        This maintains the mean rewards of all the clusters and their variances,
        activates and eliminates while performing adaptive discretization and
        optimism under uncertainty.
        :param num_actions: Maximum number of actions that can be taken
        :param evaluator: Loss function which takes the state and given evaluation
        """
        self.evaluator = evaluator
        self.num_actions = num_actions

    def search(self, _n_trials) -> None:
        """Searches the space with adaptive discretization
        :param _n_trials: Dummy variable
        """

    def act(self) -> int:
        """Searches and returns the best result
        :return: The value of the best state found
        """
        self.search(0)
        return np.random.randint(self.num_actions)
