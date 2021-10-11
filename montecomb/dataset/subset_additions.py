import numpy as np

from .meta_dataset import MetaDataset


class SubsetAdditionDataset(MetaDataset):
    def __init__(self, n: int = 10):
        self.n = n
        self.rewards = {
            np.random.randint(2 ** n): np.random.random() * 2 - 1 for _i in range(200)
        }

    def __call__(self, val):
        answer = 0
        for number, reward in self.rewards.items():
            if val & number == number:
                answer += reward
        return answer

    def step(self, state, action):
        new_state = state | (1 << action)
        new_reward = self(new_state) - self(state)
        return new_state, new_reward
