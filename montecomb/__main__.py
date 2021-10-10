import numpy as np

from .dataset.subset_additions import SubsetAdditionDatasets
from .agents.mcts import MCTSAgent


if __name__ == "__main__":
    df = SubsetAdditionDatasets(10)
    mcts = MCTSAgent(0, df.n, df)
    result = mcts.act()
    print(f"Chosen action {result} with reward {df(result)}.")
    print(f"Best reward was {np.max([df(val) for val in range(2 ** df.n)])}")
