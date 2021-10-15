"""The main script that tests all agents against the different datasets."""


import time

from .dataset.subset_additions import SubsetAdditionDataset
from .dataset.local_effects import LocalEffectsDataset
from .agents.mcts import MCTSAgent
from .agents.random import RandomAgent
from .agents.full import FullSearchAgent
from .agents.lipschitz import BiasingSamplerAgent
from .loggers.counter import FunctionCallCounter


if __name__ == "__main__":
    metrics = [FunctionCallCounter()]
    for df in [
        LocalEffectsDataset(15, 4, metrics, features_count=300),
        SubsetAdditionDataset(15, metrics, features_count=300),
    ]:
        print(
            f"--------------{len(df.__class__.__name__) * '-'}\n"
            f"Evaluating on {df.__class__.__name__}\n"
            f"--------------{len(df.__class__.__name__) * '-'}"
        )
        for cls in [BiasingSamplerAgent, MCTSAgent, RandomAgent, FullSearchAgent]:
            time.sleep(0.1)
            agent = cls(df.n, df)
            result = agent.act()
            print(
                f"{cls.__name__}: chose action {result} with reward {df(result)}"
                f" with {metrics[0].unique_calls} unique calls ({metrics[0].total_calls} total)."
            )
            metrics[0].reset()
            time.sleep(0.1)
