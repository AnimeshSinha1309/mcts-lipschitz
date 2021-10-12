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
    for df in [LocalEffectsDataset(13, 4, metrics), SubsetAdditionDataset(13, metrics)]:
        print(
            f"--------------{len(df.__class__.__name__) * '-'}\n"
            f"Evaluating on {df.__class__.__name__}\n"
            f"--------------{len(df.__class__.__name__) * '-'}"
        )
        for cls in [BiasingSamplerAgent, MCTSAgent, RandomAgent, FullSearchAgent]:
            agent = cls(df.n, df)
            result = agent.act()
            print(
                f"{cls.__name__}: chose action {result} with reward {df(result)}"
                f" with {metrics[0].count} calls."
            )
            metrics[0].reset()
            time.sleep(0.1)
