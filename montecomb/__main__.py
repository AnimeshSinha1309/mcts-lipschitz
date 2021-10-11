from .dataset.subset_additions import SubsetAdditionDataset
from .agents.mcts import MCTSAgent
from .agents.random import RandomAgent
from .agents.full import FullSearchAgent
from .agents.lipschitz import LipschitzSamplerAgent
from .loggers.counter import FunctionCallCounter


if __name__ == "__main__":
    metrics = [FunctionCallCounter()]
    df = SubsetAdditionDataset(13, metrics)
    for cls in [LipschitzSamplerAgent, MCTSAgent, RandomAgent, FullSearchAgent]:
        agent = cls(df.n, df)
        result = agent.act()
        print(
            f"{cls.__name__}: chose action {result} with reward {df(result)} with {metrics[0].count} calls.",
            flush=True,
        )
        metrics[0].reset()
