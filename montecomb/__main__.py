from .dataset.subset_additions import SubsetAdditionDatasets
from .agents.mcts import MCTSAgent
from .agents.random import RandomAgent
from .agents.full import FullSearchAgent
from .agents.lipschitz import LipschitzSamplerAgent

if __name__ == "__main__":
    df = SubsetAdditionDatasets(20)
    for cls in [MCTSAgent, RandomAgent, FullSearchAgent, LipschitzSamplerAgent]:
        agent = cls(df.n, df)
        result = agent.act()
        print(f"{cls.__name__}: chose action {result} with reward {df(result)}", flush=True)
