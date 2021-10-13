import os
import time

import wandb

import montecomb

if __name__ == "__main__":
    os.system("wandb login d43f6dc5f4f9981ac8b6bffd1ab5db7d9ac45480")
    wandb.init(
        project="mcts-lipschitz",
        name="mcts-sweep-1",
        save_code=False,
        config=dict(
            mcts_exploration_ratio=1.0,
            mcts_discount_factor=0.95,
        ),
        resume=False,
    )

    metrics = [montecomb.loggers.counter.FunctionCallCounter()]
    for df in [
        montecomb.dataset.local_effects.LocalEffectsDataset(
            15, 4, metrics, features_count=300
        ),
        montecomb.dataset.subset_additions.SubsetAdditionDataset(
            15, metrics, features_count=300
        ),
    ]:
        cls = montecomb.agents.mcts.MCTSAgent
        time.sleep(0.1)
        agent = cls(df.n, df)
        result = agent.act()
        print(
            f"{cls.__name__} on {df.__class__.__name__}: chose action {result} with reward {df(result)}"
            f" with {metrics[0].unique_calls} unique calls ({metrics[0].total_calls} total)."
        )
        wandb.log({f"unique_calls-{df.__class__.__name__}": metrics[0].unique_calls,
                   f"reward-{df.__class__.__name__}": df(result)})
        metrics[0].reset()
        time.sleep(0.1)
