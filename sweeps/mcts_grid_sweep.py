import os
import time

import numpy as np
import wandb

import montecomb

if __name__ == "__main__":
    os.system("wandb login d43f6dc5f4f9981ac8b6bffd1ab5db7d9ac45480")
    wandb.init(
        project="mcts-lipschitz",
        name="mcts-sweep-2",
        save_code=False,
        config=dict(
            mcts_exploration_ratio=1.0,
            mcts_discount_factor=0.95,
        ),
        resume=False,
    )

    metrics = [montecomb.loggers.counter.FunctionCallCounter()]
    cls = montecomb.agents.mcts.MCTSAgent
    cls.HYPER_PARAMETER_EXPLORATION = wandb.config.mcts_exploration_ratio
    cls.HYPER_PARAM_DISCOUNT_FACTOR = wandb.config.mcts_discount_factor

    unique_calls, rewards = [], []
    for _ in range(10):
        df = montecomb.dataset.local_effects.LocalEffectsDataset(15, 4, metrics, features_count=300)
        time.sleep(0.1)
        agent = cls(df.n, df)
        result = agent.act()
        print(
            f"{cls.__name__} on {df.__class__.__name__}: chose action {result} with reward {df(result)}"
            f" with {metrics[0].unique_calls} unique calls ({metrics[0].total_calls} total)."
        )
        rewards.append(df(result))
        unique_calls.append(metrics[0].unique_calls)
        wandb.log({f"unique_calls-intermediate": unique_calls[-1],
                   f"reward-intermediate": rewards[-1]})
        metrics[0].reset()
        time.sleep(0.1)
    wandb.log({f"unique_calls": np.mean(unique_calls[-1]),
               f"reward": np.mean(rewards)})
