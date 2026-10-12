from env import FL_INPUT_DIM
from federated import FederatedRiskTrainer
from rl_core import SharedQLearner


class ProposedDDQNAgent(SharedQLearner):
    def __init__(self, state_dim: int, action_dim: int, cfg, aggregator: str = "trust",
                 faults=None, seed: int = 0):
        super().__init__(state_dim, action_dim, cfg, use_double=True)
        self.aggregator = aggregator
        self.risk_estimator = FederatedRiskTrainer(cfg, FL_INPUT_DIM, aggregator=aggregator,
                                                   faults=faults, seed=seed)

    def attach(self, env):
        env.attach_risk_model(self.risk_estimator)

    def federated_round(self, env):
        return self.risk_estimator.aggregate(env.local_fl_buffers)

    def save(self, path):
        import torch
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "risk_model": self.risk_estimator.global_model.state_dict(),
                    "env_steps": self.env_steps, "train_steps": self.train_steps}, path)
