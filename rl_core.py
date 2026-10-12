import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer


def mlp(inp: int, out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out))


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(state_dim, action_dim, hidden_dim)

    def forward(self, x):
        return self.net(x)


class SharedQLearner:
    def __init__(self, state_dim: int, action_dim: int, cfg, use_double: bool):
        self.cfg = cfg
        self.state_dim, self.action_dim = state_dim, action_dim
        self.use_double = use_double
        self.device = torch.device(cfg.device)
        self.online = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size, (state_dim,))
        self.env_steps = 0
        self.train_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(1, self.cfg.epsilon_decay_env_steps))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, states, greedy: bool = False):
        eps = 0.0 if greedy else self.epsilon()
        with torch.no_grad():
            q = self.online(torch.as_tensor(states, dtype=torch.float32, device=self.device)).cpu().numpy()
        actions = q.argmax(axis=1)
        explore = np.random.rand(len(actions)) < eps
        actions[explore] = np.random.randint(0, self.action_dim, size=int(explore.sum()))
        return actions.astype(np.int64)

    def observe(self, states, actions, rewards, next_states):
        self.replay.push_many(states, actions, rewards, next_states)
        self.env_steps += 1

    def train_step(self):
        if len(self.replay) < self.cfg.min_replay_for_train:
            return None
        s, a, r, ns, d = self.replay.sample(self.cfg.batch_size)
        dev = self.device
        s, ns = torch.as_tensor(s, device=dev), torch.as_tensor(ns, device=dev)
        a = torch.as_tensor(a, device=dev).unsqueeze(1)
        r = torch.as_tensor(r, device=dev).unsqueeze(1)
        d = torch.as_tensor(d, device=dev).unsqueeze(1)
        q = self.online(s).gather(1, a)
        with torch.no_grad():
            if self.use_double:
                na = self.online(ns).argmax(dim=1, keepdim=True)        
                nq = self.target(ns).gather(1, na)                      
            else:
                nq = self.target(ns).max(dim=1, keepdim=True)[0]
            y = r + self.cfg.gamma * (1.0 - d) * nq
        loss = F.smooth_l1_loss(q, y) if self.cfg.use_huber_loss else F.mse_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "env_steps": self.env_steps, "train_steps": self.train_steps}, path)
