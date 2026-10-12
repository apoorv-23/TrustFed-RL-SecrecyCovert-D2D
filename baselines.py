
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from rl_core import SharedQLearner, mlp


#  Greedy
def greedy_policy(env):
    meas = env._meas
    order = np.argsort(-meas["snr_db"])
    taken = set()
    actions = np.zeros(env.num_agents, dtype=np.int64)
    for i in order:
        free = [k for k in range(env.num_rbs) if k not in taken] or list(range(env.num_rbs))
        rb = min(free, key=lambda k: meas["inr_db"][i, k])
        taken.add(rb)
        actions[i] = rb * env.L + (env.L - 1)
    return actions


# DQN
class DQNAgent(SharedQLearner):
    def __init__(self, state_dim: int, action_dim: int, cfg):
        super().__init__(state_dim, action_dim, cfg, use_double=False)


#  DDQN
class DDQNAgent(SharedQLearner):
    def __init__(self, state_dim: int, action_dim: int, cfg):
        super().__init__(state_dim, action_dim, cfg, use_double=True)


#  DDPG
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = mlp(state_dim, 2, hidden_dim)

    def forward(self, s):
        return torch.tanh(self.net(s))


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = mlp(state_dim + 2, 1, hidden_dim)

    def forward(self, s, u):
        return self.net(torch.cat([s, u], dim=-1))


class DDPGAgent:
    def __init__(self, state_dim: int, num_rbs: int, num_power_levels: int, cfg):
        self.cfg = cfg
        self.K, self.L = num_rbs, num_power_levels
        self.action_dim = num_rbs * num_power_levels
        self.device = torch.device(cfg.device)
        h = cfg.hidden_dim
        self.actor = Actor(state_dim, h).to(self.device)
        self.actor_t = Actor(state_dim, h).to(self.device)
        self.critic = Critic(state_dim, h).to(self.device)
        self.critic_t = Critic(state_dim, h).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.ddpg_actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.ddpg_critic_lr)
        self.replay = ReplayBuffer(cfg.replay_size, (state_dim,), (2,), np.float32)
        self.env_steps = 0
        self.train_steps = 0
        self.warmup_env_steps = int(np.ceil(cfg.min_replay_for_train / max(1, cfg.num_d2d)))
        self._last_u = None

    def epsilon(self) -> float:
        frac = min(1.0, self.env_steps / max(1, self.cfg.epsilon_decay_env_steps))
        return self.cfg.ddpg_noise_start + frac * (self.cfg.ddpg_noise_end - self.cfg.ddpg_noise_start)

    def quantize(self, u):
        k = np.minimum(np.floor((u[:, 0] + 1.0) / 2.0 * self.K), self.K - 1).astype(np.int64)
        l = np.minimum(np.floor((u[:, 1] + 1.0) / 2.0 * self.L), self.L - 1).astype(np.int64)
        return k * self.L + l

    def act(self, states, greedy: bool = False):
        n = len(states)
        if not greedy and self.env_steps < self.warmup_env_steps:
            u = np.random.uniform(-1.0, 1.0, size=(n, 2))
        else:
            with torch.no_grad():
                u = self.actor(torch.as_tensor(states, dtype=torch.float32, device=self.device)).cpu().numpy()
            if not greedy:
                u = np.clip(u + self.epsilon() * np.random.randn(n, 2), -1.0, 1.0)
        self._last_u = u.astype(np.float32)
        return self.quantize(self._last_u)

    def observe(self, states, actions, rewards, next_states):
        self.replay.push_many(states, self._last_u, rewards, next_states)
        self.env_steps += 1

    def _soft_update(self, net, target):
        tau = self.cfg.ddpg_tau
        with torch.no_grad():
            for p, pt in zip(net.parameters(), target.parameters()):
                pt.mul_(1.0 - tau).add_(tau * p)

    def train_step(self):
        if len(self.replay) < self.cfg.min_replay_for_train:
            return None
        s, u, r, ns, d = self.replay.sample(self.cfg.batch_size)
        dev = self.device
        s, ns = torch.as_tensor(s, device=dev), torch.as_tensor(ns, device=dev)
        u = torch.as_tensor(u, device=dev)
        r = torch.as_tensor(r, device=dev).unsqueeze(1)
        d = torch.as_tensor(d, device=dev).unsqueeze(1)

        with torch.no_grad():
            y = r + self.cfg.gamma * (1.0 - d) * self.critic_t(ns, self.actor_t(ns))
        q = self.critic(s, u)
        critic_loss = F.smooth_l1_loss(q, y) if self.cfg.use_huber_loss else F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_t)
        self._soft_update(self.critic, self.critic_t)
        self.train_steps += 1
        return float(critic_loss.item())

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "actor_target": self.actor_t.state_dict(), "critic_target": self.critic_t.state_dict(),
                    "env_steps": self.env_steps, "train_steps": self.train_steps}, path)
