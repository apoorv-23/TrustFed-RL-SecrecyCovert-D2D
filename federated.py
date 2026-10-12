import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class RiskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class LocalRiskDataset:

    def __init__(self, maxlen: int, dim: int = 5):
        self.maxlen = int(maxlen)
        self.X = np.zeros((self.maxlen, dim), dtype=np.float32)
        self.y = np.zeros(self.maxlen, dtype=np.float32)
        self.ptr = 0
        self.n = 0

    def add(self, x, y):
        self.X[self.ptr] = x
        self.y[self.ptr] = y
        self.ptr = (self.ptr + 1) % self.maxlen
        self.n = min(self.n + 1, self.maxlen)

    def size(self) -> int:
        return self.n

    def as_arrays(self):
        return self.X[:self.n].copy(), self.y[:self.n].copy()


def _get_vec(model) -> np.ndarray:
    return parameters_to_vector(model.parameters()).detach().cpu().numpy().astype(np.float64)


def _set_vec(model, vec: np.ndarray):
    vector_to_parameters(torch.as_tensor(vec, dtype=torch.float32), model.parameters())


class FederatedRiskTrainer:
    def __init__(self, cfg, input_dim: int, aggregator: str = "trust", faults=None, seed: int = 0,
                 local_client: int = -1):
        if aggregator not in ("trust", "fedavg", "trimmed_mean", "local_only"):
            raise ValueError(f"unknown aggregator {aggregator}")
        self.cfg = cfg
        self.aggregator = aggregator
        self.faults = dict(faults or {})
        self.rng = np.random.default_rng(seed + 7919)
        self.device = torch.device(cfg.device)
        self.global_model = RiskMLP(input_dim, cfg.fl_hidden_dim).to(self.device)
        self.rounds_done = 0
        self.prev_trust = {}            
        self.history = []
        self.num_clients = cfg.num_d2d
        self.local_client = local_client % self.num_clients
        self.class_keep = {}
        alpha = self.faults.get("dirichlet_alpha")
        if alpha:
            for c in range(self.num_clients):
                p = self.rng.dirichlet([alpha, alpha])
                self.class_keep[c] = p / p.max()

    @property
    def is_trained(self) -> bool:
        return self.rounds_done > 0

    def _client_data(self, cid, ds):
        X, y = ds.as_arrays()
        if cid in self.class_keep:
            keep = self.class_keep[cid][y.astype(int)]
            mask = self.rng.random(len(y)) < keep
            X, y = X[mask], y[mask]
        if cid in self.faults.get("label_flip_clients", []):
            y = y.copy()
            flip = self.rng.random(len(y)) < self.faults.get("label_flip_rate", 1.0)
            y[flip] = 1.0 - y[flip]
        return X, y

    def _train_local(self, X, y):
        model = copy.deepcopy(self.global_model)
        model.train()
        n = len(X)
        perm = self.rng.permutation(n)
        n_val = max(1, int(self.cfg.fl_val_fraction * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        Xt = torch.as_tensor(X[tr_idx], device=self.device)
        yt = torch.as_tensor(y[tr_idx], device=self.device).unsqueeze(1)
        Xv = torch.as_tensor(X[val_idx], device=self.device)
        yv = torch.as_tensor(y[val_idx], device=self.device).unsqueeze(1)
        opt = optim.Adam(model.parameters(), lr=self.cfg.fl_local_lr)
        bce = nn.BCELoss()
        bs = self.cfg.fl_batch_size
        for _ in range(self.cfg.fl_local_epochs):
            order = self.rng.permutation(len(tr_idx))
            for s in range(0, len(order), bs):
                idx = torch.as_tensor(order[s:s + bs])
                loss = bce(model(Xt[idx]).clamp(1e-6, 1 - 1e-6), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = bce(model(Xv).clamp(1e-6, 1 - 1e-6), yv).item()
        return _get_vec(model), val_loss, len(tr_idx)

    def _trust(self, cids, U, val_losses):
        c = self.cfg
        ref = U.mean(axis=0) if c.trust_reference == "mean" else np.median(U, axis=0)
        ref_norm = np.linalg.norm(ref) + 1e-12
        norms = np.linalg.norm(U, axis=1) + 1e-12
        consistency = np.maximum((U * ref).sum(axis=1) / (norms * ref_norm), 0.0)                   
        dev = np.linalg.norm(U - ref, axis=1)
        scale = (np.std(dev) if c.trust_dev_scale == "std" else np.median(dev)) + 1e-8
        deviation = np.exp(-dev / scale)                                               
        validation = np.exp(-np.asarray(val_losses))                                   
        raw = consistency ** c.trust_tau_c * validation ** c.trust_tau_v * deviation ** c.trust_tau_d
        trust = raw / raw.sum() if raw.sum() > 0 else np.full(len(cids), 1.0 / len(cids))
        trust = np.clip(trust, c.trust_clip_min, c.trust_clip_max)
        trust = trust / trust.sum()
        smooth = np.array([c.trust_ema_lambda * self.prev_trust.get(cid, t) + (1 - c.trust_ema_lambda) * t
                           for cid, t in zip(cids, trust)])
        smooth = smooth / smooth.sum()
        for cid, s in zip(cids, smooth):
            self.prev_trust[cid] = float(s)
        return smooth, consistency, validation, deviation

    def aggregate(self, datasets, round_idx=None):
        cfg = self.cfg
        cids = [i for i, ds in enumerate(datasets) if ds.size() >= cfg.fl_min_samples]
        p_drop = self.faults.get("dropout_prob", 0.0)
        if p_drop > 0:
            cids = [cid for cid in cids if self.rng.random() >= p_drop]
        if self.aggregator == "local_only":
            cids = [cid for cid in cids if cid == self.local_client]
        m = getattr(cfg, "fl_clients_per_round", 0)
        if m and len(cids) > m:                       
            cids = sorted(self.rng.choice(cids, size=m, replace=False).tolist())

        g = _get_vec(self.global_model)
        local, vals, ns, used = [], [], [], []
        for cid in cids:
            X, y = self._client_data(cid, datasets[cid])
            if len(X) < 16:
                continue
            vec, vl, n = self._train_local(X, y)
            local.append(vec); vals.append(vl); ns.append(n); used.append(cid)
        log = {"round": self.rounds_done, "aggregator": self.aggregator, "clients": used}
        if not used:
            self.history.append(log)
            return log

        U = np.stack(local) - g
        for k, cid in enumerate(used):
            if cid in self.faults.get("byzantine_clients", []):
                scale = self.faults.get("byzantine_scale", 5.0)
                if self.faults.get("byzantine_mode", "sign") == "sign":
                    U[k] = -scale * U[k]
                else:
                    U[k] = self.rng.normal(0.0, 1.0, U.shape[1]) * scale * (np.linalg.norm(U[k]) / np.sqrt(U.shape[1]) + 1e-8)
        ns = np.asarray(ns, dtype=np.float64)

        if self.aggregator in ("fedavg", "local_only"):
            w = ns / ns.sum()
            agg = (w[:, None] * U).sum(axis=0)
        elif self.aggregator == "trimmed_mean":
            k = int(np.floor(cfg.trimmed_mean_fraction * len(used)))
            if len(used) - 2 * k < 1:
                k = 0
            S = np.sort(U, axis=0)
            agg = S[k:len(used) - k].mean(axis=0)
            w = np.full(len(used), 1.0 / len(used))
        else:
            trust, cons, val, dev = self._trust(used, U, vals)
            w = ns * trust
            w = w / w.sum()
            agg = (w[:, None] * U).sum(axis=0)
            log.update({"trust": trust.tolist(), "consistency": cons.tolist(),
                        "validation": val.tolist(), "deviation": dev.tolist()})

        _set_vec(self.global_model, g + agg)
        self.rounds_done += 1
        log["weights"] = np.asarray(w).tolist()
        self.history.append(log)
        return log

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        self.global_model.eval()
        with torch.no_grad():
            return self.global_model(torch.as_tensor(X, device=self.device)).cpu().numpy().reshape(-1)
