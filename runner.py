import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import DDPGAgent, DDQNAgent, DQNAgent, greedy_policy
from covert import covert_sinr_threshold, kl_covert_sinr_threshold, min_detection_error
from env import FL_INPUT_DIM, SecureCovertD2DEnv
from federated import FederatedRiskTrainer
from metrics import binary_risk_metrics, mean_ci, summarize
from proposed import ProposedDDQNAgent
from utils import ensure_dir, lin_to_db, set_seed

BASELINES = ["greedy", "dqn", "ddqn", "ddpg"]                 # baselines.py
LABELS = {"greedy": "Greedy", "dqn": "DQN", "ddqn": "DDQN", "ddpg": "DDPG", "proposed": "Proposed"}
NO_FL = "no_fl"                                               
KEY_COLS = ["scheme", "aggregator", "num_d2d", "pmax_dbm", "phi_cov", "seed"]
POLICY_METRICS = ["avg_secrecy_rate_bps", "sum_secrecy_rate_bps", "avg_covert_risk", "avg_detection_error",
                  "covert_outage_prob", "secrecy_outage_prob", "cu_outage_prob", "jain_fairness",
                  "sum_d2d_throughput_bps", "median_warden_sinr_db", "total_power_w", "avg_reward"]
RISK_METRICS = ["roc_auc", "precision", "recall", "f1", "accuracy", "brier", "ece", "positive_rate"]
RATE_UNITS = {"bps": (1.0, "bits/s"), "kbps": (1e-3, "kbit/s"), "Mbps": (1e-6, "Mbit/s")}

RISK_SCENARIOS = {
    "clean": {},
    "label_flip_20pct_clients": {"label_flip_fraction": 0.2, "label_flip_rate": 1.0},
    "label_flip_40pct_clients_40pct_labels": {"label_flip_fraction": 0.4, "label_flip_rate": 0.4},
    "byzantine_sign_20pct_x5": {"byzantine_fraction": 0.2, "byzantine_mode": "sign", "byzantine_scale": 5.0},
    "byzantine_noise_20pct_x5": {"byzantine_fraction": 0.2, "byzantine_mode": "noise", "byzantine_scale": 5.0},
    "non_iid_dirichlet_0.1": {"dirichlet_alpha": 0.1},
    "dropout_30pct": {"dropout_prob": 0.3},
}
RISK_AGGREGATORS = ["local_only", "fedavg", "trimmed_mean", "trust"]


def _resolve_faults(faults, n_clients):
    f = dict(faults or {})
    for name in ("label_flip", "byzantine"):
        frac = f.pop(f"{name}_fraction", None)
        if frac:
            f[f"{name}_clients"] = list(range(int(math.ceil(frac * n_clients))))
    return f


def exploratory_policy(env):
    """Uniformly random actions. Not a baseline: used only to collect data for the risk-predictor study."""
    return np.random.randint(0, env.action_dim, size=env.num_agents)

def make_agent(cfg, env, scheme, aggregator, seed=0, faults=None):
    if scheme == "greedy":
        return None
    if scheme == "dqn":
        return DQNAgent(env.state_dim, env.action_dim, cfg)
    if scheme == "ddqn":
        return DDQNAgent(env.state_dim, env.action_dim, cfg)
    if scheme == "ddpg":
        return DDPGAgent(env.state_dim, env.num_rbs, env.L, cfg)
    if scheme == "proposed":
        return ProposedDDQNAgent(env.state_dim, env.action_dim, cfg, aggregator=aggregator,
                                 faults=_resolve_faults(faults, cfg.num_d2d), seed=seed)
    raise ValueError(f"unknown scheme {scheme}")


def train(cfg, scheme, aggregator, seed, faults=None, verbose=False):
    set_seed(seed)
    env = SecureCovertD2DEnv(cfg, use_fl_features=scheme == "proposed")
    agent = make_agent(cfg, env, scheme, aggregator, seed, faults)
    if agent is None:
        return None
    is_proposed = isinstance(agent, ProposedDDQNAgent)
    if is_proposed:
        agent.attach(env)

    step = 0
    for ep in range(cfg.episodes):
        states = env.reset()
        acc = defaultdict(float)
        for _ in range(cfg.steps_per_episode):
            actions = agent.act(states)
            next_states, rewards, done, info = env.step(actions)
            step += 1
            if is_proposed and step % cfg.fl_round_freq == 0:      
                agent.federated_round(env)
                next_states = env.build_agent_states()            
            agent.observe(states, actions, rewards, next_states)
            agent.train_step()
            for k in ("avg_reward", "avg_secrecy_rate_bps", "covert_outage_prob"):
                acc[k] += info[k]
            states = next_states
            if done:
                break
        if verbose and (ep + 1) % max(1, cfg.episodes // 5) == 0:
            T = cfg.steps_per_episode
            print(f"    [{scheme} N={cfg.num_d2d} seed={seed}] ep {ep + 1}/{cfg.episodes} "
                  f"reward={acc['avg_reward'] / T:.2f} sec={acc['avg_secrecy_rate_bps'] / T:.3g} bits/s "
                  f"cov_out={acc['covert_outage_prob'] / T:.2f}", flush=True)
    return agent


def evaluate(cfg, scheme, agent, eval_seed, episodes):
    set_seed(eval_seed)
    env = SecureCovertD2DEnv(cfg, use_fl_features=scheme == "proposed")
    if isinstance(agent, ProposedDDQNAgent):
        agent.attach(env)
    env.fl_log = []
    acc, n = defaultdict(float), 0
    for _ in range(episodes):
        states = env.reset()
        done = False
        while not done:
            actions = greedy_policy(env) if scheme == "greedy" else agent.act(states, greedy=True)
            states, _, done, info = env.step(actions)
            for k, v in info.items():
                acc[k] += v
            n += 1
    X = np.concatenate([f for f, _ in env.fl_log])
    y = np.concatenate([l for _, l in env.fl_log])
    return {k: v / n for k, v in acc.items()}, (X, y)


def make_key(scheme, aggregator, n, pmax, phi, seed):
    return (scheme, aggregator, int(n), float(pmax), float(phi), int(seed))


def _run_cfg(base_cfg, key):
    cfg = copy.deepcopy(base_cfg)
    _, _, cfg.num_d2d, cfg.d2d_max_power_dbm, cfg.phi_cov, _ = key
    return cfg


def run_one(args):
    base_cfg, key = args
    import torch
    torch.set_num_threads(1)
    scheme, aggregator, _, _, _, seed = key
    cfg = _run_cfg(base_cfg, key)
    t0 = time.time()
    agent = train(cfg, scheme, aggregator, seed)
    res, _ = evaluate(cfg, scheme, agent, 10_000 + seed, cfg.eval_episodes)
    row = dict(zip(KEY_COLS, key))
    row.update(res)
    row["train_time_s"] = round(time.time() - t0, 1)
    return row

class ResultStore:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        self.keys = set()
        for _, r in self.df.iterrows():
            self.keys.add(make_key(*[r[c] for c in KEY_COLS]))

    def __contains__(self, key):
        return key in self.keys

    def add(self, row):
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.keys.add(make_key(*[row[c] for c in KEY_COLS]))
        self.df.to_csv(self.path, index=False)


def _cfg_hash(cfg):
    skip = {"num_d2d", "d2d_max_power_dbm", "phi_cov", "output_dir", "d2d_user_values",
            "pmax_values_dbm", "pmax_num_d2d", "phi_cov_values", "rate_plot_unit"}
    d = {k: v for k, v in cfg.to_dict().items() if k not in skip}
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def collect(base_cfg, keys, store, jobs=1):
    todo = [k for k in dict.fromkeys(keys) if k not in store]
    print(f"[runs] {len(keys)} needed, {len(set(keys)) - len(todo)} cached, {len(todo)} to run "
          f"({jobs} process{'es' if jobs > 1 else ''})", flush=True)
    if not todo:
        return
    todo.sort(key=lambda k: (k[2], k[0] == "greedy"))             
    args = [(base_cfg, k) for k in todo]

    def _done(i, row):
        store.add(row)
        print(f"[runs] {i}/{len(todo)} {row['scheme']:9s} {row['aggregator']:6s} N={row['num_d2d']:<3d} "
              f"Pmax={row['pmax_dbm']:<5g} phi_cov={row['phi_cov']:<4g} seed={row['seed']} "
              f"({row['train_time_s']:.0f}s)", flush=True)

    if jobs > 1:
        with mp.get_context("spawn").Pool(jobs) as pool:
            for i, row in enumerate(pool.imap_unordered(run_one, args), 1):
                _done(i, row)
    else:
        for i, a in enumerate(args, 1):
            _done(i, run_one(a))

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _series(df, x_col, metric, **filt):
    d = df
    for k, v in filt.items():
        d = d[np.isclose(d[k], v)] if isinstance(v, float) else d[d[k] == v]
    rows = []
    for x, g in d.groupby(x_col):
        m, sd, half = mean_ci(g[metric].values)
        rows.append({x_col: x, "mean": m, "std": sd, "ci95": half, "n_seeds": len(g)})
    return pd.DataFrame(rows).sort_values(x_col)


def _plot(curves, x_col, xlabel, ylabel, path, scale=1.0, error_bars=False, data_dir=None):
    plt.figure(figsize=(7, 4.5))
    out = []
    for (label, s), mk in zip(curves.items(), MARKERS):
        if s.empty:
            continue
        if error_bars:
            plt.errorbar(s[x_col], s["mean"] * scale, yerr=s["ci95"] * scale, marker=mk, capsize=3, lw=1.8, label=label)
        else:
            plt.plot(s[x_col], s["mean"] * scale, marker=mk, lw=1.8, label=label)
        out.append(s.assign(curve=label))
    if out:
        plt.xticks(sorted(set(np.concatenate([s[x_col].values for s in out]))))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()
    if data_dir and out:
        name = os.path.splitext(os.path.basename(path))[0] + ".csv"
        pd.concat(out)[["curve", x_col, "mean", "std", "ci95", "n_seeds"]].to_csv(os.path.join(data_dir, name), index=False)

def run_figures(cfg, seeds, out, jobs=1, error_bars=False):
    ensure_dir(out)
    fig_dir, data_dir = os.path.join(out, "figures"), os.path.join(out, "figure_data")
    ensure_dir(fig_dir)
    ensure_dir(data_dir)
    store = ResultStore(os.path.join(out, f"runs_{_cfg_hash(cfg)}.csv"))

    Ns, P0, phi0 = list(cfg.d2d_user_values), float(cfg.d2d_max_power_dbm), float(cfg.phi_cov)
    Ps, N_p = list(cfg.pmax_values_dbm), cfg.pmax_num_d2d
    phis = list(cfg.phi_cov_values)
    methods = [(s, NO_FL) for s in BASELINES] + [("proposed", "trust")]

    keys = []
    for seed in seeds:
        for n in Ns:                                                 
            keys += [make_key(s, a, n, P0, phi0, seed) for s, a in methods]
        for p in Ps:                                                
            keys += [make_key(s, a, N_p, p, phi0, seed) for s, a in methods]
        for n in Ns:
            keys += [make_key("proposed", "trust", n, P0, phi, seed) for phi in phis]   
            keys += [make_key("proposed", "fedavg", n, P0, phi0, seed),                
                     make_key("proposed", "trust", n, P0, 0.0, seed)]
    collect(cfg, keys, store, jobs)

    df = store.df[store.df["seed"].isin(seeds)]
    scale, unit = RATE_UNITS[cfg.rate_plot_unit]
    kw = dict(error_bars=error_bars, data_dir=data_dir)
    xl_n = "Number of D2D users"

    def vs_n(metric):
        return {LABELS[s]: _series(df, "num_d2d", metric, scheme=s, aggregator=a, pmax_dbm=P0, phi_cov=phi0)
                for s, a in methods}

    for fname, metric, ylabel, sc in [
        ("01_avg_secrecy_rate_vs_d2d_users", "avg_secrecy_rate_bps", f"Average secrecy rate ({unit})", scale),
        ("02_avg_covert_risk_vs_d2d_users", "avg_covert_risk", "Average covert risk", 1.0),
        ("03_cellular_outage_vs_d2d_users", "cu_outage_prob", "Cellular outage probability", 1.0),
        ("04_fairness_index_vs_d2d_users", "jain_fairness", "Fairness index", 1.0),
        ("06_secrecy_outage_vs_d2d_users", "secrecy_outage_prob", "Secrecy outage probability", 1.0),
        ("07_covert_outage_vs_d2d_users", "covert_outage_prob", "Covert outage probability", 1.0),
        ("08_sum_d2d_throughput_vs_d2d_users", "sum_d2d_throughput_bps", f"Sum D2D throughput ({unit})", scale),
    ]:
        _plot(vs_n(metric), "num_d2d", xl_n, ylabel, os.path.join(fig_dir, fname + ".png"), sc, **kw)

    curves = {LABELS[s]: _series(df, "pmax_dbm", "avg_secrecy_rate_bps", scheme=s, aggregator=a,
                                 num_d2d=N_p, phi_cov=phi0) for s, a in methods}
    _plot(curves, "pmax_dbm", r"Maximum D2D transmit power $P_{\max}$ (dBm)", f"Average secrecy rate ({unit})",
          os.path.join(fig_dir, "05_avg_secrecy_rate_vs_pmax.png"), scale, **kw)

    curves = {rf"Proposed, $\varphi_{{cov}}$ = {phi:g}": _series(df, "num_d2d", "avg_covert_risk", scheme="proposed",
                                                               aggregator="trust", pmax_dbm=P0, phi_cov=float(phi))
              for phi in phis}
    _plot(curves, "num_d2d", xl_n, "Average covert risk",
          os.path.join(fig_dir, "09_avg_covert_risk_vs_d2d_users_phi_cov.png"), **kw)

    ablation = {
        "Proposed scheme w/o federated learning": dict(scheme="ddqn", aggregator=NO_FL, phi_cov=phi0),
        "Proposed scheme with FedAvg": dict(scheme="proposed", aggregator="fedavg", phi_cov=phi0),
        "Proposed scheme w/o covert penalty": dict(scheme="proposed", aggregator="trust", phi_cov=0.0),
        "Proposed scheme": dict(scheme="proposed", aggregator="trust", phi_cov=phi0),
    }
    for fname, metric, ylabel, sc in [
        ("10a_ablation_avg_secrecy_rate", "avg_secrecy_rate_bps", f"Average secrecy rate ({unit})", scale),
        ("10b_ablation_avg_covert_risk", "avg_covert_risk", "Average covert risk", 1.0),
    ]:
        curves = {lab: _series(df, "num_d2d", metric, pmax_dbm=P0, **f) for lab, f in ablation.items()}
        _plot(curves, "num_d2d", xl_n, ylabel, os.path.join(fig_dir, fname + ".png"), sc, **kw)

    summary = summarize(df, KEY_COLS[:-1], POLICY_METRICS)
    summary.to_csv(os.path.join(out, "summary_all_runs.csv"), index=False)
    print(f"[figures] saved to {fig_dir}")
    return df, summary

def _risk_test_set(cfg, seed, episodes):
    set_seed(10_000 + seed)
    env = SecureCovertD2DEnv(cfg, use_fl_features=False)
    env.fl_log = []
    for _ in range(episodes):
        env.reset()
        for _ in range(cfg.steps_per_episode):
            env.step(exploratory_policy(env))
    return np.concatenate([f for f, _ in env.fl_log]), np.concatenate([l for _, l in env.fl_log])


def run_risk(cfg, seeds, out, data_episodes=200, test_episodes=50, scenarios=None, aggregators=None):
    scenarios = scenarios or RISK_SCENARIOS
    aggregators = aggregators or RISK_AGGREGATORS
    ensure_dir(os.path.join(out, "figures"))
    rows = []
    for seed in seeds:
        X_test, y_test = _risk_test_set(cfg, seed, test_episodes)
        for sc_name, faults in scenarios.items():
            for agg in aggregators:
                set_seed(seed)
                env = SecureCovertD2DEnv(cfg, use_fl_features=False)
                fl = FederatedRiskTrainer(cfg, FL_INPUT_DIM, aggregator=agg,
                                          faults=_resolve_faults(faults, cfg.num_d2d), seed=seed)
                step = 0
                for _ in range(data_episodes):
                    env.reset()
                    for _ in range(cfg.steps_per_episode):
                        env.step(exploratory_policy(env))
                        step += 1
                        if step % cfg.fl_round_freq == 0:
                            fl.aggregate(env.local_fl_buffers)
                r = binary_risk_metrics(y_test, fl.predict(X_test))
                rows.append({"scenario": sc_name, "aggregator": agg, "seed": seed, **r})
                print(f"[risk] seed={seed} {sc_name:38s} {agg:13s} AUC={r['roc_auc']:.3f} "
                      f"F1={r['f1']:.3f} Brier={r['brier']:.3f}", flush=True)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(out, "risk_per_seed.csv"), index=False)
    summary = summarize(per_seed, ["scenario", "aggregator"], RISK_METRICS)
    summary.to_csv(os.path.join(out, "risk_summary.csv"), index=False)

    sc_list = list(scenarios)
    x = np.arange(len(sc_list))
    width = 0.8 / len(aggregators)
    plt.figure(figsize=(10, 4.5))
    for k, agg in enumerate(aggregators):
        s = summary[summary["aggregator"] == agg].set_index("scenario").reindex(sc_list)
        plt.bar(x + k * width, s["roc_auc_mean"], width, label=agg)
    plt.xticks(x + width * (len(aggregators) - 1) / 2, sc_list, rotation=25, ha="right", fontsize=8)
    plt.ylabel("Risk predictor ROC-AUC")
    plt.ylim(0.4, 1.0)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "risk_auc_by_scenario.png"), dpi=300)
    plt.close()
    return per_seed, summary


def run_robust_rl(cfg, seeds, out, scenarios=None):
    scenarios = scenarios or {k: RISK_SCENARIOS[k] for k in
                              ("clean", "label_flip_20pct_clients", "byzantine_sign_20pct_x5")}
    ensure_dir(out)
    rows = []
    for seed in seeds:
        for sc_name, faults in scenarios.items():
            for agg in ("fedavg", "trust"):
                print(f"[robust_rl] seed={seed} scenario={sc_name} aggregator={agg}", flush=True)
                agent = train(cfg, "proposed", agg, seed, faults=faults)
                res, (X, y) = evaluate(cfg, "proposed", agent, 10_000 + seed, cfg.eval_episodes)
                risk = binary_risk_metrics(y, agent.risk_estimator.predict(X))
                rows.append({"scenario": sc_name, "aggregator": agg, "seed": seed, **res,
                             "risk_roc_auc": risk["roc_auc"]})
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(os.path.join(out, "robust_rl_per_seed.csv"), index=False)
    summary = summarize(per_seed, ["scenario", "aggregator"], POLICY_METRICS + ["risk_roc_auc"])
    summary.to_csv(os.path.join(out, "robust_rl_summary.csv"), index=False)
    return per_seed, summary


def run_xi_curve(cfg, out, n_values=(50, 100, 500, 1000)):
    ensure_dir(os.path.join(out, "figures"))
    g_db = np.linspace(-40, 10, 501)
    data = {"warden_sinr_db": g_db}
    plt.figure(figsize=(6, 4))
    for n in n_values:
        xi = min_detection_error(10 ** (g_db / 10), n)
        data[f"xi_n{n}"] = xi
        plt.plot(g_db, xi, label=f"n = {n}")
    plt.axhline(1 - cfg.covert_epsilon, ls="--", c="gray", lw=0.8)
    plt.xlabel("Warden SINR (dB)")
    plt.ylabel(r"Minimum detection error $\xi^{\star}$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "xi_vs_warden_sinr.png"), dpi=300)
    plt.close()
    pd.DataFrame(data).to_csv(os.path.join(out, "xi_curve.csv"), index=False)
    th = pd.DataFrame([{"n": n, "epsilon": eps,
                        "exact_threshold_db": float(lin_to_db(covert_sinr_threshold(n, eps))),
                        "kl_threshold_db": float(lin_to_db(kl_covert_sinr_threshold(n, eps)))}
                       for n in n_values for eps in (0.05, 0.1)])
    th.to_csv(os.path.join(out, "covert_thresholds.csv"), index=False)
    print(th.to_string(index=False))
    return th
