import argparse
import json
import os

from config import Config
from runner import run_figures, run_risk, run_robust_rl, run_xi_curve
from utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="figures",
                    choices=["figures", "risk", "robust_rl", "xi_curve", "all"])
    ap.add_argument("--seeds", type=int, default=5, help="number of independent random seeds")
    ap.add_argument("--jobs", type=int, default=1, help="parallel training processes (figures experiment)")
    ap.add_argument("--episodes", type=int, default=None, help="override training episodes")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--error-bars", action="store_true", help="draw 95%% confidence intervals on the figures")
    ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args()

    cfg = Config()
    seeds = list(range(args.seeds))
    risk_data_episodes, risk_test_episodes = 200, 50
    if args.quick:
        cfg.episodes, cfg.eval_episodes, cfg.epsilon_decay_env_steps = 15, 2, 300
        seeds = seeds[:1]
        risk_data_episodes, risk_test_episodes = 10, 10
    if args.episodes:
        cfg.episodes = args.episodes
    out = args.out or (cfg.output_dir + ("_quick" if args.quick else ""))
    cfg.output_dir = out
    ensure_dir(out)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**cfg.to_dict(), "seeds": seeds}, f, indent=2)

    exp = args.experiment
    if exp in ("figures", "all"):
        run_figures(cfg, seeds, out, jobs=args.jobs, error_bars=args.error_bars)
    if exp in ("xi_curve", "all"):
        run_xi_curve(cfg, os.path.join(out, "xi_curve"))
    if exp in ("risk", "all"):
        run_risk(cfg, seeds, os.path.join(out, "risk"), risk_data_episodes, risk_test_episodes)
    if exp in ("robust_rl", "all"):
        run_robust_rl(cfg, seeds, os.path.join(out, "robust_rl"))
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    main()
