import numpy as np

from covert import covert_sinr_threshold, min_detection_error
from federated import LocalRiskDataset
from utils import (db_to_lin, dbm_to_watt, jain_fairness, lin_to_db,
                   noise_power_dbm, sample_point_in_circle)

FL_INPUT_DIM = 5
DB_SCALE = 20.0          
INR_FLOOR_DB = -30.0     


def _scale_db(x_db):
    return np.clip(np.asarray(x_db, dtype=np.float64) / DB_SCALE, -5.0, 5.0)


def _pairwise_dist(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


class SecureCovertD2DEnv:
    def __init__(self, cfg, use_fl_features: bool = True):
        if cfg.num_cus > cfg.num_rbs:
            raise ValueError("num_cus must be <= num_rbs (one CU per RB).")
        self.cfg = cfg
        self.use_fl_features = use_fl_features

        self.N, self.M, self.K = cfg.num_d2d, cfg.num_cus, cfg.num_rbs
        self.num_agents = self.N
        self.num_rbs = self.K
        self.power_levels_watt = dbm_to_watt(np.array(cfg.power_levels_dbm()))
        self.L = len(self.power_levels_watt)
        self.p_max = float(self.power_levels_watt.max())
        self.cu_tx_power = float(dbm_to_watt(cfg.cu_tx_power_dbm))
        self.B = float(cfg.rb_bandwidth_hz)

        wavelength = 3e8 / (cfg.carrier_freq_ghz * 1e9)
        self.pl_ref = (wavelength / (4.0 * np.pi)) ** 2         
        self.noise = float(dbm_to_watt(noise_power_dbm(cfg)))    
        self.cov_th = covert_sinr_threshold(cfg.warden_samples, cfg.covert_epsilon)
        self.cu_th = float(db_to_lin(cfg.cu_sinr_threshold_db))

        
        self.cu_rb = np.arange(self.M)
        self.cu_onehot = np.zeros((self.M, self.K))
        self.cu_onehot[np.arange(self.M), self.cu_rb] = 1.0

        self.state_dim = 3 * self.K + 5
        self.action_dim = self.K * self.L

        self.local_fl_buffers = [LocalRiskDataset(cfg.fl_buffer_size, FL_INPUT_DIM) for _ in range(self.N)]
        self.risk_model = None
        self.fl_log = None           
        self.reset()

    def attach_risk_model(self, model):
        self.risk_model = model

    def _shadow(self, shape):
        s = self.cfg.shadowing_std_db
        if s <= 0:
            return np.ones(shape)
        return db_to_lin(s * np.random.randn(*np.atleast_1d(shape)))

    def _pathloss(self, d):
        return self.pl_ref * np.maximum(d, 1.0) ** (-self.cfg.path_loss_exponent)

    def place_nodes(self):
        R = self.cfg.cell_radius_m
        self.cu_pos = np.array([sample_point_in_circle(R) for _ in range(self.M)])
        self.d2d_tx = np.array([sample_point_in_circle(R) for _ in range(self.N)])
        rx = []
        for tx in self.d2d_tx:
            for _ in range(100):
                ang = 2 * np.pi * np.random.rand()
                dist = np.random.uniform(self.cfg.d2d_pair_min_distance_m, self.cfg.d2d_pair_max_distance_m)
                cand = tx + dist * np.array([np.cos(ang), np.sin(ang)])
                if np.linalg.norm(cand) <= R:
                    rx.append(cand)
                    break
            else:
                rx.append(tx * (1 - self.cfg.d2d_pair_min_distance_m / max(np.linalg.norm(tx), 1e-9)))
        self.d2d_rx = np.array(rx)
        self.warden_pos = sample_point_in_circle(R)
        N, M = self.N, self.M
        self.sh = {"dd": self._shadow((N, N)), "dw": self._shadow(N), "db": self._shadow(N),
                   "cb": self._shadow(M), "cw": self._shadow(M), "cd": self._shadow((M, N))}

    def update_positions(self):
        R, v = self.cfg.cell_radius_m, self.cfg.mobility_max_step_m
        for arr in (self.cu_pos, self.d2d_tx, self.d2d_rx):
            n = len(arr)
            ang = 2 * np.pi * np.random.rand(n)
            step = v * np.random.rand(n)
            cand = arr + step[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=1)
            ok = np.linalg.norm(cand, axis=1) <= R
            arr[ok] = cand[ok]

    def update_channels(self):
        N, M = self.N, self.M
        ex = np.random.exponential
        w = self.warden_pos[None, :]
        self.G_dd = self._pathloss(_pairwise_dist(self.d2d_tx, self.d2d_rx)) * self.sh["dd"] * ex(size=(N, N))  # [j, i]: D-TX j -> D-RX i
        self.g_d2d = np.diag(self.G_dd).copy()
        self.g_d2w = self._pathloss(np.linalg.norm(self.d2d_tx - w, axis=1)) * self.sh["dw"] * ex(size=N)
        self.g_d2b = self._pathloss(np.linalg.norm(self.d2d_tx, axis=1)) * self.sh["db"] * ex(size=N)
        self.g_cb = self._pathloss(np.linalg.norm(self.cu_pos, axis=1)) * self.sh["cb"] * ex(size=M)
        self.g_cw = self._pathloss(np.linalg.norm(self.cu_pos - w, axis=1)) * self.sh["cw"] * ex(size=M)
        self.G_cd = self._pathloss(_pairwise_dist(self.cu_pos, self.d2d_rx)) * self.sh["cd"] * ex(size=(M, N))  # [m, i]

    def reset(self):
        self.place_nodes()
        self.update_channels()
        self.current_step = 0
        self.prev_rbs = -np.ones(self.N, dtype=np.int64)      
        self.prev_pidx = np.zeros(self.N, dtype=np.int64)
        self.prev_sinr_db = np.zeros(self.N)
        return self.build_agent_states()

    def decode(self, actions):
        a = np.asarray(actions, dtype=np.int64)
        return a // self.L, a % self.L

    def local_measurements(self):
        snr_db = lin_to_db(self.p_max * self.g_d2d / self.noise)                          
        inr = np.zeros((self.N, self.K))                                                  
        inr[:, self.cu_rb] = (self.cu_tx_power * self.G_cd).T / self.noise               
        inr_db = np.maximum(lin_to_db(inr), INR_FLOOR_DB)
        occ = np.zeros((self.N, self.K))
        valid = self.prev_rbs >= 0
        if valid.any():
            counts = np.bincount(self.prev_rbs[valid], minlength=self.K).astype(np.float64)
            occ = np.tile(counts, (self.N, 1))
            occ[np.where(valid)[0], self.prev_rbs[valid]] -= 1.0          
            occ /= max(1, self.N - 1)
        return {"snr_db": snr_db, "inr_db": inr_db, "occ": occ,
                "snr_s": _scale_db(snr_db), "inr_s": _scale_db(inr_db),
                "prev_sinr_s": _scale_db(self.prev_sinr_db)}

    def fl_feature_tensor(self, meas, pidx):
        """FL input features for every (agent, RB) pair: shape (N, K, 5)."""
        F = np.empty((self.N, self.K, FL_INPUT_DIM), dtype=np.float32)
        F[:, :, 0] = meas["snr_s"][:, None]
        F[:, :, 1] = meas["inr_s"]
        F[:, :, 2] = meas["occ"]
        F[:, :, 3] = (np.asarray(pidx) / max(1, self.L - 1))[:, None]
        F[:, :, 4] = meas["prev_sinr_s"][:, None]
        return F

    def risk_scores(self, meas):
        if (not self.use_fl_features or self.risk_model is None
                or not getattr(self.risk_model, "is_trained", False)):
            return np.zeros((self.N, self.K))
        X = self.fl_feature_tensor(meas, self.prev_pidx).reshape(-1, FL_INPUT_DIM)
        return self.risk_model.predict(X).reshape(self.N, self.K)

    def build_agent_states(self) -> np.ndarray:
        meas = self.local_measurements()
        self._meas = meas
        rho = self.risk_scores(meas)
        states = np.concatenate([
            meas["snr_s"][:, None],                                 
            meas["inr_s"],                                          
            meas["occ"],                                           
            ((self.prev_rbs + 1) / self.K)[:, None],               
            (self.prev_pidx / max(1, self.L - 1))[:, None],         
            meas["prev_sinr_s"][:, None],                          
            rho.mean(axis=1, keepdims=True),                       
            rho,                                                    
        ], axis=1)
        return states.astype(np.float32)

    def compute_metrics(self, rbs, pidx):
        p = self.power_levels_watt[pidx]
        same = rbs[:, None] == rbs[None, :]
        np.fill_diagonal(same, False)                                 
        A = self.cu_onehot[:, rbs]                                     

        i_dd = (p[:, None] * self.G_dd * same).sum(axis=0)
        i_cu_rx = (self.cu_tx_power * self.G_cd * A).sum(axis=0)
        legit = p * self.g_d2d / (i_dd + i_cu_rx + self.noise)

        i_dw = ((p * self.g_d2w)[:, None] * same).sum(axis=0)         
        i_cu_w = (self.cu_tx_power * self.g_cw[:, None] * A).sum(axis=0)
        eve = p * self.g_d2w / (i_dw + i_cu_w + self.noise)

        i_bs = (A * (p * self.g_d2b)[None, :]).sum(axis=1)
        cu_sinr = self.cu_tx_power * self.g_cb / (i_bs + self.noise)

        legit_rate = self.B * np.log2(1.0 + legit)                                  
        sec = np.maximum(legit_rate - self.B * np.log2(1.0 + eve), 0.0)              
        xi = min_detection_error(eve, self.cfg.warden_samples)
        return {"rbs": rbs, "pidx": pidx, "powers": p, "cu_sinr": cu_sinr, "legit_sinr": legit,
                "eve_sinr": eve, "legit_rate": legit_rate, "sec_rate": sec, "xi": xi}

    def step(self, actions):
        rbs, pidx = self.decode(actions)
        meas = self._meas                                            
        feats = self.fl_feature_tensor(meas, pidx)[np.arange(self.N), rbs]      
        m = self.compute_metrics(rbs, pidx)

        cov_out = m["eve_sinr"] > self.cov_th                       
        sec_out = m["sec_rate"] < self.cfg.secrecy_threshold_bps
        labels = (cov_out | sec_out).astype(np.float32)             
        for i in range(self.N):
            self.local_fl_buffers[i].add(feats[i], labels[i])
        if self.fl_log is not None:
            self.fl_log.append((feats, labels))

        c = self.cfg
        covert_risk = 1.0 - m["xi"]                                 
        rb_counts = np.bincount(rbs, minlength=self.K)
        cu_short = np.maximum(1.0 - m["cu_sinr"] / self.cu_th, 0.0)          
        cu_pen = (self.cu_onehot[:, rbs] * cu_short[:, None]).sum(axis=0)    
        rewards = (c.phi_sec * m["sec_rate"] / self.B
                   - c.phi_pow * pidx / max(1, self.L - 1)
                   - c.phi_cu * cu_pen
                   - c.phi_cov * covert_risk
                   - c.phi_out * sec_out
                   - c.phi_cng * (rb_counts[rbs] - 1))

        self.prev_rbs, self.prev_pidx = rbs.copy(), pidx.copy()
        self.prev_sinr_db = lin_to_db(m["legit_sinr"])
        if c.enable_mobility:
            self.update_positions()
        self.update_channels()
        self.current_step += 1
        done = self.current_step >= c.steps_per_episode
        next_states = self.build_agent_states()

        info = {
            "avg_secrecy_rate_bps": float(m["sec_rate"].mean()),
            "sum_secrecy_rate_bps": float(m["sec_rate"].sum()),
            "avg_covert_risk": float(covert_risk.mean()),
            "avg_detection_error": float(m["xi"].mean()),
            "covert_outage_prob": float(cov_out.mean()),
            "secrecy_outage_prob": float(sec_out.mean()),
            "cu_outage_prob": float((m["cu_sinr"] < self.cu_th).mean()),
            "jain_fairness": jain_fairness(m["sec_rate"]),
            "sum_d2d_throughput_bps": float(m["legit_rate"].sum()),
            "median_warden_sinr_db": float(np.median(lin_to_db(m["eve_sinr"]))),
            "total_power_w": float(m["powers"].sum()),
            "avg_reward": float(rewards.mean()),
        }
        return next_states, rewards.astype(np.float32), done, info
