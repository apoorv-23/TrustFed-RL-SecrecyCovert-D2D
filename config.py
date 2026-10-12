from dataclasses import dataclass, fields
from typing import Tuple


@dataclass
class Config:
    num_cus: int = 20                   
    num_d2d: int = 20                   
    num_rbs: int = 20                   
    cell_radius_m: float = 500.0
    d2d_pair_min_distance_m: float = 5.0
    d2d_pair_max_distance_m: float = 40.0

    carrier_freq_ghz: float = 2.0      
    path_loss_exponent: float = 3.2
    shadowing_std_db: float = 0.0
    rb_bandwidth_hz: float = 180e3      
    noise_psd_dbm_hz: float = -174.0
    noise_figure_db: float = 9.0
    cu_tx_power_dbm: float = 23.0

    d2d_max_power_dbm: float = 23.0    
    num_power_levels: int = 5           
    power_step_db: float = 5.0

    enable_mobility: bool = True
    mobility_max_step_m: float = 3.0   

    cu_sinr_threshold_db: float = 3.0
    secrecy_threshold_bps: float = 36e3 
    warden_samples: int = 100           
    covert_epsilon: float = 0.1        

    phi_sec: float = 2.0               
    phi_pow: float = 0.15               
    phi_cu: float = 3.0                 
    phi_cov: float = 5.0               
    phi_out: float = 1.5                
    phi_cng: float = 0.2                

    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 128
    replay_size: int = 50000
    min_replay_for_train: int = 2000
    target_update_freq: int = 200       
    hidden_dim: int = 128
    grad_clip: float = 5.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_env_steps: int = 10000  
    use_huber_loss: bool = True

    ddpg_actor_lr: float = 1e-4
    ddpg_critic_lr: float = 1e-3
    ddpg_tau: float = 0.005            
    ddpg_noise_start: float = 0.5       
    ddpg_noise_end: float = 0.05
  
    fl_round_freq: int = 20            
    fl_clients_per_round: int = 10      
    fl_local_epochs: int = 2            
    fl_local_lr: float = 1e-3
    fl_batch_size: int = 64
    fl_min_samples: int = 64            
    fl_buffer_size: int = 1000        
    fl_val_fraction: float = 0.2
    fl_hidden_dim: int = 64
    trust_clip_min: float = 0.01
    trust_clip_max: float = 0.8
    trust_ema_lambda: float = 0.8
    trust_tau_c: float = 1.0           
    trust_tau_v: float = 1.0           
    trust_tau_d: float = 1.0           
    trust_reference: str = "median"
    trust_dev_scale: str = "median"
    trimmed_mean_fraction: float = 0.25 

    episodes: int = 500
    steps_per_episode: int = 40
    eval_episodes: int = 50

    d2d_user_values: Tuple[int, ...] = (10, 20, 30, 40, 50)
    pmax_values_dbm: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 23.0)
    pmax_num_d2d: int = 23            
    phi_cov_values: Tuple[float, ...] = (0.1, 0.3, 0.7, 1.0)
    rate_plot_unit: str = "bps"         

    device: str = "cpu"
    output_dir: str = "outputs_secure_covert_d2d"

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type in (int, float) and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise TypeError(f"config.{f.name} must be a number, got {v!r}. "
                                f"Check config.py for a stray comma (write 50000 or 50_000, not 50,000).")
            if f.type is int and isinstance(v, float) and not v.is_integer():
                raise TypeError(f"config.{f.name} must be a whole number, got {v!r}.")

    def power_levels_dbm(self):
        L = self.num_power_levels
        return [self.d2d_max_power_dbm - self.power_step_db * (L - 1 - l) for l in range(L)]

    def to_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}
