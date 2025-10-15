import math
from tta import (
    EnergyTTA,
    Tent,
    EATA,
)

tta_configs = {
    "energy_tta": {
        "factory": EnergyTTA,
        "config": {
            # One update per batch; episodic resets as in Tent
            "steps": 1,
            "episodic": False,
            # JEM-style SGLD / replay defaults
            "buffer_size": 10000,
            "sgld_steps": 20,
            "sgld_lr": 1.0,
            "sgld_std": 0.01,
            "reinit_freq": 0.05,
            # unconditional sampling unless you explicitly pass y during sampling
            "if_cond": False,
            # dataset/model metadata
            "n_classes": 1000,
            "im_sz": 224,
            "n_ch": 3,
            "eval_mode_during_sgld": True,
            "eval_mode_during_tta": True,
        },
    },
    "tent": {
        "factory": Tent,
        "config": {
            "steps": 1,
            "episodic": False,
            "eval_mode_during_tta": True,
        },
    },
    "eata": {
        "factory": EATA,
        "config": {
            # Adaptation settings:
            "steps": 1,  # number of per-batch adaptation steps
            "episodic": False,  # no episodic reset by default
            # Entropy-based sample filtering (EATA):
            "e_margin": math.log(1000) / 2
            - 1,  # threshold for reliable samples (Eq. 3), log(1000)/2 - 1 ≈ 2.44
            "d_margin": 0.05,  # redundancy threshold (Eq. 5)
            # Anti-forgetting regularizer (Fisher-based EWC-style):
            "fishers": None,  # optional pre-computed Fisher info dict, e.g., {param_name: (F, theta_old)}
            "fisher_alpha": 2000.0,  # regularizer strength, aligning with your code default
            # Initial moving-average and sample counters (internal, not config):
            # e.g., num_samples_update_1, num_samples_update_2, current_model_probs are managed internally
            "eval_mode_during_tta": True,
        },
    },
}
