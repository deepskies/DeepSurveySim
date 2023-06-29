from utils import read_config
from ray import tune

experiment_config = read_config.ReadConfig("experiment_config.yaml")()

hyperparams = {
    "PPO": experiment_config["PPO_HYPERPARAMS"],
    "SAC": experiment_config["SAC_HYPERPARAMS"],
    "DDPG": experiment_config["DDPG_HYPERPARAMS"],
    "DDPPO": experiment_config["PPO_HYPERPARAMS"],
}

PPO_HYPERPARAMS = {
    "gamma": tune.grid_search([0.9, 0.95, 0.99]),
    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    "use_critic": tune.grid_search([True, False]),
    "use_gae": tune.grid_search([True, False]),
    "kl_coeff": tune.grid_search([0.1, 0.2, 0.3]),
    "sgd_minibatch_size": tune.grid_search([128, 256]),
    "num_sgd_iter": tune.grid_search([5, 10, 20]),
    "shuffle_sequences": tune.grid_search([True, False]),
    "vf_loss_coeff": tune.grid_search([0.1, 0.5, 1.0]),
    "entropy_coeff": tune.grid_search([0.001, 0.01, 0.1]),
    "clip_param": tune.grid_search([0.1, 0.2, 0.3]),
    "vf_clip_param": tune.grid_search([5.0, 10.0, 20.0]),
    "grad_clip": tune.grid_search([0.1, 0.5, 1.0]),
    "kl_target": tune.grid_search([0.01, 0.1, 0.5])
}

SAC_HYPERPARAMS = {
    "twin_q": True,
    "q_model_config": {},
    "policy_model_config": {},
    "tau": tune.grid_search([0.001, 0.005, 0.01]),
    "initial_alpha": tune.grid_search([0.01, 0.1, 1.0]),
    "target_entropy": "auto",
    "n_step": 1,
    "store_buffer_in_checkpoints": False,
    "replay_buffer_config": {
        "_enable_replay_buffer_api": True,
        "type": "SimpleReplayBuffer",
        "capacity": 100000,
        "max_size": 100000
    },
    "training_intensity": None,
    "clip_actions": False,
    "grad_clip": tune.grid_search([0.01, 0.05, 0.1]),
    "optimization_config": {
        "actor_learning_rate": tune.grid_search([0.0001, 0.0003, 0.001]),
        "critic_learning_rate": tune.grid_search([0.0001, 0.0003, 0.001]),
        "entropy_learning_rate": tune.grid_search([0.0001, 0.0003, 0.001])
    },
    "target_network_update_freq": 1,
    "_deterministic_loss": False,
    "_use_beta_distribution": False
}

DDPG_HYPERPARAMS = {
    "twin_q": True,
    "policy_delay": 1,
    "smooth_target_policy": True,
    "target_noise": 0.2,
    "target_noise_clip": 0.5,
    "use_state_preprocessor": False,
    "actor_hiddens": tune.grid_search([[256], [256, 256], [512, 256]]),
    "actor_hidden_activation": "relu",
    "critic_hiddens": tune.grid_search([[256], [256, 256], [512, 256]]),
    "critic_hidden_activation": "relu",
    "n_step": 1,
    "critic_lr": tune.grid_search([0.0001, 0.001, 0.01]),
    "actor_lr": tune.grid_search([0.0001, 0.001, 0.01]),
    "tau": tune.grid_search([0.001, 0.005, 0.01]),
    "use_huber": False,
    "huber_threshold": 1.0,
    "l2_reg": 0.0,
    "training_intensity": None
}


hyperparams_tune = {
    "PPO": PPO_HYPERPARAMS,
    "DDPPO": PPO_HYPERPARAMS,
    "SAC": SAC_HYPERPARAMS,
    "DDPG": DDPG_HYPERPARAMS
}
