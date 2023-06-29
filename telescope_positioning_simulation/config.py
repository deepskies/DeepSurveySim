from utils import read_config
from ray import tune

experiment_config = read_config.ReadConfig("experiment_config.yaml")()

hyperparams = {
    "PPO": experiment_config["PPO_HYPERPARAMS"],
    "SAC": experiment_config["SAC_HYPERPARAMS"],
    "DDPG": experiment_config["DDPG_HYPERPARAMS"],
    "DDPPO": experiment_config["PPO_HYPERPARAMS"],
}

ppo_hyperparams = {
    "gamma": tune.grid_search([0.9, 0.95, 0.99]),
    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
    "lr_schedule": tune.grid_search([[0, 0.0003], [10000, 0.0001]]),
    "use_critic": tune.grid_search([True, False]),
    "use_gae": tune.grid_search([True, False]),
    "kl_coeff": tune.grid_search([0.1, 0.2, 0.3]),
    "sgd_minibatch_size": tune.grid_search([128, 256]),
    "num_sgd_iter": tune.grid_search([5, 10, 20]),
    "shuffle_sequences": tune.grid_search([True, False]),
    "vf_loss_coeff": tune.grid_search([0.1, 0.5, 1.0]),
    "entropy_coeff": tune.grid_search([0.001, 0.01, 0.1]),
    "entropy_coeff_schedule": tune.grid_search([[0, 0.01], [10000, 0.001]]),
    "clip_param": tune.grid_search([0.1, 0.2, 0.3]),
    "vf_clip_param": tune.grid_search([5.0, 10.0, 20.0]),
    "grad_clip": tune.grid_search([0.1, 0.5, 1.0]),
    "kl_target": tune.grid_search([0.01, 0.1, 0.5])
}

hyperparams_tune = {
    "PPO": ppo_hyperparams,
}
