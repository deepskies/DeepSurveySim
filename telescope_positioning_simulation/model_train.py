from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms.ddppo import DDPPOConfig
from ray.tune.logger import pretty_print
import ray
from Survey.RLLib_survey import Survey
from IO.read_config import ReadConfig
from ray.tune.registry import register_env
from ray import air, tune
import argparse
import config
from ray.rllib.algorithms.algorithm import Algorithm
import os
try:
    import gymnasium as gym
    gymnasium = True
except Exception:
    import gym
    gymnasium = False

seo_config = ReadConfig(
        observator_configuration="settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="settings/equatorial_survey.yaml",
        survey=True
    )()

survey_config['location']  = {'ra': [0], 'decl': [0]}

def env_creator(env_config):
    global survey_config, seo_config
    return Survey(seo_config, survey_config)

register_env("my_env", env_creator)

algorithm_config = {
    "SAC": SACConfig,
    "PPO": PPOConfig,
    "DDPG": DDPGConfig,
    "DDPPO": DDPPOConfig
}

if __name__ == "__main__":
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=0, type=int, required=False, help="Enable cuda with a number of GPUs")  # GPU support
    parser.add_argument("-nrw", "--num_rollout_workers", default=1, type=int, required=False,
                        help="Number of rollout workers")
    parser.add_argument("-ni","--num_iterations", default=config.experiment_config["DEFAULT_TRAINING_ITERATIONS"], type=int, required=False,
                        help="Number of iterations to train algorithm")
    parser.add_argument("-cp", "--checkpoint_period", required=False, type=int,
                        default=10, help="Checkpoint saving period")
    parser.add_argument("-a", "--algorithm", required=False, default=config.experiment_config["DEFAULT_TRAINING_ALGORITHM"],
                        help="Algorithm to train", choices=algorithm_config.keys())
    parser.add_argument("-m", "--mode", required=False,
                        default="train", help="Execution mode")
    parser.add_argument("-f", "--file", required=False, help="Checkpoint file to execute in test mode")
    params = parser.parse_args()
    if params.mode == "train":
        alg_config = algorithm_config[params.algorithm]().training(**config.hyperparams[params.algorithm])
        alg_config = alg_config.environment(env="my_env")
        alg_config = alg_config.rollouts(num_rollout_workers=params.num_rollout_workers)
        alg_config = alg_config.resources(num_gpus=params.cuda)
        print("Configuration",alg_config.to_dict())
        alg_config = alg_config.build()
        for i in range(params.num_iterations):
            result = alg_config.train()
            print(pretty_print(result))
            if i % params.checkpoint_period == 0:
                checkpoint_dir = alg_config.save()
                print(f"Checkpoint saved in directory {checkpoint_dir}")
    elif params.mode == "tune":
        alg_config = algorithm_config[params.algorithm]().training(**config.hyperparams_tune[params.algorithm])
        alg_config = alg_config.environment(env="my_env")
        alg_config = alg_config.rollouts(num_rollout_workers=params.num_rollout_workers)
        alg_config = alg_config.resources(num_gpus=params.cuda)
        tuner = tune.Tuner(
            params.algorithm,
            run_config=air.RunConfig(
                stop={"episode_reward_mean": config.experiment_config["STOP_EPISODE_REWARD_MEAN"]},
                storage_path="./results", name="test_experiment"
            ),
            param_space=alg_config,
        )
        results = tuner.fit()
        best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
        print("Best result", best_result)
        best_checkpoint = best_result.checkpoint
        print("Best checkpoint", best_checkpoint)
    elif params.mode == "test":
        CHECKPOINT_PATH = ""
        env = env_creator({})
        algo = Algorithm.from_checkpoint(os.path.join(CHECKPOINT_PATH, params.file))
        algo = PPOConfig().environment(env=env).build()
        print("Weights", algo.get_policy().get_weights())
        episode_reward = 0
        terminated = truncated = False
        if gymnasium:
            obs, info = env.reset()
        else:
            obs = env.reset()
        print("Obs", obs)
        while not terminated and not truncated:
            action = algo.compute_single_action(obs)
            print("Action", action)
            if gymnasium:
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                obs, reward, terminated, info = env.step(action)
            print("Obs", obs)
            print("Reward", reward)
            episode_reward += reward
