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

def env_creator(_):
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
    parser.add_argument("-ni","--num_iterations", default=config.experiment_config["DEFAULT_TRAINING_TEST_ITERATIONS"], type=int, required=False,
                        help="Number of iterations to train/test algorithm")
    parser.add_argument("-f", "--file", required=False, help="Checkpoint file to execute in test mode")
    params = parser.parse_args()
    CHECKPOINT_PATH = os.path.join("results",params.file)
    env = env_creator({})
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH).environment(env="my_env").build()
    print(algo)
    #algo = PPOConfig().environment(env="my_env").build()
    print("Weights", algo.get_policy().get_weights())
    episode_reward = 0
    terminated = truncated = False
    if gymnasium:
        obs, info = env.reset()
    else:
         obs = env.reset()
    for i in range(params.num_iterations):
        episode_reward = 0
        while not terminated and not truncated:
            action = algo.compute_single_action(obs)
            if gymnasium:
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                obs, reward, terminated, info = env.step(action)
            episode_reward += reward
        print("Episode", str(i), "finished with reward", str(episode_reward))

