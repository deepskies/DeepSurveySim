# TO TEST

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from Survey.RLLib_survey import Survey
from IO.read_config import ReadConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm
try:
    import gymnasium as gym

    gymnasium = True
except Exception:
    import gym

    gymnasium = False


ray.init()

seo_config = ReadConfig(
        observator_configuration="settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="settings/equatorial_survey.yaml",
        survey=True
    )()

survey_config['location']  = {'ra': [0], 'decl':[0]}

def env_creator(env_config):
    return Survey(seo_config, survey_config)

register_env("my_env", env_creator)

config = PPOConfig().training(lr=tune.grid_search([0.01, 0.001, 0.0001]))

tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 150},
    ),
    param_space=config,
)

results = tuner.fit()

best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

best_checkpoint = best_result.checkpoint

checkpoint_path = ""
algo = Algorithm.from_checkpoint(checkpoint_path)

env = env_creator({})

algo = PPOConfig().environment(env).build()
algo.get_policy().get_weights()

episode_reward = 0
terminated = truncated = False

if gymnasium:
    obs, info = env.reset()
else:
    obs = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
    if gymnasium:
        obs, reward, terminated, truncated, info = env.step(action)
    else:
        obs, reward, terminated, info = env.step(action)
    episode_reward += reward