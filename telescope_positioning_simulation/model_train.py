from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from Survey.RLLib_survey import Survey
from IO.read_config import ReadConfig
from ray.tune.registry import register_env

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

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="my_env")
    .build()
)

for i in range(10):
    result = algo.train()
    print(result)
    print(pretty_print(result))
    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")