import gym
import torch
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import HParam
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from Survey.StableBaselines_survey import Survey
#from TelescopePositioningSimulation.telescope_positioning_simulation.Survey.StableBaselines_survey.py import Survey
from IO.read_config import ReadConfig
#from TelescopePositioningSimulation.telescope_positioning_simulation.IO.read_config.py import ReadConfig

seo_config = ReadConfig(
        observator_configuration="settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="settings/equatorial_survey.yaml",
        survey=True
    )()

# Define a directory for logging TensorBoard files
log_dir = "./tslog/"

survey_config['location']  = {'ra': [0], 'decl': [0]}
seo_config['use_skybright'] = False
survey_config['variables'] = ["airmass", 'alt', 'ha', 'moon_airmass', 'lst', 'sun_airmass']
survey_config['reward'] = {"monitor": "airmass", "min": True}

env = Survey(seo_config, survey_config)

writer = SummaryWriter(log_dir=log_dir)

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Specify hyperparameters
hyperparams = {
    "seed": seed,
    "device": 'cpu',
    "verbose": 1,
}

# Create the TD3 agent
model = TD3("MlpPolicy", env, **hyperparams, tensorboard_log=log_dir)

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

tensorboard_callback = TensorboardCallback()
#########################################
########  RUN 20 Thousand STEPS  ########
#########################################
TIMESTEPS = 4000
i = 0
for i in range(50):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TD3", callback=HParamCallback())
    #model.save(f"{models_dir}/{TIMESTEPS*i}")

model.save(f"{models_dir}/td3_survey_model")

writer.close()



#model.learn(100000, callback=HParamCallback())
#model.save("td3_survey_model")
#writer.close()


