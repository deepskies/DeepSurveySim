import gym
import torch
import os

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from astropy.time import Time
import random


from stable_baselines3 import DDPG
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
log_dir = "./tslogtest/"

# Define a directory for saving model checkpoints
models_dir = "models/DDPGg"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

survey_config['location']  = {'ra': [0], 'decl': [0]}
seo_config['use_skybright'] = False
survey_config['variables'] = ["airmass", 'alt', 'ha', 'moon_airmass', 'lst', 'sun_airmass']
survey_config['reward'] = {"monitor": "airmass", "min": True}

env = Survey(seo_config, survey_config)
env.reset()

writer = SummaryWriter(log_dir=log_dir)

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)

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

#########################################
###  LOAD MODEL FROM CHECKPOINT  ########
###  NEED TO CHANGE PATH FOR CHECKPOINT #
#########################################
# Create the PPO agent
model_path = f"{models_dir}/710000.zip"
model = PPO.load(model_path, env=env)


observation = env.reset()
stop = False
episodes = 1000
ep = 0
total_reward = 0

action = [random.random(), random.random(), 'g']
observation, reward, stop, truncated, log = env.step(action)

while (not stop and ep < episodes):
    if (ep%2==0):
        action, _states = model.predict(observation)
    else:
        action = [random.random(), random.random(), 'g']

    print(f"Episode {ep + 1}: Action {action}: Total reward = {rewards}")
    #print(f"Episode {ep + 1}: Action {action}")
    #print(f"Episode {ep + 1}:states {_states}")
    observation, reward, stop, truncated, log  = env.step(action)
    print(f"Episode {ep + 1}: Observation {observation}")

    total_reward += reward
    ep += 1
    #env.render()
    print("Reward",reward)
    print("Stop",stop)

print(f"Total Episodes {ep + 1}: Total reward = {total_reward}")



env.close()


writer.close()


