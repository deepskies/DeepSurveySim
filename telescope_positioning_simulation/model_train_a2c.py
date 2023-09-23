import gym
import torch
import os

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3 import A2C
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
models_dir = "models/A2Cu"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

survey_config['location']  = {'ra': [0], 'decl': [0]}
seo_config['use_skybright'] = False
survey_config['variables'] = ["airmass", 'alt', 'ha', 'moon_airmass', 'lst', 'sun_airmass']
survey_config['reward'] = {"monitor": "airmass", "min": True}

env = Survey(seo_config, survey_config)

writer = SummaryWriter(log_dir=log_dir)

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)

#########################################
######  SPECIFY HYPERPARAMETERS  ########
######  NEED TO CHANGE TO OPTIMIZE  #####
#########################################
hyperparams = {
    "learning_rate": 0.0007,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "rms_prop_eps": 0.00001,
    "use_rms_prop": True,
    "seed": seed,
    "device": 'cpu',
    "verbose": 1,
}

# Create the A2C agent
model = A2C("MlpPolicy", env, **hyperparams, tensorboard_log=log_dir)

# Define an evaluation callback for logging evaluation statistics during training
eval_callback = EvalCallback(env, best_model_save_path='./best_model/',
                             log_path=log_dir, eval_freq=512,
                             deterministic=True, render=False)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    public_airmass = 0
    public_airmass_mean = 0
    public_count = 0


    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # Access the current reward at each training step
        # The reward can be accessed via `self.locals["rewards"]`
        self.public_count += 1

        reward = self.locals["rewards"].tolist()
        airmass = 1.0
        rewardval = reward[0]
        if rewardval != 0:
            airmass = 1 / rewardval
            self.public_airmass += airmass
        self.public_airmass_mean = self.public_airmass / self.public_count
        self.logger.record("airmass", self.public_airmass_mean)
        #print(f"Reward_val: {rewardval}")
        #print(f"airmass: {self.public_airmass_mean}")
        if self.public_count % 100 == 0:
            self.public_airmass = 0
            self.public_airmass_mean = 0
            self.public_count = 0
            #print('reset airmass')
        return True
    
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
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C_u", callback=tensorboard_callback)
    #model.save(f"{models_dir}/{TIMESTEPS*i}")

model.save(f"{models_dir}/a2c_u_survey_model")

writer.close()


