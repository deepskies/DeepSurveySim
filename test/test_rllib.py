import pytest

import numpy as np
from gymnasium import spaces, Env
from telescope_positioning_simulation.Survey import Survey
from telescope_positioning_simulation.IO import ReadConfig


@pytest.fixture()
def envoriment():
    class RllibSurvey(Survey, Env):
        def __init__(self, kwarg) -> None:
            obs_config = ReadConfig()()
            survey_config = ReadConfig(survey=True)()

            survey_config["variables"] = ["airmass", "alt", "sky_magnitude", "teff"]
            survey_config["stopping"] = {"timestep": 2}
            obs_config["use_skybright"] = False
            obs_config["location"] = {"ra": [0], "decl": [0]}

            super().__init__(observatory_config=obs_config, survey_config=survey_config)

            self.action_space = spaces.Dict(
                {
                    "ra": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
                    "decl": spaces.Box(
                        low=-90.0, high=90.0, shape=(1,), dtype=np.float32
                    ),
                }
            )

            self.observation_space = spaces.Dict(
                {
                    "airmass": spaces.Box(
                        low=-100000, high=100000, shape=(1,), dtype=np.float32
                    ),
                    "alt": spaces.Box(
                        low=-100000, high=100000, shape=(1,), dtype=np.float32
                    ),
                    "sky_magnitude": spaces.Box(
                        low=-100000, high=100000, shape=(1,), dtype=np.float32
                    ),
                    "teff": spaces.Box(
                        low=-100000, high=100000, shape=(1,), dtype=np.float32
                    ),
                }
            )

        def reset(self, *, seed=None, options=None):
            super().reset()
            observation = self._observation_calculation()
            observation = {
                key: np.nan_to_num(observation[key], copy=True)
                for key in self.observation_space
            }

            return observation, {}

        def step(self, action: dict):

            new_action = {
                "time": self.time,
                "location": {"ra": action["ra"], "decl": action["decl"]},
            }
            observation, reward, stop, log = super().step(new_action)
            truncated = False  # Additional truncated flag required by RLLib

            observation = {
                key: np.array(
                    np.nan_to_num(observation[key], copy=True).ravel()[0],
                    dtype=np.float32,
                ).reshape(
                    1,
                )
                for key in self.observation_space
            }
            reward = reward.ravel()[0]
            return observation, reward, stop, truncated, log

    return RllibSurvey


def test_setup_env(envoriment):
    from ray.rllib.algorithms.ppo import PPOConfig
    import ray

    ray.init()
    alg_config = PPOConfig().training()
    alg_config = alg_config.environment(env=envoriment)
    alg_config = alg_config.rollouts(num_rollout_workers=10)
    alg_config = alg_config.resources(num_gpus=0)

    alg_config = alg_config.build()

    alg_config.train()
    ray.shutdown()
