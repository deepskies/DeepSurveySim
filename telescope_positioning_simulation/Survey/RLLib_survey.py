"""
Run a simulation, picking up the variables specified in the config file and updating the time and location based on the
"""

import numpy as np
from gymnasium import spaces
from telescope_positioning_simulation.Survey.survey import Survey as SurveyBase

class Survey(SurveyBase):
    def __init__(self, observatory_config: dict = {}, survey_config: dict = {}) -> None:
        super().__init__(observatory_config, survey_config)
        self.action_space = spaces.Dict({
            'ra': spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
            'decl': spaces.Box(low=-90.0, high=90.0, shape=(1,), dtype=np.float32),
        })

        self.observation_space = spaces.Dict({
            'airmass': spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32),
            'alt': spaces.Box(low=-90, high=90.0, shape=(1,), dtype=np.float32),
            'ha': spaces.Box(low=-360, high=360.0, shape=(1,), dtype=np.float32),
            'lst': spaces.Box(low=0, high=360.0, shape=(1,), dtype=np.float32),
            'moon_airmass': spaces.Box(low=-1000, high=1000.0, shape=(1,), dtype=np.float32),
            'sun_airmass': spaces.Box(low=-1000, high=1000.0, shape=(1,), dtype=np.float32),
        })
        # TODO: estimated for now using our offline dataset and tuned in order to make it work, to change with the correct ranges

    def _observation_calculation(self):
        observation = {}
        for var_name in self.observatory_variables:
            print(var_name, self.observatory_variables[var_name]()[var_name])
            observation[var_name] = self.observatory_variables[var_name]()[var_name]

        observation["valid"] = self.validity(observation=observation)
        observation["mjd"] = np.asarray(
            [
                self.observator.time.mjd + self.observator.delay
                for _ in self.observator.location
            ]
        )
        return observation

    def reset(self, *, seed=None, options=None):
        #location = {'ra': 0, 'decl':0} # TODO: ask how to fix and remove
        #self.observator.update(time=0, location=location, band="g")
        self.timestep = 0
        self.previous_mjd = self._start_time()
        observation = self._observation_calculation()
        info = {}
        info['invalid'] = ~self.validity(observation)
        info['mjd'] = observation["mjd"]
        observation = {key: observation[key] for key in self.observatory_variables}
        return observation, info  # info required by RLLib

    def _reward(self, observation, info):
        reward = super()._reward(observation)
        return reward

    def step(self, action: dict):
        action["time"] = self.previous_mjd
        new_action = {"time": action["time"], "location": {"ra": action["ra"], "decl": action["decl"]}}
        self.observator.update(**new_action)
        observation = self._observation_calculation()
        info = {}
        info["invalid"] = ~self.validity(observation)
        info["mjd"] = observation["mjd"]
        observation = {key: observation[key] for key in self.observatory_variables }
        #observation['invalid'] = ~self.validity(observation)
        reward = self._reward(observation, info)
        self.timestep += 1
        self.previous_mjd = info["mjd"]

        stop = self._stop_condition(observation)
        truncated = False  # Additional truncated flag required by RLLib
        return observation, reward, stop, truncated, info
