"""
Run a simulation, picking up the variables specified in the config file and updating the time and location based on the
"""

from typing import Union
import numpy as np

from telescope_positioning_simulation.Survey.observation_variables import (
    ObservationVariables,
)

from telescope_positioning_simulation.IO.read_config import ReadConfig


class Survey:
    def __init__(self, survey_config: dict = {}, obseravtory_config: dict = {}) -> None:

        default_survey = ReadConfig(None, survey=True)()
        survey_config = {**default_survey, **survey_config}

        default_obs = ReadConfig(None, survey=False)()
        obseravtory_config = {**default_obs, **obseravtory_config}

        self.observator = ObservationVariables(
            observator_configuration=obseravtory_config
        )

        self.telescope_config = obseravtory_config
        self.survey_config = survey_config

        self.reward_config = survey_config["reward"]
        self.stop_config = survey_config["stopping"]
        self.validity_config = survey_config["constaints"]

        self.timestep_size = survey_config["timestep_size"]
        self.start_time = survey_config["start_time"]
        self.invalid_penality = survey_config["invalid_penality"]

        self.save_config = survey_config["save"]

        var_dict = self.observator.name_to_function()

        self.observatory_variables = {
            key: var_dict[key] for key in survey_config["variables"]
        }
        self.timestep = 0

    def _start_time(self):
        if self.start_time == "random":
            return np.random.default_rng().integers(low=55000, high=70000)
        else:
            return self.start_time

    def reset(self):
        self.observator.update(time=0, band="g")
        self.timestep = 0
        self.time = self._start_time()

    def validity(self, observation):
        valid = True

        for condition in self.validity_config:
            if self.validity_config[condition]["lesser"]:
                valid_condition = (
                    observation[condition] >= self.validity_config[condition]["value"]
                )
            else:
                valid_condition = (
                    observation[condition] <= self.validity_config[condition]["value"]
                )

            valid = valid & valid_condition
        return valid

    def _stop_condition(self, observation):
        """Returns true when stopping condition has been met"""
        stop = False
        stop = stop and self.timestep <= self.stop_config["timestep"]

        other_conditions = [
            condition
            for condition in self.stop_config.keys()
            if condition != "timestep"
        ]
        for condition in other_conditions:
            if self.stop_config[condition]["lesser"]:
                stop_condition = (
                    observation[condition] <= self.stop_config[condition]["value"]
                )
            else:
                stop_condition = (
                    observation[condition] >= self.stop_config[condition]["value"]
                )

            stop = stop & stop_condition

        return not np.all(stop)

    def _reward(self, observation):
        metric = self.reward_config["monitor"]
        reward = observation[metric]
        if self.reward_config["min"]:
            reward = reward ** (-1)

        if "threshold" in self.reward_config:
            reward = np.where(
                reward > self.reward_config["threshold"], reward, self.invalid_penality
            )

        reward = np.where(observation["invalid"], self.invalid_penality, reward)

        return reward

    def step(self, action: dict):

        self.observator.update(**action)
        observation = self._observation_calculation()
        observation['invalid'] = ~self.validity(observation)
        reward = self._reward(observation)
        self.timestep += 1

        stop = self._stop_condition(observation)

        log = {}

        return observation, reward, stop, log

    def _observation_calculation(self):

        observation = {}
        for var_name in self.observatory_variables:
            observation[var_name] = self.observatory_variables[var_name]()[var_name]

        observation["valid"] = self.validity(observation=observation)
        observation["mjd"] = np.asarray(
            [
                self.observator.time.mjd + self.observator.delay
                for _ in self.observator.location
            ]
        )

        return observation

    def __call__(self):
        stop = False
        results = {}
        while not stop:
            observation, reward, stop, _ = self.step(self.observator.default_locations)

            results[self.observator.time.mjd] = {
                obs_var: np.array(observation[obs_var], dtype=np.float32)
                for obs_var in observation
            }
            results[self.observator.time.mjd]["reward"] = np.array(
                reward, dtype=np.float32
            )

            # TODO checkpoint functionality

        return results
