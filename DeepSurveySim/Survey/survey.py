import numpy as np

from DeepSurveySim.Survey.observation_variables import (
    ObservationVariables,
)

from DeepSurveySim.IO.read_config import ReadConfig


class Survey:
    """
    Run a survey in sequence, selecting new sites to observe and few the assocaited variables

        Args:
            obseravtory_config (dict): Setup parameters for Survey.ObservationVariables, the telescope configuration, as read by IO.ReadConfig
            survey_config (dict): Parameters for the survey, including the stopping conditions, the validity conditions, the variables to collect, as read by IO.ReadConfig

        Examples:
            >>> survey = Survey(observatory_config, survey_config)
                action_generator = ActionGenerator() # Attributary function to produce time, location pairs
                for step in range(10):
                    action_time, action_location = action_generator()
                    update_action = {"time":[action_time], "location":{"ra":[action_location["ra"]], "decl":[action_location["decl"]]}}
                    observation, reward, stop, log = survey.step(update_action)

            >>> survey = Survey(observatory_config, survey_config)
                # Run without changing the location, only stepping time forward
                survey_results = survey()
    """

    def __init__(
        self,
        observatory_config: dict,
        survey_config: dict,
    ) -> None:

        self.observator = ObservationVariables(
            observator_configuration=observatory_config
        )

        self.telescope_config = observatory_config
        self.survey_config = survey_config

        self.reward_config = survey_config["reward"]
        self.stop_config = survey_config["stopping"]
        self.validity_config = survey_config["constaints"]

        self.timestep_size = survey_config["timestep_size"]
        self.start_time = survey_config["start_time"]
        self.invalid_penality = survey_config["invalid_penality"]

        self.time = self._start_time()
        self.observator.update(time=self.time)

        self.save_config = survey_config["save"]
        self.timestep = 0

        var_dict = self.observator.name_to_function()

        self.observatory_variables = {
            key: var_dict[key] for key in survey_config["variables"]
        }

    def _start_time(self):
        if self.start_time == "random":
            return np.random.default_rng().integers(low=55000, high=70000)
        else:
            return self.start_time

    def reset(self):
        """Return the observer to its inital position, the time to the start time, and the timestep to 0."""
        self.timestep = 0
        self.time = self._start_time()
        self.observator.update(time=self.time)

    def _validity(self, observation):
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
        stop = stop or self.timestep >= self.stop_config["timestep"]

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

            stop = stop or stop_condition

        return np.any(stop)

    def _reward(self, observation):
        metric = self.reward_config["monitor"]
        reward = observation[metric]

        if self.reward_config["min"]:
            reward = reward ** (-1)

        if "threshold" in self.reward_config:
            reward = np.where(
                reward > self.reward_config["threshold"], reward, self.invalid_penality
            )

        reward = np.where(~observation["valid"], self.invalid_penality, reward)

        return reward

    def step(self, action: dict):
        """
        Move the observator forward with one action and add the reward and stop condition to the returned observation.

        Args:
            action (dict): Dictionary containing "time" (array in units Mean Julian Date) "location"(dict with ra, decl, in degrees as arrays) (optional), "band" (str of the represention of the optical filter) (optional)

        Returns:
            Tuple : observation (dict, containing survey_config["variables"], vality, Time (in mjd)), reward (array), stop (array), log (dictionary)
        """
        print(action)
        self.observator.update(**action)
        self.time = self.observator.time.mjd.mean()
        observation = self._observation_calculation()
        reward = self._reward(observation)
        self.timestep += 1

        stop = self._stop_condition(observation)

        log = {}

        return observation, reward, stop, log

    def _observation_calculation(self):

        observation = {}
        for var_name in self.observatory_variables:
            observation[var_name] = self.observatory_variables[var_name]()[var_name]

        observation["valid"] = self._validity(observation=observation)
        observation["mjd"] = np.array(self.time)

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
