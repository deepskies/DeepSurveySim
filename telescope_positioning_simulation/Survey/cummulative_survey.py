from telescope_positioning_simulation.Survey.survey import Survey
import numpy as np
import pandas as pd


class CummulativeSurvey(Survey):
    def __init__(self, observatory_config: dict, survey_config: dict) -> None:
        super().__init__(observatory_config, survey_config)
        self.all_steps = pd.DataFrame()

    def reset(self):
        self.all_steps = pd.DataFrame()
        return super().reset()

    def _subclass_reward(self):
        raise NotImplemented

    def step(self, action: dict):
        observation, reward, stop, log = super().step(action)
        observation_pd = {key: observation[key].ravel() for key in observation.keys()}

        observation_pd = pd.DataFrame(observation_pd)
        observation_pd["action"] = str(action["location"]) + str(action["band"])
        observation_pd["reward"] = reward

        self.all_steps = self.all_steps.append(observation_pd)

        reward = self._subclass_reward()
        return observation, reward, stop, log


class UniformSurvey(CummulativeSurvey):
    """
    A child survey that instead of evaluating the schedule at every step, evaluates it for a full schedule.
    This survey requires a threshold is reached for each observation before it starts recording any sort of reward.

    Args:
        obseravtory_config (dict): Setup parameters for Survey.ObservationVariables, the telescope configuration, as read by IO.ReadConfig
        survey_config (dict): Parameters for the survey, including the stopping conditions, the validity conditions, the variables to collect, as read by IO.ReadConfig
        threshold (float): Threshold the survey must pass to have its quality counted towards the total reward
        uniform (str): ["site", "quality"] - If measuring the uniformity of the number of times each site has been visited, or the uniformity of the quality of observations

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
        threshold: float = 1.0,
        uniform: str = "site",
    ) -> None:
        super().__init__(observatory_config, survey_config)

        self.threshold = threshold
        reward_function = {
            "site": self.site_reward,
            "quality": self.quality_reward,
        }
        assert uniform in reward_function

        self.reward_function = reward_function[uniform]

    def site_reward(self):
        counts = self.all_steps["action"].value_counts()
        reward_scale = 1 / (len(self.all_steps) * np.var(counts))

        current_steps = self.all_steps.copy()
        # Replace reward if it doesn't make it pass the threshold
        current_steps.loc[
            self.all_steps["action"].isin(counts.index[counts < self.threshold]),
            "reward",
        ] = 0
        reward_sum = current_steps.groupby(["mjd", "action"])["reward"].sum().sum()

        return reward_scale * reward_sum

    def quality_reward(self):
        reward_scale = 1 / (len(self.all_steps)) * np.var(self.all_steps["reward"])

        current_steps = self.all_steps.copy()
        # Replace reward if it doesn't make it pass the threshold
        current_steps.loc[current_steps["reward"] < self.threshold, "reward"] = 0
        reward_sum = current_steps["reward"].sum()
        return reward_scale * reward_sum

    def _subclass_reward(self, *args, **kwargs):
        if len(self.all_steps) != 0:
            reward = self.reward_function()
            reward = reward if not (pd.isnull(reward) or reward == -np.inf) else 0

            return reward
        else:
            return 0


class LowVisiblitySurvey(CummulativeSurvey):
    def __init__(
        self, observatory_config: dict, survey_config: dict, required_sites: dict = {}
    ) -> None:
        super().__init__(observatory_config, survey_config)

        self.all_steps = pd.DataFrame()
        self.required_sites = required_sites

    def _subclass_reward(self):
        if len(self.all_steps) != 0:
            reward_scale = 1 / len(self.all_steps)
            weighted_term = self.weight * self.all_steps["reward"].sum()
            number_of_interest_hit = ""

            reward = reward_scale * (weighted_term + number_of_interest_hit)
            reward = reward if not (pd.isnull(reward) or reward == -np.inf) else 0

            return reward
        else:
            return 0
