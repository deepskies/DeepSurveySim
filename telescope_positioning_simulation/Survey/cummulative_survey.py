from telescope_positioning_simulation.Survey.survey import Survey
import numpy as np
import pandas as pd
import json


class CummulativeSurvey(Survey):
    def __init__(
        self, observatory_config: dict, survey_config: dict, *args, **kwargs
    ) -> None:
        """
        Abstract class for a survey that evaluated across multiple steps

        Args:
            observatory_config (dict): _description_
            survey_config (dict): _description_
        """
        super().__init__(observatory_config, survey_config)
        self.all_steps = pd.DataFrame()

    def reset(self):
        """Return the survey to its initial condition, wipe out all the original steps"""
        self.all_steps = pd.DataFrame()
        return super().reset()

    def cummulative_reward(self):
        """Reward that uses the 'all_steps' class parameter."""
        raise NotImplemented

    def step(self, action: dict):
        """
        Move the observator forward with one action and add the reward and stop condition to the returned observation.
        Reward is defined with 'cummulative_reward'

        Args:
            action (dict): Dictionary containing "time" (array in units Mean Julian Date) "location"(dict with ra, decl, in degrees as arrays) (optional), "band" (str of the represention of the optical filter) (optional)

        Returns:
            Tuple : observation (dict, containing survey_config["variables"], vality, Time (in mjd)), reward (array), stop (array), log (dictionary)
        """
        observation, reward, stop, log = super().step(action)
        observation_pd = {key: observation[key].ravel() for key in observation.keys()}

        observation_pd = pd.DataFrame(observation_pd)
        observation_pd["action"] = str(
            {"location": action["location"], "band": self.observator.band}
        )
        observation_pd["band"] = self.observator.band
        observation_pd["location"] = str(action["location"])
        observation_pd["reward"] = reward

        self.all_steps = self.all_steps.append(observation_pd)

        reward = self.cummulative_reward()
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
        """
        Calculate the reward for all sites, assuming it is required for a threshold on the number of times each site is visited.
        Follows the equation:

        $$R_{S_{n}} = \frac{1}{||T||* Var(\\{||(s_i)||: s_i \\in S_n\\})}*  \\Sigma^{S_{n}}_{i=0} \begin{cases} \\Sigma^{t_{n}}_{j=0} \tau_{eff}(s_{i,j}) & ||s_i|| \\geq N \\ 0 & ||s_i|| < N  \\ \\end{cases}$$

        Returns:
            float: reward based on if number of sites a site was visited reached a threshold
        """
        counts = self.all_steps["action"].value_counts()
        reward_scale = 1 / (len(self.all_steps) * np.var(counts))

        current_steps = self.all_steps.copy()
        # Replace reward if it doesn't make it pass the threshold
        current_steps.loc[
            self.all_steps["action"].isin(counts.index[counts < self.threshold]),
            "reward",
        ] = 0
        reward_sum = (
            current_steps.groupby(["mjd", "action", "band"])["reward"].sum().sum()
        )

        return reward_scale * reward_sum

    def quality_reward(self):
        """Cummulitive reward based on if a per site reward threshold is reached.
        Defined as:

        $$R_{S_{n}} = \frac{1}{||T||* Var(\\{\tau_{eff}(s_i): s_i \\in S_n)\\}}* \\Sigma^{t_{n}}_{i=0} \begin{cases} \tau_{eff}(s_i) & T_{eff}(s_i) \\geq \theta \\ 0 & \tau_{eff}(s_i) < \theta  \\ \\end{cases}$$

        Returns:
            float: reward based on if a threshold in reward per site is reached.
        """
        reward_scale = 1 / (len(self.all_steps)) * np.var(self.all_steps["reward"])

        current_steps = self.all_steps.copy()
        # Replace reward if it doesn't make it pass the threshold
        current_steps.loc[current_steps["reward"] < self.threshold, "reward"] = 0
        reward_sum = current_steps["reward"].sum()
        return reward_scale * reward_sum

    def cummulative_reward(self, *args, **kwargs):
        """
        Quality thresholded or visit thresholded reward

        Returns:
            float: all sites reward
        """
        if len(self.all_steps) != 0:
            reward = self.reward_function()
            reward = reward if not (pd.isnull(reward) or reward == -np.inf) else 0

            return reward
        else:
            return 0


class LowVisiblitySurvey(CummulativeSurvey):
    def __init__(
        self,
        observatory_config: dict,
        survey_config: dict,
        required_sites: list = [],
        other_site_weight: float = 0.6,
        time_tolerance: float = 0.01388,
    ) -> None:
        """
        Survey that evaluates how many times a "required site" has been visited, and defines the reward of a site based on this.

        Args:
            obseravtory_config (dict): Setup parameters for Survey.ObservationVariables, the telescope configuration, as read by IO.ReadConfig
            survey_config (dict): Parameters for the survey, including the stopping conditions, the validity conditions, the variables to collect, as read by IO.ReadConfig
            required_sites (list, optional): Sites that must be visited to achieve reward. Same format as an action. Defaults to [].
            other_site_weight (float, optional): Weighting factor applied to reward gained from visiting sites not in the "required_sites" list. Defaults to 0.6.
            time_tolerance (float, optional): Tolerance given to a required time. Defaults to 0.01388 (seconds, 20 minutes).
        """
        super().__init__(observatory_config, survey_config)

        self.required_sites = required_sites
        self.time_tolerance = time_tolerance
        self.weight = other_site_weight

    def sites_hit(self):
        """Count the number of times a required site was visited. Does not follow a thresholding rule.

        Returns:
            int: times the requires sites were visited.
        """

        # TODO do this with sets and arrays instead of a loop
        hit_counter = 0
        for site in self.required_sites:

            subset = self.all_steps.copy()
            if "time" in site.keys():
                subset = subset[
                    (subset["mjd"] < site["time"][0] + self.time_tolerance)
                    & (subset["mjd"] > site["time"][0] - self.time_tolerance)
                ]

            if "band" in site.keys():
                subset = subset[subset["band"] == site["band"]]

            subset = subset[subset["location"] == str(site["location"])]
            hit_counter += len(subset)

        return hit_counter

    def cummulative_reward(self):
        """
        Reward for all visited sites as defined as
        $$  R_{s_{n}} = \frac{1}{||T||} (||\\{s_i: s_i \\in S_{interest}\\}|| + \\lambda \\Sigma^{t_{n}}_{t=0}  T_{eff_t}(s_t)) $$

        Returns:
            float: reward calculted as a result of all current sites visited in the schedule
        """
        if len(self.all_steps) != 0:

            reward_scale = 1 / len(self.all_steps)
            weighted_term = self.weight * self.all_steps["reward"].sum()
            number_of_interest_hit = self.sites_hit()

            reward = reward_scale * (weighted_term + number_of_interest_hit)
            reward = reward if not (pd.isnull(reward) or reward == -np.inf) else 0

            return reward

        else:
            return 0.0
