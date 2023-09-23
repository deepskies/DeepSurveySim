"""
Run a simulation, picking up the variables specified in the config file and updating the time and location based on the
"""
import numpy as np
from gymnasium import spaces
#from Survey.survey.py import Survey as SurveyBase
from .survey import Survey as SurveyBase
#from telescope_positioning_simulation.Survey.survey.py import Survey as SurveyBase
#from TelescopePositioningSimulation.telescope_positioning_simulation.Survey.survey.py import Survey as SurveyBase

def find_dict_structure(dictionary, depth=0):
    for key, value in dictionary.items():
        indent = "  " * depth
        print(f"{indent}- {key}: {type(value).__name__}")
        if isinstance(value, dict):
            find_dict_structure(value, depth + 1)

def print_keys(dictionary):
    keys = dictionary.keys()
    for key in keys:
        print(key)

def print_specific_keys(dictionary, keys_to_print):
    for key in keys_to_print:
        if key in dictionary:
            print(f"{key}: {dictionary[key]}")
        else:
            print(f"Key '{key}' not found in the dictionary.")



class Survey(SurveyBase):
    def __init__(self, observatory_config: dict = {}, survey_config: dict = {}) -> None:
        super().__init__(observatory_config, survey_config)
        self.action_space = spaces.box.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.ra_min, self.ra_max = 0, 360
        self.decl_min, self.decl_max = -90, 90
        self.observation_space = spaces.box.Box(low=-100000, high=100000, shape=(6,), dtype=np.float32)
        # TODO: for now tuned in order to make it work, to change with the correct ranges

    def reset(self, *, seed=None, options=None):
        self.timestep = 0
        self.previous_mjd = self._start_time()
        observation = self._observation_calculation()
        info = {}
        info['invalid'] = ~self.validity(observation)
        info['mjd'] = observation["mjd"]
        observation = {key: observation[key] for key in self.observatory_variables}
        obs_list = list(observation.values())
        obs_array = np.squeeze(obs_list)
        observation = np.array(obs_array)
        #return observation, info
        return observation

    def _reward(self, observation, info):
        reward = super()._reward(observation, info)
        if isinstance(reward, np.ndarray) and reward.ndim == 1:
            reward = reward[0]
        #print("Reward", reward)
        return reward

    def step(self, action: dict):
        '''
        #print('action')
        #print(action)
        #print(isinstance(action, dict))
        #find_dict_structure(action)
        #print('action parts')
        #print_keys(action)
        #print_specific_keys(action,["ra","decl","band"])
        #print("action parts")
        #print(action['ra'][0])
        #print(action[0])
        # remap the action space range [-1, 1] into the proper range of the variables
        #ra = (action[0] + 1) * (self.ra_max - self.ra_min) / 2 + self.ra_min  
        #ra = (action['ra'][0] + 1) * (self.ra_max - self.ra_min) / 2 + self.ra_min
        #decl = (action['decl'][0] + 1) * (self.decl_max - self.decl_min) / 2 + self.decl_min
        #ra = (action['ra'][0] + 1) * (self.ra_max - self.ra_min) / 2 + self.ra_min
        #decl = (action['decl'][0] + 1) * (self.decl_max - self.decl_min) / 2 + self.decl_min
        '''
        ra = (action[0] + 1) * (self.ra_max - self.ra_min) / 2 + self.ra_min
        decl = (action[1] + 1) * (self.decl_max - self.decl_min) / 2 + self.decl_min
        new_action = {"time": self.previous_mjd, "location": {"ra": [ra], "decl": [decl]}}
        print('new action')
        print(new_action)
        self.observator.update(**new_action)
        observation = self._observation_calculation()
        info = {}
        info["invalid"] = not self.validity(observation)
        info["mjd"] = observation["mjd"]
        observation = {key: observation[key] for key in self.observatory_variables }
        reward = self._reward(observation, info)
        self.timestep += 1
        self.previous_mjd = info["mjd"]
        stop = self._stop_condition(observation)
        truncated = False  # Additional truncated flag required by RLLib
        obs_list = list(observation.values())
        obs_array = np.squeeze(obs_list)
        observation = np.array(obs_array)
        return observation, reward, stop, truncated, info
    
    def get_reward(self, observation, info):
        reward = super()._reward(observation, info)
        if isinstance(reward, np.ndarray) and reward.ndim == 1:
            reward = reward[0]
        #print("Reward", reward)
        return reward


