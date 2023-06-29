from Survey.RLLib_survey import Survey
from IO.read_config import ReadConfig
from astropy.time import Time
import numpy as np
import random

seo_config = ReadConfig(
        observator_configuration="settings/SEO.yaml"
    )()


survey_config = ReadConfig(
        observator_configuration="settings/equatorial_survey.yaml",
        survey=True
    )()
survey_config['location']  = {'ra': [0], 'decl': [0]}

env = Survey(seo_config, survey_config)
observation = env.reset()

stop = False
while not stop:
    ra = random.randint(0, 90)
    decl = random.randint(0, 90)
    action = {
        "ra": [ra],
        "decl": [decl],
        "band": "g"
    }
    print("Action",action)
    observation, reward, stop, truncated, log = env.step(action)
    print("Observation",observation)
    print("Reward",reward)
    print("Stop",stop)