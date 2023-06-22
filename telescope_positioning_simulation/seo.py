from Survey.survey import Survey
from IO.read_config import ReadConfig
from astropy.time import Time
import numpy as np

seo_config = ReadConfig(
        observator_configuration="telescope_positioning_simulation/settings/SEO.yaml"
    )()


survey_config = ReadConfig(
        observator_configuration="telescope_positioning_simulation/settings/equatorial_survey.yaml",
        survey=True
    )()
survey_config['location']  = {'ra': [0], 'decl':[0]}



env = Survey(seo_config, survey_config)
observation = env._observation_calculation()

stop = False
while not stop:
    action = {
        "time": Time.now().mjd,
        "location": {"ra": [30], "decl": [30]},
        "band": "g"
    }
    observation, reward, stop, log = env.step(action)
    print(observation)