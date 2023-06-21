from Survey.survey import Survey
from IO.read_config import ReadConfig
from astropy.time import Time
import numpy as np

seo_config = ReadConfig(
        observator_configuration="settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="settings/equatorial_survey.yaml",
        survey=True
    )()

env = Survey(seo_config, survey_config)
observation = env._observation_calculation()

stop = False
while not stop:
    action = {
        "time": Time.now().mjd,
        #"location": {"ra": np.array([30]), "decl": np.array([30])},
        #"location": {"ra": 30, "decl": 30},
        "location": {"ra": [30], "decl": [30]},
        "band": "g"
    }
    observation, reward, stop, log = env.step(action)
    print(observation)