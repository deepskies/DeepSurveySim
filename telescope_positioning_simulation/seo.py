from Survey.RLLib_survey import Survey
from IO.read_config import ReadConfig
from astropy.time import Time

seo_config = ReadConfig(
        observator_configuration="settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="settings/equatorial_survey.yaml",
        survey=True
    )()

survey_config['location']  = {'ra': [0], 'decl':[0]}

env = Survey(seo_config, survey_config)
observation = env.reset()

stop = False
while not stop:
    action = {
        "time": Time.now().mjd,
        "ra": [50],
        "decl": [50],
        "band": "g"
    }
    observation, reward, stop, truncation, log = env.step(action)
    print(observation, reward, stop, truncation, log)