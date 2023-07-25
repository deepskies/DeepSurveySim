
![status](https://img.shields.io/badge/License-MIT-lightgrey)

# Summary

This repo is a stripped down version of the main [RLTelescope](https://github.com/deepskies/RLTelescopes); containing just the positioning simulation.


# Installation
## Install from pip
Simply run

`pip install git+https://github.com/deepskies/TelescopePositioningSimulation`

This will install the project with all its mandatory requirements.

If you wish to include the optional `skybright`, use the command:
`pip install git+https://github.com/deepskies/TelescopePositioningSimulation@development#egg=telescope-positioning-simulation[skybright]`

Not installing this will result in loss of the variables `sky_magintude`, `tau`, and `teff`, but will work on most (if not all) machines.

## Install from source

The project is built with [poetry](https://python-poetry.org/), and this is the recommended install method.
All dependencies are resolved in the `poetry.lock` file, so you can install immediately from the command

```
git clone https://github.com/deepskies/TelescopePositioningSimulation.git
poetry shell
poetry install --all-extras

```

Assuming you have poetry installed on your base environment.
This will use lock file to install all the correct versions.
To use the installed environment, use the command `poetry shell` to enter it.
The command `exit` will take you out of this environment as it would for any other type of virtual environment.

Otherwise, you can use the `pyproject.toml` with your installer of choice.

To verify all the depedencies are properly installed - run `python run pytest`.

# Example:

## To run as a live envoriment for RL

```
from telescope_positioning_simulation.Survey.survey import Survey
from telescope_positioning_simulation.IO.read_config import ReadConfig

seo_config = ReadConfig(
        observator_configuration="telescope_positioning_simulation/settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="telescope_positioning_simulation/settings/equatorial_survey.yaml",
        survey=True
    )()

env = Survey(seo_config, survey_config)
observation = env._observation_calculation()

stop = True
while not stop:
    action = model.predict_action(observation)
    observation, reward, stop, log = env.step()

```

## To generate observations

```
from telescope_positioning_simulation.Survey.survey import Survey
from telescope_positioning_simulation.IO.read_config import ReadConfig

seo_config = ReadConfig(
        observator_configuration="telescope_positioning_simulation/settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="telescope_positioning_simulation/settings/equatorial_survey.yaml",
        survey=True
    )()

env = Survey(seo_config, survey_config)
observations = env()

```


# Acknowledgement
And you <3


