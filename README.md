
![status](https://img.shields.io/badge/License-MIT-lightgrey)

# This branch has been tested with Python 3.9

# Summary

This repo is a stripped down version of the main [RLTelescope](https://github.com/deepskies/RLTelescopes); containing just the positioning simulation.


# Installation
## Install from pip
Simply run

`pip install git+https://github.com/deepskies/TelescopePositioningSimulation`

This will install the project with all its requirements.

## Install from source

The project is built with [poetry](https://python-poetry.org/), and this is the recommended install method.
All dependencies are resolved in the `poetry.lock` file, so you can install immediately from the command

```
poetry shell
poetry install

```

Assuming you have poetry installed on your base environment.
This will use lock file to install all the correct versions.
To use the installed environment, use the command `poetry shell` to enter it.
The command `exit` will take you out of this environment as it would for any other type of virtual environment.

Otherwise, you can use the `pyproject.toml` with your installer of choice.

To verify all the depedencies are properly installed - run `python run pytest`.

## Install palpy and skybright from source 

## Workaround to Install palpy
Install from source repository
Modify setup.py with the following changes
#sources = ["cpal.pxd", "pal.pyx"]
sources = ["pal.pyx"]

## Workaround for skybright

convert fit_sky.py from Python2 to Python3

missing apt_pkg.so fix
$ sudo cp apt_pkg.cpython-38-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/apt_pkg.so

## Installation of stablebaselines3 Recurrent PPO

pip install sb3-contrib

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

## Evaluate Checkpoints

run the model_test_xxx.py script

Example:  Load a PPO checkpoint
model_path = f"{models_dir}/710000.zip" #  NEED TO CHANGE PATH FOR CHECKPOIN
model = PPO.load(model_path, env=env)


# Acknowledgement
And you <3


