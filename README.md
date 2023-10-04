
[![status](https://img.shields.io/badge/License-BSD3-lightgrey)](https://github.com/deepskies/DeepSurveySim/blob/main/LICENSE)
[![test-telescope](https://github.com/deepskies/TelescopePositioningSimulation/actions/workflows/test-telescope.yaml/badge.svg?branch=main)](https://github.com/deepskies/DeepSurveySim/actions/workflows/test-telescope.yaml)
 [![PyPI version](https://badge.fury.io/py/deepsurveysim.svg)](https://badge.fury.io/py/deepsurveysim)
[![Documentation Status](https://readthedocs.org/projects/deepsurveysim/badge/?version=latest)](https://deepsurveysim.readthedocs.io/en/latest/?badge=latest)

# Summary

Modern astronomical surveys have multiple competing scientific goals.
Optimizing the observation schedule for these goals presents significant computational and theoretical challenges, and state-of-the-art methods rely on expensive human inspection of simulated telescope schedules.
Automated methods, such as reinforcement learning, have recently been explored to accelerate scheduling.
**DeepSurveySim** provides methods for tracking and approximating sky conditions for a  set of observations from a user-supplied telescope configuration.

# Documentation

### [ReadTheDocs](https://deepsurveysim.readthedocs.io/en/latest/)

### Build locally

First install the package from source, then run

```
pip install sphinx
cd docs
make html
```

The folder `docs/_build/html` will be populated with the documentation.
Navigate to `file:///<path to local install>/docs/_build/html/index.html` in any web browser to view.



# Installation
### Install from pip

Simply run

```
pip install DeepSurveySim
```

This will install the project with all its mandatory requirements.

If you wish to add the optional `skybright`, use the command:

```
pip install git+https://github.com/ehneilsen/skybright.git
```

Not installing this will result in loss of the variables `sky_magintude`, `tau`, and `teff`, but will work on most (if not all) machines.

### Install from source

The project is built with [poetry](https://python-poetry.org/), and this is the recommended install method.
All dependencies are resolved in the `poetry.lock` file, so you can install immediately from the command

```
git clone https://github.com/deepskies/DeepSurveySim.git
poetry shell
poetry install
poetry add git+https://github.com/ehneilsen/skybright.git

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
from DeepSurveySim.Survey.survey import Survey
from DeepSurveySim.IO.read_config import ReadConfig

seo_config = ReadConfig(
        observator_configuration="DeepSurveySim/settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="DeepSurveySim/settings/equatorial_survey.yaml",
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
from DeepSurveySim.Survey.survey import Survey
from DeepSurveySim.IO.read_config import ReadConfig

seo_config = ReadConfig(
        observator_configuration="DeepSurveySim/settings/SEO.yaml"
    )()

survey_config = ReadConfig(
        observator_configuration="DeepSurveySim/settings/equatorial_survey.yaml",
        survey=True
    )()

env = Survey(seo_config, survey_config)
observations = env()
```


# Acknowledgement
This work was produced by Fermi Research Alliance, LLC under Contract No. DE-AC02-07CH11359 with the U.S. Department of Energy, Office of Science, Office of High Energy Physics. Publisher acknowledges the U.S. Government license to provide public access under the DOE Public Access Plan DOE Public Access Plan.

We acknowledge the Deep Skies Lab as a community of multi-domain experts and collaborators who’ve facilitated an environment of open discussion, idea-generation, and collaboration. This community was important for the development of this project.

We thank Franco Terranova  and Shohini Rhae for their assistance in testing the preliminary version of the package, and Eric Neilsen  Jr. for his domain expertise.

# Citation

If this package is useful for your work, we request you cite us:
```

```

If the `skybright` option is used, we also encourage its citation:
```
@misc{skybright_Neilsen:2019,
    author = "Neilsen, Eric",
    title = "{skybright}",
    reportNumber = "FERMILAB-CODE-2019-01",
    doi = "10.11578/dc.20190212.1",
    month = "2",
    year = "2019"
}
```


