import pytest

from telescope_positioning_simulation.Survey import UniformSurvey, LowVisiblitySurvey
from telescope_positioning_simulation.IO import ReadConfig

action = {"location": {"ra": [0], "decl": [0]}, "band": "g"}
action_2 = {"location": {"ra": [1], "decl": [1]}, "band": "g"}


def test_uniform_site():
    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}
    uniform_survey = UniformSurvey(
        observatory_config=obs_config, survey_config=ReadConfig(survey=True)()
    )
    assert len(uniform_survey.all_steps) == 0
    assert uniform_survey._reward() == 0

    uniform_survey.step(action)
    assert uniform_survey._reward() == 0
    assert len(uniform_survey.all_steps) == 1

    uniform_survey.step(action_2)
    assert len(uniform_survey.all_steps) == 2
    assert uniform_survey._reward() == 0


def test_uniform_quality():

    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}
    uniform_survey = UniformSurvey(
        observatory_config=obs_config,
        survey_config=ReadConfig(survey=True)(),
        uniform="quality",
    )
    assert len(uniform_survey.all_steps) == 0
    assert uniform_survey._reward() == 0

    uniform_survey.step(action)
    assert uniform_survey._reward() == 0
    assert len(uniform_survey.all_steps) == 1

    uniform_survey.step(action_2)
    assert len(uniform_survey.all_steps) == 2
    assert uniform_survey._reward() == 0
