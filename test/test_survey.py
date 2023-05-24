import pytest

from telescope_positioning_simulation.Survey.survey import Survey


def test_default_init():
    survey = Survey()

    assert survey.reward_config["monitor"] == "airmass"
    assert survey.stop_config["timestep"] == 400
    assert survey.validity_config["airmass"]["value"] == 2.0

    assert survey.timestep_size == 300
    assert survey.start_time == "random"
    assert survey.invalid_penality == -100

    assert survey.save_config == "./equatorial_survey/"


def test_missing_params():
    input_config = {"nothing": 0}
    Survey(survey_config=input_config)
