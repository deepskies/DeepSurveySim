import pytest
import pandas as pd
import numpy as np


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
    survey = Survey(survey_config=input_config)

    assert survey.reward_config["monitor"] == "airmass"
    assert survey.stop_config["timestep"] == 400
    assert survey.validity_config["airmass"]["value"] == 2.0

    assert survey.timestep_size == 300
    assert survey.start_time == "random"
    assert survey.invalid_penality == -100

    assert survey.save_config == "./equatorial_survey/"


def test_valid_observeration():
    observation = {"random": np.array([10, 10, 10, 10])}
    survey_config = {"constaints": {"random": {"value": 1, "lesser": True}}}

    s = Survey(survey_config=survey_config)
    valid = s.validity(observation)
    assert np.all(valid)


def test_invalid_observeration():
    observation = {"random": np.array([10, 10, 10, 10])}
    survey_config = {"constaints": {"random": {"value": 1, "lesser": False}}}
    s = Survey(survey_config=survey_config)
    valid = s.validity(observation)
    assert not np.all(valid)


def test_mixed_invalid_observeration():
    observation = {
        "airmass": np.array([1, 10, 12, 10]),
        "alt": np.array([50, 60, 70, 60]),
    }
    survey_config = {
        "constaints": {
            "airmass": {"value": 2.0, "lesser": False},
            "alt": {"value": 20.0, "lesser": True},
        }
    }

    s = Survey(survey_config=survey_config)
    valid = s.validity(observation)
    assert np.any(~valid)


def test_dont_stop_me():
    survey_config = {
        "stopping": {"timestep": 6, "random_var": {"value": 42, "lesser": True}}
    }
    s = Survey(survey_config=survey_config)
    observation = {"random_var": np.array([80])}
    stop = s._stop_condition(observation)

    assert stop


def test_stop_me():
    survey_config = {
        "stopping": {"timestep": 6, "random_var": {"value": 42, "lesser": True}}
    }
    s = Survey(survey_config=survey_config)
    observation = {"random_var": np.array([10, 50, 2])}
    stop = s._stop_condition(observation)

    assert stop


def test_stop_me_timestep():

    survey_config = {
        "stopping": {"timestep": 6, "random_var": {"value": 42, "lesser": True}}
    }
    s = Survey(survey_config=survey_config)

    s.timestep = 50

    observation = {"random_var": np.array([70])}
    stop = s._stop_condition(observation)
    assert stop


def test_all_gather_variables():
    observation = Survey()._observation_calculation()
    observation_keys = observation.keys()
    expected_keys = [
        "airmass",
        "ha",
        "moon_airmass",
        "lst",
        "sun_airmass",
        "alt",
        "mjd",
        "valid",
    ]

    expected_shape = (10, 1)
    for key in observation_keys:
        assert observation[key].shape == expected_shape

    assert set(observation_keys) == set(expected_keys)


def test_subset_gather_variables():
    subset_config = {
        "variables": ["alt", "az"],
        "constaints": {"alt": {"value": 20, "lesser": False}},
    }
    observation = Survey(subset_config)._observation_calculation()
    observation_keys = observation.keys()
    expected_subset = ["alt", "az", "mjd", "valid"]

    expected_shape = (10, 1)
    for key in observation_keys:
        assert observation[key].shape == expected_shape

    assert set(observation_keys) == set(expected_subset)


def test_single_reward():
    pass


def test_threshold_reward():
    pass
