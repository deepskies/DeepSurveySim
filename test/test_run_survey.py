from DeepSurveySim.Survey.survey import Survey
from DeepSurveySim.IO.read_config import ReadConfig

import numpy as np
import pytest


@pytest.fixture
def survey_setup():
    seo_config = ReadConfig(
        observator_configuration="DeepSurveySim/settings/SEO.yaml"
    )()

    survey_config = ReadConfig(
        observator_configuration="DeepSurveySim/settings/equatorial_survey.yaml",
        survey=True,
    )()

    survey_config["stopping"]["timestep"] = 5
    seo_config["location"] = {"ra": [0], "decl": [0]}

    survey = Survey(seo_config, survey_config)
    print(survey.observator.location)
    observations = []
    rewards = []
    stops = []
    mjd = np.array(60000)

    for _ in range(20):
        action = {
            "location": {
                "decl": [np.random.default_rng().integers(-90, 90)],
                "ra": [np.random.default_rng().integers(0, 360)],
            },
            "time": mjd,
        }
        observation, reward, stop, _ = survey.step(action)
        observations.append(observation)
        rewards.append(reward)
        stops.append(stop)
        mjd = observation["mjd"]

    print(observations)
    return observations, rewards, stops


def test_length(survey_setup):
    assert len(survey_setup[0]) == len(survey_setup[1]) == 20


def test_has_stop(survey_setup):
    assert True in survey_setup[2]

    assert survey_setup[2][9] == survey_setup[2][4] == True


@pytest.mark.flakey()
def test_has_invalid(survey_setup):
    assert True in [obs["valid"][0][0] for obs in survey_setup[0]]


@pytest.mark.flakey()
def test_has_valid(survey_setup):
    assert False in [obs["valid"][0][0] for obs in survey_setup[0]]


def test_valid_matchup(survey_setup):
    reward = np.array([survey.ravel() for survey in survey_setup[1]])
    valid = np.array([obs["valid"][0].ravel() for obs in survey_setup[0]])
    assert (reward[~valid] == -100).all()


def test_has_reward(survey_setup):
    reward = np.array([survey.ravel() for survey in survey_setup[1]])
    valid = np.array([obs["valid"].ravel() for obs in survey_setup[0]])
    assert (reward[valid] != -100).all()
