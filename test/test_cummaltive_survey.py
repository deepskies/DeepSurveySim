import pytest

from telescope_positioning_simulation.Survey import UniformSurvey, LowVisiblitySurvey
from telescope_positioning_simulation.IO import ReadConfig

action = {"location": {"ra": [0], "decl": [0]}, "band": "g"}
action_2 = {"location": {"ra": [1], "decl": [1]}, "band": "g"}


def test_uniform_site():
    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}
    obs_config["start_time"] = 59946
    uniform_survey = UniformSurvey(
        observatory_config=obs_config, survey_config=ReadConfig(survey=True)()
    )
    assert len(uniform_survey.all_steps) == 0
    assert uniform_survey._subclass_reward() == 0

    uniform_survey.step(action)
    assert uniform_survey._subclass_reward() == 0
    assert len(uniform_survey.all_steps) == 1

    uniform_survey.step(action_2)
    assert len(uniform_survey.all_steps) == 2
    assert uniform_survey._subclass_reward() == 0


def test_uniform_quality():

    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}
    obs_config["start_time"] = 59946

    uniform_survey = UniformSurvey(
        observatory_config=obs_config,
        survey_config=ReadConfig(survey=True)(),
        uniform="quality",
    )
    assert len(uniform_survey.all_steps) == 0
    assert uniform_survey._subclass_reward() == 0

    uniform_survey.step(action)
    assert uniform_survey._subclass_reward() == 0
    assert len(uniform_survey.all_steps) == 1

    uniform_survey.step(action_2)
    assert len(uniform_survey.all_steps) == 2
    assert uniform_survey._subclass_reward() == 0


# Todo - schedule that passes the conditions in the defaults


def test_lowvis_hit_required_sites():
    required_sites = [
        {"location": {"ra": [ra], "decl": [decl]}} for ra, decl in zip([0, 10], [0, 10])
    ]
    expected_reward = 1
    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}
    survey_config = ReadConfig(survey=True)()
    survey_config["invalid_penality"] = 0

    survey = LowVisiblitySurvey(
        observatory_config=obs_config,
        survey_config=survey_config,
        required_sites=required_sites,
        other_site_weight=0,
    )
    actions = [
        {"location": {"ra": [0], "decl": [0]}, "band": "g"},
        {"location": {"ra": [10], "decl": [10]}, "band": "g"},
    ]
    for action in actions:
        survey.step(action)

    print(survey.all_steps)
    assert survey._subclass_reward() >= expected_reward


def test_lowvis_hit_required_sites_correct_time():
    required_sites = [
        {"time": [time], "location": {"ra": [ra], "decl": [decl]}}
        for ra, decl, time in zip([0, 10], [0, 10], [59946.08, 59946.1])
    ]

    expected_reward = 1
    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}
    survey_config = ReadConfig(survey=True)()
    survey_config["invalid_penality"] = 0

    survey = LowVisiblitySurvey(
        observatory_config=obs_config,
        survey_config=survey_config,
        required_sites=required_sites,
        other_site_weight=0,
    )
    for action in required_sites:
        survey.step(action)

    assert survey._subclass_reward() >= expected_reward


def test_lowvis_hit_required_sites_incorrect_time():
    required_sites = [
        {"time": [time], "location": {"ra": [ra], "decl": [decl]}}
        for ra, decl, time in zip([0, 10], [0, 10], [59946.08, 59946.1])
    ]

    expected_reward = 1
    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}
    survey_config = ReadConfig(survey=True)()
    survey_config["invalid_penality"] = 0

    survey = LowVisiblitySurvey(
        observatory_config=obs_config,
        survey_config=survey_config,
        required_sites=required_sites,
        other_site_weight=0,
    )
    actions = [
        {"time": [time], "location": {"ra": [ra], "decl": [decl]}}
        for ra, decl, time in zip([0, 10], [0, 10], [59948, 59948.1])
    ]
    for action in actions:
        survey.step(action)

    assert survey._subclass_reward() < expected_reward


def test_lowvis_hit_required_sites_incorrect_band():
    required_sites = [
        {"band": band, "location": {"ra": [ra], "decl": [decl]}}
        for ra, decl, band in zip([0, 10], [0, 10], ["g", "g"])
    ]

    expected_reward = 1
    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}
    survey_config = ReadConfig(survey=True)()
    survey_config["invalid_penality"] = 0

    survey = LowVisiblitySurvey(
        observatory_config=obs_config,
        survey_config=survey_config,
        required_sites=required_sites,
        other_site_weight=0,
    )
    actions = [
        {"band": band, "location": {"ra": [ra], "decl": [decl]}}
        for ra, decl, band in zip([0, 10], [0, 10], ["b", "v"])
    ]

    for action in actions:
        survey.step(action)

    assert survey._subclass_reward() < expected_reward


def test_lowvis_incorrect_sites():
    required_sites = [
        {"location": {"ra": [ra], "decl": [decl]}}
        for ra, decl in zip([15, 20], [10, 20])
    ]

    expected_reward = 1
    obs_config = ReadConfig(survey=False)()
    obs_config["location"] = {"ra": [0], "decl": [0]}

    survey_config = ReadConfig(survey=True)()
    survey_config["invalid_penality"] = 0

    survey = LowVisiblitySurvey(
        observatory_config=obs_config,
        survey_config=survey_config,
        required_sites=required_sites,
        other_site_weight=0,
    )
    actions = [
        {"location": {"ra": [ra], "decl": [decl]}} for ra, decl in zip([0, 10], [0, 10])
    ]

    for action in actions:
        survey.step(action)

    assert survey._subclass_reward() < expected_reward
