from telescope_positioning_simulation.IO.save_simulation import SaveSimulation
from telescope_positioning_simulation.IO.read_config import ReadConfig
from telescope_positioning_simulation.Survey.survey import Survey

import pytest
import os
import json
import yaml
import numpy as np


@pytest.fixture
def default_survey():
    default_survey_config = ReadConfig(survey=True)()
    default_observatory_config = ReadConfig(survey=False)()

    default_survey_config["save"] = "test/test_files/test_saving/"

    survey = Survey(
        survey_config=default_survey_config,
        obseravtory_config=default_observatory_config,
    )

    return survey


def test_save_expected_results_file(default_survey):
    sample_results = {}

    sample_results[default_survey.observator.time.mjd] = {
        "airmass": np.array([1.0, 10.0, 100.0], dtype=np.float32),
        "reward": np.array([1, 2, 3], dtype=np.float32),
    }

    saver = SaveSimulation(default_survey, sample_results)
    saver.save_results()

    expected_save_path = f"{saver.save_path}/survey_results.json"
    assert os.path.exists(expected_save_path)
    with open(expected_save_path, "r") as f:
        saved_results = json.load(f)

    assert len(saved_results.keys()) == len(sample_results.keys())


def test_save_config_file(default_survey):
    saver = SaveSimulation(default_survey, {})
    saver.save_config()

    expected_save_path = f"{saver.save_path}/run_config.yaml"
    assert os.path.exists(expected_save_path)

    with open(expected_save_path, "r") as f:
        saved_config = yaml.safe_load(f)

    default_survey_config = ReadConfig(survey=True)()
    default_observatory_config = ReadConfig(survey=False)()

    for key in default_survey_config.keys():
        assert key in saved_config.keys()

    for key in default_observatory_config.keys():
        assert key in saved_config.keys()


def test_id_random():
    id_1 = SaveSimulation.generate_run_id()
    id_2 = SaveSimulation.generate_run_id()

    assert id_1 != id_2
