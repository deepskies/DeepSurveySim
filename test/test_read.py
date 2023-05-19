from telescope_positioning_simulation.IO.read_config import ReadConfig
import pytest
import yaml


def test_read_telescope_default_config():
    obs_path = "telescope_positioning_simulation/settings/SEO.yaml"
    default_telescope = ReadConfig(
        observator_configuration="telescope_positioning_simulation/settings/SEO.yaml"
    )()

    with open(obs_path, "r") as f:
        default_config = yaml.safe_load(f)

    assert default_config == default_telescope


def test_read_survey_default_config():
    obs_path = "telescope_positioning_simulation/settings/equatorial_survey.yaml"
    default_telescope = ReadConfig(observator_configuration=obs_path, survey=True)()

    with open(obs_path, "r") as f:
        default_config = yaml.safe_load(f)

    assert default_config == default_telescope


def test_read_empty_config():
    empty_config = {}
    test_path = "test/test_files/empty_config.yaml"
    with open(test_path, "w") as f:
        yaml.safe_dump(empty_config, f)

    default_path = "telescope_positioning_simulation/settings/SEO.yaml"
    default_telescope = ReadConfig(test_path)()

    with open(default_path, "r") as f:
        read_config = yaml.safe_load(f)

    assert read_config == default_telescope


def test_read_half_filled_config():

    half_filled_config = {
        "location": {"n_sites": 100},
        "seeing": 2.0,
        "cloud_extinction": 1.0,
    }

    test_path = "test/test_files/test_config.yaml"
    with open(test_path, "w") as f:
        yaml.safe_dump(half_filled_config, f)

    default_path = "telescope_positioning_simulation/settings/SEO.yaml"
    read_config = ReadConfig(test_path)()

    with open(default_path, "r") as f:
        default_config = yaml.safe_load(f)

    assert read_config["location"] == half_filled_config["location"]
    assert read_config["seeing"] == half_filled_config["seeing"]
    assert read_config["cloud_extinction"] == half_filled_config["cloud_extinction"]

    for key in [
        key for key in read_config.keys() if key not in half_filled_config.keys()
    ]:
        assert read_config[key] == default_config[key]
