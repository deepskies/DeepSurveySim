import pytest
import numpy as np

from telescope_positioning_simulation.Survey import Weather
from telescope_positioning_simulation.Survey import ObservationVariables
from telescope_positioning_simulation.IO import ReadConfig


@pytest.fixture
def weather():
    weather_source_file = "./telescope_positioning_simulation/settings/3428354.csv"
    weather = Weather(weather_source_file)
    return weather


@pytest.fixture
def obsprog():
    config = ReadConfig()
    source_path = "./telescope_positioning_simulation/settings/3428354.csv"
    config["weather"] = {"include": True, "weather_source_file": source_path}
    return ObservationVariables(config)


def test_find_date_summer(weather):
    mjd = 58300  # Jul 01 2018
    date = weather.find_date(mjd)

    assert np.all(date.str.month == 7)


def test_find_date_winter(weather):
    mjd = 58119  # Jan 01 2018
    date = weather.find_date(mjd)

    assert np.all(date.str.month == 1)


def test_find_condition(weather):
    mjd = 58300
    condition = weather.condition(mjd)

    assert np.all(condition["DATE"].str.datetime.month == 7)


def test_find_seeing(weather):
    mjd = 58300
    condition = weather.condition(mjd)
    seeing = weather.seeing(condition)
    assert seeing == pytest.approx(0.9, 0.1)


def test_find_clouds(weather):
    mjd = 58300
    condition = weather.condition(mjd)
    clouds = weather.clouds(condition)

    assert clouds == pytest.approx(0.0, 0.1)


def test_init_weather_in_obser(obsprog):
    assert hasattr(obsprog, "weather")


def test_update_seeing_good_conditions(obsprog):
    obsprog.update(time=58300)
    assert obsprog.seeing == 0.9


def test_update_seeing_middling_conditions(obsprog):
    obsprog.update(time=58300)

    assert obsprog.seeing != 0.9
    assert obsprog.seeing != 0.0


def test_update_seeing_bad_conditions(obsprog):
    obsprog.update(time=58300)
    assert obsprog.seeing == 0.0


def test_update_clouds_good_conditions(obsprog):
    obsprog.update(time=58300)
    assert obsprog.clouds == 0.0


def test_update_clouds_bad_conditions(obsprog):
    original_clouds = obsprog.clouds
    obsprog.update(time=58300)

    assert obsprog.clouds != original_clouds
