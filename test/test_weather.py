import pytest

from telescope_positioning_simulation.Survey import Weather
from telescope_positioning_simulation.Survey import ObservationVariables
from telescope_positioning_simulation.IO import ReadConfig


@pytest.fixture
def weather():
    weather_source_file = ""
    weather = Weather(weather_source_file)
    return weather


@pytest.fixture
def obsprog():
    config = ReadConfig()
    config["weather"] = {"include": True, "weather_source_file": ""}
    return ObservationVariables(config)


def test_find_date_summer(weather):
    mjd = 58300  # Jul 01 2018
    date = weather.find_date(mjd)

    assert date == ""


def test_find_date_winter(weather):
    mjd = 58119  # Jan 01 2018
    date = weather.find_date(mjd)

    assert date == ""


def test_find_condition(weather):
    mjd = ""
    condition = weather.condition(mjd)
    assert condition == ""


def test_find_seeing(weather):
    mjd = ""
    condition = weather.condition(mjd)
    seeing = weather.seeing(condition)
    assert seeing == ""


def test_find_clouds(weather):
    mjd = ""
    condition = weather.condition(mjd)
    clouds = weather.clouds(condition)

    assert clouds == ""


def test_init_weather_in_obser(obsprog):
    assert hasattr(obsprog, "weather")


def test_update_seeing_good_conditions(obsprog):
    obsprog.update(time="")
    assert obsprog.seeing == ""


def test_update_seeing_bad_conditions(obsprog):
    original_seeing = obsprog.seeing
    obsprog.update(time="")

    assert obsprog.seeing == ""
    assert obsprog.seeing != original_seeing


def test_update_clouds_good_conditions(obsprog):
    obsprog.update(time="")
    assert obsprog.clouds == ""


def test_update_clouds_bad_conditions(obsprog):
    original_clouds = obsprog.clouds
    obsprog.update(time="")

    assert obsprog.clouds == ""
    assert obsprog.clouds != original_clouds
