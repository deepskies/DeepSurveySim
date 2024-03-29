import pytest
import numpy as np

from DeepSurveySim.Survey import Weather
from DeepSurveySim.Survey import ObservationVariables
from DeepSurveySim.IO import ReadConfig

weather_source_file = "./DeepSurveySim/settings/SEO_weather.csv"


@pytest.fixture
def weather():
    weather = Weather(weather_source_file)
    return weather


@pytest.fixture
def obsprog():
    config = ReadConfig()()
    config["weather_sim"] = True
    config["weather_config"] = {"weather_source_file": weather_source_file}
    return ObservationVariables(config)


def test_find_date_summer(weather):
    mjd = 58300  # Jul 01 2018
    month, day = weather._find_date(mjd)

    assert month == 7
    assert day == 1


def test_find_date_winter(weather):
    mjd = 58119  # Jan 01 2018
    month, day = weather._find_date(mjd)

    assert month == 1
    assert day == 1


def test_find_condition(weather):
    mjd = 58300
    condition = weather.condition(mjd)

    assert np.all(condition["DATE"].dt.month == 7)
    assert np.all(condition["DATE"].dt.day.isin([1, 2, 3]))


def test_find_seeing(weather):
    mjd = 58320
    condition = weather.condition(mjd)
    seeing = weather.seeing(condition)
    assert seeing == pytest.approx(0.9, 0.1)


def test_find_clouds(weather):
    mjd = 58300
    condition = weather.condition(mjd)
    clouds = weather.clouds(condition)

    assert clouds == pytest.approx(1.0, 0.1)


def test_init_weather_in_obser(obsprog):
    assert hasattr(obsprog, "weather")


def test_update_seeing_good_conditions(obsprog):
    obsprog.update(time=58320)
    assert obsprog.seeing == 0.9


def test_update_seeing_bad_conditions(obsprog):
    obsprog.update(time=58119)
    assert obsprog.seeing != 0.9


def test_update_clouds_good_conditions(obsprog):
    obsprog.update(time=58320)
    assert obsprog.clouds == 0.0


def test_update_clouds_bad_conditions(obsprog):
    original_clouds = obsprog.clouds
    obsprog.update(time=58119)

    assert obsprog.clouds != original_clouds
