import pytest
from telescope_positioning_simulation.Survey.observation_variables import (
    ObservationVariables,
)
from telescope_positioning_simulation.IO.read_config import ReadConfig

# Assuming the variables are correct from the test_read

import numpy as np
import pandas as pd
import astropy


@pytest.fixture
def seo_observatory():
    config_path = "test/test_files/empty_config.yaml"
    config = ReadConfig(config_path)()
    SEO = ObservationVariables(config)
    return SEO


@pytest.fixture
def observations(seo_observatory):
    times = np.random.default_rng().uniform(low=55000, high=70000, size=5)
    band = "g"  # TODO test other bands

    seo_observatory.update(times)

    obs = {}
    for function in seo_observatory.observator_mapping():
        obs |= function()

    obs["times"] = np.asarray(
        [seo_observatory.time.mjd for _ in seo_observatory.location]
    ).ravel()
    obs["ra"] = np.asarray(
        [seo_observatory.location.ra.value for _ in seo_observatory.time.ravel()]
    ).T.ravel()
    obs["decl"] = np.asarray(
        [seo_observatory.location.dec.value for _ in seo_observatory.time.ravel()]
    ).T.ravel()

    for key in obs.keys():
        obs[key] = obs[key].ravel()
    return pd.DataFrame(obs)


def test_variable_size(seo_observatory):
    time_size = 5
    default_sites = 10
    times = np.random.default_rng().uniform(low=55000, high=70000, size=time_size)
    seo_observatory.update(times)

    for function in seo_observatory.observator_mapping():
        variable_dictionary = function()
        for key in variable_dictionary:
            print(key, variable_dictionary[key].shape)
            assert variable_dictionary[key].shape == (default_sites, time_size)


# test all the defaults are de-faulting
def test_default_location(seo_observatory):
    latitude = 38.29
    longitude = -122.50
    elevation = 2215.00

    location = seo_observatory.observator.location

    assert elevation == np.round(location.height.value, 2)
    assert latitude == np.round(location.lat.value, 2)
    assert longitude == np.round(location.lon.value, 2)


def test_init_time(seo_observatory):
    expected_time = astropy.time.Time(60000, format="mjd")
    assert expected_time == seo_observatory.time


# test all the variables (I have set ranges for)
def test_sun_angles(observations):
    assert observations.sun_ha.values == pytest.approx(
        observations.lst.values - observations.sun_ra.values, rel=1, abs=1e-6
    )


def test_sun_seperation(seo_observatory, observations):
    times = astropy.time.Time(observations.times, format="mjd")
    sun_ap_coords = astropy.coordinates.get_sun(times)

    site = astropy.coordinates.EarthLocation.from_geodetic(
        lon=seo_observatory.observator.location.lon,
        lat=seo_observatory.observator.location.lat,
    )

    sun_ap_hadec_coords = sun_ap_coords.transform_to(
        astropy.coordinates.HADec(obstime=times, location=site)
    )

    sun_df_coords = astropy.coordinates.SkyCoord(
        observations.sun_ra, observations.sun_decl, frame="icrs", unit="deg"
    )
    sun_df_hadec_coords = astropy.coordinates.HADec(
        obstime=times,
        ha=observations["sun_ha"] * astropy.units.deg,
        dec=observations["sun_decl"] * astropy.units.deg,
        location=site,
    )

    assert sun_ap_coords.separation(sun_df_coords).deg.max() < 1
    assert sun_ap_hadec_coords.separation(sun_df_hadec_coords).deg.max() < 1


def test_horizon_coord(observations, seo_observatory):
    ra, decl = observations.ra, observations.decl
    time = astropy.time.Time(observations.times, format="mjd")

    coords = astropy.coordinates.SkyCoord(ra, decl, unit="deg")

    site = astropy.coordinates.EarthLocation.from_geodetic(
        lon=seo_observatory.observator.location.lon,
        lat=seo_observatory.observator.location.lat,
    )

    altaz = astropy.coordinates.AltAz(obstime=time, location=site)
    altaz_coords = coords.transform_to(altaz)

    assert observations["alt"].values == pytest.approx(
        altaz_coords.alt.deg, rel=1, abs=0.5
    )
    assert observations["az"].values == pytest.approx(
        altaz_coords.az.deg, rel=1, abs=0.5
    )


def test_hour_angle(seo_observatory, observations):
    ra, decl = observations.ra, observations.decl

    time = astropy.time.Time(observations.times, format="mjd")
    coords = astropy.coordinates.SkyCoord(ra, decl, unit="deg")

    site = astropy.coordinates.EarthLocation.from_geodetic(
        lon=seo_observatory.observator.location.lon,
        lat=seo_observatory.observator.location.lat,
    )

    ap_hadec_coords = coords.transform_to(
        astropy.coordinates.HADec(obstime=time, location=site)
    )

    observations_hadec_coords = astropy.coordinates.HADec(
        obstime=time,
        ha=observations["ha"] * astropy.units.deg,
        dec=decl * astropy.units.deg,
        location=site,
    )
    hadec_separation = observations_hadec_coords.separation(ap_hadec_coords).deg

    assert max(hadec_separation) < 0.75


def test_init_skybright():
    config_path = "test/test_files/empty_config.yaml"
    config = ReadConfig(config_path)()
    config["use_skybright"] = True
    SEO = ObservationVariables(config)
    results = SEO.calculate_sky_magnitude()

    assert "sky_magnitude" in results
    assert "tau" in results
    assert "teff" in results


def test_nudge_large_change():
    config = ReadConfig(config_path=None)()
    config["max_position_fuzz"] = {"decl": 0.5, "ra": 0.5}
    config["location"] = {"ra": [0], "decl": [0]}
    SEO = ObservationVariables(config)

    new_position = {"ra": [50], "decl": [50]}
    SEO.update(time="", location=new_position)

    assert pytest.approx(new_position["ra"], SEO.location["ra"], 2)
    assert pytest.approx(new_position["decl"], SEO.location["decl"], 2)

    assert pytest.approx(abs(new_position["ra"] - SEO.location["ra"]), 0.5, 0.01)
    assert pytest.approx(abs(new_position["decl"] - SEO.location["decl"]), 0.5, 0.01)


def test_nudge_small_change():

    config = ReadConfig(config_path=None)()
    config["max_position_fuzz"] = {"decl": 0.5, "ra": 0.5}
    config["location"] = {"ra": [0], "decl": [0]}
    SEO = ObservationVariables(config)

    new_position = {"ra": [0], "decl": [1]}
    SEO.update(time="", location=new_position)

    assert pytest.approx(new_position["ra"], SEO.location["ra"], 0.00001)
    assert pytest.approx(new_position["decl"], SEO.location["decl"], 0.0001)
