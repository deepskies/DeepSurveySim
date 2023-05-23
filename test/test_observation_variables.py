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
    for function in seo_observatory.variables:
        obs |= function()

    obs["times"] = np.asarray(
        [seo_observatory.time.mjd for _ in seo_observatory.location]
    ).ravel()
    obs["ra"] = np.asarray(
        [seo_observatory.time.mjd for _ in seo_observatory.location]
    ).ravel()
    obs["decl"] = np.asarray(
        [seo_observatory.time.mjd for _ in seo_observatory.location]
    ).ravel()

    for key in obs.keys():
        obs[key] = obs[key].ravel()
    return pd.DataFrame(obs)


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
    assert observations.sun_zd.values == pytest.approx(
        90 - observations.sun_alt.values, rel=1, abs=1e-6
    )
    assert observations.sun_ha.values == pytest.approx(
        observations.lst.values - observations.sun_ra.values, rel=1, abs=1e-6
    )


def test_sun_seperation(seo_observatory, observations):
    times = seo_observatory.time
    sun_ap_coords = astropy.get_sun(times)
    sun_ap_hadec_coords = sun_ap_coords.transform_to(
        astropy.HADec(obstime=times, location=seo_observatory.observator)
    )
    sun_ap_altaz_coords = sun_ap_coords.transform_to(
        astropy.AltAz(obstime=times, location=seo_observatory.observator)
    )

    sun_df_coords = astropy.SkyCoord(
        observations.sun_ra, observations.sun_decl, frame="icrs", unit="deg"
    )
    sun_df_hadec_coords = astropy.HADec(
        obstime=times,
        ha=observations["sun_ha"] * astropy.units.deg,
        dec=observations["sun_decl"] * astropy.units.deg,
        location=seo_observatory.observator,
    )
    sun_df_altaz_coords = astropy.AltAz(
        alt=observations["sun_alt"] * astropy.units.deg,
        az=observations["sun_az"] * astropy.units.deg,
        obstime=times,
        location=seo_observatory.observator,
    )

    assert sun_ap_coords.separation(sun_df_coords).deg.max() < 1
    assert sun_ap_hadec_coords.separation(sun_df_hadec_coords).deg.max() < 1
    assert sun_ap_altaz_coords.separation(sun_df_altaz_coords).deg.max() < 1


def test_airmass_range(seo_observatory, observations):

    assert np.all(observations[np.isnan(observations.airmass)].zd > 89)

    times = seo_observatory.time
    ra, decl = seo_observatory.locations["ra"], seo_observatory.locations["decl"]
    coords = astropy.SkyCoord(ra, decl, frame="icrs", unit="deg")
    altaz = astropy.AltAz(obstime=times, location=seo_observatory.observator)
    altaz_coords = coords.transform_to(altaz)

    assert altaz_coords.secz.value[secz_valid] == pytest.approx(
        observations[secz_valid].airmass, rel=99, abs=1e-2
    )

    secz_valid = observations["zd"] < 60
    secz_invalid = np.logical_and(
        ~np.isnan(observations["airmass"]), observations["zd"] > 60
    )
    assert np.all(
        observations[secz_invalid]["airmass"] > observations[secz_valid].airmass.max()
    )
    assert np.all(observations[secz_invalid]["airmass"] < 40)


def test_lst(seo_obseratory, observations):
    ref_lst = seo_obseratory.time.sidereal_time("mean", seo_observatory.observator).deg
    assert ref_lst == pytest.approx(observations["lst"].values, rel=1, abs=1e-6)


def test_horizon_coord(observations, seo_observatory):
    ra, decl = seo_observatory.locations["ra"], seo_observatory.locations["decl"]

    coords = astropy.SkyCoord(ra, decl, frame="icrs", unit="deg")

    altaz = astropy.AltAz(
        obstime=seo_observatory.time, location=seo_observatory.observator
    )
    altaz_coords = coords.transform_to(altaz)
    ref_zd = 90 - observations["alt"].values

    assert observations["alt"].values == pytest.approx(
        altaz_coords.alt.deg, rel=1, abs=1e-6
    )
    assert observations["az"].values == pytest.approx(
        altaz_coords.az.deg, rel=1, abs=1e-6
    )
    assert ref_zd == pytest.approx(observations["zd"].values, rel=1, abs=1e-6)


def test_hour_angle(seo_observatory, observations):
    ra, decl = seo_observatory.locations["ra"], seo_observatory.locations["decl"]
    coords = astropy.SkyCoord(ra, decl, frame="icrs", unit="deg")

    ap_hadec_coords = coords.transform_to(
        astropy.HADec(obstime=seo_observatory.time, location=seo_observatory.observator)
    )

    observations_hadec_coords = astropy.HADec(
        obstime=seo_observatory.time,
        ha=observations["ha"] * astropy.units.deg,
        dec=decl * astropy.units.deg,
        location=seo_observationary.observator,
    )
    hadec_separation = observations_hadec_coords.separation(ap_hadec_coords).deg
    assert max(hadec_separation) < 0.5
