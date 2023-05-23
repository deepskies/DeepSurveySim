"""
Calculate each variable needed for he simulation of the sky.
Each function takes the time as an input, and calculates each step at that time.

Requires set up of the observation location from a yaml file or a dictionary.
"""
from typing import Union
import astroplan
import astropy
import numpy as np

import numexpr


class ObservationVariables:
    def __init__(self, observator_configuration: dict):
        """
        Calculate the parameters for a specific observation

        Args:
            observator_configuration (dict): Describes the way the observatory is set up.
            parameters:
                obs_latitude_degrees (float) (required): Degree location
                obs_logitude_degrees (float) (required): Degree location
                obs_elevation_meters (float) (required): Height above sea level

                bands (dict): String indication of the band, and its associated wavelength.
                    Default {
                            "u": 380.0,
                            "g": 475.0,
                            "r": 635.0,
                            "i": 775.0,
                            "z": 925.0,
                            "Y": 1000.0,
                                }
                seeing (float) : [0, 3]; indicates the clarity of the sky.
                    3 is totally obscured, 0 is totally clear (as if observing through a vacuum). Default 0.9
                optics_fwhm (float): Default 0.45

                location (dict) :
                    dictionary containing either 'n_sites'
                        (Number of sequentically generated sites spanning the whole sky)
                    or paired "ra" and "decl", containing lists of locations.
                    Default: 10 sites.

                use_skybright (bool):
                    use the skybright program to add additional variables to the program
                    Requires an outside download of PalPy (Not included with this distirbution)
                    Default: False

            Run the program by
                specifying the init params,
                updating the location, band, and time with ObservationVariables.update(),
                calculating the requested variables

                Possible calculations are seen with ObservationVariables.variables

                All variables are returned with the dimensions (n timesteps, n observation sites)
                    in a dictionary labeled with their variable names
        """

        self.degree = astropy.units.deg
        self.radians = astropy.units.rad
        self.meters = astropy.units.m

        self.to_radians = self.degree.to(self.radians)

        self.observator = self._init_observator(
            observator_configuration["latitude"],
            observator_configuration["longitude"],
            observator_configuration["elevation"],
        )

        self.band_wavelengths = observator_configuration["wavelengths"]

        self.seeing = observator_configuration["seeing"]

        self.optics_fwhm = observator_configuration["fwhm"]

        self.default_locations = self._default_locations(
            **observator_configuration["location"]
        )

        self.time = self._time(60000)
        self.location = self.default_locations
        self.band = "g"

        self.variables = self.observator_mapping()
        if observator_configuration["use_skybright"]:
            self.init_skybright(observator_configuration["skybright"])

    def update(
        self,
        time: Union[float, list[float]],
        location: Union[dict, None] = None,
        band: Union[str, None] = None,
    ):
        if location is not None:
            assert "ra" in location.keys()
            assert "decl" in location.keys()

            assert len(location["ra"]) == len(location["decl"]) != 0

        self.time = self._time(time)
        self.band = band if band is not None else self.band
        self.location = (
            self.default_locations
            if location is None
            else self._sky_coordinates(location["ra"], location["decl"])
        )

    def _init_observator(
        self, obs_latitude_degrees, obs_logitude_degrees, obs_elevation_meters
    ):
        return astroplan.Observer(
            longitude=obs_logitude_degrees * self.degree,
            latitude=obs_latitude_degrees * self.degree,
            elevation=obs_elevation_meters * self.meters,
        )

    def _default_locations(self, ra=None, decl=None, n_sites=None):
        n_sites = n_sites if n_sites is not None else 10
        if (ra is None) & (decl is None):
            decl = list(np.arange(-90, 90, step=int((90 * 2) / n_sites)))
            ra = list(np.arange(0, 360, step=int(360 / n_sites)))

        assert len(ra) == len(decl), "Please pass pairs of ra/decl"

        return self._sky_coordinates(ra, decl)

    def _airmass(self, location):

        alt = np.array([np.array(self._alt_az(l).alt.degree) for l in location])
        cos_zd = np.cos(np.radians(90) - alt * self.to_radians)
        a = numexpr.evaluate("462.46 + 2.8121/(cos_zd**2 + 0.22*cos_zd + 0.01)")

        airmass = numexpr.evaluate("sqrt((a*cos_zd)**2 + 2*a + 1) - a * cos_zd")
        airmass[(alt * self.to_radians) < 0] = np.nan

        return airmass

    def _time(self, time):
        return astropy.time.Time(time * astropy.units.day, format="mjd")

    def _alt_az(self, coordinates):
        alt_az = self.observator.altaz(time=self.time, target=coordinates)

        return alt_az

    def _local_sidereal_time(self):
        local_sidereal_time = self.observator.local_sidereal_time(
            self.time, "mean"
        ).to_value(self.degree)
        return local_sidereal_time

    def _ha(self, location):
        lst = self._local_sidereal_time()
        ha = lst - location.ra.to_value(self.degree)
        return ha

    def _sky_coordinates(
        self,
        ra_degree: Union[float, list[float]],
        decl_degree: Union[float, list[float]],
    ):
        if type(ra_degree) == float:
            ra_degree = [ra_degree]
        if type(decl_degree) == float:
            decl_degree == [decl_degree]

        return astropy.coordinates.SkyCoord(
            ra=[ra * self.to_radians for ra in ra_degree] * self.radians,
            dec=[decl * self.to_radians for decl in decl_degree] * self.radians,
        )

    def calculate_sun_location(self):

        sun_coordinates = astropy.coordinates.get_sun(self.time)

        sun_ra = sun_coordinates.ra.to_value(self.degree)
        sun_decl = sun_coordinates.dec.to_value(self.degree)

        return {
            "sun_ra": np.asarray([sun_ra for _ in range(len(self.location))]),
            "sun_decl": np.asarray([sun_decl for _ in range(len(self.location))]),
        }

    def calculate_sun_ha(self):

        sun_coordinates = astropy.coordinates.get_sun(self.time)

        sun_ha = self._ha(sun_coordinates)
        return {"sun_ha": np.asarray([sun_ha for _ in range(len(self.location))])}

    def calculate_sun_airmass(self):

        sun_coordinates = astropy.coordinates.get_sun(self.time)
        sun_airmass = self._airmass(sun_coordinates)
        return {
            "sun_airmass": np.asarray([sun_airmass for _ in range(len(self.location))])
        }

    def calculate_moon_location(self):
        moon_location = astropy.coordinates.get_moon(self.time)
        moon_ra = moon_location.ra.to_value(self.degree)
        moon_decl = moon_location.dec.to_value(self.degree)

        return {
            "moon_ra": np.asarray([moon_ra for _ in range(len(self.location))]),
            "moon_decl": np.asarray([moon_decl for _ in range(len(self.location))]),
        }

    def calculate_moon_brightness(self):

        moon_location = astropy.coordinates.get_moon(self.time)

        moon_phase = astroplan.moon.moon_phase_angle(self.time)
        moon_illumination = self.observator.moon_illumination(self.time)

        moon_elongation = (
            astropy.coordinates.get_sun(self.time)
            .separation(moon_location)
            .to_value(self.degree)
        )
        alpha = 180.0 - moon_elongation

        # Allen's _Astrophysical Quantities_, 3rd ed., p. 144
        moon_Vmagintude = -12.73 + 0.026 * np.abs(alpha) + 4e-9 * (alpha**4)

        moon_seperation = np.asarray(
            [
                moon_location.separation(location).to_value(self.degree)
                for location in self.location
            ]
        )
        return {
            "moon_phase": np.asarray([moon_phase for _ in range(len(self.location))]),
            "moon_illumination": np.asarray(
                [moon_illumination for _ in range(len(self.location))]
            ),
            "moon_Vmagintude": np.asarray(
                [moon_Vmagintude for _ in range(len(self.location))]
            ),
            "moon_seperation": moon_seperation,
        }

    def calculate_moon_ha(self):
        moon_location = astropy.coordinates.get_moon(self.time)
        moon_ha = self._ha(moon_location)
        return {"moon_ha": np.asarray([moon_ha for _ in range(len(self.location))])}

    def calculate_moon_airmass(self):
        moon_location = astropy.coordinates.get_moon(self.time)
        moon_airmass = self._airmass(moon_location)

        return {
            "moon_airmass": np.asarray(
                [moon_airmass for _ in range(len(self.location))]
            )
        }

    def calculate_observation_angles(self):
        hzcrds = [self._alt_az(l) for l in self.location]
        alt = np.asarray([hzcrd.alt for hzcrd in hzcrds]) * self.degree
        az = np.asarray([hzcrd.az for hzcrd in hzcrds]) * self.degree

        return {"az": az, "alt": alt}

    def calculate_observation_ha(self):
        return {"ha": np.asarray([self._ha(l) for l in self.location])}

    def calculate_observation_airmass(self):
        return {"airmass": self._airmass(self.location)}

    def calculate_seeing(self):
        airmass = self._airmass(self.location)
        pt_seeing = self.seeing * airmass**0.6
        wavelength = self.band_wavelengths[self.band]
        band_seeing = pt_seeing * (500.0 / wavelength) ** 0.2
        fwhm = np.sqrt(band_seeing**2 + self.optics_fwhm**2)

        return {"pt_seeing": pt_seeing, "band_seeing": band_seeing, "fwhm": fwhm}

    def observator_mapping(self):
        return [
            self.calculate_sun_location,
            self.calculate_sun_airmass,
            self.calculate_sun_ha,
            self.calculate_observation_ha,
            self.calculate_observation_angles,
            self.calculate_observation_airmass,
            self.calculate_moon_airmass,
            self.calculate_moon_location,
            self.calculate_moon_brightness,
            self.calculate_moon_ha,
            self.calculate_seeing,
        ]

    def init_skybright(self, skybright_config):
        raise NotImplemented("Skybright integration not included in this release")
