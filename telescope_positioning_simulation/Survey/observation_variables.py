from typing import Union
import astroplan
import astropy
import numpy as np
import os

import numexpr


class ObservationVariables:
    """
    Calculate the parameters for a specific observation

    Run the program by
        specifying the init params,
        updating the location, band, and time with ObservationVariables.update(),
        calculating the requested variables
        Possible calculations are seen with ObservationVariables.variables

    All variables are returned with the dimensions (n observation times, n sites) in a dictionary labeled with their variable names

    Args:
        observator_configuration (dict): Describes the way the observatory is set up. This contains:

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

    Examples:
        >>> observer = ObservationVariables(configuration)
            observer.update({"time": [60125], location:{"ra":[20, 35]}, "decl":[0, 0]})
            ## Calculate moon location for 6/30/2023 at 20*, 35* along the equator.
            moon_location = observer.calculate_moon_location()
            `{"moon_ra":(20,20), "moon_decl":(20,20)}`

        >>> observer = ObservationVariables(configuration)
            observer.update(time = [60125], location = {"ra":[20, 35]}, "decl":[0, 0]})
            # Calculate all variables
            all_stats = {}
            for function in observator.observation_variables():
                all_stats |= function()

    """

    def __init__(self, observator_configuration: dict):

        if observator_configuration["use_skybright"]:
            self._init_skybright(observator_configuration["skybright"])

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
        self.clouds = observator_configuration["cloud_extinction"]

        self.optics_fwhm = observator_configuration["fwhm"]

        self.default_locations = self._default_locations(
            **observator_configuration["location"]
        )
        self.delay = 0

        self.time = self._time(60000)
        self.location = self.default_locations
        self.band = "g"

        self.slew_rate = observator_configuration["slew_expr"]
        self.band_change_rate = observator_configuration["filter_change_rate"]
        self.readout_seconds = observator_configuration["readout_seconds"]

    def _init_skybright(self, skybright_config):
        try:
            from skybright import skybright
            from configparser import ConfigParser

        except ModuleNotFoundError:
            print(
                "ERROR: skybright module not found, please install it from https://github.com/ehneilsen/skybright.git"
            )

        default_config_path = (
            f"{os.path.dirname(__file__).rstrip('/')}/../settings/skybright_config.conf"
        )

        config_path = (
            default_config_path
            if skybright_config["config"] == "default"
            else skybright_config["config"]
        )
        skybright_config_file = ConfigParser()
        skybright_config_file.read(config_path)

        self.skybright = skybright.MoonSkyModel(skybright_config_file)

    def update(
        self,
        time: Union[float, list[float]],
        location: Union[dict, None] = None,
        band: Union[str, None] = None,
    ):
        """
        Move the simulation forward to the next site.
        Updates the time (ObservationVariables.time), the delay between pervious time and new time, observation site (ObservationVariables.location), and optial filter band (ObservationVariables.band)

        Args:
            time (Union[float, list[float]]): Time to move forward to, in Mean Julian Date
            location (Union[dict, None], optional): Location (paired ra/delc) in degrees to move the telescope pointing. Will not change the pointing if location not specificed. Defaults to None.
            band (Union[str, None], optional): Optical filter to use for observation. Will not be changed if not specified. Select from bands specified by ObservationVariables.band_wavelengths. Defaults to None.
        """

        if location is not None:
            assert "ra" in location.keys()
            assert "decl" in location.keys()

            location = self._sky_coordinates(location["ra"], location["decl"])
            self.delay = self._delay_time(location, band)

        else:
            self.delay = self._delay_time(self.location, band)

        self.time = self._time(time)
        self.band = band if band is not None else self.band
        self.location = self.location if location is None else location

    def _angular_distance(self, location):
        ra1, ra2 = np.array(
            (self.location.ra.value * np.pi / 180) + 10**-9
        ), np.array((location.ra.value * np.pi / 180) + 10**-9)
        decl1, decl2 = np.array(
            (self.location.dec.value * np.pi / 180) + 10**-9
        ), np.array((location.dec.value * np.pi / 180) + 10**-9)

        seperation = np.arccos(
            np.sin(decl1) * np.sin(decl2)
            + np.cos(decl1) * np.cos(decl2) * np.cos(abs(ra1 - ra2))
        )
        return seperation

    def _delay_time(self, location, band):
        delay = self.slew_rate * self._angular_distance(location)

        if band != self.band:
            delay += self.band_change_rate
        delay += self.readout_seconds

        delay_days = delay * 0.00001157407
        return delay_days

    def _init_observator(
        self, obs_latitude_degrees, obs_logitude_degrees, obs_elevation_meters
    ):
        return astroplan.Observer(
            longitude=obs_logitude_degrees * self.degree,
            latitude=obs_latitude_degrees * self.degree,
            elevation=obs_elevation_meters * self.meters,
        )

    def _default_locations(self, ra=None, decl=None, n_sites=None):
        if (ra is None) & (decl is None):
            assert n_sites is not None
            decl = np.arange(-90, 90, step=int((90 * 2) / n_sites))
            ra = np.arange(0, 360, step=int(360 / n_sites))

        if type(ra) == list:
            assert len(ra) == len(decl), "Please pass pairs of ra/decl"

        return self._sky_coordinates(ra, decl)

    def _airmass(self, location):
        alt = np.array(self._alt_az(location).alt.degree)
        cos_zd = np.cos(np.radians(90) - alt * self.degree.to(self.radians))
        a = numexpr.evaluate("462.46 + 2.8121/(cos_zd**2 + 0.22*cos_zd + 0.01)")

        airmass = numexpr.evaluate("sqrt((a*cos_zd)**2 + 2*a + 1) - a * cos_zd")
        airmass[(alt * self.to_radians) < 0] = np.nan
        # airmass = airmass * self.radians.to(self.degree)
        return airmass

    def _time(self, time):
        return astropy.time.Time(np.asarray(time), format="mjd")

    def _alt_az(self, coordinates):
        site = astropy.coordinates.EarthLocation.from_geodetic(
            lon=self.observator.location.lon, lat=self.observator.location.lat
        )

        altaz_conversion = astropy.coordinates.AltAz(obstime=self.time, location=site)
        alt_az = coordinates.transform_to(altaz_conversion)

        return alt_az

    def _local_sidereal_time(self):
        local_sidereal_time = self.observator.local_sidereal_time(
            self.time, "mean"
        ).to_value(self.degree)
        return local_sidereal_time

    def _ha(self, location):
        lst = self._local_sidereal_time()
        ha = lst - location.ra.value
        return ha

    def _sky_coordinates(
        self,
        ra_degree,
        decl_degree,
    ):
        return astropy.coordinates.SkyCoord(
            ra=ra_degree * self.degree, dec=decl_degree * self.degree, unit="deg"
        )

    def calculate_lst(self):
        """
        Calculate the current pointing local sidereal time
        LST - https://en.wikipedia.org/wiki/Sidereal_time

        Returns:
            dict[array]: Local Sideral Time, shape (n observation times, n sites)
        """
        lst = self._local_sidereal_time()
        return {"lst": np.asarray([lst for _ in range(len(self.location))])}

    def calculate_sun_location(self):
        """
        Calculate the position of the sun in Right Ascension/Declination (degrees)

        Returns:
            dict[array]: Dictionary of RA/Decl of the Sun, shape (n observation times, n sites)
        """
        sun_coordinates = astropy.coordinates.get_sun(self.time)

        sun_ra = sun_coordinates.ra.to_value(self.degree)
        sun_decl = sun_coordinates.dec.to_value(self.degree)

        return {
            "sun_ra": np.asarray([sun_ra for _ in range(len(self.location))]),
            "sun_decl": np.asarray([sun_decl for _ in range(len(self.location))]),
        }

    def calculate_sun_ha(self):
        """
        Calculate the Sun's Hour Angle
        https://en.wikipedia.org/wiki/Hour_angle

        Returns:
            dict[array]: Sun HA, shape (n observation times, n sites)
        """
        sun_coordinates = astropy.coordinates.get_sun(self.time)
        sun_ha = self._ha(sun_coordinates)
        return {"sun_ha": np.asarray([sun_ha for _ in range(len(self.location))])}

    def calculate_sun_airmass(self):
        """
        Calculate the Airmass of the sun relative to the current location.
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)

        Returns:
            dict[array]: Sun Airmass, shape (n observation times, n sites)
        """
        sun_coordinates = astropy.coordinates.get_sun(self.time)
        sun_airmass = self._airmass(sun_coordinates)

        return {
            "sun_airmass": np.asarray([sun_airmass for _ in range(len(self.location))])
        }

    def calculate_moon_location(self):
        moon_location = astropy.coordinates.get_moon(self.time)
        moon_ra = moon_location.ra.to_value(self.degree)
        moon_decl = moon_location.dec.to_value(self.degree)
        """ Calculate the moon position at current time

        Returns:
            dict[array]: Moon location in degrees (Right Ascension/Declination), shape (n observation times, n sites)
        """
        return {
            "moon_ra": np.asarray([moon_ra for _ in range(len(self.location))]),
            "moon_decl": np.asarray([moon_decl for _ in range(len(self.location))]),
        }

    def calculate_moon_brightness(self):
        """
        Calculate the brightness of the moon at a the current time, as observated from the current observatory

        Returned dictionary contains
            - Moon Elongation (seperation from the sun, in degrees)
            - Moon Phase (quarters of the moon illumated)
            - Moon Illumation (% of moon face illumated)
            - Moon V Magnitude: Visible magnitude approximation as defined by Allen's Astrophysical Quantities
            - Moon Seperation: Angular distance (degrees) between point and the moon

        Returns:
            dict[array]: Array of above moon brightness variables, shape (n observation times, n sites)
        """
        moon_location = astropy.coordinates.get_moon(self.time)

        moon_phase = astroplan.moon.moon_phase_angle(self.time).to_value(self.degree)
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
            "moon_elongation": np.asarray(
                [moon_elongation for _ in range(len(self.location))]
            ),
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
        """ "
        Calculate the hour angle of the moon at the given point in time
        https://en.wikipedia.org/wiki/Hour_angle

        Returns:
            dict[array]: Moon HA shape (n observation times, n sites)
        """
        moon_location = astropy.coordinates.get_moon(self.time)
        moon_ha = self._ha(moon_location)
        return {"moon_ha": np.asarray([moon_ha for _ in range(len(self.location))])}

    def calculate_moon_airmass(self):
        """ "
        Calculate the airmass of the moon at the given point in time
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)

        Returns:
            dict[array]: Moon Airmass, shape (n observation times, n sites)
        """
        moon_location = astropy.coordinates.get_moon(self.time)
        moon_airmass = self._airmass(moon_location)
        return {
            "moon_airmass": np.array([moon_airmass for _ in range(len(self.location))])
        }

    def calculate_observation_angles(self):
        """
        Calculate the altitude and azumultial angle of the current pointing, in degrees

        Returns:
            dict[array]: Azimuthal angle (az), Altitude (alt) in degrees, shape (n observation times, n sites)
        """
        hzcrds = [self._alt_az(location) for location in self.location]
        alt = np.asarray([hzcrd.alt.degree for hzcrd in hzcrds])
        az = np.asarray([hzcrd.az.degree for hzcrd in hzcrds])

        return {
            "az": az,
            "alt": alt,
        }

    def calculate_observation_ha(self):
        """
        Calcate the current point's Hour Angle
        https://en.wikipedia.org/wiki/Hour_angle

        Returns:
            dict[array]: Pointing HA, shape (n observation times, n sites)
        """
        return {"ha": np.asarray([self._ha(location) for location in self.location])}

    def calculate_observation_airmass(self):
        """
        Calculate the current pointing's airmass
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)
        Returns:
            dict[array]: Pointing Airmass,  shape (n observation times, n sites)
        """
        return {
            "airmass": np.asarray(
                [self._airmass(location) for location in self.location]
            )
        }

    def calculate_seeing(self):
        """
        Calculate the optical visibality of the observation with the current filter/band
        fwhw defintion - https://en.wikipedia.org/wiki/Full_width_at_half_maximum

        Returns:
            dict[array]: Dictionary of Transverse seeing (pt_seeing), Seeing through the current filter (band_seeing), Full width at half maximum (fwhw) for the light signal, shape (n observation times, n sites)

        """
        airmass = self.calculate_observation_airmass()["airmass"]
        pt_seeing = self.seeing * airmass**0.6
        wavelength = self.band_wavelengths[self.band]
        band_seeing = pt_seeing * (500.0 / wavelength) ** 0.2
        fwhm = np.sqrt(band_seeing**2 + self.optics_fwhm**2)

        return {"pt_seeing": pt_seeing, "band_seeing": band_seeing, "fwhm": fwhm}

    def calculate_sky_magnitude(self):
        """
        If skybright is both installed and set up, calculate the sky brightness/magnitude of brightness
        Please view  https://github.com/ehneilsen/skybright/blob/b0e2d7e6e25131393ee76ce334ce1df1521e3659/skybright/skybright.py#L173 for details

        Returns:
            dict[array]: Dictionary of "sky magnitude", "tau", "teff", shape (n observation times, n sites)
        """
        if hasattr(self, "skybright"):
            m0 = self.skybright.m_zen[self.band]
            nu = 10 ** (-1 * self.clouds / 2.5)
            fwhm500 = self.calculate_seeing()["fwhm"]

            sky_mag = np.asarray(
                [
                    self.skybright(
                        self.time.mjd.mean(),
                        location.ra.degree,
                        location.dec.degree,
                        self.band,
                        moon_crds=astropy.coordinates.get_moon(self.time),
                        moon_elongation=moon_elongation,
                        sun_crds=astropy.coordinates.get_sun(self.time),
                    )
                    for location, moon_elongation in zip(
                        self.location,
                        self.calculate_moon_brightness()["moon_elongation"],
                    )
                ]
            )

            tau = ((nu * (0.9 / fwhm500)) ** 2) * (10 ** ((sky_mag - m0) / 2.5))

            teff = tau * self.readout_seconds * 0.00001157407 * 86400

            return {
                "sky_magnitude": np.array(sky_mag),
                "tau": np.array(tau),
                "teff": np.array(teff),
            }
        else:
            return {}

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
            self.calculate_lst,
            self.calculate_sky_magnitude,
        ]

    def name_to_function(self):
        """
        Map between the name of the variable and the function used to produce it.
        Call this to find what variables are avaliable for your instance of ObservationVaraibles.

        Returns:
            dict: map between variable names and their functions.
        """
        names = {}
        for function in self.observator_mapping():
            function_result = function()
            function_map = {name: function for name in function_result.keys()}
            names = {**function_map, **names}
        return names
