Configurations
===============

Observator/Telescope Configuration
-----------------------------------

The observator is the core of this program, it calculates all the variables needed to make predictions on what action is optimial.
Many of these variables are dense, so they are given default values, but explained briefly here, and broken into major groups


.. attribute:: Observatory

    Variables relating to the location of the observatory/telescope itself

    :param name: (optional) Name of the obseratory being simulated
    :type name: str
    :param latitude: latitude position of the obseratory, in decimal degrees
    :type latitude: float
    :param longitude: longitude position of the obseratory, in decimal degrees
    :type longitude: float
    :param elevation: elevation above sea level, in meters
    :type elevation: float

.. code-block:: yaml

    name: "Stone Edge Observatory"
    latitude: 38.28869
    longitude:  -122.504
    elevation: 2215.0

.. attribute:: Slew

    Measuring the amount of time needed to transition between each pointing

    :param default_transition_time: (optional) Automatic time between observations (in seconds)
    :type name: float
    :param slew_expr: Time required to transition between sites (in seconds per degree)
    :type name: float

.. code-block:: yaml

    default_transition_time: 0.0
    slew_expr: 1.7

.. attribute:: Bands

    Optical settings for the filters used during observations

    :param wavelengths: Pairing of filter names and wavelengths they observe
    :type name: dictionary
    :param filter_change_rate: Time required to change filters (in seconds)
    :type name: float
    :param fwhm: Full width at half maximum; https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    :type name: float

.. code-block:: yaml

    wavelengths:
        - 'u': 380.0
        - 'g': 475.0
        - 'r': 635.0
        - 'i': 775.0
        - 'z': 925.0
        - 'Y': 1000.0
    filter_change_rate: 0.0
    fwhm: 0.45

.. attribute:: Weather

    Constants for deteriming sight due to weather conditions

    :param seeing: % visablity
    :type name: float
    :param cloud_extinction: Rate clouds appear
    :type name: float
    :param weather_sim: Include a rudimentary weather simulation based on historical data
    :type name: boolean 
    :param weather_config: Setting for the `Weather` engine class
    :type name: dictionary

.. code-block:: yaml

    seeing: 0.9
    cloud_extinction: 0.0
    weather_sim: False


.. attribute:: Camera

    Constants for the time it takes for the camera behind the telescope to operate

    :param shutter_seconds: Time shutter requires to move from open to closed, thus preventing a clear observation (in seconds)
    :type name: float
    :param readout_seconds: Time required for the camera to process the data, so another observation cannot be taken directly after another (in seconds)
    :type name: float

.. code-block:: yaml

    shutter_seconds: 0.0
    readout_seconds: 27.0


.. attribute:: Location

    Observation taken as the default

    :param location: Define the default locations. Either a pair of Right Ascension and Declination arrays (in degrees), or 'n_sites' to define a random selection.
    :type name: dictionary

.. code-block:: yaml

    location : {'n_sites': 10}

.. attribute:: Skybright

    Parameters to use the package `SkyBright` to
    `SkyBright` can be found here- https://github.com/ehneilsen/skybright

    :param use_skybright: Define if skybright is used
    :type name: boolean
    :param skybright: Path to the configuration file used to initalize
    :type name: dictionary

.. code-block:: yaml

    use_skybright: False
    skybright: {"config":'default'}



Survey Configuration
---------------------

This configuration file deterimes how the survey is executed, what sites are considered "valid", what the stopping condition is, and what deterimes the quality of each observation.
It also sets inital conditions.

.. attribute:: Timestep

    How many seconds are between each step of the survey

    :param timestep_size: Timestep (in seconds)
    :type name: float, int

.. code-block:: yaml

    timestep_size: 300


.. attribute:: Start Time

    When to start the survey

    :param start_time: If decided, the time in Mean Julian Date. If not, the string "random".
    :type name: float, str

.. code-block:: yaml

    start_time: "random"


.. attribute:: Penality

    The reward given to a site that is considered invalid at the current timestep.

    :param invalid_penality: Penality value
    :type name: float, int

.. code-block:: yaml

    invalid_penality: -100

.. attribute:: Reward

    Reward for each observation

    :param monitor: Variable to use as reward, must be in `variables`
    :type monitor: str
    :param min: If a smaller reward is better
    :type monitor: boolean
    :param threshold: (optional) Value the reward must clear in order to be considered a `valid` reward
    :type monitor: float, int

.. code-block:: yaml

    reward
        - "monitor": "airmass"
        - "min": False


.. attribute:: Constaints

    Dictionary of constraints the observations must fall under to be considered `valid`

    :param [variable name]: Variable to used as a metric for validity, must be in `variables`
    :type monitor: str
    :param value: Threshold value to consider
    :type monitor: float, int
    :param lesser: If a value smaller than the threshold is considered valid
    :type monitor: boolean

.. code-block:: yaml

    constaints:
        - "airmass":
            - "value": 2.0
            - "lesser": True
        - "alt":
            - "value": 20.0
            - "lesser": False

.. attribute:: Stopping

    Dictionary of values used to deterime the conditions used to stop the simulation
    `Timestep` deterimes the number of iterations the simulation will run, the rest are optional.

    :param [variable name]: Variable to used as a metric for validity, must be in `variables`
    :type monitor: str
    :param value: Threshold value to consider
    :type monitor: float, int
    :param lesser: If a value smaller than the threshold is considered valid
    :type monitor: boolean

.. code-block:: yaml

    stopping:
        - "timestep": 400
        - "reward":
            - "value": 200.0
            - "lesser": False

.. attribute:: Save

    Location to save resulting survey, if so desired.

    :param save: String of the path to save the results and configuration file
    :type monitor: str

.. code-block:: yaml

    save: "./equatorial_survey/"

.. attribute:: Variables

    List of variables used in the survey.

    All possible variables are:

    ['lst', 'pt_seeing', 'band_seeing', 'fwhm', 'moon_ha', 'moon_elongation', 'moon_phase', 'moon_illumination', 'moon_Vmagintude', 'moon_seperation', 'moon_ra', 'moon_decl', 'moon_airmass', 'airmass', 'az', 'alt', 'ha', 'sun_ha', 'sun_airmass', 'sun_ra', 'sun_decl']

    :param variables: List of string names of the variables used in the survey.
    :type monitor: list

.. code-block:: yaml

    variables: ["airmass", 'alt', 'ha', 'moon_airmass', 'lst', 'sun_airmass']
