Using the simulator to generate data
===============================================

While this simulation is designed to be used to sequencially generate a site at a time,
it can also be used to produce multiple observations at once,
so that data may be generated for multiple sites
for each timestep without having to iterate over each site of interest.

For this purpose, code can be left mostly unchanged,
and the only modifications are in the form of custom configuration files.


Specifying variables of interest
----------------------------------
The only requirements to change the variables included in the simulation is to specify them by name in the `survey_configuration` file.
Modifying the field `variables` will set the contents of the simulation.

For example;

.. code-block:: yaml

    variables: ["airmass", 'alt', 'ha']

See :ref:`the survey configuration documention<Survey Configuration>` for further details.


Specifying observation sites
-----------------------------
To set which sites are included,
they can either be set in the `observator` program configuration,
or manually changing them in the `step` function of the survey program.

Using The Configuration File
"""""""""""""""""""""""""""""
If you wish to calculate variables for the same Right Ascension/Declination pairs throughout the course of the survey, these sites can be given in the `observator` confuguration file.

There are two formats - `n_sites` or `ra`/ `delc` pairs. `n_sites` evenly distributes them across the sky, and `ra` / `decl` supports specific location assignement.

The format of this follows:

.. code-block:: yaml

    location : {'n_sites': 10}

for uniformally distributed locations, and

.. code-block:: yaml

    location : {
        'ra': [0, 10, 20, 30],
        'decl': [0, 0, 0, 0]
        }

for site specifiications. Note that these must be given in pairs.
See :ref:`the survey configuration documention<Survey Configuration>` for further details.


Using the `step` function
""""""""""""""""""""""""""
If using different observation sites for each timestep, the pairs of ra/decl must be given in the `step` function.

In this case, this is done during code execution using the format:

.. code-block:: python

    for timestep in all_timesteps:
        next_location = {
            "ra": [0, 10, 20, 30],
            "decl":[0, 0, 0, 0]
        }
        next_action = {
            "time": [timestep],
            "location": next_location
        }

        new_observation, reward, stop, log = Survey.step(next_action)


Specifying observation times
-----------------------------

Using `stop`
"""""""""""""""""
To set the survey to run until a certain condition is met, within the `survey` configuration step -

.. code-block:: yaml

    stopping:
        - "timestep": 400,
        - "airmass":
            - "value": 2
            - "lesser": False
    start_time: 59946

For example - This configuration block sets a survey to run for 400 timesteps, or until airmass is greater than 2.
The start time is also specified to be January 1, 2023.

Using `step`
"""""""""""""""""
Using `step` provides more fine-control. The format follows:

.. code-block:: python

    for timestep in timesteps_in_mjd:
        next_action = {
            "time": [timestep],
        }
        new_observation, reward, stop, log = Survey.step(next_action)


Specifying observation filters
------------------------------
Different filters are only avalible through use of the `step` function.
The specifics of the wavelength each filter is given in the `observator` configuration file,
a as a pair of filter name and associated wavelength.

For example:
.. code-block:: yaml

    wavelengths: {'u': 380.0, 'g': 475.0}

This allows filters 'u' and 'g' to be used in the step function to change the filter.
Then the `step` function can be used as follows.

.. code-block:: python

    for timestep in timesteps_in_mjd:
        next_action = {
            "time": [timestep],
            "filter":"g"
        }
        new_observation, reward, stop, log = Survey.step(next_action)


Running the simulation
-----------------------

To run the simulation, simply run `results = Survey(survey_config, observer_config)()`.

Manual control is still done with the `step` function.

Accessing and saving the output
--------------------------------
Use the IO module to order to save the results of the survey and the configuration files.
View See :ref:`the io module documention<IO>` for further details.
