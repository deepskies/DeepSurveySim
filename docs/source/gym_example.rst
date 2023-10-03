Using the simulation with ``gym.Env``
===================================
`gym.Env <https://gymnasium.farama.org/api/env/>`_ is a framework to produce a live simulation that can be incremented in parallel.
Most commonly this is used in RL applications, like for Rllib or Stable Baselines, as their high computational cost requires the efficiency of parallelization.

In order to change the the simulation into a gym.Env, subclass ``gym.Env`` and the ``DeepSurveySim.Survey`` functionality.
It is recommended to check your work with gym's `Envoriment Checker <https://gymnasium.farama.org/api/utils/#environment-checking>`_ before using it in any application.

Initialization
---------------

The initalization method of the subclass has to initalize both the ``Survey`` and ``Env`` super classes.
In the case of the ``Survey``, this just means calling the ``super().__init__()`` method with the necessary configuration files.
For ``Env``, this means including the ``self.action_space`` and ``self.observation_space`` attributes.

The bare requirements for an initalized action space is as follows:

.. code-block:: python

    def __init__(self, obs_config, survey_config):
        super().__init__(observatory_config=obs_config, survey_config=survey_config)

        self.action_space = self.generate_action_space()
        self.observation_space = self.generation_observation_space()


Action Space
--------------------------
These ``spaces`` must be either ``spaces.Box``, ``spaces.Discrete`` instances,  (in the case of actions that are single variables), or ``spaces.Dict`` instances (where they are made up of multiple variables).
`View the spaces documentation for further details. <https://gymnasium.farama.org/api/spaces/>`_

For this specific program, the logical action space is made up of the variables that are input into ``Survey.update()``, the continious ``ra``, ``decl`` and the discete ``band``.
This would produce an action space

.. code-block:: python

    spaces.Dict(
        {
        "ra": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
        "decl": spaces.Box(low=-90.0, high=90.0, shape=(1,), dtype=np.float32),
        "band": spaces.Discrete(len(self.observator.band_wavelengths))
        }
    )

However, some of these actions can be held constant, and thus the action space becomes:

.. code-block:: python

    spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)

Or if you wish to descetize the action space (the below example assumes the ``step`` function maps the actions to 10 unique ra/decl options; it would also be useful to store this map in the envoriment's ``__init__``):

.. code-block:: python

    spaces.Dict(
        {
        "ra": spaces.Discrete(10),
        "decl": spaces.Discrete(10),
        "band": spaces.Discrete(len(self.observator.band_wavelengths))
        }
    )

Observation Space
-------------------

The observation space behaves very simularly to the action space in terms of programming.
The only difference is that this defines the format of the expected output of ``step`` method.
By consequence, it is encouraged to only define the observation in terms of the variables in ``Survey.observatory_variables``.

For example, a configuration file that contains the line

.. code-block:: yaml

    variables: ["airmass", 'ha']

Would logically have the ``observation_space``:

.. code-block:: python

    spaces.Dict(
        {
        "airmass": spaces.Box(low=-100000, high=100000, shape=(1,), dtype=np.float32),
        "ha": spaces.Box(low=-100000, high=100000, shape=(1,), dtype=np.float32),
        }
    )

This space is much larger than is stricitly required for these variables, but if you wish to define the spaces automatically, using this wider range is encouraged.

Step and Reset
---------------
``step`` and ``reset`` are also required by ``gym.Env``, and are the core of the program.

``step`` defines how the simulation is updated and what is returned (and what format), and ``reset`` returns the program back to its inital condition.

The ``super().step(action)`` from ``TelescopePositioningSimulator.Survey`` already handles updating the simulation, so all that is required of the subclass is formatting.
The ``action`` argument of ``step`` requests a dictionary containing ``time``, ``location``, ``band``.
This can be achieved by formatting the passed action into:

.. code-block:: python

    {
    "time": self.time,
    "location": {"ra": action["ra"], "decl": action["decl"]},
    "band": action["band"]
    }

``time`` is the only required parameter, so if ra, declination, or band are held constant, they need not be passed.
The ``action`` given to the ``step`` function depends on the variables defined in the ``action_space``

The ``super().step()`` returns the calculated observation (containing the variables from ``Survey.observation_variables``) as a dictionary of arrays, the reward, as an array, the stop condition, as an array, and a 'log' (dictionary with possible diagonistic data).
The format step will need to return depends on the framework being used and the specifics of your ``observation_space``.

For example, a discrete observation space will require:

.. code-block:: python

    new_observation = {
        key: mapping_rule(observation[key]) for key in self.observation_space
    }
Where the ``mapping_rule`` defines how the variable ``obvervation[key]`` is discetized.

Or a continious ``np.ndarray`` observation space will be:

.. code-block:: python

    new_observation = {
        key: np.array(np.nan_to_num(observation[key], copy=True).ravel(), dtype=np.float32,) for key in self.observation_space
    }


``reset`` also requires an observation be formated as defined in ``self.observation_space``, but also requires the ``super().reset()`` method is called.
The state of the simulation can then be accessed with ``self._observation_calculation()``

.. code-block:: python

     def reset(self, *, seed=None, options=None):
            super().reset()
            observation = self._observation_calculation()
            return observation

Example
--------
The below example shows a bare-bones envoriment with outputs designed for an `rllib` trained algorithm to interact with.

.. code-block:: python

    import numpy as np
    from gymnasium import spaces, Env
    from DeepSurveySim.Survey import Survey
    from DeepSurveySim.IO import ReadConfig

    class GymSurvey(Survey, Env):
        def __init__(self, kwarg):
            obs_config = ReadConfig(kwarg["observatory_config"])()
            survey_config = ReadConfig(kwarg["survey_config"], survey=True)()

            super().__init__(observatory_config=obs_config, survey_config=survey_config)

            self.action_space = spaces.Dict(
                {
                    "ra": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
                    "decl": spaces.Box(
                        low=-90.0, high=90.0, shape=(1,), dtype=np.float32
                    ),
                }
            )

            self.observation_space = spaces.Dict(
                {
                    "airmass": spaces.Box(
                        low=-100000, high=100000, shape=(1,), dtype=np.float32
                    ),
                    "alt": spaces.Box(
                        low=-100000, high=100000, shape=(1,), dtype=np.float32
                    ),
                    "sky_magnitude": spaces.Box(
                        low=-100000, high=100000, shape=(1,), dtype=np.float32
                    ),
                    "teff": spaces.Box(
                        low=-100000, high=100000, shape=(1,), dtype=np.float32
                    ),
                }
            )

        def reset(self, *, seed=None, options=None):
            super().reset()
            observation = self._observation_calculation()
            observation = {
                key: np.nan_to_num(observation[key], copy=True)
                for key in self.observation_space
            }

            return observation, {}

        def step(self, action: dict):

            new_action = {
                "time": self.time,
                "location": {"ra": action["ra"], "decl": action["decl"]},
            }
            observation, reward, stop, log = super().step(new_action)
            truncated = False  # Additional truncated flag required by RLLib

            observation = {
                key: np.array(
                    np.nan_to_num(observation[key], copy=True).ravel()[0],
                    dtype=np.float32,
                ).reshape(
                    1,
                )
                for key in self.observation_space
            }
            reward = reward.ravel()[0]
            return observation, reward, stop, truncated, log
