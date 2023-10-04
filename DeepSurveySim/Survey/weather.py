import pandas as pd
from astropy.time import Time


class Weather:
    def __init__(
        self,
        weather_source_file: str,
        seeing_tolerance: float = 0.5,
        cloud_tolerance: float = 0.5,
        base_seeing: float = 0.9,
        base_clouds: float = 0,
        csv_configuration: dict = {},
        **kwargs
    ) -> None:
        """Run a deteriministic weather simulation based on historical data.
                Assumes all weather conditions are a function of the month.
                Update either "clouds" or "seeing" based on given tolerances.

                Args:
                    weather_source_file (str): Path to csv file containing historical weather data.
                    seeing_tolerance (float, optional): Allowed percent of cloudiness before seeing is impacted. Defaults to 0.5.
                    cloud_tolerance (float, optional): Allowed percent of cloudiness before clouds are set to full extiniction. Defaults to 0.5.
                    base_seeing (float, optional): Best allowed seeing conditions. Defaults to 0.9.
                    base_clouds (float, optional): Best allowed cloud conditions. Defaults to 0.
                    csv_configuration (dict, optional): Instructions on how to read the csv, add in any parameters to the configuration dictionary with this argument. Defaults to {"seeing": "HourlySkyConditions","date": "DATE", "allowed_conditions": ["FEW", "CLR"]}
                }
        .
        """

        csv_configuration = {**csv_configuration, **self._default_configuration()}
        self.seeing_name = csv_configuration["seeing"]
        self.date_name = csv_configuration["date"]

        self.seeing_tolerance = seeing_tolerance
        self.clouds_tolerance = cloud_tolerance

        self.reference_seeing = base_seeing
        self.reference_clouds = base_clouds

        self.weather_source = pd.read_csv(weather_source_file)[
            [self.seeing_name, self.date_name]
        ]
        self._format_source(csv_configuration)

    def _default_configuration(self):
        return {
            "seeing": "HourlySkyConditions",
            "date": "DATE",
            "allowed_conditions": ["FEW", "CLR"],
        }

    def _format_source(self, configuration):

        self.weather_source.dropna(inplace=True)

        allowed = "|".join(configuration["allowed_conditions"])

        self.weather_source[self.seeing_name] = (
            self.weather_source[self.seeing_name].str.contains(allowed).astype(int)
        )

        self.weather_source[self.date_name] = pd.to_datetime(
            self.weather_source[self.date_name], infer_datetime_format=True
        )

    def _find_date(self, mjd):
        date = Time(mjd, format="mjd")
        month = date.datetime.month
        day = date.datetime.day
        return month, day

    def condition(self, mjd):
        """
        Return the sky conditions for all data with the same month as the supplied mjd.

        Args:
            mjd (Union(int, float)): Date in MJD to get sky conditions for.

        Returns:
           pd.Series: Conditions of the sky with the same month as the supplied mjd.
        """
        month, day = self._find_date(mjd)
        matching = self.weather_source[
            self.weather_source[self.date_name].dt.month == month
        ]
        matching = matching[
            matching[self.date_name].dt.day.isin([day, day + 1, day + 2])
        ]
        return matching

    def seeing(self, condition):
        """
        Approximate seeing conditions. If the seeing condition is above the tolerance level, seeing is scaled by the percent removed from perfect seeing

        Args:
            condition (pd.Series): Series to arggergate into a seeing score

        Returns:
            float: Updated seeing conditions
        """
        seeing_conditions = 1 - condition[self.seeing_name].mean()

        if seeing_conditions >= self.seeing_tolerance:
            seeing = self.reference_seeing * (1 - round(seeing_conditions, 1))
        else:
            seeing = self.reference_seeing

        return seeing

    def clouds(self, condition):
        """
        Approximate cloud extiction. If the cloud conditions are above the tolerance levels, clouds are set to 1. Else returns them to base levels.

        Args:
            condition (pd.Series): Series to arggergate into a cloud score

        Returns:
            float: clouds for the passed condition
        """
        cloud_condition = 1 - condition[self.seeing_name].mean()
        if cloud_condition >= self.clouds_tolerance:
            clouds = 1
        else:
            clouds = self.reference_clouds

        return clouds
