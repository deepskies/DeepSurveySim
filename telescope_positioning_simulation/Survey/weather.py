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

        csv_configuration = {**csv_configuration, **self.default_configuration()}
        self.seeing_name = csv_configuration["seeing"]
        self.date_name = csv_configuration["date"]

        self.seeing_tolerance = seeing_tolerance
        self.clouds_tolerance = cloud_tolerance

        self.reference_seeing = base_seeing
        self.reference_clouds = base_clouds

        self.weather_source = pd.read_csv(weather_source_file)[
            [self.seeing_name, self.date_name]
        ]
        self.format_source(csv_configuration)

    def default_configuration(self):
        return {
            "seeing": "HourlySkyConditions",
            "reference_clouds": 29.92,
            "date": "DATE",
            "allowed_conditions": ["FEW", "CLR"],
        }

    def format_source(self, configuration):

        self.weather_source.dropna(inplace=True)

        allowed = "|".join(configuration["allowed_conditions"])

        self.weather_source[self.seeing_name] = self.weather_source[
            self.seeing_name].str.contains(allowed).astype(int)

        self.weather_source[self.date_name] = pd.to_datetime(
            self.weather_source[self.date_name], infer_datetime_format=True
        )

    def find_month(self, mjd):
        date = Time(mjd, format="mjd")
        month = date.datetime.month
        return month

    def condition(self, mjd):
        month = self.find_month(mjd)
        matching = self.weather_source[
            self.weather_source[self.date_name].dt.month == month
        ]
        return matching

    def seeing(self, condition):
        seeing_conditions = 1 - condition[self.seeing_name].mean()

        if seeing_conditions >= self.seeing_tolerance:
            seeing = self.reference_seeing * (1 - round(seeing_conditions, 1))
        else:
            seeing = self.reference_seeing

        return seeing

    def clouds(self, condition):
        cloud_condition = 1 -  condition[self.seeing_name].mean()
        if cloud_condition >= self.clouds_tolerance:
            clouds = 1
        else:
            clouds = self.reference_clouds

        return clouds
