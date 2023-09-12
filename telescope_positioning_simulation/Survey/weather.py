import pandas as pd
from astropy.time import Time


class Weather:
    def __init__(
        self,
        weather_source_file: str,
        seeing_tolerance: float = 0.25,
        cloud_tolerance: float = 0.1,
        base_seeing: float = 0.9,
        base_clouds: float = 0,
        csv_configuration: dict = {},
    ) -> None:

        csv_configuration = {**csv_configuration, **self.default_configuration()}
        self.seeing_name = csv_configuration["seeing"]
        self.clouds_name = csv_configuration["clouds"]
        self.date_name = csv_configuration["date"]

        self.seeing_tolerance = seeing_tolerance
        self.clouds_tolerance = cloud_tolerance

        self.reference_seeing = base_seeing
        self.reference_clouds = base_clouds

        self.weather_source = pd.read_csv(weather_source_file)[
            [self.seeing_name, self.date_name, self.clouds_name]
        ]
        self.format_source(csv_configuration)

    def default_configuration(self):
        return {
            "seeing": "HourlySkyConditions",
            "clouds": "HourlyStationPressure",
            "reference_clouds": 29.92,
            "date": "DATE",
            "allowed_conditions": ["FEW", "CLR"],
        }

    def format_source(self, configuration):
        allowed = "|".join(configuration["allowed_conditions"])
        self.weather_source[
            self.weather_source[self.seeing_name].str.contains(allowed) == True
        ][self.seeing_name] = 1

        self.weather_source[
            self.weather_source[self.seeing_name].str.contains(allowed) == False
        ][self.seeing_name] = 0

        print(self.weather_source[self.seeing_name])
        self.weather_source[self.date_name] = pd.to_datetime(
            self.weather_source[self.date_name], infer_datetime_format=True
        )

        self.weather_source[self.clouds_name] = self.weather_source[
            self.clouds_name
        ].str.replace(r"[^0-9]+", "")
        self.weather_source[self.clouds_name] = abs(
            self.weather_source[self.clouds_name].astype(float)
            - configuration["reference_clouds"]
        )

        self.weather_source.dropna(inplace=True)

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

        if seeing_conditions >= self.cover_tolerance:
            seeing = 0
        else:
            seeing = self.reference_seeing * (1 - round(seeing_conditions, 1))

        return seeing

    def clouds(self, condition):
        cloud_condition = condition[self.clouds_name].mean()

        if cloud_condition >= self.clouds_tolerance:
            clouds = 1
        else:
            clouds = self.reference_clouds

        return clouds
