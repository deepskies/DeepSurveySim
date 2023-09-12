import pandas as pd


class Weather:
    def __init__(self, weather_source_file: str, csv_configuration: dict = {}) -> None:
        csv_configuration = {**csv_configuration, **self.default_configuration()}
        self.conditions = csv_configuration["sky_conditions"]
        self.visiblity = csv_configuration["visiblity"]
        self.date = csv_configuration["date"]

        self.weather_source = pd.read_csv(weather_source_file)
        self.format_source()

    def default_configuration(self):
        return {
            "sky_conditions": "HourlySkyConditions",
            "visiblity": "HourlyVisibility",
            "date": "DATE",
            "date_format": "YYYY-MM-DD",
        }

    def format_source(self):
        pass

    def find_date(self, mjd):
        pass

    def condition(self, mjd):
        pass

    def seeing(self, condition):
        pass

    def clouds(self, condition):
        pass
