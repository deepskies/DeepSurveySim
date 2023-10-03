import astropy
import numpy as np


class Action:
    def __init__(
        self, time: float, ra: float = 0, decl: float = 0, band: str = "g"
    ) -> None:
        self.mjd = time
        self.band = band

        assert np.any(0 <= ra) and np.any(ra <= 360)
        self.ra = ra

        assert np.any(-90 <= decl) and np.any(decl <= 90)
        self.decl = decl

        self.degree = astropy.units.deg
        self.radians = astropy.units.rad

        self.location = self.sky_coordinates()
        self.time = self._time()

    def asdict(self):
        return {
            "time": self.mjd,
            "location": {"ra": self.ra, "decl": self.decl},
            "band": self.band,
        }

    def sky_coordinates(self):
        return astropy.coordinates.SkyCoord(
            ra=self.ra * self.degree, dec=self.decl * self.degree, unit="deg"
        )

    def _time(self):
        return astropy.time.Time(np.asarray(self.mjd), format="mjd")
