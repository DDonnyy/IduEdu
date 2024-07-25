from enum import Enum


class Transport(Enum):
    """
    Enumeration class for edge types in graphs.
    """
    SUBWAY = "subway"
    BUS = "bus"
    TRAM = "tram"
    TROLLEYBUS = "trolleybus"
    WALK = "walk"
    DRIVE = "car"

    @property
    def russian_name(self) -> str:
        names = {
            Transport.SUBWAY: "Метро",
            Transport.BUS: "Автобус",
            Transport.TRAM: "Трамвай",
            Transport.TROLLEYBUS: "Троллейбус",
            Transport.WALK: "Пешком",
            Transport.DRIVE: "Автомобиль",
        }
        return names[self]

    @property
    def avg_speed(self) -> float:
        """
        Average speed in m/min.
        """
        speeds = {
            Transport.SUBWAY: 40 * 1000 / 60,
            Transport.BUS: 20 * 1000 / 60,
            Transport.TRAM: 20 * 1000 / 60,
            Transport.TROLLEYBUS: 18 * 1000 / 60,
            Transport.WALK: 5 * 1000 / 60,
            Transport.DRIVE: 50 * 1000 / 60,
        }
        return speeds[self]
