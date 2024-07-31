from enum import Enum


class PublicTrasport(Enum):
    """
    Enumeration class for edge types in graphs.
    """
    SUBWAY = "subway"
    BUS = "bus"
    TRAM = "tram"
    TROLLEYBUS = "trolleybus"

    @property
    def russian_name(self) -> str:
        names = {
            PublicTrasport.SUBWAY: "Метро",
            PublicTrasport.BUS: "Автобус",
            PublicTrasport.TRAM: "Трамвай",
            PublicTrasport.TROLLEYBUS: "Троллейбус",
        }
        return names[self]

    @property
    def avg_speed(self) -> float:
        """
        Average speed in m/min.
        """
        speeds = {
            PublicTrasport.SUBWAY: 40 * 1000 / 60,
            PublicTrasport.BUS: 20 * 1000 / 60,
            PublicTrasport.TRAM: 20 * 1000 / 60,
            PublicTrasport.TROLLEYBUS: 18 * 1000 / 60,
        }
        return speeds[self]


class Transport(Enum):
    """
    Enumeration class for edge types in graphs.
    """
    WALK = "walk"
    DRIVE = "car"

    @property
    def russian_name(self) -> str:
        names = {
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
            Transport.WALK: 5 * 1000 / 60,
            Transport.DRIVE: 50 * 1000 / 60,
        }
        return speeds[self]
