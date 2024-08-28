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
            PublicTrasport.SUBWAY: 40,
            PublicTrasport.BUS: 20,
            PublicTrasport.TRAM: 20,
            PublicTrasport.TROLLEYBUS: 18,
        }
        return speeds[self] * 1000 / 60
