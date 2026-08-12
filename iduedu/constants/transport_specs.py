from dataclasses import dataclass, replace
from math import sqrt


@dataclass(frozen=True, slots=True)
class TransportSpec:  # pylint: disable=too-many-instance-attributes
    """
    Configuration of a single public-transport mode used to estimate travel time on graph edges.

    Each transport specification defines technical and operational characteristics of a mode
    (e.g. bus, tram, subway) that are used to compute per-edge travel time based on segment length,
    road speed limits, acceleration/braking behavior, and traffic conditions.

    Attributes:
        name (str):
            Transport type identifier, usually matching the OSM ``route=*`` value
            (e.g. ``"bus"``, ``"tram"``, ``"subway"``).
        vmax_tech_kmh (float):
            Technical maximum speed of the vehicle in kilometers per hour.
        accel_dist_m (float):
            Typical distance (meters) required to accelerate from standstill to cruising speed.
        brake_dist_m (float):
            Typical distance (meters) required to decelerate from cruising speed to standstill.
        traffic_coef (float):
            Share of the road speed limit the vehicle actually realizes on top of
            ``base_speed_kmh``. Free-flow speed is affine in the limit:
            ``v_free = min(vmax_tech_kmh, base_speed_kmh + traffic_coef * speed_limit)``.
            Rail modes have a high coefficient (the tagged limit governs them), while
            surface modes have a low one (they are governed by congestion and stops).
        base_speed_kmh (float):
            Speed the mode sustains regardless of the road limit, in kilometers per hour.
            Defaults to ``0``, which reduces the affine model to the purely multiplicative
            one and preserves the behavior of registries built before this field existed.
        dwell_min (float):
            Time lost standing at a stop, in minutes. Added once per segment. Defaults to ``0``.
        avg_wait_time_min (float):
            Average passenger waiting time in minutes. It is assigned to directed
            ``boarding`` edges built from OSM data.

    See also:
        https://iduclub.github.io/IduEdu/examples/transport_registry.html
    """

    name: str
    vmax_tech_kmh: float
    accel_dist_m: float
    brake_dist_m: float
    traffic_coef: float = 1.0
    avg_wait_time_min: float = 1.0
    base_speed_kmh: float = 0.0
    dwell_min: float = 0.0

    def validate(self) -> None:
        """Validate transport specification fields.

        Raises:
            ValueError: If a required field is missing or outside the supported
            range.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("TransportSpec.name must be a non-empty string")

        for field in (
            "vmax_tech_kmh",
            "accel_dist_m",
            "brake_dist_m",
            "traffic_coef",
            "avg_wait_time_min",
            "base_speed_kmh",
            "dwell_min",
        ):
            v = getattr(self, field)
            if v is None:
                raise ValueError(f"{field} must not be None")
            if not isinstance(v, (float, int)):
                raise ValueError(f"{field} must be numeric, got {type(getattr(self, field))}")

        if self.vmax_tech_kmh <= 0:
            raise ValueError("vmax_tech_kmh must be > 0")
        if self.accel_dist_m < 0 or self.brake_dist_m < 0:
            raise ValueError("accel_dist_m and brake_dist_m must be >= 0")
        if not (0 < self.traffic_coef <= 1.5):
            raise ValueError("traffic_coef must be in (0, 1.5]")
        if self.avg_wait_time_min < 0:
            raise ValueError("avg_wait_time_min must be >= 0")
        if self.base_speed_kmh < 0:
            raise ValueError("base_speed_kmh must be >= 0")
        if self.dwell_min < 0:
            raise ValueError("dwell_min must be >= 0")

    def travel_time_min(
        self,
        segment_len_m: float,
        *,
        speed_limit_mpm: float | None = None,
        min_speed_mpm: float = 60.0,  # 60 m/min = 1 m/s = 3.6 km/h
    ) -> float:
        """
        Compute travel time (minutes) for a single graph segment.

        The method estimates traversal time using a simplified kinematic model that accounts for:
        - the transport mode technical maximum speed;
        - an optional road speed limit, entering the free-flow speed affinely;
        - time lost standing at a stop (``dwell_min``);
        - time lost on acceleration and braking.

        For short segments where the vehicle cannot reach cruising speed, a reduced peak speed
        is assumed and the segment is traversed using an acceleration-deceleration profile
        without a cruising phase. The peak speed scales as ``sqrt(L / span)`` because under
        constant acceleration the distance covered grows with the square of the speed reached;
        the two branches meet continuously at ``L == span``.

        Parameters:
            segment_len_m (float):
                Segment length in meters. Must be positive.
            speed_limit_mpm (float | None):
                Optional road speed limit in meters per minute. If provided, the effective speed
                will not exceed this value.
            min_speed_mpm (float):
                Lower bound for effective speed (meters per minute), used to avoid unrealistically
                large travel times on very short segments.

        Returns:
            float:
                Estimated travel time for the segment in minutes.
        """
        segment_len_m = float(segment_len_m)
        if segment_len_m <= 0:
            return 0.0

        vmax = float(self.vmax_tech_kmh) * 1000.0 / 60.0

        limit = vmax
        if speed_limit_mpm is not None and float(speed_limit_mpm) > 0:
            limit = min(vmax, float(speed_limit_mpm))

        velocity = float(self.base_speed_kmh) * 1000.0 / 60.0 + float(self.traffic_coef) * limit
        velocity = min(velocity, vmax)
        velocity = max(velocity, float(min_speed_mpm))  # avoid zero speed

        d_acc = max(float(self.accel_dist_m), 0.0)
        d_brk = max(float(self.brake_dist_m), 0.0)

        span = d_acc + d_brk
        dwell = float(self.dwell_min)

        if span > 1e-9 and segment_len_m < span:
            v_peak = velocity * sqrt(segment_len_m / span)
            v_peak = max(v_peak, float(min_speed_mpm))

            return dwell + (2.0 * segment_len_m) / v_peak

        # acceleration and braking together cost the same as covering `span` twice,
        # so the whole segment reduces to (L + span) / v
        return dwell + (segment_len_m + span) / velocity


class TransportRegistry:
    """
    Registry of available public-transport modes and their specifications.

    The registry stores ``TransportSpec`` objects indexed by normalized transport type names
    (lowercase). It provides utilities for validating transport types, updating parameters,
    and ensuring that unknown types encountered during parsing are assigned reasonable defaults.

    The registry is used throughout graph construction to compute per-edge travel times
    consistently across different transport modes.

    See also:
        https://iduclub.github.io/IduEdu/examples/transport_registry.html
    """

    def __init__(self, specs: dict[str, TransportSpec] | None = None):
        """Initialize the registry with optional transport specifications."""
        self._specs: dict[str, TransportSpec] = {}
        if specs:
            for k, v in specs.items():
                self.add(v if isinstance(v, TransportSpec) else TransportSpec(**v))

    @staticmethod
    def _norm_key(name: str) -> str:
        return name.strip().lower()

    def get(self, name: str) -> TransportSpec:
        """Return a transport specification by name.

        Args:
            name: Transport type name. Matching is case-insensitive and ignores
                surrounding whitespace.

        Raises:
            KeyError: If the transport type is unknown.
        """
        key = self._norm_key(name)
        try:
            return self._specs[key]
        except KeyError as e:
            raise KeyError(f"Unknown transport type: {name!r}") from e

    def try_get(self, name: str) -> TransportSpec | None:
        """Return a transport specification, or ``None`` if it is unknown."""
        return self._specs.get(self._norm_key(name))

    def add(self, spec: TransportSpec, *, overwrite: bool = False) -> None:
        """Add a transport specification to the registry.

        Args:
            spec: Specification to add.
            overwrite: If true, replace an existing specification with the
                same normalized name.

        Raises:
            ValueError: If ``spec`` is invalid or already exists.
        """
        spec = replace(spec, name=self._norm_key(spec.name))
        spec.validate()
        if (spec.name in self._specs) and not overwrite:
            raise ValueError(f"Transport {spec.name!r} already exists (use overwrite=True)")
        self._specs[spec.name] = spec

    def update(self, transport_type: str, **fields) -> TransportSpec:
        """Update fields of an existing transport specification.

        Args:
            transport_type: Existing transport type name.
            **fields: Dataclass fields to update.

        Returns:
            Updated transport specification.

        Raises:
            KeyError: If ``transport_type`` is unknown.
            ValueError: If updated fields are invalid or rename conflicts.
        """
        key = self._norm_key(transport_type)
        cur = self.get(key)
        if "name" in fields:
            fields["name"] = self._norm_key(fields["name"])
        nxt = replace(cur, **fields)
        nxt.validate()

        if nxt.name != key:
            if nxt.name in self._specs:
                raise ValueError(f"Cannot rename to {nxt.name!r}: already exists")
            del self._specs[key]
        self._specs[nxt.name] = nxt
        return nxt

    def remove(self, name: str) -> None:
        """Remove a transport specification by name."""
        key = self._norm_key(name)
        del self._specs[key]

    def ensure(self, name: str, *, defaults: TransportSpec | None = None) -> TransportSpec:
        """Return an existing spec or create one from defaults.

        Args:
            name: Transport type name.
            defaults: Optional specification used when the name is missing.

        Returns:
            Existing or newly registered transport specification.
        """
        key = self._norm_key(name)
        spec = self._specs.get(key)
        if spec:
            return spec

        if defaults is None:
            defaults = TransportSpec(
                name=key,
                vmax_tech_kmh=25.0,
                accel_dist_m=500.0,
                brake_dist_m=500.0,
                traffic_coef=0.8,
            )
        self.add(defaults, overwrite=False)
        return self._specs[key]

    def list_types(self):
        """Return registered transport type names."""
        return list(self._specs.keys())


# Speed parameters are fitted against observed GTFS run times: segments are matched to OSM
# travel edges by both endpoints, which yields triples of (length, scheduled time, road limit).
# Waiting times are harmonic means of boarding-edge times, taken per city and then aggregated
# by the median across cities: within a city routing takes the minimum over available
# departures, between cities no such minimum exists.
_DEFAULT_TRANSPORT_SPECS = {
    "bus": TransportSpec(
        "bus",
        vmax_tech_kmh=90,
        accel_dist_m=25.9,
        brake_dist_m=24.1,
        traffic_coef=0.375,
        avg_wait_time_min=8.0,
        base_speed_kmh=21.0,
        dwell_min=0.4,
    ),
    "trolleybus": TransportSpec(
        "trolleybus",
        vmax_tech_kmh=70,
        accel_dist_m=25.9,
        brake_dist_m=24.1,
        traffic_coef=0.15,
        avg_wait_time_min=10.0,
        base_speed_kmh=18.0,
        dwell_min=0.55,
    ),
    "tram": TransportSpec(
        "tram",
        vmax_tech_kmh=75,
        accel_dist_m=52.6,
        brake_dist_m=47.4,
        traffic_coef=0.30,
        avg_wait_time_min=5.0,
        base_speed_kmh=26.0,
        dwell_min=1.30,
    ),
    "subway": TransportSpec(
        "subway",
        vmax_tech_kmh=80,
        accel_dist_m=25.0,
        brake_dist_m=25.0,
        traffic_coef=0.625,
        avg_wait_time_min=3.0,
        base_speed_kmh=1.5,
        dwell_min=0.375,
    ),
}
_TRAIN_SPEC = TransportSpec(
    "train",
    vmax_tech_kmh=140,
    accel_dist_m=114.3,
    brake_dist_m=85.7,
    traffic_coef=0.675,
    avg_wait_time_min=11.0,
    base_speed_kmh=7.5,
    dwell_min=1.05,
)

DEFAULT_REGISTRY = TransportRegistry(_DEFAULT_TRANSPORT_SPECS)
DEFAULT_REGISTRY_W_TRAIN = TransportRegistry({**_DEFAULT_TRANSPORT_SPECS, "train": _TRAIN_SPEC})
