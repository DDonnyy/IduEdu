import math

import pytest

from iduedu.constants.transport_specs import (
    DEFAULT_REGISTRY,
    DEFAULT_REGISTRY_W_TRAIN,
    TransportRegistry,
    TransportSpec,
)

pytestmark = pytest.mark.unit


def approx(a: float, b: float, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


@pytest.mark.parametrize(
    "spec",
    [
        TransportSpec(name="", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, traffic_coef=1.0),
        TransportSpec(name="   ", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, traffic_coef=1.0),
    ],
)
def test_transport_spec_validate_name(spec):
    with pytest.raises(ValueError):
        spec.validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name="bus", vmax_tech_kmh=0, accel_dist_m=1, brake_dist_m=1, traffic_coef=1.0),
        dict(name="bus", vmax_tech_kmh=-1, accel_dist_m=1, brake_dist_m=1, traffic_coef=1.0),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=-1, brake_dist_m=1, traffic_coef=1.0),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=-1, traffic_coef=1.0),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, traffic_coef=0),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, traffic_coef=2.0),
        dict(name="bus", vmax_tech_kmh=10, accel_dist_m=1, brake_dist_m=1, traffic_coef=1.0, avg_wait_time_min=-1),
    ],
)
def test_transport_spec_validate_ranges(kwargs):
    spec = TransportSpec(**kwargs)
    with pytest.raises(ValueError):
        spec.validate()


def test_transport_spec_validate_str():
    spec = TransportSpec(name="bus", vmax_tech_kmh="50", accel_dist_m="10", brake_dist_m="10", traffic_coef="1.0")
    with pytest.raises(ValueError):
        spec.validate()

    bad = TransportSpec(name="bus", vmax_tech_kmh="fast", accel_dist_m=10, brake_dist_m=10, traffic_coef=1.0)
    with pytest.raises(ValueError):
        bad.validate()

    bad_wait = TransportSpec(
        name="bus",
        vmax_tech_kmh=50,
        accel_dist_m=10,
        brake_dist_m=10,
        traffic_coef=1.0,
        avg_wait_time_min="slow",
    )
    with pytest.raises(ValueError):
        bad_wait.validate()


def test_travel_time_zero_or_negative_length():
    spec = TransportSpec("bus", 60, 100, 100, 1.0)
    assert spec.travel_time_min(0) == 0.0
    assert spec.travel_time_min(-5) == 0.0


def test_travel_time_long_segment_with_cruise():
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=100, brake_dist_m=50, traffic_coef=1.0)
    L = 1000.0
    V = 60 * 1000.0 / 60.0  # 1000 m/min
    span = 150.0
    expected = (2 * 100 / V) + (2 * 50 / V) + ((L - span) / V)
    got = spec.travel_time_min(L)
    assert approx(got, expected, rel=1e-12, abs_=1e-12)


def test_travel_time_short_segment_no_cruise_uses_v_peak():
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=100, brake_dist_m=100, traffic_coef=1.0)
    L = 50.0
    V = 60 * 1000.0 / 60.0  # 1000 m/min
    span = 200.0
    v_peak = V * math.sqrt(L / span)  # 500 m/min
    expected = (2 * L) / v_peak
    got = spec.travel_time_min(L)
    assert approx(got, expected, rel=1e-12, abs_=1e-12)


def test_travel_time_grows_with_length_on_short_segments():
    """Peak speed scaling linearly in L made every sub-span segment cost the same."""
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=700, brake_dist_m=650, traffic_coef=0.7)
    times = [spec.travel_time_min(L, speed_limit_mpm=50 * 1000 / 60) for L in (150, 400, 800, 1200)]
    assert times == sorted(times)
    assert times[-1] > times[0] * 2


def test_travel_time_branches_meet_at_span():
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=100, brake_dist_m=100, traffic_coef=1.0)
    span = 200.0
    below = spec.travel_time_min(span - 1e-6)
    above = spec.travel_time_min(span + 1e-6)
    assert approx(below, above, rel=1e-6, abs_=1e-6)


def test_travel_time_respects_speed_limit():
    spec = TransportSpec("bus", vmax_tech_kmh=80, accel_dist_m=0, brake_dist_m=0, traffic_coef=1.0)
    L = 1000.0

    # без лимита: V = 80km/h -> 1333.333.. m/min
    t_free = spec.travel_time_min(L)

    # с лимитом 600 m/min (36 km/h) время должно быть больше: t = L/V
    t_limited = spec.travel_time_min(L, speed_limit_mpm=600.0)
    assert t_limited > t_free
    assert approx(t_limited, L / 600.0, rel=1e-12, abs_=1e-12)


def test_travel_time_respects_traffic_coef():
    base = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=0, brake_dist_m=0, traffic_coef=1.0)
    jam = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=0, brake_dist_m=0, traffic_coef=0.5)
    L = 1200.0
    assert jam.travel_time_min(L) > base.travel_time_min(L)


def test_travel_time_min_speed_floor_on_short_segment():
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=1000, brake_dist_m=1000, traffic_coef=1.0)
    L = 1.0
    min_speed = 60.0
    got = spec.travel_time_min(L, min_speed_mpm=min_speed)
    expected = (2 * L) / min_speed
    assert approx(got, expected, rel=1e-12, abs_=1e-12)


def test_free_speed_is_affine_in_the_speed_limit():
    spec = TransportSpec("bus", vmax_tech_kmh=90, accel_dist_m=0, brake_dist_m=0, traffic_coef=0.2, base_speed_kmh=12)
    L = 3000.0
    for limit_kmh in (30, 50, 90):
        expected_v = (12 + 0.2 * limit_kmh) * 1000 / 60
        got = spec.travel_time_min(L, speed_limit_mpm=limit_kmh * 1000 / 60)
        assert approx(got, L / expected_v, rel=1e-12, abs_=1e-12)


def test_free_speed_is_capped_by_technical_maximum():
    spec = TransportSpec("tram", vmax_tech_kmh=40, accel_dist_m=0, brake_dist_m=0, traffic_coef=1.0, base_speed_kmh=30)
    L = 2000.0
    got = spec.travel_time_min(L, speed_limit_mpm=90 * 1000 / 60)
    assert approx(got, L / (40 * 1000 / 60), rel=1e-12, abs_=1e-12)


def test_dwell_is_added_once_on_both_branches():
    kwargs = dict(vmax_tech_kmh=60, accel_dist_m=100, brake_dist_m=100, traffic_coef=1.0)
    plain = TransportSpec("bus", **kwargs)
    with_dwell = TransportSpec("bus", **kwargs, dwell_min=0.5)
    for L in (50.0, 1000.0):  # below and above span
        assert approx(with_dwell.travel_time_min(L), plain.travel_time_min(L) + 0.5, rel=1e-12, abs_=1e-12)


def test_base_speed_defaults_preserve_multiplicative_behaviour():
    """Registries built before base_speed_kmh existed must be unaffected."""
    spec = TransportSpec("bus", vmax_tech_kmh=60, accel_dist_m=0, brake_dist_m=0, traffic_coef=0.7)
    L = 1000.0
    limit = 40 * 1000 / 60
    assert approx(spec.travel_time_min(L, speed_limit_mpm=limit), L / (0.7 * limit), rel=1e-12, abs_=1e-12)


def test_registry_add_and_get_normalizes_key():
    reg = TransportRegistry()
    reg.add(TransportSpec("  BuS  ", 60, 1, 1, 1.0))
    assert reg.get("bus").name == "bus"
    assert reg.get(" BUS ").name == "bus"


def test_registry_add_duplicate_without_overwrite_raises():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    with pytest.raises(ValueError):
        reg.add(TransportSpec("bus", 70, 1, 1, 1.0))


def test_registry_add_duplicate_with_overwrite():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.add(TransportSpec("bus", 70, 1, 1, 1.0), overwrite=True)
    assert reg.get("bus").vmax_tech_kmh == 70


def test_registry_try_get_returns_none():
    reg = TransportRegistry()
    assert reg.try_get("unknown") is None


def test_registry_remove():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.remove("bus")
    with pytest.raises(KeyError):
        reg.get("bus")


def test_registry_update_field():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    updated = reg.update("bus", traffic_coef=0.7)
    assert updated.traffic_coef == 0.7
    assert reg.get("bus").traffic_coef == 0.7


def test_registry_update_wait_time():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    updated = reg.update("bus", avg_wait_time_min=4.5)
    assert updated.avg_wait_time_min == 4.5
    assert reg.get("bus").avg_wait_time_min == 4.5


def test_registry_update_rename_moves_key():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.update("bus", name="express_bus")
    with pytest.raises(KeyError):
        reg.get("bus")
    assert reg.get("express_bus").name == "express_bus"


def test_registry_update_rename_to_existing_raises():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.add(TransportSpec("tram", 50, 1, 1, 1.0))
    with pytest.raises(ValueError):
        reg.update("bus", name="tram")


def test_registry_ensure_existing_returns_same():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    spec = reg.ensure("bus")
    assert spec.name == "bus"
    assert reg.get("bus") is spec


def test_registry_ensure_creates_default_when_missing():
    reg = TransportRegistry()
    spec = reg.ensure("ferry")
    assert spec.name == "ferry"
    # дефолтные значения из ensure()
    assert spec.vmax_tech_kmh > 0
    assert reg.get("ferry").name == "ferry"


def test_registry_ensure_uses_provided_defaults():
    reg = TransportRegistry()
    defaults = TransportSpec("funicular", 30, 10, 10, 0.9)
    spec = reg.ensure("funicular", defaults=defaults)
    assert reg.get("funicular").vmax_tech_kmh == 30
    assert spec.traffic_coef == 0.9


def test_registry_list_types_contains_added():
    reg = TransportRegistry()
    reg.add(TransportSpec("bus", 60, 1, 1, 1.0))
    reg.add(TransportSpec("tram", 50, 1, 1, 1.0))
    types = reg.list_types()
    assert "bus" in types
    assert "tram" in types


def test_default_registry_has_expected_keys():
    types = set(DEFAULT_REGISTRY.list_types())
    assert {"bus", "tram", "trolleybus", "subway"} <= types
    assert "train" not in types


def test_default_registry_has_mode_specific_wait_times():
    assert DEFAULT_REGISTRY.get("bus").avg_wait_time_min == 8.0
    assert DEFAULT_REGISTRY.get("trolleybus").avg_wait_time_min == 10.0
    assert DEFAULT_REGISTRY.get("tram").avg_wait_time_min == 5.0
    assert DEFAULT_REGISTRY.get("subway").avg_wait_time_min == 3.0
    assert DEFAULT_REGISTRY_W_TRAIN.get("train").avg_wait_time_min == 11.0


def test_default_registry_keeps_wait_times_ordered_by_mode():
    """The ordering disappears silently if the constants are ever collapsed into one value."""
    waits = {t: DEFAULT_REGISTRY_W_TRAIN.get(t).avg_wait_time_min for t in DEFAULT_REGISTRY_W_TRAIN.list_types()}
    assert waits["subway"] < waits["tram"] < waits["bus"] <= waits["trolleybus"] < waits["train"]


def test_default_registry_keeps_buses_faster_on_faster_roads():
    bus = DEFAULT_REGISTRY.get("bus")
    L = 3000.0
    city = bus.travel_time_min(L, speed_limit_mpm=50 * 1000 / 60)
    highway = bus.travel_time_min(L, speed_limit_mpm=90 * 1000 / 60)
    assert highway < city


def test_default_registry_rail_reacts_to_the_road_limit_more_than_buses():
    """Measured behaviour: rail responds to a tagged limit more than buses, but both respond.

    An earlier version of this test demanded a fivefold difference, on constants
    fitted over 28 cities where the bus coefficient came out near zero. Refitting
    over 65 cities of matched segments raised it from 0.05 to 0.375: buses do
    follow road class, roughly two thirds as strongly as the metro. The ordering
    survived, the magnitude did not, and the test now guards the ordering.
    """
    L = 3000.0

    def gain_kmh(spec):
        city = L / spec.travel_time_min(L, speed_limit_mpm=50 * 1000 / 60) * 60 / 1000
        highway = L / spec.travel_time_min(L, speed_limit_mpm=90 * 1000 / 60) * 60 / 1000
        return highway - city

    assert gain_kmh(DEFAULT_REGISTRY.get("subway")) > gain_kmh(DEFAULT_REGISTRY.get("bus")) > 5.0


def test_default_registry_with_train_has_expected_keys():
    types = set(DEFAULT_REGISTRY_W_TRAIN.list_types())
    assert {"bus", "tram", "trolleybus", "subway", "train"} <= types


def test_default_registry_specs_are_valid():
    for t in DEFAULT_REGISTRY.list_types():
        DEFAULT_REGISTRY.get(t).validate()
    for t in DEFAULT_REGISTRY_W_TRAIN.list_types():
        DEFAULT_REGISTRY_W_TRAIN.get(t).validate()


def test_default_registries_are_independent():
    assert DEFAULT_REGISTRY_W_TRAIN is not DEFAULT_REGISTRY
    assert DEFAULT_REGISTRY.try_get("train") is None
    assert DEFAULT_REGISTRY_W_TRAIN.try_get("train") is not None
