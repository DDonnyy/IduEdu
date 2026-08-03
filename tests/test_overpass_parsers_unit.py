import pandas as pd
import pytest

from iduedu.overpass.parsers import (
    infer_role_from_tags,
    overpass_routes_to_df,
    overpass_subway2edgenode,
    parse_maxspeed_to_m_per_min,
    parse_overpass_subway_data,
)

pytestmark = pytest.mark.unit

LOCAL_CRS = "EPSG:32636"


# ---------------------------------------------------------------------------
# parse_maxspeed_to_m_per_min
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected_kmh",
    [
        (50, 50.0),
        ("50", 50.0),
        ("50 km/h", 50.0),
        ("50kmh", 50.0),
        ("60kph", 60.0),
        ("30 mph", 30 * 1.60934),
    ],
)
def test_parse_maxspeed_known_formats(raw, expected_kmh):
    result = parse_maxspeed_to_m_per_min(raw)
    assert result == pytest.approx(expected_kmh * 1000.0 / 60.0)


@pytest.mark.parametrize("raw", [None, "", "   ", "walk", "fast", "50 knots"])
def test_parse_maxspeed_unparseable_returns_none(raw):
    assert parse_maxspeed_to_m_per_min(raw) is None


# ---------------------------------------------------------------------------
# infer_role_from_tags
# ---------------------------------------------------------------------------


def test_infer_role_empty_tags():
    assert infer_role_from_tags({}) == ""


@pytest.mark.parametrize(
    "tags, expected",
    [
        ({"public_transport": "station"}, "station"),
        ({"railway": "station"}, "station"),
        ({"station": "subway"}, "station"),
        ({"public_transport": "platform"}, "platform"),
        ({"railway": "platform"}, "platform"),
        ({"public_transport": "stop_position"}, "stop"),
        ({"railway": "halt"}, "stop"),
        ({"railway": "subway_entrance"}, "entrance"),
        ({"entrance": "yes"}, "entrance"),
        ({"entrance": "entry"}, "entry_only"),
        ({"entrance": "exit"}, "exit_only"),
        ({"entrance": "yes", "entry": "yes"}, "entry_only"),
        ({"entrance": "yes", "exit": "yes"}, "exit_only"),
        ({"amenity": "cafe"}, ""),
    ],
)
def test_infer_role_from_tags(tags, expected):
    assert infer_role_from_tags(tags) == expected


# ---------------------------------------------------------------------------
# overpass_routes_to_df
# ---------------------------------------------------------------------------


def test_overpass_routes_to_df_empty_input_has_bool_flag_columns():
    df = overpass_routes_to_df([], enable_subway_details=True)
    assert df.empty
    for col in ("is_stop_area", "is_stop_area_group", "is_station"):
        assert col in df.columns


def test_overpass_routes_to_df_flags_way_data_and_speed():
    routes = [
        {"type": "relation", "id": 1, "tags": {"route": "bus"}},
        {"type": "way", "id": 2, "tags": {"highway": "primary", "maxspeed": "60"}},
    ]
    df = overpass_routes_to_df(routes, enable_subway_details=False)

    way_row = df[df["id"] == 2].iloc[0]
    assert bool(way_row["is_way_data"]) is True
    assert way_row["way_speed_m_per_min"] == pytest.approx(60 * 1000.0 / 60.0)

    rel_row = df[df["id"] == 1].iloc[0]
    assert rel_row["transport_type"] == "bus"


def test_overpass_routes_to_df_marks_subway_station_details():
    routes = [
        {"type": "relation", "id": 10, "tags": {"public_transport": "stop_area"}},
        {"type": "relation", "id": 11, "tags": {"public_transport": "station"}},
    ]
    df = overpass_routes_to_df(routes, enable_subway_details=True)

    assert bool(df[df["id"] == 10].iloc[0]["is_stop_area"]) is True
    assert bool(df[df["id"] == 11].iloc[0]["is_station"]) is True
    # station-context relations are re-labelled as subway
    assert set(df["transport_type"]) == {"subway"}


# ---------------------------------------------------------------------------
# parse_overpass_subway_data / overpass_subway2edgenode: surface access typing
#
# Node type carries the surface-access contract consumed by the intermodal builder:
# "platform" means the object is reachable from the street and must be projected onto the
# walk graph, "subway_platform" means it is reachable only through a station or an entrance.
# ---------------------------------------------------------------------------


def _member(ref, role, lat, lon):
    return {"type": "node", "ref": ref, "role": role, "lat": lat, "lon": lon}


def _stop_area_row(area_id, members, **flags):
    row = {
        "id": area_id,
        "tags": {},
        "members": members,
        "is_stop_area": True,
        "is_stop_area_group": False,
        "is_station": False,
    }
    row.update(flags)
    return row


def _parse_stop_area(members):
    stop_areas = pd.DataFrame([_stop_area_row(10, members)])
    empty = pd.DataFrame(columns=["id", "tags", "members"])
    edges, nodes = parse_overpass_subway_data(stop_areas, empty, empty, LOCAL_CRS)
    return edges, dict(zip(nodes["ref_id"], nodes["type"]))


def _stop_area_members(*, station=True, entrance=True):
    # Refs: 1 station, 2 platform, 3 stop, 4 entrance.
    members = []
    if station:
        members.append(_member(1, "station", 59.9000, 30.3000))
    members.append(_member(2, "platform", 59.9001, 30.3001))
    members.append(_member(3, "stop", 59.9001, 30.30011))
    if entrance:
        members.append(_member(4, "entrance", 59.9002, 30.3002))
    return members


def test_parse_subway_stop_area_with_entrance_keeps_platform_underground():
    edges, types = _parse_stop_area(_stop_area_members())

    assert types[1] == "subway_station"
    assert types[2] == "subway_platform"
    assert types[4] == "subway_entry_exit"
    assert {"subway_entrance", "subway_exit", "subway_station", "boarding"} <= set(edges["type"])


def test_parse_subway_stop_area_without_entrance_marks_station_as_surface_platform():
    _, types = _parse_stop_area(_stop_area_members(entrance=False))

    # Without entrances the station itself becomes the surface access point.
    assert types[1] == "platform"
    assert types[2] == "subway_platform"


def test_parse_subway_stop_area_without_station_and_entrance_marks_platform_as_surface():
    _, types = _parse_stop_area(_stop_area_members(station=False, entrance=False))

    assert types[2] == "platform"


def test_parse_subway_stop_area_without_station_links_entrances_to_platform():
    edges, types = _parse_stop_area(_stop_area_members(station=False))

    assert types[2] == "subway_platform"
    entrance_edges = edges[edges["type"] == "subway_entrance"]
    assert set(zip(entrance_edges["u_ref"], entrance_edges["v_ref"])) == {(4, 2)}


def _route_members(*, platforms=True):
    # Two stops along one way, each with a platform 8 m aside unless platforms are disabled.
    members = [_member(301, "stop", 59.9100, 30.3102), _member(303, "stop", 59.9100, 30.3140)]
    if platforms:
        members += [_member(302, "platform", 59.91007, 30.3102), _member(304, "platform", 59.91007, 30.3140)]
    members.append(
        {
            "type": "way",
            "ref": 305,
            "role": "",
            "geometry": [{"lat": 59.9100, "lon": 30.3095}, {"lat": 59.9100, "lon": 30.3145}],
        }
    )
    return members


def _route_row(route_id, members):
    return {
        "id": route_id,
        "tags": {"ref": "M1"},
        "members": members,
        "is_stop_area": False,
        "is_stop_area_group": False,
        "is_station": False,
    }


def test_overpass_subway2edgenode_marks_route_platforms_without_station_as_surface():
    subway_data = pd.DataFrame([_stop_area_row(10, _stop_area_members()), _route_row(20, _route_members())])

    _, nodes = overpass_subway2edgenode(subway_data, LOCAL_CRS)
    types = dict(zip(nodes["node_id"], nodes["type"]))

    # Route platforms belong to no stop area, so nothing links them to a station or an entrance.
    assert types[302] == "platform"
    assert types[304] == "platform"
    # The stop-area platform is reachable through its station and stays underground.
    assert types[1] == "subway_station"
    assert types[2] == "subway_platform"


def test_overpass_subway2edgenode_parses_routes_without_stop_areas():
    # Territories where subway stop areas are not mapped produce no stop-area edges at all.
    subway_data = pd.DataFrame([_route_row(20, _route_members())])

    edges, nodes = overpass_subway2edgenode(subway_data, LOCAL_CRS)
    types = dict(zip(nodes["node_id"], nodes["type"]))

    assert types[301] == "subway"
    assert types[302] == "platform"
    assert {"boarding", "subway"} <= set(edges["type"])


def test_overpass_subway2edgenode_builds_platforms_for_stops_without_platform_members():
    # Stops without a platform are paired with a generated one, which reads the stop-area edges.
    subway_data = pd.DataFrame([_route_row(20, _route_members(platforms=False))])

    edges, nodes = overpass_subway2edgenode(subway_data, LOCAL_CRS)
    types = dict(zip(nodes["node_id"], nodes["type"]))

    assert types["from_301"] == "platform"
    assert types["from_303"] == "platform"
    assert ("301", "from_301") in {(str(u), str(v)) for u, v in zip(edges["u"], edges["v"])}
