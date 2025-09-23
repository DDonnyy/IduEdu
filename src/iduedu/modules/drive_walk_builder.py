from typing import Literal

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import line_merge, MultiLineString, LineString, Point, Polygon
from shapely.geometry.multipolygon import MultiPolygon

from iduedu.enums.network_enums import Network
from iduedu.enums.highway_enums import HighwayType
from iduedu.modules.overpass_downloaders import get_network_by_filters
from iduedu.utils.utils import estimate_crs_for_bounds, keep_largest_strongly_connected_component
from iduedu.modules.overpass_downloaders import get_4326_boundary
from iduedu import config

logger = config.logger


def get_highway_properties(highway) -> tuple[str, float]:
    default = (HighwayType.UNCLASSIFIED.reg_status, HighwayType.UNCLASSIFIED.max_speed)

    if not highway:
        return default

    highway_list = highway if isinstance(highway, list) else [highway]
    reg_values, speed_values = [], []
    for ht in highway_list:
        try:
            enum_value = HighwayType[ht.upper()]
        except KeyError:
            continue
        reg_values.append(enum_value.reg_status)
        speed_values.append(enum_value.max_speed)

    if not reg_values or not speed_values:
        return default

    rank = {"local": 0, "regional": 1, "federal": 2}
    lowest_reg = min(reg_values, key=lambda s: rank.get(s, 999))

    min_speed = min(speed_values) if speed_values else default[1]

    return lowest_reg, min_speed


def _build_edges_from_overpass(polygon: Polygon, way_filter: str, simplify: bool = True) -> tuple[GeoDataFrame, CRS]:

    data = get_network_by_filters(polygon, way_filter)
    ways = data[data["type"] == "way"].copy()

    # Собираем координаты каждой линии (lon, lat)
    coords_list = [np.asarray([(p["lon"], p["lat"]) for p in pts], dtype="f8") for pts in ways["geometry"].values]

    # сегментация на отрезки
    starts = np.concatenate([a[:-1] for a in coords_list], axis=0)
    ends = np.concatenate([a[1:] for a in coords_list], axis=0)

    lengths = np.array([a.shape[0] for a in coords_list], dtype=int)
    seg_counts = np.maximum(lengths - 1, 0)
    way_idx = np.repeat(ways.index.values, seg_counts)

    geoms = [LineString([tuple(s), tuple(e)]) for s, e in zip(starts, ends)]

    # локальная проекция
    local_crs = estimate_crs_for_bounds(*polygon.bounds)

    edges = gpd.GeoDataFrame({"way_idx": way_idx}, geometry=geoms, crs=4326).to_crs(local_crs)
    edges = edges.join(ways[["id", "tags"]], on="way_idx")

    if simplify:
        # сшиваем ребра и переносим атрибуты через midpoints → nearest
        merged = list(line_merge(MultiLineString(edges.geometry.to_list()), directed=True).geoms)
        lines = gpd.GeoDataFrame(geometry=merged, crs=local_crs)

        mid = lines.copy()
        mid.geometry = mid.interpolate(lines.length / 2)

        # TODO Вот тут подумать надо, теряются аттрибуты, надо ли складывать их "по умному", в листы.
        joined = gpd.sjoin_nearest(mid[["geometry"]], edges, how="left", max_distance=1)
        joined = joined.reset_index().drop_duplicates(subset="index").set_index("index")
        lines = lines.join(joined[["tags", "id", "way_idx"]])

        edges = lines

    return edges, local_crs


def get_drive_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
    keep_largest_subgraph: bool = True,
    simplify: bool = True,
    road_type: Literal["drive", "drive_service", "custom"] = "drive",
    custom_filter: str | None = None,
    add_road_category: bool = True,
    keep_edge_geometry: bool = True,
    clip_by_territory: bool = False, # TODO клипать или нет, вроде просто .clip гдфку
) -> nx.MultiDiGraph:

    polygon4326 = get_4326_boundary(osm_id, territory)

    filters = {
        "drive": Network.DRIVE.filter,
        "drive_service": Network.DRIVE_SERVICE.filter,
        "custom": custom_filter,
    }
    try:
        road_filter = filters[road_type]
    except KeyError:
        raise ValueError(f"Unknown road_type: {road_type!r}")
    if road_type == "custom" and road_filter is None:
        raise ValueError("For road_type='custom' you must provide custom_filter")

    logger.info("Downloading drive network via Overpass ...")
    edges, local_crs = _build_edges_from_overpass(polygon4326, road_filter, simplify=simplify)

    if osm_edge_tags is None:
        needed_tags = set(config.drive_useful_edges_attr)
    else:
        needed_tags = set(osm_edge_tags)

    tags_to_retrieve = set(needed_tags) | {"oneway", "maxspeed", "highway"}

    tags_df = pd.DataFrame.from_records(
        ({k: v for k, v in d.items() if k in tags_to_retrieve} for d in edges["tags"]),
        index=edges.index,
    )
    edges = edges.join(tags_df)

    # двусторонние — дублируем с реверсом
    two_way = edges[edges["oneway"] != "yes"].copy()
    two_way.geometry = two_way.geometry.reverse()
    edges = pd.concat([edges, two_way], ignore_index=True)

    coords = edges.geometry.get_coordinates().to_numpy()
    counts = edges.geometry.count_coordinates()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1
    starts = coords[first_idx]
    ends = coords[last_idx]
    edges["start"] = list(map(tuple, starts))
    edges["end"] = list(map(tuple, ends))

    all_endpoints = pd.Index(edges["start"]).append(pd.Index(edges["end"]))
    labels, uniques = pd.factorize(all_endpoints)
    n = len(edges)
    u = labels[:n]
    v = labels[n:]
    edges["u"] = u
    edges["v"] = v

    edges[["category", "maxspeed_mpm"]] = edges["highway"].apply(lambda h: pd.Series(get_highway_properties(h)))

    maxspeed_osm_mpm = (pd.to_numeric(edges["maxspeed"], errors="coerce") * 1000.0 / 60.0).round(3)

    edges["speed_mpm"] = maxspeed_osm_mpm.fillna(edges["maxspeed_mpm"])

    edges["length_meter"] = edges.geometry.length.round(3)
    edges["time_min"] = (edges["length_meter"] / edges["speed_mpm"]).round(3)

    graph = nx.MultiDiGraph()

    graph.add_nodes_from((i, {"x": float(x), "y": float(y)}) for i, (x, y) in enumerate(uniques))

    if add_road_category:
        needed_tags |= {"category"}
    if keep_edge_geometry:
        needed_tags |= {"geometry"}
    needed_tags |= {"length_meter", "time_min"}
    edge_attr_cols = list(needed_tags)
    attrs_iter = edges[edge_attr_cols].to_dict("records")
    graph.add_edges_from((int(uu), int(vv), d) for uu, vv, d in zip(u, v, attrs_iter))

    if not keep_largest_subgraph:
        graph = keep_largest_strongly_connected_component(graph)

    mapping = {old: new for new, old in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph.graph["crs"] = local_crs
    graph.graph["type"] = road_type
    logger.debug("Drive graph built.")
    return graph


def get_walk_graph(
    osm_id: int | None = None,
    polygon=None,
    walk_speed: float = 5 * 1000 / 60,  # m/min
    retain_all: bool = False,
    simplify: bool = True,
    road_filter: str | None = None,
) -> nx.MultiDiGraph:

    polygon = get_4326_boundary(osm_id, polygon)
    if road_filter is None:
        road_filter = Network.WALK.filter

    logger.info("Downloading walk network via Overpass ...")
    edges, local_crs = _build_edges_from_overpass(polygon, road_filter, simplify=simplify)

    # длина/время
    edges["length_meter"] = edges.geometry.length.round(3)
    edges["time_min"] = (edges["length_meter"] / walk_speed).round(3)

    # u/v и узлы
    coords = edges.geometry.get_coordinates().to_numpy()
    counts = edges.geometry.count_coordinates()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1
    starts = coords[first_idx]
    ends = coords[last_idx]
    edges["start"] = list(map(tuple, starts))
    edges["end"] = list(map(tuple, ends))

    all_endpoints = pd.Index(edges["start"]).append(pd.Index(edges["end"]))
    labels, uniques = pd.factorize(all_endpoints)
    n = len(edges)
    u = labels[:n]
    v = labels[n:]

    G = nx.MultiDiGraph()
    for i, (x, y) in enumerate(uniques):
        G.add_node(i, x=float(x), y=float(y), geometry=Point(x, y))

    attrs_iter = edges[["length_meter", "time_min", "geometry"]].to_dict("records")
    G.add_edges_from((int(uu), int(vv), d) for uu, vv, d in zip(u, v, attrs_iter))

    if not retain_all:
        G = keep_largest_strongly_connected_component(G)

    mapping = {old: new for new, old in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    G.graph["crs"] = local_crs
    G.graph["walk_speed"] = walk_speed
    G.graph["type"] = "walk"
    logger.debug("Walk graph built.")
    return G
