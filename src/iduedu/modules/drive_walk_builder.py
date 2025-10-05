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
    simplify: bool = True,
    add_road_category: bool = True,
    clip_by_territory: bool = False,
    keep_largest_subgraph: bool = True,
    network_type: Literal["drive", "drive_service", "custom"] = "drive",
    custom_filter: str | None = None,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
    keep_edge_geometry: bool = True,
) -> nx.MultiDiGraph:

    polygon4326 = get_4326_boundary(osm_id, territory)

    filters = {
        "drive": Network.DRIVE.filter,
        "drive_service": Network.DRIVE_SERVICE.filter,
        "custom": custom_filter,
    }
    try:
        road_filter = filters[network_type]
    except KeyError:
        raise ValueError(f"Unknown road_type: {network_type!r}")
    if network_type == "custom" and road_filter is None:
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

    if clip_by_territory:
        clip_poly_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs=4326).to_crs(local_crs)
        edges = edges.clip(clip_poly_gdf, keep_geom_type=True)

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
    edges["type"] = "drive"

    graph = nx.MultiDiGraph()

    graph.add_nodes_from((i, {"x": float(x), "y": float(y)}) for i, (x, y) in enumerate(uniques))

    if add_road_category:
        needed_tags |= {"category"}
    if keep_edge_geometry:
        needed_tags |= {"geometry"}
    needed_tags |= {"length_meter", "time_min", "type"}
    edge_attr_cols = list(needed_tags)
    attrs_iter = edges[edge_attr_cols].to_dict("records")
    graph.add_edges_from((int(uu), int(vv), d) for uu, vv, d in zip(u, v, attrs_iter))

    if keep_largest_subgraph:
        graph = keep_largest_strongly_connected_component(graph)

    mapping = {old: new for new, old in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph.graph["crs"] = local_crs
    graph.graph["type"] = network_type
    logger.debug("Drive graph built.")
    return graph


def get_walk_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    simplify: bool = True,
    clip_by_territory: bool = False,
    keep_largest_subgraph: bool = True,
    walk_speed: float = 5 * 1000 / 60,  # m/min
    network_type: Literal["walk", "custom"] = "walk",
    custom_filter: str | None = None,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
    keep_edge_geometry: bool = True,
) -> nx.MultiDiGraph:

    polygon4326 = get_4326_boundary(osm_id, territory)

    filters = {
        "walk": Network.WALK.filter,
        "custom": custom_filter,
    }
    try:
        road_filter = filters[network_type]
    except KeyError:
        raise ValueError(f"Unknown road_type: {network_type!r}")
    if network_type == "custom" and road_filter is None:
        raise ValueError("For road_type='custom' you must provide custom_filter")

    logger.info("Downloading walk network via Overpass ...")
    edges, local_crs = _build_edges_from_overpass(polygon4326, road_filter, simplify=simplify)

    if osm_edge_tags is None:
        needed_tags = set(config.walk_useful_edges_attr)
    else:
        needed_tags = set(osm_edge_tags)

    tags_df = pd.DataFrame.from_records(
        ({k: v for k, v in d.items() if k in needed_tags} for d in edges["tags"]),
        index=edges.index,
    )
    edges = edges.join(tags_df)

    if clip_by_territory:
        clip_poly_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs=4326).to_crs(local_crs)
        edges = edges.clip(clip_poly_gdf, keep_geom_type=True)

    two_way = edges.copy()
    two_way.geometry = two_way.geometry.reverse()
    edges = pd.concat([edges, two_way], ignore_index=True)

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

    edges["length_meter"] = edges.geometry.length.round(3)
    edges["time_min"] = (edges["length_meter"] / float(walk_speed)).round(3)
    edges["type"] = "walk"

    # Сборка графа
    graph = nx.MultiDiGraph()
    graph.add_nodes_from((i, {"x": float(x), "y": float(y)}) for i, (x, y) in enumerate(uniques))

    edge_attrs = set(needed_tags)
    if keep_edge_geometry:
        edge_attrs |= {"geometry"}
    edge_attrs |= {"length_meter", "time_min", "type"}

    attrs_iter = edges[list(edge_attrs)].to_dict("records")
    graph.add_edges_from((int(uu), int(vv), d) for uu, vv, d in zip(u, v, attrs_iter))

    if keep_largest_subgraph:
        graph = keep_largest_strongly_connected_component(graph)

    mapping = {old: new for new, old in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    graph.graph["crs"] = local_crs
    graph.graph["walk_speed"] = float(walk_speed)
    graph.graph["type"] = network_type
    logger.debug("Walk graph built.")
    return graph
