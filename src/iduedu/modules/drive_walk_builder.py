import re

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import MultiPolygon, Polygon, unary_union
from loguru import logger
from iduedu.enums.drive_enums import HighwayType
from iduedu.modules.downloaders import get_boundary_by_name, get_boundary_by_osm_id
from iduedu.utils.utils import estimate_crs_for_bounds
from tqdm.auto import tqdm

base_filter = (
    "['highway'~'motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link"
    "|trunk_link|primary_link|secondary_link|tertiary_link|living_street']"
)


def highway_type_to_reg(highway_type) -> int:
    """
     Determine the reg_status based on highway type.

    Parameters:
    highway_type: The type of highway.

    Returns:
    int: The REG value.
    """
    try:
        if isinstance(highway_type, list):
            reg_values = [HighwayType[ht.upper()].reg_status for ht in highway_type]
            return min(reg_values)
        return HighwayType[highway_type.upper()].reg_status
    except KeyError:
        return 3


def determine_reg(name_roads, highway_type=None) -> int:
    """
    Determine the reg_status based on road_name.

    Parameters:
    name_roads: The input name_roads.
    highway_type: The type of highway.

    Returns:
    int: The value of REG.
    """

    if isinstance(name_roads, list):
        for item in name_roads:
            if re.match(r"^[МАР]", str(item)):
                return 1
            elif re.match(r"^\d.*[A-Za-zА-Яа-я]", str(item)):
                return 2
        return 3
    elif pd.isna(name_roads):
        # Выставление значения по типу дороги, если значение NaN
        if highway_type:
            return highway_type_to_reg(highway_type)
        return 3
    if re.match(r"^[МАР]", str(name_roads)):
        return 1
    elif re.match(r"^\d.*[A-Za-zА-Яа-я]", str(name_roads)):
        return 2
    else:
        return 3


def get_max_speed(highway_types) -> float:
    """
    Получение максимальной скорости для типов дорог.

    Parameters:
    highway_types: Тип(ы) дорог.

    Returns:
    float: Максимальная скорость.
    """

    # Проверяем, является ли highway_types списком.
    try:

        if isinstance(highway_types, list):
            return max([HighwayType[ht.upper()].max_speed for ht in highway_types])
        else:
            return HighwayType[highway_types.upper()].max_speed
    except KeyError:
        return 40 * 1000 / 60


def get_drive_graph_by_poly(polygon: Polygon | MultiPolygon, filter: str = None) -> nx.MultiDiGraph:
    if not filter:
        filter = base_filter
    if isinstance(polygon, MultiPolygon):
        polygon = unary_union(polygon)
        if isinstance(polygon, MultiPolygon):
            polygon = polygon.convex_hull

    graph = ox.graph_from_polygon(
        polygon,
        network_type="drive",
        custom_filter=filter,
        truncate_by_edge=False,
    )
    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    nodes, edges = ox.graph_to_gdfs(graph)
    edges: gpd.GeoDataFrame
    edges.reset_index(inplace=True)
    edges["reg"] = edges.apply(lambda row: determine_reg(row["ref"], row["highway"]), axis=1)

    nodes.to_crs(local_crs, inplace=True)
    nodes[["x", "y"]] = nodes.apply(lambda row: (row.geometry.x, row.geometry.y), axis=1, result_type="expand")
    edges.to_crs(local_crs, inplace=True)

    edges["maxspeed"] = edges["highway"].apply(lambda x: get_max_speed(x))

    edges[["length", "time_min"]] = edges.apply(
        lambda row: (round(row.geometry.length, 3), round(row.geometry.length / row.maxspeed, 3)),
        axis=1,
        result_type="expand",
    )
    edges = edges[
        [
            "u",
            "v",
            "key",
            # "highway",
            # "maxspeed",
            "reg",
            # "ref",
            "length",
            "time_min",
            "geometry",
        ]
    ]
    edges.set_index(["u", "v", "key"], inplace=True)
    graph = ox.graph_from_gdfs(nodes, edges)
    graph.graph["crs"] = local_crs
    return graph


def get_drive_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
):
    if osm_id is None and territory_name is None and polygon is None:
        raise ValueError("Either osm_id or name or polygon must be specified")
    if osm_id:
        polygon: Polygon = get_boundary_by_osm_id(osm_id)
    elif territory_name:
        polygon: Polygon = get_boundary_by_name(territory_name)

    if isinstance(polygon, MultiPolygon):
        polygon: Polygon = polygon.convex_hull

    return get_drive_graph_by_poly(polygon)


def get_walk_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    walk_speed: float = 5 * 1000 / 60,
):
    if osm_id is None and territory_name is None and polygon is None:
        raise ValueError("Either osm_id or name or polygon must be specified")
    if osm_id:
        polygon: Polygon = get_boundary_by_osm_id(osm_id)
    elif territory_name:
        polygon: Polygon = get_boundary_by_name(territory_name)

    if isinstance(polygon, MultiPolygon):
        polygon: Polygon = polygon.convex_hull

    logger.debug(f"Downloading walk graph from OSM ...")
    graph = ox.graph_from_polygon(polygon, network_type="walk", truncate_by_edge=False, simplify=True)
    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    logger.debug(f"Calculating the weights of the graph ...")
    nodes, edges = ox.graph_to_gdfs(graph)
    nodes.to_crs(local_crs, inplace=True)
    nodes[["x", "y"]] = nodes.apply(lambda row: (row.geometry.x, row.geometry.y), axis=1, result_type="expand")
    edges.reset_index(inplace=True)
    edges.to_crs(local_crs, inplace=True)

    edges[["length", "time_min"]] = edges.apply(
        lambda row: (round(row.geometry.length, 3), round(row.geometry.length / walk_speed, 3)),
        axis=1,
        result_type="expand",
    )
    edges = edges[
        [
            "u",
            "v",
            "key",
            "length",
            "time_min",
            "geometry",
        ]
    ]
    edges.set_index(["u", "v", "key"], inplace=True)
    graph = ox.graph_from_gdfs(nodes, edges)
    graph.graph["crs"] = local_crs
    graph.graph["walk_speed"] = walk_speed
    return graph
