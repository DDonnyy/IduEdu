import multiprocessing

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import MultiPolygon, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from .downloaders import (
    get_routes_by_poly,
    get_boundary,
)
from iduedu.modules.routes_parser import parse_overpass_to_edgenode
from iduedu.enums.pt_enums import PublicTrasport
from iduedu.utils.utils import clip_nx_graph, estimate_crs_for_bounds


def graph_data_to_nx(graph_df) -> nx.DiGraph:

    platforms = graph_df[graph_df["desc"] == "platform"]
    platforms = platforms.groupby("point").agg({"node_id": list, "route": list, "desc": "first"}).reset_index()

    stops = graph_df[graph_df["desc"] == "stop"][["point", "node_id", "route", "desc"]]
    stops["node_id"] = stops["node_id"].apply(lambda x: [x])
    stops["route"] = stops["route"].apply(lambda x: [x])
    all_nodes = pd.concat([platforms, stops], ignore_index=True).reset_index(drop=True)
    mapping = {}
    for i, row in all_nodes.iterrows():
        index = row.name
        node_id_list = row["node_id"]
        for node_id in node_id_list:
            mapping[node_id] = int(index)

    def replace_with_mapping(value):
        return mapping.get(value)

    # Применение функции замены к каждому столбцу DataFrame
    graph_df["u"] = graph_df["u"].apply(replace_with_mapping)
    graph_df["v"] = graph_df["v"].apply(replace_with_mapping)
    graph_df["node_id"] = graph_df["node_id"].apply(replace_with_mapping)

    edges = graph_df[graph_df["desc"].isna()].copy()

    def calc_len_time(row):
        if row.type == "boarding":
            return 0, 0
        length = round(row.geometry.length, 3)
        return length, round(length / PublicTrasport[row.type.upper()].avg_speed, 3)

    edges[["length", "time_min"]] = edges.apply(
        calc_len_time,
        axis=1,
        result_type="expand",
    )
    graph = nx.DiGraph()
    for i, node in all_nodes.iterrows():
        route = ",".join(map(str, set(node["route"])))
        graph.add_node(i, x=node["point"][0], y=node["point"][1], desc=node["desc"], route=route)
    for i, edge in edges.iterrows():
        graph.add_edge(
            edge["u"],
            edge["v"],
            route=edge["route"],
            type=edge["type"],
            geometry=edge["geometry"],
            length=edge["length"],
            time_min=edge["time_min"],
        )

    return graph


def _get_multi_routes_by_poly(args):
    polygon, transport = args
    return get_routes_by_poly(polygon, transport)


def _multi_process_row(args):
    row, local_crs = args
    return parse_overpass_to_edgenode(row, local_crs)


def get_single_public_transport_graph(
    public_transport_type: str | PublicTrasport,
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    clip_by_bounds: bool = False,
):
    """

    :param clip_by_bounds:
    :param public_transport_type:
    :param osm_id:
    :param territory_name:
    :param polygon: should be in valid crs (4326)
    :return:
    """
    polygon = get_boundary(osm_id, territory_name, polygon)

    overpass_data = get_routes_by_poly(polygon, public_transport_type)

    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    if overpass_data.shape[0] > 100:
        # Если много маршрутов - обрабатываем в параллели
        n_cpus = multiprocessing.cpu_count()
        rows = [(row, local_crs) for _, row in overpass_data.iterrows()]
        chunksize = max(1, len(rows) // n_cpus)
        edgenode_for_routes = process_map(_multi_process_row, rows, desc="Parsing routes", chunksize=chunksize)
    else:
        tqdm.pandas(desc="Parsing routes data")
        edgenode_for_routes = overpass_data.progress_apply(
            lambda x: parse_overpass_to_edgenode(x, local_crs), axis=1
        ).tolist()
    graph_df = pd.concat(edgenode_for_routes, ignore_index=True)
    to_return = graph_data_to_nx(graph_df)
    to_return.graph["crs"] = local_crs
    if clip_by_bounds:
        polygon = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).unary_union
        return clip_nx_graph(to_return, polygon)
    return to_return


def get_all_public_transport_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    clip_by_bounds: bool = False,
):

    polygon = get_boundary(osm_id, territory_name, polygon)

    transports = [transport.value for transport in PublicTrasport]
    args_list = [(polygon, transport) for transport in transports]

    overpass_data = pd.concat(
        process_map(_get_multi_routes_by_poly, args_list, desc="Downloading routes"), ignore_index=True
    ).reset_index(drop=True)

    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    if overpass_data.shape[0] > 100:
        # Если много маршрутов - обрабатываем в параллели
        rows = [(row, local_crs) for _, row in overpass_data.iterrows()]
        n_cpus = multiprocessing.cpu_count()
        chunksize = max(1, len(rows) // n_cpus)
        edgenode_for_routes = process_map(_multi_process_row, rows, desc="Parsing routes", chunksize=chunksize)
        # pandarallel usecase
        # edgenode_for_routes = overpass_data.parallel_apply(lambda x: parse_overpass_to_edgenode(x, local_crs), axis=1)
    else:
        tqdm.pandas(desc="Parsing routes data...")
        edgenode_for_routes = overpass_data.progress_apply(
            lambda x: parse_overpass_to_edgenode(x, local_crs), axis=1
        ).tolist()
    graph_df = pd.concat(edgenode_for_routes, ignore_index=True)
    to_return = graph_data_to_nx(graph_df)
    to_return.graph["crs"] = local_crs
    if clip_by_bounds:
        polygon = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).unary_union
        return clip_nx_graph(to_return, polygon)
    return to_return
