import multiprocessing

import networkx as nx
import pandas as pd
from shapely import MultiPolygon, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from ..utils.utils import estimate_crs_for_overpass
from ..enums.enums import PublicTrasport, Transport
from .downloaders import (
    get_boundary_by_name,
    get_boundary_by_osm_id,
    get_routes_by_osm_id,
    get_routes_by_poly,
    get_routes_by_terr_name,
)
from .routes_parser import parse_overpass_to_edgenode


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
            mapping[node_id] = index

    def replace_with_mapping(value):
        return mapping.get(value)

    # Применение функции замены к каждому столбцу DataFrame
    graph_df["u"] = graph_df["u"].apply(replace_with_mapping)
    graph_df["v"] = graph_df["v"].apply(replace_with_mapping)
    graph_df["node_id"] = graph_df["node_id"].apply(replace_with_mapping)

    edges = graph_df[graph_df["desc"].isna()]

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
            geometry=str(edge["geometry"]),
        )
    return graph


def _get_multi_routes_by_poly(args):
    polygon, transport = args
    return get_routes_by_poly(polygon, transport)


def _multi_process_row(args):
    row, local_crs = args
    return parse_overpass_to_edgenode(row, local_crs)


def get_single_public_transport_graph(
    public_transport_type: str | Transport,
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
):
    if osm_id is None and territory_name is None and polygon is None:
        raise ValueError("Either osm_id or name or polygon must be specified")

    if osm_id:
        overpass_data = get_routes_by_osm_id(osm_id, public_transport_type)
    elif territory_name:
        overpass_data = get_routes_by_terr_name(territory_name, public_transport_type)
    else:
        if isinstance(polygon, MultiPolygon):
            polygon = polygon.convex_hull
        overpass_data = get_routes_by_poly(polygon, public_transport_type)
    local_crs = estimate_crs_for_overpass(overpass_data)

    if overpass_data.shape[0] > 100:
        # Если много маршрутов - обрабатываем в параллели
        n_cpus = multiprocessing.cpu_count()
        rows = [(row, local_crs) for _, row in overpass_data.iterrows()]
        chunksize = max(1, len(rows) // n_cpus)
        edgenode_for_routes = process_map(_multi_process_row, rows, desc="Parsing routes",chunksize=chunksize)
    else:
        tqdm.pandas(desc="Parsing routes data")
        edgenode_for_routes = overpass_data.progress_apply(
            lambda x: parse_overpass_to_edgenode(x, local_crs), axis=1
        ).tolist()
    graph_df = pd.concat(edgenode_for_routes, ignore_index=True)
    to_return = graph_data_to_nx(graph_df)
    to_return.graph['crs'] = local_crs
    return to_return


def get_all_public_transport_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
):

    if osm_id is None and territory_name is None and polygon is None:
        raise ValueError("Either osm_id or name or polygon must be specified")
    transports = [transport.value for transport in PublicTrasport]
    if osm_id:
        polygon = get_boundary_by_osm_id(osm_id)
    elif territory_name:
        polygon = get_boundary_by_name(territory_name)

    if isinstance(polygon, MultiPolygon):
        polygon = polygon.convex_hull

    args_list = [(polygon, transport) for transport in transports]
    overpass_data = pd.concat(
        process_map(_get_multi_routes_by_poly, args_list, desc="Downloading routes"), ignore_index=True
    ).reset_index(drop=True)
    local_crs = estimate_crs_for_overpass(overpass_data)

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
    to_return.graph["crs"] = local_crs.to_epsg()
    return to_return
