import multiprocessing

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import MultiPolygon, Polygon
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from iduedu import config
from iduedu.enums.pt_enums import PublicTrasport
from iduedu.modules.routes_parser import parse_overpass_to_edgenode, parse_overpass_subway_data
from iduedu.utils.utils import clip_nx_graph, estimate_crs_for_bounds

from .overpass_downloaders import (
    get_4326_boundary,
    get_routes_by_poly,
)

logger = config.logger


def _graph_data_to_nx(graph_df, keep_geometry: bool = True, additional_data=None) -> nx.DiGraph:

    nodes_col = ["node_id", "point", "route", "type", "ref_id", "extra_data"]
    edges_col = ["u", "v","type","extra_data", "geometry"]
    graph_nodes = graph_df[~graph_df["node_id"].isna()][nodes_col].copy()
    graph_edges = graph_df[graph_df["node_id"].isna()][edges_col].copy()
    if additional_data is not None:
        edges_to_add, nodes_to_add = additional_data
        graph_nodes_combined = graph_nodes.merge(nodes_to_add, left_on="ref_id", right_on="ref_id", how="outer")
        for column in nodes_col:
            if column in graph_nodes_combined.columns:
                continue
            else:
                graph_nodes_combined[column] = graph_nodes_combined[f"{column}_x"].combine_first(
                    graph_nodes_combined[f"{column}_y"]
                )
        graph_nodes_combined = graph_nodes_combined[nodes_col]

        no_point_refs = graph_nodes_combined[(graph_nodes_combined['point'].isna())&(~graph_nodes_combined['ref_id'].isna())][['ref_id']].copy()
        no_point_refs = no_point_refs.merge(edges_to_add[["v_ref",'u_ref']], left_on="ref_id", right_on="v_ref")

        gnm = graph_nodes.drop_duplicates("ref_id").dropna(subset="ref_id")
        refid2nodeid = gnm.set_index("ref_id")["node_id"]

        nodeid2refid = gnm[gnm['type']=='subway_platform'].set_index("node_id")["ref_id"]


        no_point_refs["u_node_id"] = no_point_refs["u_ref"].map(refid2nodeid)
        no_point_refs = no_point_refs.dropna(subset='u_node_id').merge(graph_edges[['u','v']], left_on="u_node_id", right_on="u", how="left").drop(columns=["u"])
        no_point_refs['potential'] = no_point_refs["v"].map(nodeid2refid) # v_ref -> from... замену надо сделать #TODO

    platforms = graph_df[graph_df["type"] == "platform"].copy()
    platforms["point_group"] = platforms["point"].apply(lambda x: (round(x[0]), round(x[1])))
    platforms = platforms.groupby("point_group", as_index=False).agg(
        {"point": "first", "node_id": list, "route": list, "ref_id": "first"}
    )
    platforms["type"] = "platform"

    stops = graph_df[(graph_df["type"] != "platform") & (graph_df["u"].isna())][
        ["point", "node_id", "route", "type", "ref_id"]
    ]
    stops["point_group"] = stops["point"].apply(lambda x: (round(x[0]), round(x[1])))
    stops = stops.groupby(["point_group", "route", "type"], as_index=False).agg(
        {"point": "first", "node_id": list, "ref_id": "first"}
    )
    stops["route"] = stops["route"].apply(lambda x: [x])

    all_nodes = pd.concat([platforms, stops], ignore_index=True).drop(columns="point_group").reset_index(drop=True)
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

    edges = graph_df[~graph_df["u"].isna()][["route", "type", "u", "v", "geometry", "extra_data"]].copy()

    def calc_len_time(row):
        if row.type == "boarding":
            return 0, 0
        length = round(row.geometry.length, 3)
        return length, round(length / PublicTrasport[row.type.upper()].avg_speed, 3)

    edges[["length_meter", "time_min"]] = edges.apply(
        calc_len_time,
        axis=1,
        result_type="expand",
    )
    graph = nx.DiGraph()
    for i, node in all_nodes.iterrows():
        route = list(set(node["route"]))
        if len(route) == 1:
            route = route[0]
        graph.add_node(i, x=node["point"][0], y=node["point"][1], type=node["type"], route=route, ref_id=node["ref_id"])
    for i, edge in edges.iterrows():
        graph.add_edge(
            edge["u"],
            edge["v"],
            route=edge["route"],
            type=edge["type"],
            geometry=edge["geometry"] if keep_geometry else None,
            length_meter=edge["length_meter"],
            time_min=edge["time_min"],
            **edge["extra_data"],
        )

    return graph


def _multi_get_routes_by_poly(args):
    return get_routes_by_poly(*args)


def _multi_parse_overpass_to_edgenode(args):
    return parse_overpass_to_edgenode(*args)


def _get_public_transport_graph(
    osm_id: int,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None,
    transport_types: list[str],
    osm_edge_tags: list[str],
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
):

    polygon = get_4326_boundary(osm_id, territory)

    args_list = [(polygon, transport) for transport in transport_types]

    # Если парсим метро - ожидаем в ответе информацию о станциях
    platform_stop_data_use = False
    if "subway" in transport_types:
        platform_stop_data_use = True

    if not config.enable_tqdm_bar:
        logger.debug("Downloading pt routes")
    overpass_data = pd.concat(
        thread_map(
            _multi_get_routes_by_poly,
            args_list,
            desc="Downloading public transport routes from OSM",
            disable=not config.enable_tqdm_bar,
        ),
        ignore_index=True,
    ).reset_index(drop=True)

    if overpass_data.shape[0] == 0:
        logger.warning("No routes found for public transport.")
        return nx.Graph()

    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    # необходимые osm теги из relation маршрута
    if osm_edge_tags is None:
        needed_tags = set(config.transport_useful_edges_attr)
    else:
        needed_tags = set(osm_edge_tags)

    # Отделяем станции от маршрутов при необходимости
    if platform_stop_data_use:
        routes_data = overpass_data[
            ~((overpass_data["is_stop_area"]) | (overpass_data["is_stop_area_group"]) | (overpass_data["is_station"]))
        ].copy()
        stop_areas = overpass_data[overpass_data["is_stop_area"]].copy()
        stop_areas_group = overpass_data[overpass_data["is_stop_area_group"]].copy()
        stations_data = overpass_data[overpass_data["is_station"]].copy()
        add_data = parse_overpass_subway_data(stop_areas, stop_areas_group, stations_data, local_crs)
    else:
        routes_data = overpass_data.copy()
        add_data = None

    if not config.enable_tqdm_bar:
        logger.debug("Parsing public transport routes")
    if overpass_data.shape[0] > 100:
        # Если много маршрутов - обрабатываем в параллели
        edgenode_for_routes = process_map(
            _multi_parse_overpass_to_edgenode,
            [(row, local_crs, needed_tags) for _, row in routes_data.iterrows()],
            desc="Parsing public transport routes",
            chunksize=1,
            disable=not config.enable_tqdm_bar,
        )
    else:
        tqdm.pandas(desc="Parsing public transport routes", disable=not config.enable_tqdm_bar)
        edgenode_for_routes = [
            data
            for data in routes_data.progress_apply(
                lambda x: parse_overpass_to_edgenode(x, local_crs, needed_tags), axis=1
            ).tolist()
            if data is not None
        ]

    if len(edgenode_for_routes) == 0:
        logger.warning("No routes were parsed for public transport.")
        return nx.DiGraph()
    graph_df = pd.concat(edgenode_for_routes, ignore_index=True)
    to_return = _graph_data_to_nx(graph_df, keep_geometry=keep_edge_geometry, additional_data=add_data)
    to_return.graph["crs"] = local_crs
    to_return.graph["type"] = "public_trasport"

    if clip_by_territory:
        polygon = gpd.GeoSeries([polygon], crs=4326).to_crs(local_crs).union_all()
        to_return = clip_nx_graph(to_return, polygon)

    logger.debug("Done!")
    return to_return


def get_single_public_transport_graph(
    public_transport_type: str | PublicTrasport,
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
):
    """
    Generate a graph of a specific type of public transport network within a given territory or polygon.
    The graph can optionally be clipped by the specified bounds and retain original geometries.

    Parameters
    ----------
    public_transport_type : str | PublicTransport
        Type of public transport to generate the graph for. Examples include "bus", "tram", "subway", etc.
    osm_id : int, optional
        OpenStreetMap ID of the territory. Either this or `territory_name` must be provided.
    territory : Polygon | MultiPolygon | gpd.GeoDataFrame |, optional
        A custom polygon or MultiPolygon to define the area for the transport network. Must be in CRS 4326.
    clip_by_territory : bool, optional
        If True, clips the resulting graph to the bounds of the provided polygon. Defaults to False.
    keep_edge_geometry : bool, optional
        If True, retains the original geometry of the transport routes. Defaults to True.
    # TODO  add docstrings
    Returns
    -------
    networkx.Graph
        A public transport network graph for the specified transport type and territory or polygon.

    Raises
    ------
    ValueError
        If no valid `osm_id`, `territory_name`, or `polygon` is provided.

    Examples
    --------
    >>> bus_graph = get_single_public_transport_graph(public_transport_type="bus", osm_id=1114252)
    >>> tram_graph = get_single_public_transport_graph(territory_name="Санкт-Петербург", public_transport_type="tram")
    >>> metro_graph = get_single_public_transport_graph(polygon=poly, public_transport_type="bus", clip_by_territory=True)

    Notes
    -----
    The function uses OpenStreetMap data and processes public transport routes in parallel.
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """
    public_transport_type = (
        public_transport_type.value() if isinstance(public_transport_type, PublicTrasport) else public_transport_type
    )
    return _get_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=[public_transport_type],
        osm_edge_tags=osm_edge_tags,
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
    )


def get_all_public_transport_graph(
    *,
    osm_id: int | None = None,
    territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None,
    clip_by_territory: bool = False,
    keep_edge_geometry: bool = True,
    transport_types: list[PublicTrasport] = None,
    osm_edge_tags: list[str] | None = None,  # overrides default tags
) -> nx.Graph:
    """
    Generate a public transport network graph that includes all types of transport within a specified territory.
    The graph can optionally be clipped by bounds and retain original geometries.

    Parameters
    ----------
    osm_id : int, optional
        OpenStreetMap ID of the territory. Either this or `territory_name` must be provided.
    territory : Polygon | MultiPolygon | gpd.GeoDataFrame, optional
        A custom polygon or MultiPolygon to define the area for the transport network. Must be in CRS 4326.
    clip_by_territory : bool, optional
        If True, clips the resulting graph to the bounds of the provided polygon. Defaults to False.
    keep_edge_geometry : bool, optional
        If True, retains the original geometry of the transport routes. Defaults to True.
    transport_types: list[PublicTransport], optional
        By default `[PublicTrasport.TRAM, PublicTrasport.BUS, PublicTrasport.TROLLEYBUS, PublicTrasport.SUBWAY]`,
        can be any combination of PublicTransport Enums.
    # TODO  add docstrings
    Returns
    -------
    networkx.Graph
        A public transport network graph for all transport types (e.g., bus, tram, subway) in the specified territory.

    Raises
    ------
    ValueError
        If no valid `osm_id`, `territory_name`, or `polygon` is provided.

    Warnings
    --------
    Logs a warning if no public transport routes are found in the specified area.

    Examples
    --------
    >>> all_transport_graph = get_all_public_transport_graph(osm_id=1114252)
    >>> all_transport_graph = get_all_public_transport_graph(territory_name="Санкт-Петербург", clip_by_territory=True)
    >>> all_transport_graph = get_all_public_transport_graph(polygon=some_polygon, keep_edge_geometry=False)

    Notes
    -----
    This function retrieves routes for all public transport types using OpenStreetMap data.
    The CRS for the graph is estimated based on the bounds of the provided/downloaded polygon, stored in G.graph['crs'].
    """

    if transport_types is None:
        transport_types = [PublicTrasport.TRAM, PublicTrasport.BUS, PublicTrasport.TROLLEYBUS, PublicTrasport.SUBWAY]
    else:
        for transport_type in transport_types:
            if not isinstance(transport_type, PublicTrasport):
                raise ValueError(f"transport_type {transport_type} is not a valid transport type.")

    transports = [transport.value for transport in transport_types if isinstance(transport, PublicTrasport)]

    return _get_public_transport_graph(
        osm_id=osm_id,
        territory=territory,
        transport_types=transports,
        osm_edge_tags=osm_edge_tags,
        clip_by_territory=clip_by_territory,
        keep_edge_geometry=keep_edge_geometry,
    )
