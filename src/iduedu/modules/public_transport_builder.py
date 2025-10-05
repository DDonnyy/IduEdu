import time

import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
from shapely import MultiPolygon, Polygon, LineString
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
    def avg_point(points):
        arr = np.asarray([p for p in points if p is not None], dtype=float)
        return tuple(arr.mean(axis=0)) if len(arr) else None

    def merge_dicts_last(dicts):
        out = {}
        for d in dicts.dropna():
            for k, v in d.items():
                out[k] = v
        return out

    nodes_col = ["node_id", "point", "route", "type", "ref_id", "extra_data"]
    edges_col = ["u", "v", "type", "extra_data", "route", "geometry"]
    graph_nodes = graph_df[~graph_df["node_id"].isna()][nodes_col].copy()
    graph_edges = graph_df[graph_df["node_id"].isna()][edges_col].copy()

    if additional_data is not None:
        additional_edges, additional_nodes = additional_data
        graph_nodes_combined = graph_nodes.merge(additional_nodes, left_on="ref_id", right_on="ref_id", how="outer")
        for column in nodes_col:
            if column in graph_nodes_combined.columns:
                continue
            else:
                graph_nodes_combined[column] = graph_nodes_combined[f"{column}_y"].combine_first(
                    graph_nodes_combined[f"{column}_x"]
                )
        graph_nodes_combined = graph_nodes_combined[nodes_col]

        no_point_refs = graph_nodes_combined[
            (graph_nodes_combined["point"].isna()) & (~graph_nodes_combined["ref_id"].isna())
        ][["ref_id"]].copy()
        if len(no_point_refs) > 0:
            no_point_refs = no_point_refs.merge(
                additional_edges[["v_ref", "u_ref"]], left_on="ref_id", right_on="v_ref"
            )[["v_ref", "u_ref"]].drop_duplicates(subset="u_ref")
            no_point_refs = no_point_refs.merge(
                graph_nodes[["node_id", "type", "ref_id"]], left_on="u_ref", right_on="ref_id"
            ).drop(columns=["ref_id"])
            no_point_refs = no_point_refs.merge(
                graph_edges[["u", "v"]], left_on="node_id", right_on="u", how="left"
            ).drop(columns=["u"])
            no_point_refs = no_point_refs.merge(
                graph_nodes[["node_id", "type", "ref_id"]].add_prefix("potential_"),
                left_on="v",
                right_on="potential_node_id",
            )
            no_point_refs = no_point_refs[no_point_refs["potential_type"] == "platform"]
            remap_refs = no_point_refs.set_index("potential_ref_id")["v_ref"].to_dict()
            s = graph_nodes_combined["ref_id"].copy()
            mapped = s.map(remap_refs)
            mask = mapped.notna()
            graph_nodes_combined.loc[mask, "ref_id"] = mapped[mask]
            graph_nodes_combined.loc[mask, "type"] = "subway_platform"

            graph_nodes = graph_nodes_combined.dropna(subset=["node_id", "point"], how="all").copy()
            graph_nodes["route"] = graph_nodes["route"].fillna("subway_transit")
            # TODO delete duplicated platform node

    platforms = graph_nodes[graph_nodes["type"] == "platform"].copy()
    platforms["point_group"] = platforms["point"].apply(lambda p: (round(p[0]), round(p[1])))
    platforms = platforms.groupby("point_group", as_index=False).agg(
        point=("point", "first"),
        node_id=("node_id", lambda s: tuple(s.dropna())),
        route=("route", lambda s: tuple(s)),
        ref_id=("ref_id", lambda s: tuple(s.dropna())),
        extra_data=("extra_data", merge_dicts_last),
    )
    platforms["type"] = "platform"

    not_platforms = graph_nodes[graph_nodes["type"] != "platform"].copy()
    not_platforms["point_group"] = not_platforms["point"].apply(lambda p: (round(p[0]), round(p[1])))
    not_platforms = not_platforms.groupby(["point_group", "route", "type"], as_index=False).agg(
        point=("point", "first"),
        node_id=("node_id", lambda s: tuple(s.dropna())),
        ref_id=("ref_id", lambda s: tuple(set(s.dropna()))),
        extra_data=("extra_data", merge_dicts_last),
    )
    not_platforms = not_platforms.groupby(["ref_id", "route", "type"], as_index=False).agg(
        point=("point", avg_point),
        node_id=("node_id", "sum"),
        extra_data=("extra_data", merge_dicts_last),
    )

    all_nodes = pd.concat([platforms, not_platforms], ignore_index=True)

    map_nodeid_to_idx = {}
    map_refid_to_idx = {}

    for idx, row in all_nodes.iterrows():
        for nid in row["node_id"] or []:
            map_nodeid_to_idx[nid] = idx
        rids = row["ref_id"]
        if not isinstance(rids, (list, tuple)):
            rids = [rids]
        for rid in rids or []:
            map_refid_to_idx[rid] = idx

    edges_existing = graph_edges[["route", "type", "u", "v", "geometry", "extra_data"]].copy()
    edges_existing["u"] = edges_existing["u"].map(map_nodeid_to_idx)
    edges_existing["v"] = edges_existing["v"].map(map_nodeid_to_idx)
    edges_existing = edges_existing.dropna(subset=["u", "v"]).copy()

    if additional_data is not None:
        additional_edges["u"] = additional_edges["u_ref"].map(map_refid_to_idx)
        additional_edges["v"] = additional_edges["v_ref"].map(map_refid_to_idx)
        additional_edges = additional_edges.dropna(subset=["u", "v"]).copy()

        def _ensure_geom(row):
            if row.get("geometry") is not None:
                return row["geometry"]
            pu = all_nodes.loc[int(row["u"]), "point"]
            pv = all_nodes.loc[int(row["v"]), "point"]
            return LineString([pu, pv])

        additional_edges["geometry"] = additional_edges.apply(_ensure_geom, axis=1)
    else:
        additional_edges = pd.DataFrame()

    edges = pd.concat([edges_existing, additional_edges], ignore_index=True)

    if "length_meter" not in edges.columns:
        edges["length_meter"] = np.nan
    if "time_min" not in edges.columns:
        edges["time_min"] = np.nan

    def calc_len_time(row):
        if row.type == "boarding":
            return 0.0, 0.0
        geom = row.geometry
        if geom is None:
            return 0.0, 0.0
        length = float(round(geom.length, 3))
        speed = PublicTrasport[row.type.upper()].avg_speed
        return length, float(round(length / speed, 3))

    mask_missing = edges["length_meter"].isna() | edges["time_min"].isna()

    vals = edges.loc[mask_missing].apply(calc_len_time, axis=1, result_type="expand")
    vals.columns = ["length_meter", "time_min"]
    edges.loc[mask_missing, ["length_meter", "time_min"]] = vals

    graph = nx.DiGraph()

    for idx, node in all_nodes.iterrows():
        route = list(set(node["route"])) if isinstance(node["route"], tuple) else [node["route"]]
        if len(route) == 1:
            route = route[0]
        graph.add_node(
            idx,
            x=float(node["point"][0]),
            y=float(node["point"][1]),
            type=node["type"],
            route=route,
            ref_id=(node["ref_id"][0] if isinstance(node["ref_id"], tuple) and node["ref_id"] else node["ref_id"]),
            **(node["extra_data"] if isinstance(node["extra_data"], dict) else {}),
        )

    for _, e in edges.iterrows():
        graph.add_edge(
            int(e["u"]),
            int(e["v"]),
            route=e["route"],
            type=e["type"],
            geometry=(e["geometry"] if keep_geometry else None),
            length_meter=e["length_meter"],
            time_min=e["time_min"],
            **(e["extra_data"] if isinstance(e["extra_data"], dict) else {}),
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
        for column in ["is_stop_area", "is_stop_area_group", "is_station"]:
            overpass_data[column] = overpass_data[column].astype("boolean").fillna(False)
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
