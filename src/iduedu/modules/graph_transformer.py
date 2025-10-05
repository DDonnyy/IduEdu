import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from shapely import LineString
from shapely.geometry.point import Point


def _edges_to_gdf(graph: nx.Graph, crs) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.
    """
    graph_df = pd.DataFrame(list(graph.edges(data=True)), columns=["u", "v", "data"])
    edge_data_expanded = pd.json_normalize(graph_df["data"])
    graph_df = pd.concat([graph_df.drop(columns=["data"]), edge_data_expanded], axis=1)
    graph_df = gpd.GeoDataFrame(graph_df, geometry="geometry", crs=crs).set_index(["u", "v"])
    graph_df["geometry"] = graph_df["geometry"].fillna(LineString())
    return graph_df


def _nodes_to_gdf(graph: nx.Graph, crs: int) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as nodes.
    """

    ind, data = zip(*graph.nodes(data=True))
    node_geoms = (Point(d["x"], d["y"]) for d in data)
    gdf_nodes = gpd.GeoDataFrame(data, index=ind, crs=crs, geometry=list(node_geoms))

    return gdf_nodes


def _restore_edges_geom(nodes_gdf, edges_gdf) -> gpd.GeoDataFrame:
    edges_wout_geom = edges_gdf[edges_gdf["geometry"].is_empty].reset_index()
    edges_wout_geom["geometry"] = [
        LineString((s, e))
        for s, e in zip(
            nodes_gdf.loc[edges_wout_geom["u"], "geometry"], nodes_gdf.loc[edges_wout_geom["v"], "geometry"]
        )
    ]
    edges_wout_geom.set_index(["u", "v"], inplace=True)
    edges_gdf.update(edges_wout_geom)
    return edges_gdf


def graph_to_gdf(
    graph: nx.MultiDiGraph, edges: bool = True, nodes: bool = True, restore_edge_geom=False
) -> gpd.GeoDataFrame | None:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The graph to convert.
    edges: bool, default to True
        Keep edges in GoeDataFrame.
    nodes: bool, default to True
        Keep nodes in GoeDataFrame.
    restore_edge_geom: bool, default to False
        if True, will try to restore edge geometry from nodes.
    Returns
    -------
    gpd.GeoDataFrame
        Graph representation in GeoDataFrame format
    """
    try:
        crs = graph.graph["crs"]
    except KeyError as exc:
        raise ValueError("Graph does not have crs attribute and no crs was provided") from exc
    if not edges and not nodes:
        logger.debug("Neither edges or nodes were selected, graph_to_gdf returning None")
        return None
    if nodes and not edges:
        nodes_gdf = _nodes_to_gdf(graph, crs)
        return nodes_gdf
    if not nodes and edges:
        edges_gdf = _edges_to_gdf(graph, crs)
        if restore_edge_geom:
            nodes_gdf = _nodes_to_gdf(graph, crs)
            edges_gdf = _restore_edges_geom(nodes_gdf, edges_gdf)
        return edges_gdf

    nodes_gdf = _nodes_to_gdf(graph, crs)
    edges_gdf = _edges_to_gdf(graph, crs)
    if restore_edge_geom:
        edges_gdf = _restore_edges_geom(nodes_gdf, edges_gdf)
    full_gdf = pd.concat([nodes_gdf, edges_gdf])
    return full_gdf

def gdf_to_graph(
    gdf: gpd.GeoDataFrame, project_gdf_attr=True, reproject_to_utm_crs=True, speed=5, check_intersections=True
) -> nx.DiGraph:
    """
    Converts a GeoDataFrame of LineStrings into a directed graph (nx.DiGraph).

    This function transforms a set of linear geometries (which may or may not form a planar graph)
    into a directed graph where each edge corresponds to a LineString (or its segment) from the GeoDataFrame.
    Intersections are optionally checked and merged. Attributes from the original GeoDataFrame
    can be projected onto the graph edges using spatial matching.

    Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame containing at least one LineString geometry.
        project_gdf_attr (bool): If True, attributes from the input GeoDataFrame will be spatially
            projected onto the resulting graph edges. This can be an expensive operation for large datasets.
        reproject_to_utm_crs (bool): If True, reprojects the GeoDataFrame to the estimated local UTM CRS to ensure
            accurate edge length calculations in meters. If False, edge lengths are still computed in UTM CRS,
            but the final graph will remain in the original CRS of the input GeoDataFrame.
        speed (float): Assumed travel speed in km/h used to compute edge traversal time (in minutes).
        check_intersections (bool): If True, merges geometries to ensure topological correctness.
            Can be disabled if the input geometries already form a proper planar graph with no unintended intersections.

    Returns:
        (nx.DiGraph): A directed graph where each edge corresponds to a line segment from the input GeoDataFrame.
            Edge attributes include geometry, length in meters, travel time (in minutes), and any additional projected
            attributes from the original GeoDataFrame.

    Raises:
        ValueError: If the input GeoDataFrame contains no valid LineStrings.
    """

    def unique_list(agg_vals):
        agg_vals = list(set(agg_vals.dropna()))
        if len(agg_vals) == 1:
            return agg_vals[0]
        return agg_vals

    original_crs = gdf.crs
    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    gdf = gdf.explode(ignore_index=True)
    gdf = gdf[gdf.geom_type == "LineString"]

    if len(gdf) == 0:
        raise ValueError("Provided GeoDataFrame contains no valid LineStrings")

    if check_intersections:
        lines = line_merge(node(MultiLineString(gdf.geometry.to_list())))
    else:
        lines = line_merge(MultiLineString(gdf.geometry.to_list()))

    lines = gpd.GeoDataFrame(geometry=list(lines.geoms), crs=gdf.crs)

    if len(gdf.columns) > 1 and project_gdf_attr:
        lines_centroids = lines.copy()
        lines_centroids.geometry = lines_centroids.apply(
            lambda row: row.geometry.line_interpolate_point(row.geometry.length / 2), axis=1
        ).buffer(0.05, resolution=2)
        lines_with_attrs = gpd.sjoin(lines_centroids, gdf, how="left", predicate="intersects")
        aggregated_attrs = (
            lines_with_attrs.drop(columns=["geometry", "index_right"])  # убрать геометрию буфера
            .groupby(lines_with_attrs.index)
            .agg(unique_list)
        )
        lines = pd.concat([lines, aggregated_attrs], axis=1)

    lines["length_meter"] = np.round(lines.length, 2)
    if not reproject_to_utm_crs:
        lines = lines.to_crs(original_crs)

    coords = lines.geometry.get_coordinates()
    coords_grouped_by_index = coords.reset_index(names="old_index").groupby("old_index")
    start_coords = coords_grouped_by_index.head(1).apply(lambda a: (a.x, a.y), axis=1).rename("start")
    end_coords = coords_grouped_by_index.tail(1).apply(lambda a: (a.x, a.y), axis=1).rename("end")
    coords = pd.concat([start_coords.reset_index(), end_coords.reset_index()], axis=1)[["start", "end"]]
    lines = pd.concat([lines, coords], axis=1)
    unique_coords = pd.concat([coords["start"], coords["end"]], ignore_index=True).unique()
    coord_to_index = {coord: idx for idx, coord in enumerate(unique_coords)}

    lines["u"] = lines["start"].map(coord_to_index)
    lines["v"] = lines["end"].map(coord_to_index)

    speed = speed * 1000 / 60
    lines["time_min"] = np.round(lines["length_meter"] / speed, 2)

    graph = nx.Graph()
    for coords, node_id in coord_to_index.items():
        x, y = coords
        graph.add_node(node_id, x=float(x), y=float(y))

    columns_to_attr = lines.columns.difference(["start", "end", "u", "v"])
    for _, row in lines.iterrows():
        edge_attrs = {}
        for col in columns_to_attr:
            edge_attrs[col] = row[col]
        graph.add_edge(row.u, row.v, **edge_attrs)

    graph.graph["crs"] = lines.crs
    graph.graph["speed m/min"] = speed
    return nx.DiGraph(graph)
