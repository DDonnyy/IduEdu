import geopandas as gpd
import networkit as nk
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.spatial import KDTree

from iduedu import config

logger = config.logger


def get_closest_nodes(gdf_from: gpd.GeoDataFrame, to_nx_graph: nx.Graph):
    assert gdf_from.crs == CRS.from_epsg(
        to_nx_graph.graph["crs"]
    ), f'CRS mismatch , gdf_from.crs = {gdf_from.crs.to_epsg()}, graph["crs"] = {to_nx_graph.graph["crs"]}'
    mapping = dict((u, id) for (id, u) in zip(to_nx_graph.nodes(), range(to_nx_graph.number_of_nodes())))
    points = gdf_from.representative_point()
    coordinates = [(data["x"], data["y"]) for node, data in to_nx_graph.nodes(data=True)]
    tree = KDTree(coordinates)
    target_coord = [(p.x, p.y) for p in points]
    _, indices = tree.query(target_coord)
    return [mapping.get(x) for x in indices]


def get_dist_matrix(graph: nx.Graph, nodes_from: [], nodes_to: [], weight: str, dtype: np.dtype) -> pd.DataFrame:
    if graph.is_directed() & graph.is_multigraph():
        nx_graph = nx.DiGraph(graph)
    else:
        nx_graph = graph
    mapping = dict((id, u) for (id, u) in zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))
    nk_graph = nk.nxadapter.nx2nk(nx_graph, weight)
    spsp = nk.distance.SPSP(nk_graph, sources=[mapping.get(x) for x in nodes_from])
    spsp.setTargets(targets=[mapping.get(x) for x in nodes_to])
    spsp.run()
    distance_matrix = pd.DataFrame(spsp.getDistances(asarray=True), index=nodes_from, columns=nodes_to, dtype=dtype)
    return distance_matrix


def get_adj_matrix_gdf_to_gdf(
    gdf_from: gpd.GeoDataFrame,
    gft_to: gpd.GeoDataFrame,
    nx_graph: nx.Graph,
    weight: str = "length_meter",
    dtype: np.dtype = np.float16,
) -> pd.DataFrame:
    """
    Compute an adjacency matrix representing the shortest path distances between two sets of points (GeoDataFrames)
    based on a provided graph. Distances are calculated using the specified edge weight attribute.

    Parameters
    ----------
    gdf_from : gpd.GeoDataFrame
        The GeoDataFrame containing the origin points for the distance matrix calculation.
    gft_to : gpd.GeoDataFrame
        The GeoDataFrame containing the destination points for the distance matrix calculation.
    nx_graph : nx.Graph
        A NetworkX graph with geographic data where each edge has the specified `weight` (e.g., 'length_meter').
    weight : str, optional
        The edge attribute to use for calculating the shortest paths. Defaults to 'length_meter'.
    dtype : np.dtype, optional
        The data type for the adjacency matrix. Defaults to np.float16, but can be changed to avoid precision loss.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the adjacency matrix where rows correspond to `gdf_from` and columns to `gft_to`.
        Each value in the matrix represents the shortest path distance between the corresponding points,
        based on the provided graph.

    Raises
    ------
    AssertionError
        If the coordinate reference systems (CRS) of `gdf_from`, `gft_to`, and the graph do not match.

    Notes
    -----
    - The function computes the closest graph nodes for both `gdf_from` and `gft_to` before calculating the matrix.
    - If `gdf_from` and `gft_to` are equal, the distance matrix will be square.
    - Ensure that all edges in the graph have the specified `weight` attribute; otherwise, the calculation may fail.

    Examples
    --------
    >>> adj_matrix = get_adj_matrix_gdf_to_gdf(origins_gdf, destinations_gdf, graph, weight='time', dtype=np.float32)
    """

    assert gdf_from.crs == gft_to.crs == CRS.from_epsg(nx_graph.graph["crs"]), (
        f"CRS mismatch, gdf_from.crs = {gdf_from.crs.to_epsg()},"
        f" gft_to.crs = {gft_to.crs.to_epsg()},"
        f' graph["crs"] = {nx_graph.graph["crs"]}'
    )
    if gdf_from.equals(gft_to):
        closest_nodes = get_closest_nodes(gdf_from, nx_graph)
        adj_matrix = get_dist_matrix(nx_graph, closest_nodes, closest_nodes, weight, dtype)
        adj_matrix.columns = gdf_from.index
        adj_matrix.index = gdf_from.index
        return adj_matrix
    closest_nodes_from = get_closest_nodes(gdf_from, nx_graph)
    closest_nodes_to = get_closest_nodes(gft_to, nx_graph)
    adj_matrix = get_dist_matrix(nx_graph, closest_nodes_from, closest_nodes_to, weight, dtype)
    adj_matrix.columns = gft_to.index
    adj_matrix.index = gdf_from.index
    return adj_matrix
