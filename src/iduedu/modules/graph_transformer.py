import networkx as nx
import pandas as pd
import geopandas as gpd
from loguru import logger
from shapely.geometry.point import Point


def _edges_to_gdf(G: nx.MultiDiGraph, crs: int) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.
    """

    edges_list = []

    for data in list(G.edges.data()):
        current_dict = {
            'id': (data[0], data[1])
        }
        current_dict.update(data[2])
        edges_list.append(current_dict)

    edges_gdf = gpd.GeoDataFrame(
        data=edges_list, geometry='geometry', crs=crs
    )

    return edges_gdf


def _nodes_to_gdf(G: nx.MultiDiGraph, crs: int) -> gpd.GeoDataFrame:
    """
    Converts nx graph to gpd.GeoDataFrame as nodes.
    """

    nodes_list = []
    for data in list(G.nodes.data()):
        current_dict = {
            'id': data[0]
        }
        geometry = Point([data[1]['x'], data[1]['y']])
        current_dict['geometry'] = geometry
        current_dict.update(data[1])
        current_dict.pop('x')
        current_dict.pop('y')
        nodes_list.append(current_dict)

    nodes_gdf = gpd.GeoDataFrame(
        data=nodes_list, geometry='geometry', crs=crs
    )

    return nodes_gdf


def graph_to_gdf(
        G: nx.MultiDiGraph,
        crs: int,
        edges: bool = True,
        nodes: bool = True,
) -> gpd.GeoDataFrame | None:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The graph to convert.
    crs: int
        Crs with which graph is provided.
    edges: bool, default to True
        Keep edges in GoeDataFrame.
    nodes: bool, default to True
        Keep nodes in GoeDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        Graph representation in GeoDataFrame format
    """

    if not edges and not nodes:
        logger.debug("Neither edges or nodes were selected, graph_to_gdf returning None")
        return None
    else:
        if nodes and not edges:
            nodes_gdf = _nodes_to_gdf(G, crs)
            return nodes_gdf
        elif not nodes and edges:
            edges_gdf = _edges_to_gdf(G, crs)
            return  edges_gdf
        else:
            nodes_gdf = _nodes_to_gdf(G, crs)
            edges_gdf = _edges_to_gdf(G, crs)
            full_gdf = pd.concat([nodes_gdf, edges_gdf])
            return  full_gdf