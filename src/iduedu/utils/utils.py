import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely import Point, Polygon


def clip_nx_graph(graph: nx.Graph, polygon: Polygon) -> nx.Graph:
    """

    :param graph:
    :param polygon: should be in same crs as coordinates in graph
    :return:
    """
    crs = graph.graph["crs"]
    points = gpd.GeoDataFrame(
        data=[{"id": p_id, "geometry": Point(data["x"], data["y"])} for p_id, data in graph.nodes(data=True)], crs=crs
    ).clip(polygon, True)
    clipped = graph.subgraph(points["id"].tolist())
    return clipped


def reproject_nx_graph(graph: nx.Graph, crs_to=None):
    """

    :param graph:
    :param crs_to: by default None, will be determine automatically
    :return:
    """
    raise NotImplemented
    return None


def estimate_crs_for_bounds(minx, miny, maxx, maxy):
    x_center = np.mean([minx, maxx])
    y_center = np.mean([miny, maxy])
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=x_center,
            south_lat_degree=y_center,
            east_lon_degree=x_center,
            north_lat_degree=y_center,
        ),
    )
    crs = CRS.from_epsg(utm_crs_list[0].code)
    logger.info(f"Estimated CRS for territory {crs}")
    return crs
