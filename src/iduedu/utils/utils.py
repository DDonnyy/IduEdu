import networkx as nx
import pandas as pd
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely import Polygon


def clip_nx_graph(graph: nx.Graph, polygon: Polygon):
    # TODO Добавить обрезку по полигону
    print(1)


def reproject_nx_graph(graph: nx.Graph, crs=None):
    """

    :param graph:
    :param crs: by default None, will be determine automatically
    :return:
    """
    print(1)


def estimate_crs_for_bounds(min_lat, min_lon, max_lat, max_lon):
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=min_lon,
            south_lat_degree=min_lat,
            east_lon_degree=max_lon,
            north_lat_degree=max_lat,
        ),
    )
    return CRS.from_epsg(utm_crs_list[0].code)


def estimate_crs_for_overpass(overpass_data):
    def find_bounds(bounds):
        df_expanded = pd.json_normalize(bounds)
        min_lat = df_expanded["minlat"].min()
        min_lon = df_expanded["minlon"].min()
        max_lat = df_expanded["maxlat"].max()
        max_lon = df_expanded["maxlon"].max()
        return min_lat, min_lon, max_lat, max_lon

    return estimate_crs_for_bounds(*find_bounds(overpass_data["bounds"]))
