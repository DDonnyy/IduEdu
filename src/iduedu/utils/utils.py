import geopandas as gpd
import networkx as nx
import numpy as np
from loguru import logger
from pyproj import CRS
from pyproj.aoi import AreaOfInterest

# pylint: disable=no-name-in-module
from pyproj.database import query_utm_crs_info
from shapely import Point, Polygon


def clip_nx_graph(graph: nx.Graph, polygon: Polygon) -> nx.Graph:
    crs = graph.graph["crs"]
    points = gpd.GeoDataFrame(
        data=[{"id": p_id, "geometry": Point(data["x"], data["y"])} for p_id, data in graph.nodes(data=True)], crs=crs
    ).clip(polygon, True)
    clipped = graph.subgraph(points["id"].tolist())
    return clipped


def _fmt_top_sizes(sizes, top_k: int = 5) -> str:
    ss = sorted(sizes, reverse=True)
    if len(ss) <= top_k:
        return "[" + ", ".join(map(str, ss)) + "]"
    return "[" + ", ".join(map(str, ss[:top_k])) + ", …]"


def keep_largest_strongly_connected_component(
    graph: nx.DiGraph,
    *,
    top_k_wcc_sizes: int = 5,
) -> nx.DiGraph:
    graph = graph.copy()

    weakly_connected_components = list(nx.weakly_connected_components(graph))
    if len(weakly_connected_components) > 1:
        sizes = [len(c) for c in weakly_connected_components]
        logger.warning(
            f"Graph contains {len(weakly_connected_components)} weakly connected components. "
            f"This means the graph has disconnected groups if edge directions are ignored. "
            f"Component sizes:: {_fmt_top_sizes(sizes, top_k=top_k_wcc_sizes)}"
        )

    all_scc = sorted(nx.strongly_connected_components(graph), key=len)
    nodes_to_del = set().union(*all_scc[:-1])

    if nodes_to_del:
        logger.warning(
            f"Removing {len(nodes_to_del)} nodes from {len(all_scc) - 1} smaller strongly connected components. "
            f"These are subgraphs where nodes are internally reachable but isolated from the rest. "
            f"Retaining only the largest strongly connected component ({len(all_scc[-1])} nodes)."
        )
        graph.remove_nodes_from(nodes_to_del)

    return graph


def estimate_crs_for_bounds(minx, miny, maxx, maxy) -> CRS:
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
    logger.debug(f"Estimated CRS for territory {crs}")
    return crs
