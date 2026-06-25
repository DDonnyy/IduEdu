from typing import Literal

import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from iduedu.graph.urban_graph import UrbanGraph


def estimate_crs_for_bounds(minx, miny, maxx, maxy) -> CRS:
    """
    Estimate a local UTM CRS for the given lon/lat bounds.
    """
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
    return CRS.from_epsg(utm_crs_list[0].code)


def simplify_urban_graph_multiedges(
    graph: UrbanGraph, *, weight: str = "time_min", rule: Literal["min", "max"] = "min"
) -> UrbanGraph:
    """
    Схлопывает мультиграф ``UrbanGraph`` до обычного графа.

    Для каждой пары узлов выбирается одно ребро по весу ``weight``. Правило
    ``min`` оставляет ребро с минимальным весом, ``max`` - с максимальным.
    Метод :meth:`lprp.models.graph.graph.UrbanGraph.simplify_multiedges`
    вызывает эту функцию внутри.

    Args:
        graph: Исходный городской граф.
        weight: Колонка веса для выбора ребра.
        rule: Правило выбора ребра: ``min`` или ``max``.

    Returns:
        Новый ``UrbanGraph`` с ``is_multigraph=False``.

    Raises:
        TypeError: Если ``graph`` не является ``UrbanGraph``.
        KeyError: Если колонки ``weight`` нет в ребрах.
        ValueError: Если вес содержит ``NaN`` или правило неизвестно.
    """

    if not isinstance(graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(graph).__name__}")
    if not isinstance(weight, str):
        raise TypeError(f"weight must be str, got {type(weight).__name__}")
    if rule not in {"min", "max"}:
        raise ValueError(f"rule must be 'min' or 'max', got {rule!r}")

    if not graph.is_multigraph:
        return UrbanGraph(
            nodes_gdf=graph.nodes_gdf.copy(),
            edges_gdf=graph.edges_gdf.copy(),
            is_multigraph=False,
            is_directed=graph.is_directed,
            edge_direction_column=graph.edge_direction_column,
        )

    edges = graph.edges_gdf.copy()

    if weight not in edges.columns:
        raise KeyError(f"edges_gdf has no weight column {weight!r}")
    if edges[weight].isna().any():
        raise ValueError(f"edges_gdf[{weight!r}] contains NaN")

    if graph.is_directed:
        group_cols = ["u", "v"]
        work = edges
    else:
        work = edges.copy()
        uv_min = pd.concat([work["u"], work["v"]], axis=1).min(axis=1)
        uv_max = pd.concat([work["u"], work["v"]], axis=1).max(axis=1)
        work["_u_simplify"] = uv_min
        work["_v_simplify"] = uv_max
        group_cols = ["_u_simplify", "_v_simplify"]

    ascending = rule == "min"

    selected_idx = (
        work.sort_values(weight, ascending=ascending, kind="mergesort").groupby(group_cols, sort=False).head(1).index
    )

    simple_edges = edges.loc[selected_idx].copy()

    if "k" in simple_edges.columns:
        simple_edges = simple_edges.drop(columns="k")

    return UrbanGraph(
        nodes_gdf=graph.nodes_gdf.copy(),
        edges_gdf=simple_edges,
        is_multigraph=False,
        is_directed=graph.is_directed,
        edge_direction_column=graph.edge_direction_column,
    )
