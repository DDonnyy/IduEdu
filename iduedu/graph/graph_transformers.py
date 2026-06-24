import importlib
import importlib.util
import warnings
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, Point

from iduedu.graph.urban_graph import UrbanGraph


def _require_networkx():
    if importlib.util.find_spec("networkx") is None:
        raise ImportError(
            "NetworkX is required for conversions to/from NetworkX graphs. "
            "Install optional dependency `networkx` to use this function."
        )

    return importlib.import_module("networkx")


def nx_graph2urban_graph(
    nx_graph: Any,
    restore_edge_geom: bool = True,
    *,
    check_single_direction: bool = True,
    single_direction_column: str = "single_direction",
) -> UrbanGraph:
    """
    Преобразует граф NetworkX в ``UrbanGraph``.

    Функция нужна для импорта графов из библиотек, которые отдают результат в
    формате NetworkX, например из IduEdu. В графе должны быть координаты узлов
    ``x``/``y``, атрибут ``crs`` и реберные атрибуты ``geometry``,
    ``length_meter`` и ``time_min``.

    Args:
        nx_graph: ``networkx.Graph`` или ``networkx.MultiGraph``. Если
            передан ``networkx.DiGraph`` или ``networkx.MultiDiGraph``,
            функция выдаст предупреждение: стабильная работа с directed
            NetworkX-графами не гарантируется.
        restore_edge_geom: Если ``True``, пустые геометрии ребер будут
            восстановлены прямой линией между узлами. Если ``False``, ребра с
            пустой геометрией будут удалены.
        check_single_direction: Если ``True`` и в ребрах есть колонка
            ``single_direction_column``, она будет использована для
            частично направленного графа.
        single_direction_column: Имя булевой колонки односторонности ребра.
            ``True`` означает движение только ``u -> v``, ``False`` — в обе
            стороны.

    Returns:
        Экземпляр :class:`lprp.models.graph.graph.UrbanGraph`.

    Raises:
        TypeError: Если передан не NetworkX-граф.
        ValueError: Если в графе нет CRS, узлов или ребер.
    """

    nx = _require_networkx()

    if not isinstance(nx_graph, (nx.Graph, nx.DiGraph)):
        raise TypeError(
            f"graph must be nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph, " f"got {type(nx_graph).__name__}"
        )
    if nx_graph.is_directed():
        warnings.warn(
            "Directed NetworkX graphs are accepted for compatibility, but stable behavior is not guaranteed. "
            "Prefer Graph/MultiGraph with edge geometry oriented from u to v and one-way movement stored in a "
            f"boolean edge column such as {single_direction_column!r}.",
            UserWarning,
            stacklevel=2,
        )

    try:
        crs = nx_graph.graph["crs"]
    except KeyError as exc:
        raise ValueError("Graph does not have crs attribute and no crs was provided") from exc
    if nx_graph.number_of_nodes() == 0:
        raise ValueError("graph has no nodes")
    if nx_graph.number_of_edges() == 0:
        raise ValueError("graph has no edges")

    try:
        node_index, node_data = zip(*nx_graph.nodes(data=True))
        node_geoms = [Point(data["x"], data["y"]) for data in node_data]
    except KeyError as exc:
        raise ValueError(f"NetworkX nodes must have required {exc.args[0]!r} attribute") from exc

    nodes_gdf = gpd.GeoDataFrame(index=node_index, crs=crs, geometry=node_geoms)

    if nx_graph.is_multigraph():
        edges_df = pd.DataFrame(list(nx_graph.edges(data=True, keys=True)), columns=["u", "v", "k", "data"])
    else:
        edges_df = pd.DataFrame(list(nx_graph.edges(data=True)), columns=["u", "v", "data"])

    edge_data_expanded = pd.json_normalize(edges_df["data"])
    edges_df = pd.concat([edges_df.drop(columns=["data"]), edge_data_expanded], axis=1)
    if "geometry" not in edges_df.columns:
        edges_df["geometry"] = LineString()
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry="geometry", crs=crs)
    edges_gdf["geometry"] = edges_gdf["geometry"].fillna(LineString())
    edge_direction_column = (
        single_direction_column if check_single_direction and single_direction_column in edges_gdf.columns else None
    )

    empty_geometry = edges_gdf.geometry.is_empty
    if restore_edge_geom:
        edges_gdf.loc[empty_geometry, "geometry"] = [
            LineString((nodes_gdf.loc[u, "geometry"], nodes_gdf.loc[v, "geometry"]))
            for u, v in edges_gdf.loc[empty_geometry, ["u", "v"]].itertuples(index=False, name=None)
        ]
    else:
        edges_gdf = edges_gdf.loc[~empty_geometry].copy()
        if edges_gdf.empty:
            raise ValueError("graph has no edges with non-empty geometry")

    non_linestring = edges_gdf.geometry.geom_type != "LineString"
    if non_linestring.any():
        bad_types = edges_gdf.loc[non_linestring, "geometry"].geom_type.head(10).tolist()
        raise ValueError(f"All edge geometries must be LineString, got {bad_types}")

    node_xy = pd.DataFrame(
        {"node_x": nodes_gdf.geometry.x, "node_y": nodes_gdf.geometry.y},
        index=nodes_gdf.index,
    )
    edge_node_xy = edges_gdf[["u", "v"]].join(node_xy.add_prefix("u_"), on="u").join(node_xy.add_prefix("v_"), on="v")
    missing_node_xy = edge_node_xy[["u_node_x", "u_node_y", "v_node_x", "v_node_y"]].isna().any(axis=1)
    if missing_node_xy.any():
        missing_preview = edges_gdf.loc[missing_node_xy, ["u", "v"]].head(10).to_dict("records")
        raise ValueError(f"Some edge endpoints are missing node coordinates: {missing_preview}")

    coords = edges_gdf.geometry.get_coordinates().to_numpy()
    counts = edges_gdf.geometry.count_coordinates().to_numpy()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1
    starts = coords[first_idx]
    ends = coords[last_idx]

    u_xy = edge_node_xy[["u_node_x", "u_node_y"]].to_numpy(dtype=float)
    v_xy = edge_node_xy[["v_node_x", "v_node_y"]].to_numpy(dtype=float)
    endpoint_tolerance = np.maximum(1e-6, 1e-9 * np.maximum(edges_gdf.geometry.length.to_numpy(), 1.0))

    start_matches_u = np.isclose(starts, u_xy, rtol=0, atol=endpoint_tolerance[:, None]).all(axis=1)
    swap_uv = ~start_matches_u
    if swap_uv.any():
        edges_gdf.loc[swap_uv, ["u", "v"]] = edges_gdf.loc[swap_uv, ["v", "u"]].to_numpy()
        u_xy[swap_uv], v_xy[swap_uv] = v_xy[swap_uv].copy(), u_xy[swap_uv].copy()

    start_matches_u = np.isclose(starts, u_xy, rtol=0, atol=endpoint_tolerance[:, None]).all(axis=1)
    end_matches_v = np.isclose(ends, v_xy, rtol=0, atol=endpoint_tolerance[:, None]).all(axis=1)
    bad_endpoint_match = ~(start_matches_u & end_matches_v)
    if bad_endpoint_match.any():
        bad_idx = np.flatnonzero(bad_endpoint_match)[:10]
        start_distance = np.hypot(*(starts - u_xy).T)
        end_distance = np.hypot(*(ends - v_xy).T)
        bad_edges = edges_gdf.iloc[bad_idx][["u", "v"]].assign(
            start_to_u_distance=start_distance[bad_idx],
            end_to_v_distance=end_distance[bad_idx],
        )
        raise ValueError(
            "Some edge geometry endpoints do not match node coordinates: " f"{bad_edges.to_dict('records')}"
        )

    return UrbanGraph(
        nodes_gdf=nodes_gdf,
        edges_gdf=edges_gdf,
        is_multigraph=nx_graph.is_multigraph(),
        is_directed=nx_graph.is_directed() or edge_direction_column is not None,
        edge_direction_column=edge_direction_column,
    )


def urban_graph2nx_graph(urban_graph: UrbanGraph) -> Any:
    """
    Преобразует ``UrbanGraph`` в NetworkX-граф.

    В обычной работе сервиса этот формат не обязателен, но полезен для
    интеграции с библиотеками, которые принимают ``networkx.Graph``.

    Args:
        urban_graph: Табличный городской граф.

    Returns:
        ``networkx.Graph``, ``networkx.DiGraph``, ``networkx.MultiGraph`` или
        ``networkx.MultiDiGraph`` с атрибутами узлов и ребер из таблиц
        ``UrbanGraph``.
    """

    nx = _require_networkx()

    nodes_gdf = urban_graph.nodes_gdf.copy()
    edges_gdf = urban_graph.edges_gdf.copy()

    graph_attrs = {
        "crs": edges_gdf.crs if isinstance(edges_gdf, gpd.GeoDataFrame) else None,
        "edge_direction_column": urban_graph.edge_direction_column,
    }

    is_multi = urban_graph.is_multigraph
    directed = urban_graph.is_directed and urban_graph.edge_direction_column is None

    nx_graph = (
        nx.MultiDiGraph()
        if directed and is_multi
        else (nx.MultiGraph() if is_multi else nx.DiGraph() if directed else nx.Graph())
    )
    nx_graph.graph.update(graph_attrs)

    nodes_gdf["x"] = nodes_gdf.geometry.x
    nodes_gdf["y"] = nodes_gdf.geometry.y
    df_nodes = nodes_gdf.drop(columns=nodes_gdf.active_geometry_name)

    if is_multi:
        edges_gdf = edges_gdf.set_index(["u", "v", "k"], drop=True)
        attr_names = edges_gdf.columns.to_list()

        for (u, v, k), attr_vals in zip(edges_gdf.index, edges_gdf.to_numpy(), strict=True):
            data_all = zip(attr_names, attr_vals, strict=True)
            data = {name: val for name, val in data_all if isinstance(val, list) or pd.notna(val)}
            nx_graph.add_edge(u, v, key=k, **data)

    else:
        edges_gdf = edges_gdf.set_index(["u", "v"], drop=True)
        attr_names = edges_gdf.columns.to_list()

        for (u, v), attr_vals in zip(edges_gdf.index, edges_gdf.to_numpy(), strict=True):
            data_all = zip(attr_names, attr_vals, strict=True)
            data = {name: val for name, val in data_all if isinstance(val, list) or pd.notna(val)}
            nx_graph.add_edge(u, v, **data)

    nx_graph.add_nodes_from(set(df_nodes.index) - set(nx_graph.nodes))

    for col in df_nodes.columns:
        nx.set_node_attributes(nx_graph, values=df_nodes[col].dropna().to_dict(), name=col)

    return nx_graph


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


def simplify_nx_graph_multiedges(
    graph: Any,
    *,
    weight: str = "length_meter",
    rule: Literal["min", "max"] = "min",
) -> nx.Graph | nx.DiGraph:
    """
    Схлопывает мультиграф NetworkX до обычного графа.

    Функция является NetworkX-аналогом
    :func:`simplify_urban_graph_multiedges`: для каждой пары узлов оставляет
    одно ребро по правилу ``min`` или ``max``.

    Args:
        graph: Исходный NetworkX-граф.
        weight: Атрибут ребра, по которому выбирается лучшее ребро.
        rule: ``min`` для минимального веса или ``max`` для максимального.

    Returns:
        Упрощенный ``networkx.Graph`` или ``networkx.DiGraph``.

    Raises:
        TypeError: Если передан не NetworkX-граф.
        KeyError: Если хотя бы у одного ребра нет атрибута ``weight``.
        ValueError: Если ``weight`` пустой или правило неизвестно.
    """

    nx = _require_networkx()

    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise TypeError(
            f"graph must be nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph, " f"got {type(graph).__name__}"
        )
    if not isinstance(weight, str):
        raise TypeError(f"weight must be str, got {type(weight).__name__}")
    if not weight:
        raise ValueError("weight must not be empty")
    if rule not in {"min", "max"}:
        raise ValueError(f"rule must be 'min' or 'max', got {rule!r}")

    if not graph.is_multigraph():
        return graph.copy()

    simple_graph: Any = nx.DiGraph() if graph.is_directed() else nx.Graph()
    simple_graph.graph.update(graph.graph)
    simple_graph.add_nodes_from(graph.nodes(data=True))

    for u, v, key, edge_attrs in graph.edges(keys=True, data=True):
        if weight not in edge_attrs:
            raise KeyError(f"Edge ({u!r}, {v!r}, {key!r}) has no weight attribute {weight!r}")

        if not simple_graph.has_edge(u, v):
            simple_graph.add_edge(u, v, **edge_attrs)
            continue

        current_attrs = simple_graph[u][v]

        should_replace = (
            edge_attrs[weight] < current_attrs[weight] if rule == "min" else edge_attrs[weight] > current_attrs[weight]
        )

        if should_replace:
            current_attrs.clear()
            current_attrs.update(edge_attrs)

    return simple_graph
