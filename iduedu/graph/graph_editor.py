from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring

from iduedu.graph.urban_graph import UrbanGraph


@dataclass(slots=True)
class UrbanGraphChanges:
    """
    Набор изменений, которые можно применить к ``UrbanGraph``.

    Класс используется как промежуточный результат проецирования объектов на
    граф. Он хранит новые узлы, новые ребра и ключи ребер, которые нужно
    удалить из исходного графа перед добавлением разрезанных сегментов.

    Args:
        edges_gdf: Новые ребра графа.
        nodes_gdf: Новые узлы графа.
        edges_to_delete: Таблица ключей ребер, которые нужно удалить.
        is_multigraph: Должен совпадать с ``UrbanGraph.is_multigraph``.
        is_directed: Флаг направленного графа. Для направленных графов ключи
            ребер сопоставляются строго как ``u -> v``.

    Raises:
        TypeError: Если таблицы переданы в неподдерживаемом типе.
        ValueError: Если нарушены ключи или геометрии изменений.
    """

    edges_gdf: gpd.GeoDataFrame | None = None
    nodes_gdf: gpd.GeoDataFrame | None = None
    edges_to_delete: pd.DataFrame | None = None
    is_multigraph: bool = False
    is_directed: bool = False

    def __post_init__(self) -> None:
        if self.nodes_gdf is not None:
            self._validate_nodes()
        if self.edges_gdf is not None:
            self._validate_edges()
        if self.edges_to_delete is not None:
            self._validate_edges_to_delete()

    def _edge_key_cols(self) -> list[str]:
        return ["u", "v", "k"] if self.is_multigraph else ["u", "v"]

    def _validate_nodes(self) -> None:
        nodes = self.nodes_gdf
        if not isinstance(nodes, gpd.GeoDataFrame):
            raise TypeError(f"nodes_gdf must be GeoDataFrame, got {type(nodes).__name__}")
        if nodes.empty:
            raise ValueError("nodes_gdf is empty")
        if nodes.index.has_duplicates:
            raise ValueError("nodes_gdf.index must be unique")
        if nodes.geometry.isna().any():
            raise ValueError("nodes_gdf.geometry contains NaN")
        if (~nodes.geometry.geom_type.isin(["Point"])).any():
            raise ValueError("All nodes_gdf geometries must be Point")

    def _validate_edges(self) -> None:
        edges = self.edges_gdf
        if not isinstance(edges, gpd.GeoDataFrame):
            raise TypeError(f"edges_gdf must be GeoDataFrame, got {type(edges).__name__}")
        if edges.empty:
            raise ValueError("edges_gdf is empty")

        key_cols = self._edge_key_cols()
        required = set(key_cols + ["geometry"])
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"edges_gdf missing required columns: {sorted(missing)}")

        if edges[key_cols].isna().any().any():
            raise ValueError(f"edges_gdf columns {key_cols} must not contain NaN")
        if edges[key_cols].duplicated().any():
            raise ValueError(f"edges_gdf must have unique {key_cols} keys")
        if (~edges.geometry.geom_type.isin(["LineString"])).any():
            raise ValueError("All edges_gdf geometries must be LineString")

    def _validate_edges_to_delete(self) -> None:
        etd = self.edges_to_delete
        if not isinstance(etd, pd.DataFrame):
            raise TypeError(f"edges_to_delete must be DataFrame, got {type(etd).__name__}")
        if etd.empty:
            return
        key_cols = self._edge_key_cols()
        missing = set(key_cols) - set(etd.columns)
        if missing:
            raise ValueError(f"edges_to_delete missing required columns: {sorted(missing)}")
        if etd[key_cols].isna().any().any():
            raise ValueError(f"edges_to_delete columns {key_cols} must not contain NaN")


def project_objects2urban_graph(
    graph_gdf: UrbanGraph,
    objects_gdf: gpd.GeoDataFrame,
    speed_m_per_min: float,
) -> tuple[UrbanGraphChanges, pd.Series]:
    """
    Готовит изменения графа для подключения объектов к ближайшим ребрам.

    Функция не меняет исходный граф. Она вычисляет, какие узлы и ребра нужно
    добавить, какие исходные ребра нужно удалить, и возвращает ``object2node``
    маппинг. Для backend-сервиса это основной сценарий: ``changes`` можно
    записать в таблицы графа в БД, а ``object2node_map`` - в таблицу объектов
    как ``graph_node_id``. Для локальной копии графа результат можно применить
    через :func:`apply_urban_graph_changes`.

    Проецирование выполняется по ``representative_point()`` объекта. Поэтому
    функция одинаково подходит для зданий, сервисов и любых других слоев,
    если у объектов есть уникальный индекс и геометрия.

    Args:
        graph_gdf: Исходный городской граф.
        objects_gdf: Объекты для подключения к графу. Индекс должен быть
            стабильным идентификатором объекта, например ``object_id`` или
            ``building_id``.
        speed_m_per_min: Скорость движения по добавляемым соединительным
            ребрам в метрах в минуту. Например, ``5 * 1000 / 60`` для 5 км/ч.

    Returns:
        Пара ``(changes, object2node_map)``. ``changes.nodes_gdf`` содержит
        новые узлы, ``changes.edges_gdf`` - новые ребра,
        ``changes.edges_to_delete`` - ключи заменяемых ребер.
        ``object2node_map`` - ``Series`` с исходным индексом объектов и
        идентификаторами узлов графа.

    Raises:
        TypeError: Если ``objects_gdf`` не является ``GeoDataFrame``.
        ValueError: Если объекты пустые или скорость неположительная.
    """

    gdf_edges = graph_gdf.edges_gdf.copy()
    gdf_nodes = graph_gdf.nodes_gdf.copy()

    is_multigraph = graph_gdf.is_multigraph
    is_directed = graph_gdf.is_directed
    edge_direction_column = graph_gdf.edge_direction_column

    if not isinstance(objects_gdf, gpd.GeoDataFrame):
        raise TypeError(f"objects_gdf must be GeoDataFrame, got {type(objects_gdf).__name__}")
    if objects_gdf.empty:
        raise ValueError("objects_gdf is empty")
    if objects_gdf.index.has_duplicates:
        raise ValueError("objects_gdf.index must be unique")
    if speed_m_per_min <= 0:
        raise ValueError(f"default_speed_m_per_min must be > 0, got {speed_m_per_min}")

    original_crs = gdf_edges.crs
    local_crs = gdf_nodes.estimate_utm_crs()

    if gdf_edges.crs != local_crs:
        gdf_edges = gdf_edges.to_crs(local_crs)
    if gdf_nodes.crs != local_crs:
        gdf_nodes = gdf_nodes.to_crs(local_crs)
    if objects_gdf.crs != local_crs:
        objects_gdf = objects_gdf.to_crs(local_crs)

    rep_gdf = objects_gdf[["geometry"]].copy()
    rep_gdf.geometry = rep_gdf.geometry.representative_point()

    new_nodes_records: list[dict[str, Any]] = []
    new_edges_records: list[dict[str, Any]] = []
    edges_to_delete_records: list[dict[str, Any]] = []

    next_node_id = max(gdf_nodes.index) + 1 if len(gdf_nodes) else 0

    def _edge_record(u, v, geometry: LineString, k=0, attrs: dict[str, Any] | None = None) -> dict[str, Any]:
        record = dict(attrs) if attrs is not None else {}
        length = round(geometry.length, 3)
        time = round(length / speed_m_per_min, 2)
        record.update({"u": u, "v": v, "length_meter": length, "time_min": time, "geometry": geometry})
        if is_multigraph:
            record["k"] = k
        return record

    def _add_connector_edges(object_node, connect_node, object_point: Point, connect_point: Point) -> None:
        object_to_graph = LineString([object_point, connect_point])
        if object_to_graph.length <= 0:
            return
        connector_attrs = {edge_direction_column: False} if edge_direction_column is not None else None
        new_edges_records.append(_edge_record(object_node, connect_node, object_to_graph, attrs=connector_attrs))
        if is_directed and edge_direction_column is None:
            graph_to_object = LineString([connect_point, object_point])
            new_edges_records.append(_edge_record(connect_node, object_node, graph_to_object))

    gdf_edges["edge_geometry"] = gdf_edges["geometry"]
    projection_join = rep_gdf.reset_index(names="object_index").sjoin_nearest(gdf_edges, how="left")
    projection_join = projection_join.rename_geometry("object_geometry")
    projection_join = projection_join.rename(columns={"index_right": "edge_index"})

    projection_join["project_dist"] = projection_join["edge_geometry"].project(projection_join["object_geometry"])
    edge_lengths = projection_join["edge_geometry"].length
    endpoint_snap_tolerance = (1e-5 * edge_lengths).clip(lower=1e-6)

    start_mask = projection_join["project_dist"].abs() <= endpoint_snap_tolerance
    end_mask = (projection_join["project_dist"] - edge_lengths).abs() <= endpoint_snap_tolerance

    projection_join["node2connect"] = pd.NA
    projection_join.loc[start_mask, "node2connect"] = projection_join.loc[start_mask, "u"]
    projection_join.loc[end_mask, "node2connect"] = projection_join.loc[end_mask, "v"]

    endpoint_candidate = projection_join["node2connect"].notna()
    endpoint_object_index = projection_join.loc[endpoint_candidate, "object_index"].drop_duplicates()

    if not endpoint_object_index.empty:
        endpoint_rows = projection_join[projection_join["object_index"].isin(endpoint_object_index)]

        mixed_projection_object_index = endpoint_rows.loc[endpoint_rows["node2connect"].isna(), "object_index"].unique()
        if len(mixed_projection_object_index) > 0:
            raise ValueError(
                "Some objects have both endpoint and non-endpoint nearest edges: "
                f"{mixed_projection_object_index[:10].tolist()}"
            )

        nodes_per_object = endpoint_rows.groupby("object_index", sort=False)["node2connect"].nunique(dropna=True)
        bad_object_index = nodes_per_object[nodes_per_object != 1].index
        if len(bad_object_index) > 0:
            raise ValueError(
                "Some endpoint-projected objects connect to multiple graph nodes: " f"{bad_object_index[:10].tolist()}"
            )

        object_to_existing_node = endpoint_rows.groupby("object_index", sort=False)["node2connect"].first()

        # Привязка обьектов к углам эджей - существующим нодам

        for object_index, connect_node in object_to_existing_node.items():
            object_node = next_node_id
            next_node_id += 1

            object_point = rep_gdf.loc[object_index, "geometry"]
            connect_point = gdf_nodes.loc[connect_node, "geometry"]

            new_nodes_records.append({"node_id": object_node, "object_index": object_index, "geometry": object_point})
            _add_connector_edges(object_node, connect_node, object_point, connect_point)

        projection_join = projection_join[~projection_join["object_index"].isin(endpoint_object_index)].copy()

    projection_join["project_point"] = gpd.GeoSeries(
        projection_join["edge_geometry"].interpolate(projection_join["project_dist"]), crs=local_crs
    )
    projection_join["groupby_point"] = projection_join["project_point"].set_precision(10)
    projection_join["object_pair"] = list(zip(projection_join["object_index"], projection_join["object_geometry"]))

    projection_join["projection_group"] = projection_join.groupby(["groupby_point", "edge_index"], sort=False).ngroup()

    grouped_proj_points = projection_join.groupby(by="projection_group", as_index=False, sort=False).agg(
        {
            "project_point": "first",
            "project_dist": "first",
            "object_pair": lambda x: list(dict.fromkeys(x)),
        }
    )

    new_projection_count = len(grouped_proj_points)
    grouped_proj_points["projection_node_id"] = range(next_node_id, next_node_id + new_projection_count)
    next_node_id += new_projection_count

    grouped_proj_points["projection_node_id"] = grouped_proj_points["projection_node_id"].astype(int)
    projection_join = projection_join.merge(
        grouped_proj_points[["projection_group", "projection_node_id", "project_dist"]],
        on="projection_group",
        how="left",
        suffixes=("", "_group"),
    )
    projection_join["project_dist"] = projection_join["project_dist_group"]
    projection_join = projection_join.drop(columns="project_dist_group")

    # Привязка обьектов к первой в группе(если образовалась) точке группировки

    for _, row in grouped_proj_points.iterrows():
        projection_node = row.projection_node_id
        projection_point: Point = row.project_point
        new_nodes_records.append({"node_id": projection_node, "geometry": projection_point})
        for object_index, object_point in row["object_pair"]:
            object_node = next_node_id
            next_node_id += 1
            new_nodes_records.append({"node_id": object_node, "object_index": object_index, "geometry": object_point})
            _add_connector_edges(object_node, projection_node, object_point, projection_point)

    edge_group_agg = {
        "projection_node_id": tuple,
        "project_dist": tuple,
    }
    edge_lookup_cols = ["u", "v", "edge_geometry"]
    if is_multigraph:
        edge_lookup_cols.append("k")

    points_grouped_by_edge = projection_join.groupby(by="edge_index", as_index=False).agg(edge_group_agg)
    points_grouped_by_edge = points_grouped_by_edge.merge(
        gdf_edges[edge_lookup_cols],
        left_on="edge_index",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    for _, row in points_grouped_by_edge.iterrows():
        u = row["u"]
        v = row["v"]
        edge: LineString = row["edge_geometry"]
        edge_key = row["k"] if is_multigraph else 0
        split_attrs = {"edge_index": row["edge_index"]}

        if is_multigraph:
            edges_to_delete_records.append({"u": u, "v": v, "k": edge_key})
        else:
            edges_to_delete_records.append({"u": u, "v": v})

        eps = 1e-6 * edge.length

        dist_project = sorted(zip(row["project_dist"], row["projection_node_id"]), key=lambda x: x[0])

        grouped: list[list[Any]] = []
        for dist, node_id in dist_project:
            if not grouped:
                grouped.append([dist, [node_id]])
                continue

            last_dist, node_ids = grouped[-1]
            if abs(dist - last_dist) <= eps:
                node_ids.append(node_id)
            else:
                grouped.append([dist, [node_id]])

        last_dist = 0.0
        last_u = u

        for dist, node_ids in grouped:
            main_node = node_ids[0]
            if abs(dist) <= eps:
                last_u = u
                last_dist = 0.0
                continue

            if abs(dist - edge.length) <= eps:
                line = substring(edge, last_dist, edge.length)
                if line.length > 0 and last_u != v:
                    new_edges_records.append(_edge_record(last_u, v, line, k=edge_key, attrs=split_attrs))

                last_u = v
                last_dist = edge.length
                continue

            line = substring(edge, last_dist, dist)
            if line.length > 0 and last_u != main_node:
                new_edges_records.append(_edge_record(last_u, main_node, line, k=edge_key, attrs=split_attrs))

            last_u = main_node
            last_dist = dist

        if last_dist < edge.length:
            line = substring(edge, last_dist, edge.length)
            if line.length > 0 and last_u != v:
                new_edges_records.append(_edge_record(last_u, v, line, k=edge_key, attrs=split_attrs))

    new_nodes = gpd.GeoDataFrame(new_nodes_records, geometry="geometry", crs=local_crs)

    if not new_nodes.empty and "object_index" in new_nodes.columns:
        object2node_map = (
            new_nodes[["node_id", "object_index"]]
            .dropna(subset=["object_index"])
            .astype({"object_index": int, "node_id": int})
            .set_index("object_index")["node_id"]
        )
    else:
        object2node_map = pd.Series(dtype=int)

    if not new_nodes.empty:
        new_nodes = new_nodes.set_index("node_id", drop=True)
        if "object_index" in new_nodes.columns:
            new_nodes = new_nodes.drop(columns="object_index")

    new_edges = gpd.GeoDataFrame(new_edges_records, geometry="geometry", crs=local_crs)
    if not new_edges.empty and "edge_index" in new_edges.columns:
        changed_edge_columns = {"u", "v", "k", "geometry", "edge_geometry", "length_meter", "time_min", "edge_index"}
        edge_attr_cols = [col for col in gdf_edges.columns if col not in changed_edge_columns]
        edge_attrs = gdf_edges[edge_attr_cols].reset_index(names="edge_index")
        new_edges = new_edges.merge(edge_attrs, on="edge_index", how="left", suffixes=("", "_source"))

        for col in edge_attr_cols:
            source_col = f"{col}_source"
            if source_col in new_edges.columns:
                new_edges[col] = new_edges[col].combine_first(new_edges[source_col])
                new_edges = new_edges.drop(columns=source_col)
        new_edges = new_edges.drop(columns="edge_index")
        new_edges = gpd.GeoDataFrame(new_edges, geometry="geometry", crs=local_crs)
    edges_to_delete = pd.DataFrame(edges_to_delete_records).drop_duplicates()

    new_edges = new_edges.to_crs(original_crs)
    new_nodes = new_nodes.to_crs(original_crs)

    return (
        UrbanGraphChanges(
            edges_gdf=new_edges,
            nodes_gdf=new_nodes,
            edges_to_delete=edges_to_delete,
            is_multigraph=is_multigraph,
            is_directed=is_directed,
        ),
        object2node_map,
    )


def apply_urban_graph_changes(graph_gdf: UrbanGraph, changes: UrbanGraphChanges) -> UrbanGraph:
    """
    Применяет подготовленные изменения к ``UrbanGraph``.

    Функция удаляет ребра из ``changes.edges_to_delete``, добавляет новые узлы
    и ребра, затем возвращает новый ``UrbanGraph``. После применения изменений
    сохраненная матрица смежности считается устаревшей; перед расчетом
    доступности рекомендуется вручную вызвать
    :meth:`lprp.models.graph.graph.UrbanGraph.update_adjacency_matrix`.

    Args:
        graph_gdf: Исходный граф.
        changes: Изменения, подготовленные
            :func:`project_objects2urban_graph` или созданные вручную.

    Returns:
        Новый ``UrbanGraph`` с примененными изменениями.

    Raises:
        ValueError: Если тип графа и тип изменений не совпадают, либо
            ``edges_to_delete`` содержит отсутствующие в графе ребра.
    """

    if graph_gdf.is_multigraph != changes.is_multigraph:
        raise ValueError("graph.is_multigraph and changes.is_multigraph mismatch")
    if graph_gdf.is_directed != changes.is_directed:
        raise ValueError("graph.is_directed and changes.is_directed mismatch")

    nodes = graph_gdf.nodes_gdf.copy()
    edges = graph_gdf.edges_gdf.copy()

    new_nodes = changes.nodes_gdf
    new_edges = changes.edges_gdf
    edges_to_delete = changes.edges_to_delete

    key_cols = ["u", "v", "k"] if graph_gdf.is_multigraph else ["u", "v"]

    if edges_to_delete is not None and not edges_to_delete.empty:

        delete_keys = edges_to_delete[key_cols].drop_duplicates()
        edge_keys = edges[key_cols].drop_duplicates()

        missing_delete_keys = delete_keys.merge(edge_keys, on=key_cols, how="left", indicator=True)

        missing_delete_keys = missing_delete_keys[missing_delete_keys["_merge"] == "left_only"]

        if not missing_delete_keys.empty:
            raise ValueError(
                "Some edges_to_delete are not present in graph.edges_gdf: "
                f"{missing_delete_keys[key_cols].head(10).to_dict('records')}"
            )

        edges = edges.merge(delete_keys.assign(_delete_marker=True), on=key_cols, how="left")
        edges = edges[edges["_delete_marker"].isna()].drop(columns="_delete_marker")

    if new_nodes is not None and not new_nodes.empty:
        intersect = nodes.index.intersection(new_nodes.index)
        if len(intersect) > 0:
            raise ValueError(f"Node id collision: {intersect[:10].tolist()}")

        nodes = pd.concat([nodes, new_nodes], axis=0)

    if new_edges is not None and not new_edges.empty:
        edges = pd.concat([edges, new_edges], axis=0, ignore_index=True)

    return UrbanGraph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        is_multigraph=graph_gdf.is_multigraph,
        is_directed=graph_gdf.is_directed,
        edge_direction_column=graph_gdf.edge_direction_column,
    )
