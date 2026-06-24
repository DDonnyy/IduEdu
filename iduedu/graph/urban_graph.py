from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse


class UrbanGraph:
    """
    Табличное представление городского транспортного графа.

    ``UrbanGraph`` хранит узлы и ребра графа в ``GeoDataFrame``-таблицах и
    строит CSR-матрицу смежности для быстрых расчетов OD-матриц доступности.
    По умолчанию основным весом считается ``time_min``: это время движения по
    ребру в минутах. ``length_meter`` используется как альтернативный вес в
    метрах.

    Args:
        nodes_gdf: Таблица узлов. Индекс таблицы является идентификатором узла
            и должен быть уникальным. Для географических сценариев ожидается
            ``GeoDataFrame`` с точечной геометрией.
        edges_gdf: Таблица ребер. Обязательные колонки: ``u``, ``v``,
            ``geometry``, ``length_meter`` и ``time_min``. Для мультиграфа
            дополнительно нужна колонка ``k``. Колонки ``u`` и ``v`` ссылаются
            на индекс ``nodes_gdf``.
        is_multigraph: Флаг мультиграфа. Если ``True``, между двумя узлами
            может быть несколько ребер с разными ключами ``k``.
        is_directed: Флаг направленного графа. Если ``True``, матрица
            смежности строится только по направлению ``u -> v``.
        edge_direction_column: Имя булевой колонки ребер с признаком
            одностороннего движения. ``True`` означает только ``u -> v``,
            ``False`` означает движение в обе стороны. Если задана, граф
            считается направленным для расчетов матрицы смежности.
        adjacency_weight: Вес, который будет использоваться при построении
            матрицы смежности по умолчанию.

    Raises:
        TypeError: Если таблицы имеют неподдерживаемый тип.
        ValueError: Если нарушены контракты таблиц графа.
    """

    __slots__ = (
        "nodes_gdf",
        "edges_gdf",
        "is_multigraph",
        "is_directed",
        "edge_direction_column",
        "adjacency_matrix",
        "adjacency_nodelist",
        "node_to_adjacency_pos",
        "adjacency_weight",
    )

    def __init__(
        self,
        nodes_gdf: gpd.GeoDataFrame | pd.DataFrame,
        edges_gdf: gpd.GeoDataFrame | pd.DataFrame,
        is_multigraph: bool,
        is_directed: bool,
        *,
        edge_direction_column: str | None = None,
        adjacency_weight: str = "time_min",
    ):
        self.nodes_gdf = nodes_gdf
        self.edges_gdf = edges_gdf
        self.is_multigraph = is_multigraph
        self.edge_direction_column = edge_direction_column
        self.is_directed = is_directed or edge_direction_column is not None
        self.adjacency_matrix = None
        self.adjacency_nodelist = []
        self.node_to_adjacency_pos = {}
        self.adjacency_weight = adjacency_weight

        self._validate()

    def __repr__(self) -> str:
        return (
            f"UrbanGraph(nodes={len(self.nodes_gdf)}, edges={len(self.edges_gdf)}, "
            f"is_multigraph={self.is_multigraph}, is_directed={self.is_directed}, "
            f"edge_direction_column={self.edge_direction_column!r})"
        )

    def _validate_nodes(self) -> None:
        nodes = self.nodes_gdf
        if not isinstance(nodes, (gpd.GeoDataFrame, pd.DataFrame)):
            raise TypeError(f"nodes_gdf must be DataFrame or GeoDataFrame, got {type(nodes).__name__}")
        if nodes.empty:
            raise ValueError("nodes_gdf is empty")
        if nodes.index.has_duplicates:
            raise ValueError("nodes_gdf.index must be unique")
        if isinstance(nodes, gpd.GeoDataFrame):
            if nodes.geometry.isna().any():
                raise ValueError("nodes_gdf.geometry contains NaN")
            if (~nodes.geometry.geom_type.isin(["Point"])).any():
                raise ValueError("All nodes_gdf geometries must be Point")

    def _validate_edges(self) -> None:
        edges = self.edges_gdf
        if not isinstance(edges, (gpd.GeoDataFrame, pd.DataFrame)):
            raise TypeError(f"edges_gdf must be DataFrame or GeoDataFrame, got {type(edges).__name__}")
        if edges.empty:
            raise ValueError("edges_gdf is empty")
        required = {"u", "v", "geometry", "length_meter", "time_min"}
        missing = required - set(edges.columns)
        if missing:
            raise ValueError(f"edges_gdf missing required columns: {sorted(missing)}")
        if edges[["u", "v"]].isna().any().any():
            raise ValueError("edges_gdf columns ['u', 'v'] must not contain NaN")
        if self.edge_direction_column is not None:
            if self.edge_direction_column not in edges.columns:
                raise ValueError(f"edges_gdf missing edge_direction_column {self.edge_direction_column!r}")
            if edges[self.edge_direction_column].isna().any():
                raise ValueError(f"edges_gdf[{self.edge_direction_column!r}] contains NaN")
            values = set(edges[self.edge_direction_column].dropna().unique())
            if not values <= {False, True, 0, 1}:
                raise ValueError(f"edges_gdf[{self.edge_direction_column!r}] must contain only boolean values")
        if self.is_multigraph:
            if "k" not in edges.columns:
                raise ValueError("edges_gdf must contain 'k' for multigraph")
            if edges[["u", "v", "k"]].duplicated().any():
                raise ValueError("edges_gdf must have unique ['u','v','k']")
        else:
            if edges[["u", "v"]].duplicated().any():
                raise ValueError("edges_gdf must have unique ['u','v'] for non-multigraph")
        if isinstance(edges, gpd.GeoDataFrame):
            if edges.geometry.isna().any():
                raise ValueError("edges_gdf.geometry contains NaN")
            if (~edges.geometry.geom_type.isin(["LineString"])).any():
                raise ValueError("All edges_gdf geometries must be LineString")

    def _validate_nodes_edges(self) -> None:
        nodes = self.nodes_gdf
        edges = self.edges_gdf

        if isinstance(nodes, gpd.GeoDataFrame) and isinstance(edges, gpd.GeoDataFrame):
            if nodes.crs != edges.crs:
                raise ValueError(f"nodes and edges crs mismatch: nodes.crs={nodes.crs}, edges.crs={edges.crs}")

        edge_nodes = pd.Index(pd.concat([edges["u"], edges["v"]], ignore_index=True).unique())
        missing_nodes = edge_nodes.difference(nodes.index)

        if not missing_nodes.empty:
            raise ValueError(f"Some edge endpoints are missing in nodes_gdf.index: {missing_nodes.tolist()[:10]}")

    def _validate(self) -> None:
        self._validate_nodes()
        self._validate_edges()
        self._validate_nodes_edges()

    def copy(self) -> "UrbanGraph":
        """
        Возвращает независимую копию графа.

        Копируются таблицы узлов и ребер, а также уже построенная матрица
        смежности, если она была рассчитана ранее.

        Returns:
            Новый объект ``UrbanGraph`` с тем же состоянием.
        """

        graph = UrbanGraph(
            nodes_gdf=self.nodes_gdf.copy(),
            edges_gdf=self.edges_gdf.copy(),
            is_multigraph=self.is_multigraph,
            is_directed=self.is_directed,
            edge_direction_column=self.edge_direction_column,
            adjacency_weight=self.adjacency_weight,
        )

        if self.adjacency_matrix is not None:
            graph.adjacency_matrix = self.adjacency_matrix.copy()
            graph.adjacency_nodelist = list(self.adjacency_nodelist)
            graph.node_to_adjacency_pos = dict(self.node_to_adjacency_pos)

        return graph

    def _replace_state_from(self, other: "UrbanGraph") -> None:
        if not isinstance(other, UrbanGraph):
            raise TypeError(f"other must be UrbanGraph, got {type(other).__name__}")
        self.nodes_gdf = other.nodes_gdf
        self.edges_gdf = other.edges_gdf
        self.is_multigraph = other.is_multigraph
        self.is_directed = other.is_directed
        self.edge_direction_column = other.edge_direction_column
        self._validate()
        self.adjacency_matrix = None
        self.adjacency_nodelist = []
        self.node_to_adjacency_pos = {}

    def _build_adjacency_matrix(
        self, *, nodelist: Iterable[Any], weight: str, multigraph_rule: Literal["min", "max"] = "min"
    ) -> sparse.csr_matrix:
        nodelist = list(nodelist)

        if len(nodelist) == 0:
            raise ValueError("nodelist must not be empty")
        if weight not in self.edges_gdf.columns:
            raise KeyError(f"edges_gdf has no weight column {weight!r}")
        if self.edges_gdf[weight].isna().any():
            raise ValueError(f"edges_gdf[{weight!r}] contains NaN")
        if multigraph_rule not in {"min", "max"}:
            raise ValueError(f"Unsupported multigraph_rule={multigraph_rule!r}")

        node_to_pos = {node: i for i, node in enumerate(nodelist)}

        edges = self.edges_gdf
        mask = edges["u"].isin(nodelist) & edges["v"].isin(nodelist)
        edge_cols = ["u", "v", weight]
        if self.edge_direction_column is not None:
            edge_cols.append(self.edge_direction_column)
        edges = edges.loc[mask, edge_cols].copy()

        if self.edge_direction_column is not None:
            forward_edges = edges[["u", "v", weight]].rename(columns={"u": "source", "v": "target"})
            reverse_edges = edges.loc[~edges[self.edge_direction_column].astype(bool), ["u", "v", weight]].rename(
                columns={"v": "source", "u": "target"}
            )
            graph_edges = pd.concat([forward_edges, reverse_edges], ignore_index=True)
            graph_edges = graph_edges.groupby(["source", "target"], as_index=False, sort=False)[weight].agg(
                multigraph_rule
            )
            u = graph_edges["source"].to_numpy()
            v = graph_edges["target"].to_numpy()
            w = graph_edges[weight].to_numpy()
        elif self.is_multigraph:
            edges = edges.groupby(["u", "v"], as_index=False, sort=False)[weight].agg(multigraph_rule)
            u = edges["u"].to_numpy()
            v = edges["v"].to_numpy()
            w = edges[weight].to_numpy()
        else:
            u = edges["u"].to_numpy()
            v = edges["v"].to_numpy()
            w = edges[weight].to_numpy()

        row = np.fromiter((node_to_pos[x] for x in u), dtype=np.int64, count=len(u))
        col = np.fromiter((node_to_pos[x] for x in v), dtype=np.int64, count=len(v))

        if self.edge_direction_column is None and not self.is_directed:
            row, col = np.concatenate([row, col]), np.concatenate([col, row])
            data = np.concatenate([w, w])
        else:
            data = w

        return sparse.coo_matrix(
            (data, (row, col)),
            shape=(len(nodelist), len(nodelist)),
        ).tocsr()

    def update_adjacency_matrix(
        self,
        *,
        nodelist: Iterable[Any] | None = None,
        weight: str | None = None,
        multigraph_rule: Literal["min", "max"] = "min",
    ) -> sparse.csr_matrix:
        """
        Перестраивает и сохраняет матрицу смежности графа.

        Метод нужен после любых изменений ``nodes_gdf`` или ``edges_gdf``:
        проецирования объектов на граф, удаления или добавления ребер, смены
        веса расчета. Сохраненная матрица используется
        :func:`lprp.models.accessibility.graph_od_matrix.get_od_matrix`.

        Args:
            nodelist: Список узлов, которые нужно включить в матрицу. Если
                ``None``, используются все узлы графа.
            weight: Колонка веса ребра. Обычно ``time_min`` для времени в
                минутах или ``length_meter`` для расстояния в метрах.
            multigraph_rule: Правило схлопывания параллельных ребер в
                мультиграфе: ``min`` выбирает минимальный вес, ``max`` -
                максимальный.

        Returns:
            CSR-матрица смежности ``scipy.sparse.csr_matrix``.

        Raises:
            KeyError: Если колонки ``weight`` нет в ``edges_gdf``.
            ValueError: Если список узлов пустой или вес содержит ``NaN``.
        """

        if nodelist is None:
            nodelist = self.nodes_gdf.index.to_list()
        else:
            nodelist = list(nodelist)

        if weight is None:
            weight = self.adjacency_weight

        self.adjacency_matrix = self._build_adjacency_matrix(
            nodelist=nodelist, weight=weight, multigraph_rule=multigraph_rule
        )
        self.adjacency_nodelist = nodelist
        self.node_to_adjacency_pos = {node: i for i, node in enumerate(nodelist)}
        self.adjacency_weight = weight

        return self.adjacency_matrix

    def to_csr(
        self,
        *,
        nodelist: Iterable[Any] | None = None,
        weight: str | None = None,
        multigraph_rule: Literal["min", "max"] = "min",
    ) -> sparse.csr_matrix:
        """
        Строит CSR-матрицу смежности без изменения состояния графа.

        В отличие от :meth:`update_adjacency_matrix`, этот метод просто
        возвращает матрицу и не обновляет ``adjacency_matrix`` внутри
        ``UrbanGraph``. Его удобно использовать для разовых расчетов или
        проверки альтернативного веса.

        Args:
            nodelist: Список узлов для матрицы. Если ``None``, используются все
                узлы.
            weight: Колонка веса ребра. Если ``None``, используется
                ``adjacency_weight``.
            multigraph_rule: Правило выбора ребра в мультиграфе: ``min`` или
                ``max``.

        Returns:
            CSR-матрица смежности ``scipy.sparse.csr_matrix``.
        """

        if nodelist is None:
            nodelist = self.nodes_gdf.index.to_list()
        else:
            nodelist = list(nodelist)

        if weight is None:
            weight = self.adjacency_weight

        return self._build_adjacency_matrix(nodelist=nodelist, weight=weight, multigraph_rule=multigraph_rule)

    def single_source_dijkstra_path_length(
        self,
        source_node: Any,
        *,
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        dtype: np.dtype = np.float32,
    ) -> pd.Series:
        """Run single-source Dijkstra shortest path search on this graph."""

        from .graph_search import single_source_dijkstra_path_length

        return single_source_dijkstra_path_length(
            self,
            source_node,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            dtype=dtype,
        )

    def multi_source_dijkstra_path_length(
        self,
        *,
        source_nodes: Iterable[Any] | None = None,
        gdf_sources: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        dtype: np.dtype = np.float32,
    ) -> pd.Series:
        """Run multi-source Dijkstra shortest path search on this graph."""

        from .graph_search import multi_source_dijkstra_path_length

        return multi_source_dijkstra_path_length(
            self,
            source_nodes=source_nodes,
            gdf_sources=gdf_sources,
            graph_node_column=graph_node_column,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            dtype=dtype,
        )

    def multi_source_dijkstra_nearest_source(
        self,
        *,
        source_nodes: Iterable[Any] | None = None,
        gdf_sources: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        dtype: np.dtype = np.float32,
    ) -> pd.DataFrame:
        """Find the nearest source node and distance for each reachable graph node."""

        from .graph_search import multi_source_dijkstra_nearest_source

        return multi_source_dijkstra_nearest_source(
            self,
            source_nodes=source_nodes,
            gdf_sources=gdf_sources,
            graph_node_column=graph_node_column,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            dtype=dtype,
        )

    def dijkstra_path_length_parallel(
        self,
        *,
        source_nodes: Iterable[Any] | None = None,
        gdf_sources: pd.DataFrame | None = None,
        graph_node_column: str = "graph_node_id",
        weight: Literal["length_meter", "time_min"] = "time_min",
        cutoff: float | None = None,
        reverse: bool = False,
        dtype: np.dtype = np.float32,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Run independent Dijkstra searches for multiple source nodes."""

        from .graph_search import dijkstra_path_length_parallel

        return dijkstra_path_length_parallel(
            self,
            source_nodes=source_nodes,
            gdf_sources=gdf_sources,
            graph_node_column=graph_node_column,
            weight=weight,
            cutoff=cutoff,
            reverse=reverse,
            dtype=dtype,
            max_workers=max_workers,
        )

    def get_od_matrix(
        self,
        *,
        gdf_origins: pd.DataFrame | None = None,
        gdf_destinations: pd.DataFrame | None = None,
        origins_nodes: Iterable[Any] | None = None,
        destination_nodes: Iterable[Any] | None = None,
        graph_node_column: str = "graph_node_id",
        weight: Literal["length_meter", "time_min"] = "time_min",
        dtype: np.dtype = np.float32,
        threshold: float | None = None,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """Calculate an OD matrix of shortest paths on this graph."""

        from .graph_search import get_od_matrix

        return get_od_matrix(
            self,
            gdf_origins=gdf_origins,
            gdf_destinations=gdf_destinations,
            origins_nodes=origins_nodes,
            destination_nodes=destination_nodes,
            graph_node_column=graph_node_column,
            weight=weight,
            dtype=dtype,
            threshold=threshold,
            max_workers=max_workers,
        )

    @classmethod
    def from_nx_graph(
        cls,
        nx_graph,
        restore_edge_geom: bool = False,
        *,
        check_single_direction: bool = True,
        single_direction_column: str = "single_direction",
    ) -> "UrbanGraph":
        """
        Создает ``UrbanGraph`` из графа NetworkX.

        Этот конструктор полезен для графов, полученных из внешних библиотек,
        например IduEdu, если они уже содержат координаты узлов, CRS и
        атрибуты ребер ``length_meter`` и ``time_min``. Фактическое
        преобразование выполняет
        :func:`lprp.models.graph.graph_transformers.nx_graph2urban_graph`.

        Args:
            nx_graph: Объект ``networkx.Graph``, ``networkx.DiGraph``,
                ``networkx.MultiGraph`` или ``networkx.MultiDiGraph``.
            restore_edge_geom: Если ``True``, пустая геометрия ребра будет
                восстановлена прямым отрезком между узлами.
            check_single_direction: Если ``True`` и в ребрах есть колонка
                ``single_direction_column``, она будет использована для
                частично направленной матрицы смежности.
            single_direction_column: Имя булевой колонки односторонности.

        Returns:
            Экземпляр ``UrbanGraph``.
        """

        from .graph_transformers import nx_graph2urban_graph

        return nx_graph2urban_graph(
            nx_graph,
            restore_edge_geom=restore_edge_geom,
            check_single_direction=check_single_direction,
            single_direction_column=single_direction_column,
        )

    def to_nx_graph(self):
        """
        Преобразует ``UrbanGraph`` в граф NetworkX.

        Метод вызывает :func:`lprp.models.graph.graph_transformers.urban_graph2nx_graph`
        и сохраняет атрибуты ребер и узлов в формате NetworkX.

        Returns:
            ``networkx.Graph``, ``networkx.DiGraph``, ``networkx.MultiGraph``
            или ``networkx.MultiDiGraph`` в зависимости от флагов графа.
        """

        from .graph_transformers import urban_graph2nx_graph

        return urban_graph2nx_graph(self)

    def simplify_multiedges(
        self, *, weight: str = "time_min", rule: Literal["min", "max"] = "min", inplace: bool = False
    ) -> "UrbanGraph":
        """
        Схлопывает мультиграф до обычного графа.

        Для каждой пары узлов выбирается одно ребро по колонке ``weight``.
        Правило ``min`` оставляет ребро с минимальным весом, ``max`` - с
        максимальным. Подробная функция:
        :func:`lprp.models.graph.graph_transformers.simplify_urban_graph_multiedges`.

        Args:
            weight: Колонка веса для выбора ребра.
            rule: Правило выбора: ``min`` или ``max``.
            inplace: Если ``True``, текущий объект будет заменен упрощенным
                графом. Если ``False``, будет возвращен новый граф.

        Returns:
            Упрощенный ``UrbanGraph``.
        """

        from .graph_transformers import simplify_urban_graph_multiedges

        simplified = simplify_urban_graph_multiedges(self, weight=weight, rule=rule)
        simplified.adjacency_weight = self.adjacency_weight

        if not inplace:
            return simplified

        self._replace_state_from(simplified)
        return self

    def project_objects(
        self, objects_gdf: gpd.GeoDataFrame, speed_m_per_min: float, *, inplace: bool = False
    ) -> tuple["UrbanGraph", pd.Series]:
        """
        Добавляет объекты на ближайшие ребра графа.

        Метод создает для каждого объекта собственный узел графа, проецирует
        его ``representative_point()`` на ближайшее ребро, разрезает это ребро
        при необходимости и добавляет соединительное ребро. Это основной
        способ подготовить здания, сервисы или любые другие объекты к расчету
        графовой OD-матрицы в in-memory сценариях.

        Для backend-сервиса, где изменения графа нужно сохранить в БД, обычно
        удобнее напрямую использовать
        :func:`lprp.models.graph.graph_editor.project_objects2urban_graph`.
        Этот метод является оберткой, которая сразу применяет изменения к
        локальной копии графа.

        Функциональный аналог:
        :func:`lprp.models.graph.graph_editor.project_objects2urban_graph` +
        :func:`lprp.models.graph.graph_editor.apply_urban_graph_changes`.

        Args:
            objects_gdf: Объекты с уникальным индексом и геометрией. Индекс
                станет индексом ``object2node_map``.
            speed_m_per_min: Скорость движения по соединительным ребрам в
                метрах в минуту. Для 5 км/ч можно использовать
                ``5 * 1000 / 60``.
            inplace: Если ``True``, изменения применяются к текущему графу.
                Если ``False``, возвращается новый граф.

        Returns:
            Пара ``(graph, object2node_map)``. ``object2node_map`` -
            ``Series``, где индексом является исходный индекс объекта, а
            значением - идентификатор добавленного узла графа.
        """

        from .graph_editor import apply_urban_graph_changes, project_objects2urban_graph

        graph = self if inplace else self.copy()
        changes, object2node_map = project_objects2urban_graph(graph, objects_gdf, speed_m_per_min)

        changed_graph = apply_urban_graph_changes(graph, changes)
        changed_graph.adjacency_weight = graph.adjacency_weight

        if not inplace:
            return changed_graph, object2node_map

        self._replace_state_from(changed_graph)
        return self, object2node_map
