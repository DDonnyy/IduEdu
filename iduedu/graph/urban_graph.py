from typing import Any, Iterable, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from iduedu.graph.adjacency import build_adjacency_matrix
from iduedu.graph.validation import (
    gdf_crs,
    sync_graph_crs,
    validate_edges,
    validate_nodes,
    validate_nodes_edges,
)


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
        "crs",
        "type",
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
        crs: Any | None = None,
        graph_type: str | None = None,
    ):
        self.nodes_gdf = nodes_gdf
        self.edges_gdf = edges_gdf
        self.crs = crs
        self.type = graph_type
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
            f"edge_direction_column={self.edge_direction_column!r}, crs={self.crs!r}, type={self.type!r})"
        )

    @classmethod
    def empty(
        cls,
        *,
        crs: Any | None = None,
        is_multigraph: bool = True,
        is_directed: bool = False,
        edge_direction_column: str | None = None,
        adjacency_weight: str = "time_min",
        graph_type: str | None = None,
    ) -> "UrbanGraph":
        nodes_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)
        edge_columns = ["u", "v", "geometry", "length_meter", "time_min"]
        if is_multigraph:
            edge_columns.insert(2, "k")
        if edge_direction_column is not None:
            edge_columns.append(edge_direction_column)
        edges_gdf = gpd.GeoDataFrame(columns=edge_columns, geometry="geometry", crs=crs)
        if edge_direction_column is not None:
            edges_gdf[edge_direction_column] = pd.Series(dtype=bool)
        return cls(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            is_multigraph=is_multigraph,
            is_directed=is_directed,
            edge_direction_column=edge_direction_column,
            adjacency_weight=adjacency_weight,
            crs=crs,
            graph_type=graph_type,
        )

    @staticmethod
    def _gdf_crs(frame) -> Any | None:
        return gdf_crs(frame)

    def _sync_crs(self) -> None:
        sync_graph_crs(self)

    def _validate_nodes(self) -> None:
        validate_nodes(self)

    def _validate_edges(self) -> None:
        validate_edges(self)

    def _validate_nodes_edges(self) -> None:
        validate_nodes_edges(self)

    def _validate(self) -> None:
        self._validate_nodes()
        self._validate_edges()
        self._sync_crs()
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
            crs=self.crs,
            graph_type=self.type,
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
        self.crs = other.crs
        self.type = other.type
        self._validate()
        self._sync_crs()
        self.adjacency_matrix = None
        self.adjacency_nodelist = []
        self.node_to_adjacency_pos = {}

    def _build_adjacency_matrix(
        self, *, nodelist: Iterable[Any], weight: str, multigraph_rule: Literal["min", "max"] = "min"
    ) -> sparse.csr_matrix:
        return build_adjacency_matrix(self, nodelist=nodelist, weight=weight, multigraph_rule=multigraph_rule)

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

    def connected_components(self) -> list[set[Any]]:
        """Return connected components for an undirected graph."""

        from iduedu.graph.components import connected_components

        return connected_components(self)

    def weakly_connected_components(self) -> list[set[Any]]:
        """Return weakly connected components, ignoring edge direction."""

        from iduedu.graph.components import weakly_connected_components

        return weakly_connected_components(self)

    def strongly_connected_components(self) -> list[set[Any]]:
        """Return strongly connected components."""

        from iduedu.graph.components import strongly_connected_components

        return strongly_connected_components(self)

    def largest_component(self, *, mode: Literal["auto", "connected", "weak", "strong"] = "auto") -> set[Any]:
        """Return the largest component according to the selected mode."""

        from iduedu.graph.components import largest_component

        return largest_component(self, mode=mode)

    def subgraph_by_nodes(self, nodes: Iterable[Any]) -> "UrbanGraph":
        """Return the node-induced subgraph for ``nodes``."""

        from iduedu.graph.editors import subgraph_by_nodes

        return subgraph_by_nodes(self, nodes)

    def keep_largest_connected_component(
        self,
        *,
        mode: Literal["auto", "connected", "weak", "strong"] = "auto",
        inplace: bool = False,
    ) -> "UrbanGraph":
        """Keep only the largest graph component."""

        from iduedu.graph.transformers import keep_largest_connected_component

        filtered = keep_largest_connected_component(self, mode=mode)

        if not inplace:
            return filtered

        self._replace_state_from(filtered)
        return self

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

        from iduedu.graph.shortest_paths import single_source_dijkstra_path_length

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

        from iduedu.graph.shortest_paths import multi_source_dijkstra_path_length

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

        from iduedu.graph.shortest_paths import multi_source_dijkstra_nearest_source

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

        from iduedu.graph.shortest_paths import dijkstra_path_length_parallel

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

    def od_matrix(
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

        from iduedu.graph.shortest_paths import od_matrix

        return od_matrix(
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
        check_oneway: bool = True,
        oneway_column: str = "oneway",
    ) -> "UrbanGraph":
        """
        Создает ``UrbanGraph`` из графа NetworkX.

        Этот конструктор полезен для графов, полученных из внешних библиотек,
        например IduEdu, если они уже содержат координаты узлов, CRS и
        атрибуты ребер ``length_meter`` и ``time_min``. Фактическое
        преобразование выполняет
        :func:`iduedu.graph.adapters.nx_graph2urban_graph`.

        Args:
            nx_graph: Объект ``networkx.Graph``, ``networkx.DiGraph``,
                ``networkx.MultiGraph`` или ``networkx.MultiDiGraph``.
            restore_edge_geom: Если ``True``, пустая геометрия ребра будет
                восстановлена прямым отрезком между узлами.
            check_oneway: Если ``True`` и в ребрах есть колонка
                ``oneway_column``, она будет использована для
                частично направленной матрицы смежности.
            oneway_column: Имя булевой колонки односторонности ребра.

        Returns:
            Экземпляр ``UrbanGraph``.
        """

        from .adapters import nx_graph2urban_graph

        return nx_graph2urban_graph(
            nx_graph,
            restore_edge_geom,
            check_oneway=check_oneway,
            oneway_column=oneway_column,
        )

    def to_nx_graph(self):
        """
        Преобразует ``UrbanGraph`` в граф NetworkX.

        Метод вызывает :func:`iduedu.graph.adapters.urban_graph2nx_graph`
        и сохраняет атрибуты ребер и узлов в формате NetworkX.

        Returns:
            ``networkx.Graph``, ``networkx.DiGraph``, ``networkx.MultiGraph``
            или ``networkx.MultiDiGraph`` в зависимости от флагов графа.
        """

        from .adapters import urban_graph2nx_graph

        return urban_graph2nx_graph(self)

    def simplify_multiedges(
        self, *, weight: str = "time_min", rule: Literal["min", "max"] = "min", inplace: bool = False
    ) -> "UrbanGraph":
        """
        Схлопывает мультиграф до обычного графа.

        Для каждой пары узлов выбирается одно ребро по колонке ``weight``.
        Правило ``min`` оставляет ребро с минимальным весом, ``max`` - с
        максимальным. Подробная функция:
        :func:`iduedu.graph.transformers.simplify_multiedges`.

        Args:
            weight: Колонка веса для выбора ребра.
            rule: Правило выбора: ``min`` или ``max``.
            inplace: Если ``True``, текущий объект будет заменен упрощенным
                графом. Если ``False``, будет возвращен новый граф.

        Returns:
            Упрощенный ``UrbanGraph``.
        """

        from iduedu.graph.transformers import simplify_multiedges

        simplified = simplify_multiedges(self, weight=weight, rule=rule)
        simplified.adjacency_weight = self.adjacency_weight

        if not inplace:
            return simplified

        self._replace_state_from(simplified)
        return self

    def relabel(self, *, inplace: bool = False) -> "UrbanGraph":
        """
        Перенумеровывает узлы графа в плотный ``RangeIndex``.

        Функциональный аналог:
        :func:`iduedu.graph.editors.relabel_urban_graph`.

        Args:
            inplace: Если ``True``, текущий объект будет заменен
                перенумерованным графом. Если ``False``, будет возвращен
                новый граф.

        Returns:
            ``UrbanGraph`` с обновленными индексами узлов и концами ребер.
        """

        from .editors import relabel_urban_graph

        relabeled = relabel_urban_graph(self)

        if not inplace:
            return relabeled

        self._replace_state_from(relabeled)
        return self

    def clip(self, polygon, *, inplace: bool = False) -> "UrbanGraph":
        """
        Обрезает граф по геометрии, сохраняя только узлы внутри нее.

        Ребра сохраняются только если оба их конца остались в графе. Индексы
        узлов не перенумеровываются; при необходимости вызовите
        :meth:`relabel`.

        Функциональный аналог:
        :func:`iduedu.graph.editors.clip_urban_graph`.

        Args:
            polygon: Геометрия Shapely в CRS графа.
            inplace: Если ``True``, текущий объект будет заменен обрезанным
                графом. Если ``False``, будет возвращен новый граф.

        Returns:
            Обрезанный ``UrbanGraph``.
        """

        from .editors import clip_urban_graph

        clipped = clip_urban_graph(self, polygon)

        if not inplace:
            return clipped

        self._replace_state_from(clipped)
        return self

    def join(
        self,
        other: "UrbanGraph",
        *,
        graph_type: str | None = None,
        node_conflict: str = "left",
        inplace: bool = False,
    ) -> "UrbanGraph":
        """
        Объединяет текущий граф с другим ``UrbanGraph``.

        Общие индексы узлов допускаются; атрибуты таких узлов выбираются
        параметром ``node_conflict``. Ребра с одинаковыми ключами ``u/v/k`` для
        мультиграфа или ``u/v`` для обычного графа считаются конфликтом.

        Функциональный аналог:
        :func:`iduedu.graph.editors.join_urban_graphs`.

        Args:
            other: Второй граф для объединения.
            graph_type: Тип результирующего графа. Если ``None``, сохраняется
                тип текущего графа.
            node_conflict: Какая сторона выигрывает при совпадении индексов
                узлов: ``"left"`` или ``"right"``.
            inplace: Если ``True``, текущий объект будет заменен объединенным
                графом. Если ``False``, будет возвращен новый граф.

        Returns:
            Объединенный ``UrbanGraph``.
        """

        from iduedu.graph.editors import join_urban_graphs

        joined = join_urban_graphs(self, other, graph_type=graph_type, node_conflict=node_conflict)

        if not inplace:
            return joined

        self._replace_state_from(joined)
        return self

    def to_directed(
        self,
        *,
        edge_direction_column: str = "oneway",
        default_direction_value: bool = False,
        inplace: bool = False,
    ) -> "UrbanGraph":
        """
        Возвращает направленную версию графа с булевой колонкой направления.

        Функциональный аналог:
        :func:`iduedu.graph.transformers.to_directed`.

        Args:
            edge_direction_column: Имя колонки односторонности ребра.
            default_direction_value: Значение для ребер, где колонка
                отсутствует или содержит пропуск.
            inplace: Если ``True``, текущий объект будет заменен направленным
                графом. Если ``False``, будет возвращен новый граф.

        Returns:
            Направленный ``UrbanGraph``.
        """

        from iduedu.graph.transformers import to_directed

        directed = to_directed(
            self,
            edge_direction_column=edge_direction_column,
            default_direction_value=default_direction_value,
        )

        if not inplace:
            return directed

        self._replace_state_from(directed)
        return self

    def to_undirected(self, *, inplace: bool = False) -> "UrbanGraph":
        """
        Возвращает ненаправленную версию графа.

        Функциональный аналог:
        :func:`iduedu.graph.transformers.to_undirected`.

        Args:
            inplace: Если ``True``, текущий объект будет заменен
                ненаправленным графом. Если ``False``, будет возвращен новый
                граф.

        Returns:
            Ненаправленный ``UrbanGraph``.
        """

        from .transformers import to_undirected

        undirected = to_undirected(self)

        if not inplace:
            return undirected

        self._replace_state_from(undirected)
        return self

    def project_objects(
        self,
        objects_gdf: gpd.GeoDataFrame,
        speed_m_per_min: float,
        *,
        max_dist: float | None = None,
        add_link_edge: bool = True,
        inplace: bool = False,
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
        :func:`iduedu.graph.editors.project_objects2urban_graph`.
        Этот метод является оберткой, которая сразу применяет изменения к
        локальной копии графа.

        Функциональный аналог:
        :func:`iduedu.graph.editors.project_objects2urban_graph` +
        :func:`iduedu.graph.editors.apply_urban_graph_changes`.

        Args:
            objects_gdf: Объекты с уникальным индексом и геометрией. Индекс
                станет индексом ``object2node_map``.
            speed_m_per_min: Скорость движения по соединительным ребрам в
                метрах в минуту. Для 5 км/ч можно использовать
                ``5 * 1000 / 60``.
            max_dist: Максимальная дистанция до ближайшего ребра. Если
                ``None``, ограничение не применяется.
            add_link_edge: Если ``True``, создается отдельный узел объекта и
                соединительное ребро до точки проекции. Если ``False``, объект
                сопоставляется с точкой проекции на графе.
            inplace: Если ``True``, изменения применяются к текущему графу.
                Если ``False``, возвращается новый граф.

        Returns:
            Пара ``(graph, object2node_map)``. ``object2node_map`` -
            ``Series``, где индексом является исходный индекс объекта, а
            значением - идентификатор добавленного узла графа.
        """

        from iduedu.graph.editors import apply_urban_graph_changes, project_objects2urban_graph

        graph = self if inplace else self.copy()
        changes, object2node_map = project_objects2urban_graph(
            graph,
            objects_gdf,
            speed_m_per_min,
            max_dist=max_dist,
            add_link_edge=add_link_edge,
        )

        changed_graph = apply_urban_graph_changes(graph, changes)
        changed_graph.adjacency_weight = graph.adjacency_weight

        if not inplace:
            return changed_graph, object2node_map

        self._replace_state_from(changed_graph)
        return self, object2node_map
