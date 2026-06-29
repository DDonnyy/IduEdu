"""Базовые алгоритмы поиска по :class:`iduedu.graph.urban_graph.UrbanGraph`.

Модуль является тонкой публичной оберткой над numba-реализациями из
``iduedu._numba``. Здесь собрана неалгоритмическая подготовка, которую иначе
пришлось бы повторять в каждом вызывающем методе:

* проверка ``UrbanGraph`` и выбранной колонки веса;
* приведение входных объектов или готовых ids к узлам графа;
* построение или переиспользование матрицы смежности;
* перевод весов в целочисленное представление для numba;
* обратная сборка разреженных результатов numba в pandas ``Series`` /
  ``DataFrame``.

Все расстояния возвращаются в исходных единицах ``weight``: минуты для
``time_min`` и метры для ``length_meter``. Недостижимые узлы обозначаются
``np.inf``, а результаты с длинами путей используют разреженные dtype pandas.
"""

import logging
from typing import Any, Iterable, Literal

import numba as nb
import numpy as np
import pandas as pd
from scipy import sparse

from iduedu._numba.csr import coo_rows_to_arrays, sparse_row2numba_matrix
from iduedu._numba.shortest_paths import (
    dijkstra_numba_od_parallel,
    dijkstra_numba_path_length_parallel,
    multi_source_dijkstra_numba_nearest_source,
    multi_source_dijkstra_numba_path_length,
    single_source_dijkstra_numba_path_length,
)
from iduedu.graph.graph_inputs import resolve_graph_nodes_input
from iduedu.graph.urban_graph import UrbanGraph

logger = logging.getLogger(__name__)

NODE_INDEX_NAME = "node"
DIST_COLUMN = "dist"
SOURCE_NODE_COLUMN = "source_node"
SOURCE_NODES_ATTR = "source_nodes"

WEIGHT_SCALE = 100.0


def _int_weight2float(value) -> np.ndarray | float:
    """Возвращает вес из целочисленного формата numba в исходные единицы."""

    return value.astype(float) / WEIGHT_SCALE if hasattr(value, "astype") else float(value) / WEIGHT_SCALE


def _cutoff2int(weight_value_cutoff: float | None) -> np.int32:
    """Переводит ограничение поиска в формат numba; ``None`` означает без ограничения."""

    if weight_value_cutoff is None:
        return np.int32(np.iinfo(np.int32).max)
    return np.int32(round(float(weight_value_cutoff) * WEIGHT_SCALE))


def _validate_max_workers(max_workers: int | None) -> None:
    if max_workers is None:
        return
    if not isinstance(max_workers, int):
        raise TypeError(f"max_workers must be int | None, got {type(max_workers).__name__}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")


def _node_positions(urban_graph: UrbanGraph, nodes: Iterable[Any]) -> np.ndarray:
    node_to_pos = urban_graph.node_to_adjacency_pos
    nodes = list(nodes)
    missing_nodes = [node for node in nodes if node not in node_to_pos]
    if missing_nodes:
        preview = missing_nodes[:10]
        raise ValueError(
            f"source_nodes contain nodes that are absent in graph: {preview}"
            + (" ..." if len(missing_nodes) > 10 else "")
        )
    return np.fromiter((node_to_pos[node] for node in nodes), dtype=np.int32, count=len(nodes))


def _pos_to_node_array(urban_graph: UrbanGraph) -> np.ndarray:
    pos_to_node = np.empty(len(urban_graph.adjacency_nodelist), dtype=object)
    pos_to_node[:] = urban_graph.adjacency_nodelist
    return pos_to_node


def _prepare_numba_graph(
    urban_graph: UrbanGraph,
    *,
    weight: str,
    cutoff: float | None,
    reverse: bool,
):
    if not isinstance(urban_graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(urban_graph).__name__}")
    if len(urban_graph.nodes_gdf) == 0:
        raise ValueError("graph is empty")
    if urban_graph.nodes_gdf.index.has_duplicates:
        raise ValueError("graph.nodes_gdf.index must be unique")
    if weight not in urban_graph.edges_gdf.columns:
        raise KeyError(f"graph.edges_gdf has no weight column {weight!r}")
    if cutoff is not None and cutoff < 0:
        raise ValueError(f"weight_value_cutoff must be >= 0, got {cutoff}")

    graph_nodelist = urban_graph.nodes_gdf.index.to_list()
    if (
        urban_graph.adjacency_matrix is None
        or urban_graph.adjacency_weight != weight
        or urban_graph.adjacency_nodelist != graph_nodelist
    ):
        urban_graph.update_adjacency_matrix(nodelist=graph_nodelist, weight=weight)
    if urban_graph.adjacency_matrix.shape[0] == 0:
        raise ValueError("graph adjacency_matrix is empty")

    sparse_row_scipy = urban_graph.adjacency_matrix.copy().tocsr()
    sparse_row_scipy.data = np.round(sparse_row_scipy.data * WEIGHT_SCALE).astype(np.int32)
    if reverse and urban_graph.is_directed:
        sparse_row_scipy = sparse_row_scipy.transpose().tocsc().tocsr()
    return sparse_row2numba_matrix(sparse_row_scipy)


def _path_length_series(
    reachable_pairs,
    *,
    pos_to_node: np.ndarray,
    dtype: np.dtype,
) -> pd.Series:
    if len(reachable_pairs) == 0:
        return pd.Series(
            [],
            index=pd.Index([], name=NODE_INDEX_NAME),
            name=DIST_COLUMN,
            dtype=pd.SparseDtype(dtype, fill_value=np.inf),
        )

    reachable_pairs_arr = np.asarray(reachable_pairs, dtype=np.int32)
    return pd.Series(
        _int_weight2float(reachable_pairs_arr[:, 1]).astype(dtype),
        index=pd.Index(pos_to_node[reachable_pairs_arr[:, 0]], name=NODE_INDEX_NAME),
        name=DIST_COLUMN,
    ).astype(pd.SparseDtype(dtype, fill_value=np.inf))


def single_source_dijkstra_path_length(
    urban_graph: UrbanGraph,
    source_node: Any,
    *,
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    dtype: np.dtype = np.float32,
) -> pd.Series:
    """Считает длины кратчайших путей от одного источника до всех узлов графа.

    Args:
        urban_graph: Городской граф с таблицами узлов, ребер и весами ребер.
        source_node: Существующий id узла из ``graph.nodes_gdf.index``.
        weight: Колонка веса ребра. Обычно ``"time_min"`` или
            ``"length_meter"``.
        cutoff: Максимальная стоимость пути. Узлы дальше этого значения
            остаются ``np.inf`` в результате.
        reverse: Если ``True`` и граф направленный, расчет идет по
            транспонированной матрице смежности. Это нужно для задач покрытия
            в логике "кто может доехать до назначения". Для ненаправленного
            графа параметр не влияет на расчет.
        dtype: Вещественный dtype возвращаемой разреженной серии.

    Returns:
        Разреженный ``Series`` с индексом из всех узлов графа. Значения -
        длины путей от ``source_node``; недостижимые узлы равны ``np.inf``.
    """

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_pos = _node_positions(urban_graph, [source_node])[0]
    reachable_pairs = single_source_dijkstra_numba_path_length(
        numba_adj_matrix, np.int32(source_pos), _cutoff2int(cutoff)
    )
    return _path_length_series(reachable_pairs, pos_to_node=_pos_to_node_array(urban_graph), dtype=dtype)


def multi_source_dijkstra_path_length(
    urban_graph: UrbanGraph,
    *,
    source_nodes: Iterable[Any] | None = None,
    gdf_sources: pd.DataFrame | None = None,
    graph_node_column: str = "graph_node_id",
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    dtype: np.dtype = np.float32,
) -> pd.Series:
    """Считает расстояние от ближайшего источника до каждого узла графа.

    Аналог ``networkx.multi_source_dijkstra_path_length``: все источники
    кладутся в одну очередь Дейкстры, поэтому каждый узел получает только
    лучшее расстояние до ближайшего источника. Информация о победившем
    источнике здесь не сохраняется; для этого используйте
    :func:`multi_source_dijkstra_nearest_source`.

    Args:
        urban_graph: Городской граф с таблицами узлов, ребер и весами ребер.
        source_nodes: Узлы-источники. Передается либо этот параметр, либо
            ``gdf_sources``.
        gdf_sources: Таблица/GeoDataFrame с объектами-источниками. Если есть
            ``graph_node_column``, узлы берутся из этой колонки; иначе
            ближайшие узлы графа ищутся по геометрии.
        graph_node_column: Колонка с ids узлов графа в ``gdf_sources``.
        weight: Колонка веса ребра.
        cutoff: Максимальная стоимость пути.
        reverse: Если ``True`` и граф направленный, расчет идет по обратной
            матрице смежности.
        dtype: Вещественный dtype возвращаемой разреженной серии.

    Returns:
        Разреженный ``Series`` с индексом из всех узлов графа. Нормализованное
        соответствие исходных объектов узлам дополнительно сохраняется в
        ``result.attrs["source_nodes"]``.
    """

    source_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=source_nodes,
        gdf=gdf_sources,
        graph_node_column=graph_node_column,
        nodes_name="source_nodes",
        gdf_name="gdf_sources",
    )

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_positions = _node_positions(urban_graph, pd.Index(source_nodes_s.to_numpy()).unique())
    reachable_pairs = multi_source_dijkstra_numba_path_length(
        numba_adj_matrix,
        source_positions,
        _cutoff2int(cutoff),
    )
    result = _path_length_series(reachable_pairs, pos_to_node=_pos_to_node_array(urban_graph), dtype=dtype)
    result.attrs[SOURCE_NODES_ATTR] = source_nodes_s
    return result


def multi_source_dijkstra_nearest_source(
    urban_graph: UrbanGraph,
    *,
    source_nodes: Iterable[Any] | None = None,
    gdf_sources: pd.DataFrame | None = None,
    graph_node_column: str = "graph_node_id",
    weight: Literal["length_meter", "time_min"] = "time_min",
    cutoff: float | None = None,
    reverse: bool = False,
    dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    """Находит ближайший источник и расстояние до него для каждого узла графа.

    Аргументы совпадают с :func:`multi_source_dijkstra_path_length`.

    Returns:
        ``DataFrame`` с индексом из всех узлов графа и колонками:

        * ``source_node`` - ближайший источник или ``pd.NA`` для недостижимых
          узлов;
        * ``dist`` - разреженное расстояние до источника, ``np.inf`` для
          недостижимых узлов.

        Нормализованное соответствие исходных объектов узлам сохраняется в
        ``result.attrs["source_nodes"]``.
    """

    source_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=source_nodes,
        gdf=gdf_sources,
        graph_node_column=graph_node_column,
        nodes_name="source_nodes",
        gdf_name="gdf_sources",
    )

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_positions = _node_positions(urban_graph, pd.Index(source_nodes_s.to_numpy()).unique())
    pos_to_node = _pos_to_node_array(urban_graph)
    reachable_triplets = multi_source_dijkstra_numba_nearest_source(
        numba_adj_matrix, source_positions, _cutoff2int(cutoff)
    )

    if len(reachable_triplets) == 0:
        result = pd.DataFrame(
            {
                SOURCE_NODE_COLUMN: pd.Series([], index=pd.Index([], name=NODE_INDEX_NAME), dtype=object),
                DIST_COLUMN: pd.Series([], index=pd.Index([], name=NODE_INDEX_NAME), dtype=dtype),
            }
        )
    else:
        reachable_triplets_arr = np.asarray(reachable_triplets, dtype=np.int32)
        reachable_index = pd.Index(pos_to_node[reachable_triplets_arr[:, 0]], name=NODE_INDEX_NAME)
        result = pd.DataFrame(
            {
                SOURCE_NODE_COLUMN: pos_to_node[reachable_triplets_arr[:, 1]],
                DIST_COLUMN: _int_weight2float(reachable_triplets_arr[:, 2]).astype(dtype),
            },
            index=reachable_index,
        )
    result[DIST_COLUMN] = result[DIST_COLUMN].astype(pd.SparseDtype(dtype, fill_value=np.inf))
    result.attrs[SOURCE_NODES_ATTR] = source_nodes_s
    return result


def dijkstra_path_length_parallel(
    urban_graph: UrbanGraph,
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
    """Запускает независимый поиск Дейкстры для каждого источника.

    В отличие от :func:`multi_source_dijkstra_path_length`, источники не
    объединяются в одну очередь. Функция считает отдельную строку расстояний
    для каждого исходного объекта/узла, поэтому подходит как базовый метод для
    изохрон по отдельным origin.

    Args:
        urban_graph: Городской граф с таблицами узлов, ребер и весами ребер.
        source_nodes: Узлы-источники. Передается либо этот параметр, либо
            ``gdf_sources``.
        gdf_sources: Таблица/GeoDataFrame с объектами-источниками.
        graph_node_column: Колонка с ids узлов графа в ``gdf_sources``.
        weight: Колонка веса ребра.
        cutoff: Максимальная стоимость пути.
        reverse: Если ``True`` и граф направленный, расчет идет по обратной
            матрице смежности.
        dtype: Вещественный dtype возвращаемого разреженного датафрейма.
        max_workers: Число потоков numba для расчета.

    Returns:
        Разреженный ``DataFrame``: строки - исходные объекты, колонки - все
        узлы графа. Нормализованное соответствие исходных объектов узлам
        сохраняется в ``result.attrs["source_nodes"]``.
    """

    _validate_max_workers(max_workers)
    source_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=source_nodes,
        gdf=gdf_sources,
        graph_node_column=graph_node_column,
        nodes_name="source_nodes",
        gdf_name="gdf_sources",
    )

    numba_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=cutoff, reverse=reverse)
    source_positions = _node_positions(urban_graph, source_nodes_s.to_numpy())
    if max_workers is not None:
        nb.set_num_threads(max_workers)

    reachable_rows = dijkstra_numba_path_length_parallel(numba_adj_matrix, source_positions, _cutoff2int(cutoff))
    rows, cols, values = coo_rows_to_arrays(reachable_rows)

    if len(values) > 0:
        reachable_col_positions, compact_cols = np.unique(cols, return_inverse=True)
    else:
        reachable_col_positions = np.array([], dtype=np.int32)
        compact_cols = np.array([], dtype=np.int32)
    reachable_columns = pd.Index(
        _pos_to_node_array(urban_graph)[reachable_col_positions],
        name=NODE_INDEX_NAME,
    )

    path_matrix = sparse.coo_matrix(
        (_int_weight2float(values).astype(dtype), (rows, compact_cols)),
        shape=(len(source_nodes_s), len(reachable_columns)),
    ).tocsr()
    dense_result = np.full(path_matrix.shape, np.inf, dtype=dtype)
    if len(values) > 0:
        dense_result[rows, compact_cols] = _int_weight2float(values).astype(dtype)

    result = pd.DataFrame(
        dense_result,
        index=source_nodes_s.index,
        columns=reachable_columns,
    ).astype(pd.SparseDtype(dtype, fill_value=np.inf))
    result.attrs[SOURCE_NODES_ATTR] = source_nodes_s
    return result


def od_matrix(
    urban_graph: UrbanGraph,
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
    """Считает OD-матрицу кратчайших путей по городскому графу.

    Функция принимает либо готовые идентификаторы узлов графа
    ``origins_nodes`` / ``destination_nodes``, либо таблицы
    ``gdf_origins`` / ``gdf_destinations``. Если в таблице есть колонка
    ``graph_node_column``, узлы берутся из нее. Если колонки нет, ближайшие
    узлы графа ищутся по геометрии. Возвращает разреженный ``DataFrame``
    расстояний или времени в пути.

    Если в ``UrbanGraph`` еще нет актуальной матрицы смежности с выбранным
    весом, функция вызовет
    :meth:`iduedu.graph.urban_graph.UrbanGraph.update_adjacency_matrix`
    автоматически. Для сервиса лучше делать это явно после изменения графа,
    чтобы контролировать тяжелый этап подготовки.

    Args:
        urban_graph: Городской граф с таблицами узлов, ребер и матрицей
            смежности.
        gdf_origins: Таблица объектов-источников.
        gdf_destinations: Таблица объектов-назначений.
        origins_nodes: Узлы-источники.
        destination_nodes: Узлы-назначения.
        graph_node_column: Имя колонки узла графа в обеих таблицах.
        weight: Вес ребра для расчета: ``time_min`` или ``length_meter``.
        dtype: Тип значений в возвращаемой матрице.
        threshold: Максимальная стоимость пути. Единица измерения зависит от
            ``weight``: минуты для ``time_min`` и метры для ``length_meter``.
            Пары без пути или дальше порога получают ``np.inf``.
        max_workers: Число параллельных потоков Numba для расчета.

    Returns:
        Разреженный ``pandas.DataFrame``. Если переданы ``gdf_origins`` и
        ``gdf_destinations``, индекс и колонки соответствуют индексам этих
        таблиц. Если переданы ``origins_nodes`` и ``destination_nodes``,
        индекс и колонки соответствуют спискам узлов.

    Raises:
        TypeError: Если ``graph`` имеет неверный тип или ``max_workers``
            не является целым числом.
        ValueError: Если списки узлов пустые, порог отрицательный или узлы
            отсутствуют в графе.
    """
    _validate_max_workers(max_workers)

    if (gdf_origins is not None and graph_node_column not in gdf_origins.columns) or (
        gdf_destinations is not None and graph_node_column not in gdf_destinations.columns
    ):
        logger.info(
            "OD matrix is calculated between nearest graph nodes. "
            "For more precise object-to-object distances, project objects into UrbanGraph first."
        )

    origin_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=origins_nodes,
        gdf=gdf_origins,
        graph_node_column=graph_node_column,
        nodes_name="origins_nodes",
        gdf_name="gdf_origins",
    )
    destination_nodes_s = resolve_graph_nodes_input(
        urban_graph=urban_graph,
        nodes=destination_nodes,
        gdf=gdf_destinations,
        graph_node_column=graph_node_column,
        nodes_name="destination_nodes",
        gdf_name="gdf_destinations",
    )
    origins_nodes = origin_nodes_s.to_list()
    destination_nodes = destination_nodes_s.to_list()

    transposed = len(destination_nodes) < len(origins_nodes)
    if transposed:
        calc_origins = destination_nodes
        calc_destinations = origins_nodes
    else:
        calc_origins = origins_nodes
        calc_destinations = destination_nodes

    csr_adj_matrix = _prepare_numba_graph(urban_graph, weight=weight, cutoff=threshold, reverse=transposed)

    if max_workers is not None:
        nb.set_num_threads(max_workers)

    origins_pos = _node_positions(urban_graph, calc_origins)
    destinations_pos = _node_positions(urban_graph, calc_destinations)

    dijkstra_numba_od_parallel(
        numba_adj_matrix=csr_adj_matrix, origins=origins_pos[:1], destinations=destinations_pos[:1], cutoff=np.int32(0)
    )

    coo_rows = dijkstra_numba_od_parallel(
        numba_adj_matrix=csr_adj_matrix,
        origins=origins_pos,
        destinations=destinations_pos,
        cutoff=_cutoff2int(threshold),
    )

    rows, cols, values = coo_rows_to_arrays(coo_rows)
    od_matrix = sparse.coo_matrix(
        (_int_weight2float(values).astype(dtype), (rows, cols)), shape=(len(calc_origins), len(calc_destinations))
    ).tocsr()

    if transposed:
        od_matrix = od_matrix.T.tocsr()

    return pd.DataFrame.sparse.from_spmatrix(
        od_matrix,
        index=origin_nodes_s.index,
        columns=destination_nodes_s.index,
    ).astype(pd.SparseDtype(dtype, fill_value=np.inf))
