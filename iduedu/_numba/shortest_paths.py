from heapq import heappop, heappush

import numba as nb
import numpy as np
from numba.typed import List

from iduedu._numba import UI32CSRMatrix, ui32csr_type

coo_pair_type = nb.types.UniTuple(nb.int32, 2)
coo_triplet_type = nb.types.UniTuple(nb.int32, 3)
coo_row_type = nb.types.ListType(coo_pair_type)
coo_triplet_row_type = nb.types.ListType(coo_triplet_type)
coo_rows_type = nb.types.ListType(coo_row_type)
dijkstra_owner_heap_item_type = nb.types.UniTuple(nb.int32, 3)


def _njit(signature=None, **kwargs):
    if signature is None:
        return nb.njit(**kwargs)
    return nb.njit(signature, **kwargs)


_dijkstra_from_fringe_signature = None
_dijkstra_distances_signature = None
_single_source_signature = None
_multi_source_signature = None
_nearest_source_signature = None
_od_parallel_signature = None
_path_parallel_signature = None
if ui32csr_type is not None:
    _dijkstra_from_fringe_signature = nb.types.Array(nb.int32, 1, "C")(
        ui32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        coo_row_type,
        nb.int32,
    )
    _dijkstra_distances_signature = nb.types.Array(nb.int32, 1, "C")(
        ui32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.int32,
    )
    _single_source_signature = coo_row_type(
        ui32csr_type,
        nb.int32,
        nb.int32,
    )
    _multi_source_signature = coo_row_type(
        ui32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.int32,
    )
    _nearest_source_signature = coo_triplet_row_type(
        ui32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.int32,
    )
    _od_parallel_signature = coo_rows_type(
        ui32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.int32,
    )
    _path_parallel_signature = coo_rows_type(
        ui32csr_type,
        nb.types.Array(nb.int32, 1, "C"),
        nb.int32,
    )


@_njit(_dijkstra_from_fringe_signature, cache=True)
def _dijkstra_numba_distances_from_fringe(
    numba_adj_matrix: UI32CSRMatrix, seen: np.ndarray, fringe: list, cutoff: np.int32
):
    distances = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)

    while len(fringe) > 0:
        d, v = heappop(fringe)
        if distances[v] != -1:
            continue
        distances[v] = d
        cols = numba_adj_matrix.get_cols(v)
        vals = numba_adj_matrix.get_vals(v)
        for edge_i in range(len(cols)):
            u = cols[edge_i]
            weight = vals[edge_i]
            uv_dist = np.int32(d + weight)
            if uv_dist > cutoff:
                continue
            if seen[u] == -1 or uv_dist < seen[u]:
                seen[u] = uv_dist
                heappush(fringe, (uv_dist, np.int32(u)))

    return distances


@_njit(_dijkstra_distances_signature, cache=True)
def _dijkstra_numba_distances(numba_adj_matrix: UI32CSRMatrix, origins: np.ndarray, cutoff: np.int32):
    seen = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)
    fringe = List.empty_list(coo_pair_type)

    for origin in origins:
        if seen[origin] == -1:
            seen[origin] = np.int32(0)
            heappush(fringe, (np.int32(0), np.int32(origin)))

    return _dijkstra_numba_distances_from_fringe(numba_adj_matrix, seen, fringe, cutoff)


@_njit(_single_source_signature, cache=True)
def single_source_dijkstra_numba_path_length(numba_adj_matrix: UI32CSRMatrix, origin: np.int32, cutoff: np.int32):
    """Run single-source Dijkstra and return reachable node distances."""
    seen = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)
    fringe = List.empty_list(coo_pair_type)
    seen[origin] = np.int32(0)
    fringe.append((np.int32(0), np.int32(origin)))
    distances = _dijkstra_numba_distances_from_fringe(numba_adj_matrix, seen, fringe, cutoff)

    row = List.empty_list(coo_pair_type)
    for node in range(len(distances)):
        distance = distances[node]
        if distance >= 0:
            row.append((np.int32(node), np.int32(distance)))
    return row


@_njit(_multi_source_signature, cache=True)
def multi_source_dijkstra_numba_path_length(numba_adj_matrix: UI32CSRMatrix, sources: np.ndarray, cutoff: np.int32):
    """Run multi-source Dijkstra and return nearest-source distances."""
    distances = _dijkstra_numba_distances(numba_adj_matrix, sources, cutoff)

    row = List.empty_list(coo_pair_type)
    for node in range(len(distances)):
        distance = distances[node]
        if distance >= 0:
            row.append((np.int32(node), np.int32(distance)))
    return row


@_njit(_nearest_source_signature, cache=True)
def multi_source_dijkstra_numba_nearest_source(numba_adj_matrix: UI32CSRMatrix, origins: np.ndarray, cutoff: np.int32):
    """Run multi-source Dijkstra and return nearest source ids with distances."""
    distances = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)
    seen = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)
    seen_source = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)
    fringe = List.empty_list(dijkstra_owner_heap_item_type)

    for origin in origins:
        if seen[origin] == -1 or origin < seen_source[origin]:
            seen[origin] = np.int32(0)
            seen_source[origin] = origin
            heappush(fringe, (np.int32(0), np.int32(origin), np.int32(origin)))

    while len(fringe) > 0:
        d, owner, v = heappop(fringe)
        if distances[v] != -1:
            continue
        distances[v] = d
        cols = numba_adj_matrix.get_cols(v)
        vals = numba_adj_matrix.get_vals(v)
        for edge_i in range(len(cols)):
            u = cols[edge_i]
            weight = vals[edge_i]
            uv_dist = np.int32(d + weight)
            if uv_dist > cutoff:
                continue
            if seen[u] == -1 or uv_dist < seen[u] or (uv_dist == seen[u] and owner < seen_source[u]):
                seen[u] = uv_dist
                seen_source[u] = owner
                heappush(fringe, (uv_dist, owner, np.int32(u)))

    row = List.empty_list(coo_triplet_type)
    for node in range(len(distances)):
        if distances[node] >= 0:
            row.append((np.int32(node), seen_source[node], distances[node]))
    return row


@_njit(_od_parallel_signature, cache=True, parallel=True)
def dijkstra_numba_od_parallel(
    numba_adj_matrix: UI32CSRMatrix, origins: np.ndarray, destinations: np.ndarray, cutoff: np.int32
):
    """Compute sparse OD rows for many origins in parallel."""
    result = List.empty_list(coo_row_type)
    for _ in range(len(origins)):
        result.append(List.empty_list(coo_pair_type))
    for i in nb.prange(len(origins)):
        seen = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)
        fringe = List.empty_list(coo_pair_type)
        source = origins[i]
        seen[source] = np.int32(0)
        fringe.append((np.int32(0), np.int32(source)))
        distances = _dijkstra_numba_distances_from_fringe(numba_adj_matrix, seen, fringe, cutoff)
        row = List.empty_list(coo_pair_type)
        for j in range(len(destinations)):
            distance = distances[destinations[j]]
            if distance >= 0:
                row.append((np.int32(j), np.int32(distance)))
        result[i] = row
    return result


@_njit(_path_parallel_signature, cache=True, parallel=True)
def dijkstra_numba_path_length_parallel(numba_adj_matrix: UI32CSRMatrix, origins: np.ndarray, cutoff: np.int32):
    """Compute sparse path-length rows for many origins in parallel."""
    result = List.empty_list(coo_row_type)
    for _ in range(len(origins)):
        result.append(List.empty_list(coo_pair_type))
    for i in nb.prange(len(origins)):
        seen = np.full(numba_adj_matrix.tot_rows, -1, dtype=np.int32)
        fringe = List.empty_list(coo_pair_type)
        source = origins[i]
        seen[source] = np.int32(0)
        fringe.append((np.int32(0), np.int32(source)))
        distances = _dijkstra_numba_distances_from_fringe(numba_adj_matrix, seen, fringe, cutoff)
        row = List.empty_list(coo_pair_type)
        for node in range(len(distances)):
            distance = distances[node]
            if distance >= 0:
                row.append((np.int32(node), np.int32(distance)))
        result[i] = row
    return result
