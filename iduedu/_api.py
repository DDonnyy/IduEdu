# pylint: disable=unused-import
from .constants.transport_specs import DEFAULT_REGISTRY, TransportRegistry, TransportSpec
from .graph_builders.drive_walk_builders import get_drive_graph, get_walk_graph
from .overpass.overpass_downloaders import get_4326_boundary


def _missing_networkx_function(name: str, import_error: ModuleNotFoundError):
    def _missing(*args, **kwargs):
        raise ImportError(
            f"{name} still depends on optional NetworkX-based modules and has not been migrated to UrbanGraph yet."
        ) from import_error

    return _missing


try:
    from .graph_builders.intermodal_builders import get_intermodal_graph, join_pt_walk_graph
except ModuleNotFoundError as exc:
    if exc.name != "networkx":
        raise
    get_intermodal_graph = _missing_networkx_function("get_intermodal_graph", exc)
    join_pt_walk_graph = _missing_networkx_function("join_pt_walk_graph", exc)

try:
    from .graph_builders.public_transport_builders import get_public_transport_graph
except ModuleNotFoundError as exc:
    if exc.name != "networkx":
        raise
    get_public_transport_graph = _missing_networkx_function("get_public_transport_graph", exc)


try:
    from .matrix.matrix_builder import get_adj_matrix_gdf_to_gdf, get_closest_nodes, get_od_matrix_gdf_to_gdf
except ModuleNotFoundError as exc:
    if exc.name != "networkx":
        raise
    get_adj_matrix_gdf_to_gdf = _missing_networkx_function("get_adj_matrix_gdf_to_gdf", exc)
    get_closest_nodes = _missing_networkx_function("get_closest_nodes", exc)
    get_od_matrix_gdf_to_gdf = _missing_networkx_function("get_od_matrix_gdf_to_gdf", exc)

try:
    from .modules.graph_transformers import (
        clip_nx_graph,
        gdf_to_graph,
        graph_to_gdf,
        keep_largest_connected_component,
        read_gml,
        reproject_graph,
        write_gml,
    )
except ModuleNotFoundError as exc:
    if exc.name != "networkx":
        raise
    clip_nx_graph = _missing_networkx_function("clip_nx_graph", exc)
    gdf_to_graph = _missing_networkx_function("gdf_to_graph", exc)
    graph_to_gdf = _missing_networkx_function("graph_to_gdf", exc)
    keep_largest_connected_component = _missing_networkx_function("keep_largest_connected_component", exc)
    read_gml = _missing_networkx_function("read_gml", exc)
    reproject_graph = _missing_networkx_function("reproject_graph", exc)
    write_gml = _missing_networkx_function("write_gml", exc)
