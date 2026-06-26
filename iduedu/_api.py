# pylint: disable=unused-import
from .constants.transport_specs import DEFAULT_REGISTRY, TransportRegistry, TransportSpec
from .graph import (
    UrbanGraph,
    UrbanGraphChanges,
    apply_urban_graph_changes,
    clip_urban_graph,
    join_urban_graphs,
    nx_graph2urban_graph,
    project_objects2urban_graph,
    relabel_urban_graph,
    simplify_multiedges,
    to_directed,
    to_undirected,
    urban_graph2nx_graph,
)
from .graph_builders.drive_walk_builders import get_drive_graph, get_walk_graph
from .graph_builders.intermodal_builders import get_intermodal_graph, join_pt_walk_graph
from .graph_builders.public_transport_builders import get_public_transport_graph
from .overpass.overpass_downloaders import get_4326_boundary
from .routing import (
    dijkstra_path_length_parallel,
    multi_source_dijkstra_nearest_source,
    multi_source_dijkstra_path_length,
    od_matrix,
    single_source_dijkstra_path_length,
)


def _missing_optional_function(name: str, import_error: ModuleNotFoundError):
    def _missing(*args, **kwargs):
        raise ImportError(f"{name} requires optional NetworkX support.") from import_error

    return _missing


try:
    from .graph.nx_graph_utils import (
        clip_nx_graph,
        gdf2graph,
    )
    from .graph.nx_graph_utils import gml2nx_graph as read_gml
    from .graph.nx_graph_utils import (
        graph2gdf,
        keep_largest_connected_component,
    )
    from .graph.nx_graph_utils import nx_graph2gml as write_gml
    from .graph.nx_graph_utils import reproject_nx_graph as reproject_graph
except ModuleNotFoundError as exc:
    if exc.name != "networkx":
        raise
    clip_nx_graph = _missing_optional_function("clip_nx_graph", exc)
    gdf2graph = _missing_optional_function("gdf2graph", exc)
    graph2gdf = _missing_optional_function("graph2gdf", exc)
    keep_largest_connected_component = _missing_optional_function("keep_largest_connected_component", exc)
    read_gml = _missing_optional_function("read_gml", exc)
    reproject_graph = _missing_optional_function("reproject_graph", exc)
    write_gml = _missing_optional_function("write_gml", exc)
