Graph utilities
---------------

.. currentmodule:: iduedu

.. autosummary::
    :toctree: generated
    :nosignatures:

    connected_components
    weakly_connected_components
    strongly_connected_components
    number_connected_components
    number_weakly_connected_components
    number_strongly_connected_components
    largest_connected_component
    largest_weakly_connected_component
    largest_strongly_connected_component
    largest_component
    relabel_urban_graph
    subgraph_by_nodes
    clip_urban_graph
    join_urban_graphs
    project_objects2urban_graph
    apply_urban_graph_changes

Optional NetworkX utilities
---------------------------

The following helpers are available when NetworkX support is installed.

.. autosummary::
    :toctree: generated
    :nosignatures:

    graph2gdf
    gdf2graph
    keep_largest_nx_component
    clip_nx_graph
    read_gml
    write_gml
    reproject_graph
