API Reference
=============

.. currentmodule:: iduedu._api

High-level functions
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_drive_graph
   get_walk_graph
   get_all_public_transport_graph
   get_single_public_transport_graph
   get_intermodal_graph
   join_pt_walk_graph

Graph utilities
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   graph_to_gdf
   gdf_to_graph
   clip_nx_graph
   keep_largest_strongly_connected_component
   write_gml
   read_gml
   reproject_graph

Matrices
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_adj_matrix_gdf_to_gdf
   get_closest_nodes

Overpass helpers
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_4326_boundary