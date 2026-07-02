import pytest

from iduedu import graph2gdf

pytestmark = [pytest.mark.network, pytest.mark.slow]


def test_graph_to_gdf_restore_geom_integration(intermodal_graph):
    graph_gdf = graph2gdf(intermodal_graph, restore_edge_geom=True)
    assert graph_gdf is not None
    assert not graph_gdf.empty
