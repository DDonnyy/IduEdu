import concurrent.futures

from shapely import Polygon, MultiPolygon

from iduedu.modules.drive_walk_builder import get_walk_graph
from iduedu.modules.downloaders import get_boundary
from iduedu.modules.pt_walk_joiner import join_pt_walk_graph
from iduedu.modules.public_transport_builder import get_all_public_transport_graph

from loguru import logger


def get_intermodal_graph(
    osm_id: int | None = None,
    territory_name: str | None = None,
    polygon: Polygon | MultiPolygon | None = None,
    clip_by_bounds=False,
):
    """
    intermodal graph - composition of public transport and walk graphes, where platforms are the connecting link between two
    :param clip_by_bounds:
    :param territory_name:
    :param osm_id:
    :param polygon should be in CRS 4326
    :return:
    """

    boundary = get_boundary(osm_id, territory_name, polygon)
    with concurrent.futures.ProcessPoolExecutor() as executor:  # TODO Прокинуть логи наверх если реально...
        walk_graph_future = executor.submit(get_walk_graph, polygon=boundary)
        logger.info("Started downloading and parsing walk graph...")
        pt_graph_future = executor.submit(
            get_all_public_transport_graph, polygon=boundary, clip_by_bounds=clip_by_bounds
        )
        logger.info("Started downloading and parsing public trasport graph...")
        pt_G = pt_graph_future.result()
        logger.info("Public trasport graph done!")
        walk_G = walk_graph_future.result()
        logger.info("Walk graph done!")
    intermodal = join_pt_walk_graph(pt_G, walk_G)
    return intermodal
