import networkx as nx
import pandas as pd
from shapely import Point, LineString
import geopandas as gpd
from loguru import logger
from shapely.ops import substring


def join_pt_walk_graph(public_transport_G: nx.Graph, walk_G: nx.Graph, max_dist=20) -> nx.Graph:

    assert public_transport_G.graph["crs"] == walk_G.graph["crs"], "CRS mismatching."
    logger.info("Composing intermodal graph...")
    num_nodes_G1 = len(public_transport_G.nodes)
    mapping_G1 = {node: idx for idx, node in enumerate(public_transport_G.nodes)}
    mapping_G2 = {node: idx + num_nodes_G1 for idx, node in enumerate(walk_G.nodes)}

    transport: nx.MultiDiGraph = nx.relabel_nodes(public_transport_G, mapping_G1)
    walk: nx.MultiDiGraph = nx.relabel_nodes(walk_G, mapping_G2)

    platforms = pd.DataFrame.from_dict(dict(transport.nodes(data=True)), orient="index")
    platforms = platforms[platforms["desc"] == "platform"]
    platforms["geometry"] = platforms.apply(lambda x: Point(x.x, x.y), axis=1)
    platforms = gpd.GeoDataFrame(platforms, crs=transport.graph["crs"])
    walk_edges = gpd.GeoDataFrame(
        nx.to_pandas_edgelist(walk, source="u", target="v", edge_key="k"), crs=walk.graph["crs"]
    )

    projection_join = platforms.sjoin_nearest(walk_edges, max_distance=max_dist, distance_col="dist")

    points_grouped_by_edge = (
        projection_join.reset_index()
        .groupby("index_right", group_keys=True)
        .agg({"index": tuple, "geometry": tuple, "u": "first", "v": "first", "k": "first"})
    )

    try:
        speed = walk.graph["walk_speed"]
    except KeyError:
        logger.warning(
            "Нету скорости, выставляю по умолчанию 83 м/мин"
        )  # TODO посчитать примерную скорость по length timemin для любой эджи
        speed = 83

    edges_to_del = []

    for _, row in points_grouped_by_edge.iterrows():
        u, v, k = row[["u", "v", "k"]]
        edge: LineString = walk_edges.loc[row.name].geometry

        # Если платформа одна единственная на эдж
        if len(row["index"]) == 1:
            dist = edge.project(row.geometry[0])
            projected_point = edge.interpolate(dist)
            if dist == 0:
                # Если платформа проецируется на начало эджа
                mapping = {u: row["index"][0]}
                nx.relabel_nodes(walk, mapping, copy=False)

            elif dist == edge.length:
                # Если на конец
                mapping = {v: row["index"][0]}
                nx.relabel_nodes(walk, mapping, copy=False)

            else:

                line1 = substring(edge, 0, dist)
                line2 = substring(edge, dist, edge.length)
                # Убираем старую эджу и добавляем новые, в серединке нода - наша платформа для соединения
                # walk.remove_edge(u, v, k)
                edges_to_del.append((u, v, k))
                walk.add_node(row["index"][0], x=projected_point.x, y=projected_point.y)
                walk.add_edge(
                    u,
                    row["index"][0],
                    geometry=line1,
                    length=round(line1.length, 3),
                    time_min=round(line1.length / speed, 3),
                    type="walk",
                )  # TODO заменить type на walk
                walk.add_edge(
                    row["index"][0],
                    v,
                    geometry=line2,
                    length=round(line2.length, 3),
                    time_min=round(line2.length / speed, 3),
                    type="walk",
                )
        # Если платформм несколько на эдж
        else:
            edges_to_del.append((u, v, k))
            dist_project = [
                (edge.project(platform), edge.interpolate(edge.project(platform)), ind)
                for ind, platform in zip(row["index"], row.geometry)
            ]
            dist_project.sort(key=lambda x: x[0])
            last_dist = 0
            last_u = u
            for dist, projected_point, cur_index in dist_project:

                if dist == 0:
                    # Если платформа проецируется на начало эджа
                    mapping = {u: cur_index}
                    nx.relabel_nodes(walk, mapping, copy=False)

                elif dist == edge.length:
                    # Если на конец
                    mapping = {v: cur_index}
                    nx.relabel_nodes(walk, mapping, copy=False)

                else:
                    # По очереди добавляем ноды/линии

                    line = substring(edge, last_dist, dist)
                    #
                    walk.add_node(cur_index, x=projected_point.x, y=projected_point.y)
                    walk.add_edge(
                        last_u,
                        cur_index,
                        geometry=line,
                        length=round(line.length, 3),
                        time_min=round(line.length / speed, 3),
                        type="walk",
                    )  # TODO заменить type на walk

                    last_u = cur_index
                last_dist = dist
            # Если последняя остановка спроецировалась не на конец эджа, надо добавить остаток
            if last_dist != edge.length:
                line = substring(edge, last_dist, edge.length)
                walk.add_edge(
                    last_u,
                    v,
                    geometry=line,
                    length=round(line.length, 3),
                    time_min=round(line.length / speed, 3),
                    type="walk",
                )
    walk.remove_edges_from(edges_to_del)
    intermodal = nx.compose(nx.MultiDiGraph(transport), nx.MultiDiGraph(walk))
    logger.info("Done composing!")
    return intermodal
