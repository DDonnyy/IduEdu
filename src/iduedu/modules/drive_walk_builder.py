def get_graph_from_polygon(
    polygon: gpd.GeoDataFrame, filter: str = None, crs: int = 3857, country_polygon=None
) -> nx.MultiDiGraph:
    """Получение графа на основе полигона."""
    buffer_polygon = buffer_and_transform_polygon(polygon, crs)
    if not filter:
        filter = f"['highway'~'motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street']"
    graph = ox.graph_from_polygon(
        buffer_polygon,
        network_type="drive",
        custom_filter=filter,
        truncate_by_edge=True,
    )
    graph.graph["approach"] = "primal"
    nodes, edges = momepy.nx_to_gdf(
        graph, points=True, lines=True, spatial_weights=False
    )
    edges = add_geometry_to_edges(nodes, edges)
    edges["reg"] = edges.apply(
        lambda row: determine_reg(row["ref"], row["highway"]), axis=1
    )
    nodes = nodes.to_crs(epsg=crs)
    edges = edges.to_crs(epsg=crs)
    edges["exit"] = 0
    edges = update_edges_with_geometry(edges, polygon, crs)
    edges["maxspeed"] = edges["highway"].apply(lambda x: get_max_speed(x))
    nodes_coord = update_nodes_with_geometry(edges, {})
    edges = edges[
        [
            "highway",
            "node_start",
            "node_end",
            "geometry",
            "maxspeed",
            "reg",
            "ref",
            "exit",
        ]
    ]
    edges["type"] = "car"
    edges["geometry"] = edges["geometry"].apply(
        lambda x: LineString(
            [tuple(round(c, 6) for c in n) for n in x.coords] if x else None
        )
    )
    G = create_graph(edges, nodes_coord, "car")
    G = set_node_attributes(
        G, nodes_coord, polygon, crs, country_polygon=country_polygon
    )
    G.graph["crs"] = "epsg:" + str(crs)
    G.graph["approach"] = "primal"
    G.graph["graph_type"] = "car graph"
    return G