import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely import line_merge, MultiLineString, LineString, Point
from iduedu.enums.network_enums import Network
from iduedu.enums.drive_enums import HighwayType
from iduedu.modules.overpass_downloaders import get_network_by_filters
from iduedu.utils.utils import estimate_crs_for_bounds, keep_largest_strongly_connected_component
from iduedu.modules.overpass_downloaders import get_boundary
from iduedu import config

logger = config.logger


def get_highway_properties(highway) -> tuple[int, float]:
    default_speed = 40 * 1000 / 60  # m/min
    if not highway:
        return 3, default_speed
    highway_list = highway if isinstance(highway, list) else [highway]
    reg_values, speed_values = [], []
    for ht in highway_list:
        try:
            enum_value = HighwayType[ht.upper()]
            reg_values.append(enum_value.reg_status)
            speed_values.append(enum_value.max_speed)
        except KeyError:
            continue
    if not reg_values or not speed_values:
        return 3, default_speed
    return min(reg_values), max(speed_values)


def _build_edges_from_overpass(polygon, way_filter: str, simplify: bool = True) -> gpd.GeoDataFrame:

    data = get_network_by_filters(polygon, way_filter)
    ways = data[data["type"] == "way"].copy()

    # Собираем координаты каждой линии (lon, lat)
    coords_list = [np.asarray([(p["lon"], p["lat"]) for p in pts], dtype="f8") for pts in ways["geometry"].values]

    # сегментация на отрезки
    starts = np.concatenate([a[:-1] for a in coords_list], axis=0)
    ends = np.concatenate([a[1:] for a in coords_list], axis=0)

    lengths = np.array([a.shape[0] for a in coords_list], dtype=int)
    seg_counts = np.maximum(lengths - 1, 0)
    way_idx = np.repeat(ways.index.values, seg_counts)

    geoms = [LineString([tuple(s), tuple(e)]) for s, e in zip(starts, ends)]

    # локальная проекция
    local_crs = estimate_crs_for_bounds(*polygon.bounds).to_epsg()

    edges = gpd.GeoDataFrame({"way_idx": way_idx}, geometry=geoms, crs=4326).to_crs(local_crs)
    edges = edges.join(ways[["id", "tags"]], on="way_idx")

    if simplify:
        # сшиваем ребра и переносим атрибуты через midpoints → nearest
        merged = list(line_merge(MultiLineString(edges.geometry.to_list()), directed=True).geoms)
        lines = gpd.GeoDataFrame(geometry=merged, crs=local_crs)

        mid = lines.copy()
        mid.geometry = mid.interpolate(lines.length / 2)

        # TODO Вот тут подумать надо, теряются аттрибуты, надо ли складывать их "по умному", в листы.
        joined = gpd.sjoin_nearest(mid[["geometry"]], edges, how="left", max_distance=1)
        joined = joined.reset_index().drop_duplicates(subset="index").set_index("index")
        lines = lines.join(joined[["tags", "id", "way_idx"]])

        edges = lines

    return edges, local_crs


def get_drive_graph(
    osm_id: int | None = None,
    polygon=None,
    additional_edgedata: list[str] | str | None = None,
    retain_all: bool = False,
    simplify: bool = True,
    road_filter: str | None = None,
) -> nx.MultiDiGraph:


    if additional_edgedata is None:
        additional_edgedata = []

    polygon = get_boundary(osm_id, polygon)
    if road_filter is 'drive_service':
        road_filter = Network.DRIVE_SERVICE.filter

    if road_filter is None:
        road_filter = Network.DRIVE.filter

    logger.info("Downloading drive network via Overpass ...")
    edges, local_crs = _build_edges_from_overpass(polygon, road_filter, simplify=simplify)

    TAG_WHITELIST = {"oneway", "highway", "name", "maxspeed", "lanes"}
    tags_df = pd.DataFrame.from_records(
        ({k: v for k, v in d.items() if k in TAG_WHITELIST} for d in edges["tags"]),
        index=edges.index,
    )
    edges = edges.join(tags_df)

    # двусторонние — дублируем с реверсом
    two_way = edges[edges["oneway"] != "yes"].copy()
    two_way.geometry = two_way.geometry.reverse()
    edges = pd.concat([edges, two_way], ignore_index=True)

    # u/v и узлы
    coords = edges.geometry.get_coordinates().to_numpy()
    counts = edges.geometry.count_coordinates()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1
    starts = coords[first_idx]
    ends = coords[last_idx]
    edges["start"] = list(map(tuple, starts))
    edges["end"] = list(map(tuple, ends))

    all_endpoints = pd.Index(edges["start"]).append(pd.Index(edges["end"]))
    labels, uniques = pd.factorize(all_endpoints)
    n = len(edges)
    u = labels[:n]
    v = labels[n:]
    edges["u"] = u
    edges["v"] = v

    # длины и время (drive): скорость из HighwayType
    # (maxspeed в теге может быть строкой; используем enum через 'highway')
    # сначала определим reg и скорость (m/min)
    reg_speed = edges["highway"].map(lambda h: pd.Series(get_highway_properties(h)), na_action=None)
    edges[["reg", "maxspeed_mpm"]] = reg_speed.values
    edges["length_meter"] = edges.geometry.length.round(3)
    # защищаемся от деления на 0
    edges["time_min"] = (edges["length_meter"] / edges["maxspeed_mpm"].replace({0: np.nan})).round(3)

    # дополнительные поля
    if additional_edgedata != "save_all":
        for col in additional_edgedata:
            if col not in edges.columns:
                edges[col] = None

    # строим MultiDiGraph
    G = nx.MultiDiGraph()

    # TODO fix без for
    # Узлы с координатами
    for i, (x, y) in enumerate(uniques):
        G.add_node(i, x=float(x), y=float(y), geometry=Point(x, y))

    # TODO fix проработать additional data
    # Список атрибутов для ребра
    edge_attr_cols = [
        "length_meter",
        "time_min",
        "geometry",
        "oneway",
        "highway",
        "name",
        "maxspeed",
        "lanes",
        "reg",
        "maxspeed_mpm",
        "id",
        "way_idx",
    ]
    if additional_edgedata == "save_all":
        # всё уже в edges; выберем всё кроме служебных u/v/start/end
        edge_attr_cols = [c for c in edges.columns if c not in {"u", "v", "start", "end"}]
    else:
        edge_attr_cols += [c for c in additional_edgedata if c not in edge_attr_cols]

    attrs_iter = edges[edge_attr_cols].to_dict("records")\
    # TODO fix
    G.add_edges_from((int(uu), int(vv), d) for uu, vv, d in zip(u, v, attrs_iter))

    if not retain_all:
        G = keep_largest_strongly_connected_component(G)

    # перенумерация узлов (как в осмнксовском билдере)
    mapping = {old: new for new, old in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    G.graph["crs"] = local_crs
    G.graph["type"] = "drive"
    logger.debug("Drive graph built.")
    return G


def get_walk_graph(
    osm_id: int | None = None,
    polygon=None,
    walk_speed: float = 5 * 1000 / 60,  # m/min
    retain_all: bool = False,
    simplify: bool = True,
    road_filter: str | None = None,
) -> nx.MultiDiGraph:

    polygon = get_boundary(osm_id, polygon)
    if road_filter is None:
        road_filter = Network.WALK.filter

    logger.info("Downloading walk network via Overpass ...")
    edges, local_crs = _build_edges_from_overpass(polygon, road_filter, simplify=simplify)

    # длина/время
    edges["length_meter"] = edges.geometry.length.round(3)
    edges["time_min"] = (edges["length_meter"] / walk_speed).round(3)

    # u/v и узлы
    coords = edges.geometry.get_coordinates().to_numpy()
    counts = edges.geometry.count_coordinates()
    cuts = np.cumsum(counts)
    first_idx = np.r_[0, cuts[:-1]]
    last_idx = cuts - 1
    starts = coords[first_idx]
    ends = coords[last_idx]
    edges["start"] = list(map(tuple, starts))
    edges["end"] = list(map(tuple, ends))

    all_endpoints = pd.Index(edges["start"]).append(pd.Index(edges["end"]))
    labels, uniques = pd.factorize(all_endpoints)
    n = len(edges)
    u = labels[:n]
    v = labels[n:]

    G = nx.MultiDiGraph()
    for i, (x, y) in enumerate(uniques):
        G.add_node(i, x=float(x), y=float(y), geometry=Point(x, y))

    attrs_iter = edges[["length_meter", "time_min", "geometry"]].to_dict("records")
    G.add_edges_from((int(uu), int(vv), d) for uu, vv, d in zip(u, v, attrs_iter))

    if not retain_all:
        G = keep_largest_strongly_connected_component(G)

    mapping = {old: new for new, old in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    G.graph["crs"] = local_crs
    G.graph["walk_speed"] = walk_speed
    G.graph["type"] = "walk"
    logger.debug("Walk graph built.")
    return G
