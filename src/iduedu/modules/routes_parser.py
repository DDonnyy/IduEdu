import math
import warnings

from pandas import DataFrame
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import pandas as pd
from pyproj import Transformer
import numpy as np
from shapely import Point
from shapely.geometry import LineString
from shapely.ops import substring

PLATFORM_ROLES = ['platform_entry_only', 'platform', 'platform_exit_only']
STOPS_ROLES = ['stop', 'stop_exit_only', 'stop_entry_only']


def estimate_crs_for_overpass(overpass_data):
    def find_bounds(bounds):
        df_expanded = pd.json_normalize(bounds)
        min_lat = df_expanded['minlat'].min()
        min_lon = df_expanded['minlon'].min()
        max_lat = df_expanded['maxlat'].max()
        max_lon = df_expanded['maxlon'].max()
        return min_lat, min_lon, max_lat, max_lon

    min_lat, min_lon, max_lat, max_lon = find_bounds(overpass_data['bounds'])
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=min_lon,
            south_lat_degree=min_lat,
            east_lon_degree=max_lon,
            north_lat_degree=max_lat,
        ),
    )
    return CRS.from_epsg(utm_crs_list[0].code)


def _link_unconnected(connected_ways, threshold) -> list:
    # Считаем связи между линиями
    connect_points = [point for coords in connected_ways for point in (coords[0], coords[-1])]
    distances = cdist(connect_points, connect_points)
    n = distances.shape[0]
    mask = (np.arange(n)[:, None] // 2) == (np.arange(n) // 2)
    distances[mask] = np.inf
    indexes = []
    for i in range(len(connected_ways) - 1):
        min_index = np.unravel_index(np.argmin(distances), distances.shape)
        if distances[min_index] > threshold:
            way_inds = [way_ind for x in indexes for way_ind in (x[0] // 2, x[1] // 2)]
            return _link_unconnected([way for i, way in enumerate(connected_ways) if i in way_inds], threshold)
        distances[min_index[0], :] = np.inf
        distances[min_index[1], :] = np.inf
        distances[:, min_index[0]] = np.inf
        distances[:, min_index[1]] = np.inf
        indexes.append(min_index)

    # Определяем "начальную линию"
    way_inds = [way_ind for x in indexes for way_ind in (x[0] // 2, x[1] // 2)]
    first = next((elem for elem in way_inds if way_inds.count(elem) == 1))

    find_ind = first * 2, first * 2 + 1
    new_connected_ways = None

    # Соединяем воедино
    for i in range(len(connected_ways) - 1):
        index = next((i for i, t in enumerate(indexes) if (find_ind[0] in t or find_ind[1] in t)))
        connection = indexes.pop(index)
        if new_connected_ways is None:
            first = next((x for x in connection if x in find_ind))
            if first % 2 != 0:
                new_connected_ways = connected_ways[first // 2]
            else:
                new_connected_ways = connected_ways[first // 2][::-1]

        next_line = next(x for x in connection if x not in find_ind)
        find_ind = (next_line, next_line + 1) if next_line % 2 == 0 else (next_line, next_line - 1)

        if next_line % 2 == 0:
            new_connected_ways += connected_ways[next_line // 2]
        else:

            new_connected_ways += connected_ways[next_line // 2][::-1]
    return new_connected_ways


def parse_overpass_route_response(loc: dict, crs: CRS) -> pd.Series:
    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)

    def transform_geometry(loc):
        if isinstance(loc['geometry'], float):
            return transformer.transform(loc["lon"], loc["lat"])
        else:
            p = LineString([transformer.transform(coords["lon"], coords["lat"]) for coords in loc['geometry']]).centroid
            return p.x, p.y

    def process_roles(route, roles):
        filtered = route[route['role'].isin(roles)]
        if len(filtered) == 0:
            return None
        else:
            return filtered.apply(transform_geometry, axis=1).tolist()

    if 'ref' in loc['tags'].keys():
        transport_name = loc['tags']['ref']
    elif 'name' in loc['tags'].keys():
        transport_name = loc['tags']['name']
    else:
        transport_name = None

    route = pd.DataFrame(loc["members"])

    platforms = process_roles(route, PLATFORM_ROLES)
    stops = process_roles(route, STOPS_ROLES)

    ways = route[(route["type"] == "way") & (route["role"] == '')]

    if len(ways) > 0:
        ways = ways["geometry"].reset_index(drop=True)
        ways = ways.apply(lambda x: ([transformer.transform(coords["lon"], coords["lat"]) for coords in x])).tolist()

        connected_ways = [[]]
        cur_way = 0
        for coords in ways:
            # Соединяем маршруты, если всё ок, идут без пропусков
            if not connected_ways[cur_way]:
                connected_ways[cur_way] += coords
                continue

            if coords[0] == coords[-1]:
                # Круговое движение зацикленное зачастую в осм, можно отработать, но сходу не придумал
                continue
            if connected_ways[cur_way][-1] == coords[0]:
                connected_ways[cur_way] += coords[1:]
            elif connected_ways[cur_way][-1] == coords[-1]:
                connected_ways[cur_way] += coords[::-1][1:]
            elif connected_ways[cur_way][0] == coords[0]:
                connected_ways[cur_way] = coords[1:][::-1] + connected_ways[cur_way]
            elif connected_ways[cur_way][0] == coords[-1]:
                connected_ways[cur_way] = coords + connected_ways[cur_way][1:]
            # Случай если нету соединяющей координаты
            else:
                connected_ways += [coords]
                cur_way += 1
        # Соединяем линии по ближайшим точкам этих линий
        if len(connected_ways) > 1:
            # Check if any loops in data and remove it
            to_del = [i for i, data in enumerate(connected_ways) if (data[0] == data[-1])]
            connected_ways = [i for j, i in enumerate(connected_ways) if j not in to_del]
        if len(connected_ways) > 1:
            connected_ways = _link_unconnected(connected_ways, threshold=500)
        else:
            connected_ways = connected_ways[0]

    else:
        connected_ways = None

    return pd.Series({"path": connected_ways, "platforms": platforms, 'stops': stops, 'route': transport_name})


def geometry_to_graph_edge_node_df(loc: pd.Series, transport_type) -> DataFrame | None:
    graph_data = []
    node_id = 0
    name = loc.route
    last_dist = None
    last_projected_stop_id = None
    platforms = loc.platforms
    stops = loc.stops
    path = loc.path

    def add_node(desc, x, y, transport=None):
        if not transport:
            graph_data.append({'node_id': (loc.name, node_id), 'point': (x, y), 'desc': desc, 'route': name})
        else:
            graph_data.append(
                {'node_id': (loc.name, node_id), 'point': (x, y), 'desc': desc, 'route': name, 'type': transport_type})

    def add_edge(u, v, geometry=None, desc=None, transport=None):
        if not transport:
            graph_data.append(
                {'u': (loc.name, u), 'v': (loc.name, v), 'geometry': geometry, 'desc': desc, 'route': name})
        else:
            graph_data.append(
                {'u': (loc.name, u), 'v': (loc.name, v), 'geometry': geometry, 'desc': desc, 'route': name,
                 'type': transport_type})

    def offset_direction(point):
        # 1 if left 0 if right для определения с какой стороны от линии точки
        dist = path.project(point)

        d1 = dist - 1 if dist - 1 > 0 else 0
        d2 = dist + 1 if dist + 1 < path.length else path.length
        line = substring(path, d1, d2)
        x1, y1 = line.coords[0]
        x2, y2 = line.coords[-1]
        x, y = point.coords[0]

        cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross_product > 0:
            return 1
        elif cross_product < 0:
            return 0

    def offset_point(point, direction, distance=5):
        # для размещения платформы по одну сторону от пути на расстоянии
        dist = path.project(point)
        d1 = dist - 1 if dist - 1 > 0 else 0
        d2 = dist + 1.1 if dist + 1.1 < path.length else path.length
        nearest_pt_on_line = path.interpolate(dist)
        line = substring(path, d1, d2)

        x1, y1 = line.coords[0]
        x2, y2 = line.coords[-1]

        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx ** 2 + dy ** 2)
        dx, dy = dx / length, dy / length

        if direction == 0:  # Вправо
            nx, ny = dy, -dx
        else:  # Влево
            nx, ny = -dy, dx

        # Смещенная точка
        offset_x = nearest_pt_on_line.x + nx * distance
        offset_y = nearest_pt_on_line.y + ny * distance
        return offset_x, offset_y

    def process_platform(platform):
        nonlocal node_id, last_dist, last_projected_stop_id

        dist = path.project(platform)
        projected_stop = path.interpolate(dist)
        platform_to_stop = LineString([platform, projected_stop])

        add_node('stop', projected_stop.x, projected_stop.y, transport=True)
        if last_dist is not None:
            cur_path = substring(path, last_dist, dist)
            add_edge(last_projected_stop_id, node_id, desc='routing', geometry=cur_path, transport=True)
        last_projected_stop_id = node_id

        node_id += 1
        add_node('platform', platform.x, platform.y)
        add_edge(node_id - 1, node_id, desc='boarding', geometry=platform_to_stop)
        add_edge(node_id, node_id - 1, desc='boarding', geometry=platform_to_stop)
        node_id += 1
        return dist, last_projected_stop_id, node_id

    if not path:
        warnings.warn("В одном из маршрутов нет пути, пока не работаю с этим", FutureWarning)
        return None
    path = LineString(path)
    # Если нет платформ
    if not platforms:
        if not stops:  # Строим маршрут по пути начало - конец
            platforms = [offset_point(path.interpolate(0), 1, 7), offset_point(path.interpolate(path.length), 1, 7)]
        else:  # Если есть только остановки - превращаем их в платформы
            platforms = [offset_point(Point(stop), 1, 7) for stop in stops]
            stops = None

    stops = [] if not stops else stops

    # Если остановок больше чем платформ - найти остановки без платформ и добавить новые платформы
    if len(stops) > len(platforms):
        stop_tree = cKDTree(stops)
        distances, indices = stop_tree.query(platforms)
        connection = [(platforms[platform], stop) for platform, stop in enumerate(indices)]
        connection += [(-1, stop) for stop in set(range(len(stops))) ^ set(indices)]
        connection.sort(key=lambda x: x[1])
        direction = offset_direction(Point(platforms[len(platforms) // 2]))
        stops_to_platforms = {stop: offset_point(Point(stops[stop]), direction, 7) for stop in
                              set(range(len(stops))) ^ set(indices)}
        platforms = [coord if (coord != -1) else (stops_to_platforms.get(ind)) for coord, ind in connection]
        stops = []

    # Если получилось только одна платформа
    if len(platforms) == 1:
        platform = Point(platforms[0])
        dist = path.project(platform)
        if dist == path.length or dist == 0:  # Если платформа является конечной
            platforms = [offset_point(path.interpolate(0), 1, 7), offset_point(path.interpolate(path.length), 1, 7)]
        else:  # Если платформа не является конечной
            platforms = [offset_point(path.interpolate(0), 1, 7), platform,
                         offset_point(path.interpolate(path.length), 1, 7)]
    platforms = [Point(coords) for coords in platforms]
    if len(platforms) >= len(stops):
        for platform in platforms:
            if not last_dist:
                last_dist, last_projected_stop_id, node_id = process_platform(platform)
                if last_dist > path.length / 2:
                    path = path.reverse()
                    last_dist = path.project(platform)
            else:
                last_dist, last_projected_stop_id, node_id = process_platform(platform)

    to_return = pd.DataFrame(graph_data)
    return to_return



