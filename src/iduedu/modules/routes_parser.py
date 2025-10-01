import math
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely import Point
from shapely.geometry import LineString
from shapely.ops import substring

PLATFORM_ROLES = ["platform_entry_only", "platform", "platform_exit_only"]
STOPS_ROLES = ["stop", "stop_exit_only", "stop_entry_only"]
THRESHOLD_METERS = 100


def _link_unconnected(disconnected_ways) -> list:
    # Считаем связи между линиями
    connect_points = [point for coords in disconnected_ways for point in (coords[0], coords[-1])]
    distances = cdist(connect_points, connect_points)
    n = distances.shape[0]
    mask = (np.arange(n)[:, None] // 2) == (np.arange(n) // 2)
    distances[mask] = np.inf

    def relative_point(point: int):
        rel_point = point // 2 * 2
        return rel_point if rel_point != point else rel_point + 1

    connected_ways = []
    # pylint: disable=unbalanced-tuple-unpacking
    first_con_1, first_con_2 = np.unravel_index(np.argmin(distances), distances.shape)
    distances[first_con_1, :] = np.inf
    distances[first_con_2, :] = np.inf
    distances[:, first_con_2] = np.inf
    distances[:, first_con_1] = np.inf
    first_con_1_rel, first_con_2_rel = [relative_point(x) for x in (first_con_1, first_con_2)]
    distances[first_con_1_rel, first_con_2_rel] = np.inf
    distances[first_con_2_rel, first_con_1_rel] = np.inf
    distances[:, first_con_1_rel] = np.inf
    distances[:, first_con_2_rel] = np.inf

    line1, line2 = first_con_1 // 2, first_con_2 // 2

    if first_con_1 % 2 == 1:
        connected_ways += disconnected_ways[line1]
    else:
        connected_ways += disconnected_ways[line1][::-1]
    if first_con_2 % 2 == 0:
        connected_ways += disconnected_ways[line2]
    else:
        connected_ways += disconnected_ways[line2][::-1]

    extreme_points = [first_con_1_rel, first_con_2_rel]

    for _ in range(len(disconnected_ways) - 2):
        # pylint: disable=invalid-sequence-index
        position, ind = np.unravel_index(np.argmin(distances[extreme_points]), (2, n))
        next_con = (extreme_points[position], ind)
        rel_point = relative_point(next_con[1])

        line = ind // 2
        if position == 0:
            connected_ways = (
                disconnected_ways[line] + connected_ways
                if ind % 2 == 1
                else (disconnected_ways[line][::-1] + connected_ways)
            )
            extreme_points = [rel_point, extreme_points[1]]
        else:
            connected_ways = (
                connected_ways + disconnected_ways[line]
                if ind % 2 == 0
                else (connected_ways + disconnected_ways[line][::-1])
            )
            extreme_points = [extreme_points[0], rel_point]

        # Чтоб нельзя было зациклиться
        distances[:, rel_point] = np.inf
        distances[next_con[0], :] = np.inf
        distances[next_con[1], :] = np.inf
        distances[:, next_con[0]] = np.inf
        distances[:, next_con[1]] = np.inf
        rel_point_2 = relative_point(next_con[0])
        distances[rel_point, rel_point_2] = np.inf
        distances[rel_point_2, rel_point] = np.inf

    return connected_ways


def parse_overpass_route_response(loc: dict, crs: CRS, needed_tags: list[str]) -> pd.Series:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def transform_geometry(loc):
        if isinstance(loc["geometry"], float):
            return transformer.transform(loc["lon"], loc["lat"])
        p = LineString([transformer.transform(coords["lon"], coords["lat"]) for coords in loc["geometry"]]).centroid
        return p.x, p.y

    def process_roles(route, roles):
        filtered = route[route["role"].isin(roles)]
        if len(filtered) == 0:
            return None

        return filtered.apply(transform_geometry, axis=1).tolist(), filtered["ref"].tolist()

    def extract_needed(loc_obj: dict, keys: list[str]) -> dict | None:
        out = {}
        tags = loc_obj.get("tags", {}) if isinstance(loc_obj.get("tags"), dict) else {}
        for k in keys:
            value_found = None
            if "." in k:
                # dotted path
                parts = k.split(".")
                cur = loc_obj
                ok = True
                for part in parts:
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        ok = False
                        break
                if ok:
                    value_found = cur
            else:
                if k in tags:
                    value_found = tags[k]
                elif k in loc_obj:
                    value_found = loc_obj[k]
            if value_found is not None:
                out[k] = value_found
        return out or None

    tags = loc.get("tags", {}) if isinstance(loc.get("tags"), dict) else {}

    transport_name = None
    if "ref" in tags:
        transport_name = tags["ref"]
    elif "name" in tags:
        transport_name = tags["name"]

    members = loc.get("members", [])
    route = pd.DataFrame(members) if isinstance(members, list) else pd.DataFrame()

    for col in ("geometry", "lat", "lon", "role", "type"):
        if col not in route.columns:
            route[col] = np.nan

    route = route.dropna(subset=["lat", "lon", "geometry"], how="all")

    platforms, platforms_refs = process_roles(route, PLATFORM_ROLES)
    stops, stops_refs = process_roles(route, STOPS_ROLES)

    ways_df = route[(route["type"] == "way") & (route["role"].fillna("") == "")]

    if not ways_df.empty:
        ways = (
            ways_df["geometry"]
            .reset_index(drop=True)
            .apply(lambda g: [transformer.transform(pt["lon"], pt["lat"]) for pt in g])
            .tolist()
        )
        connected_ways = [[]]
        cur_way = 0
        for coords in ways:
            if not coords:
                continue
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

            # Случай если нету соединяющей точки между линиями маршрута
            else:
                connected_ways += [coords]
                cur_way += 1
        # Соединяем линии по ближайшим точкам этих линий

        if len(connected_ways) > 1:
            # удаляем все круговые движения
            to_del = [i for i, data in enumerate(connected_ways) if data[0] == data[-1]]
            # Если кол-во удалений == кол-во путей, надо оставить хотя бы самый большой
            if len(to_del) == len(connected_ways):
                # Найдём индекс самого длинного пути среди замкнутых
                longest_index = max(to_del, key=lambda i: len(connected_ways[i]))
                # Удалим все, кроме самого длинного
                to_del.remove(longest_index)
            connected_ways = [i for j, i in enumerate(connected_ways) if j not in to_del]
        if len(connected_ways) > 1:
            path = _link_unconnected(connected_ways)
        else:
            if not connected_ways or not connected_ways[0]:
                raise Exception("No connected ways")
            path = connected_ways[0]

    else:
        path = None

    extra = extract_needed(loc, needed_tags)

    return pd.Series(
        {
            "path": path,
            "platforms": platforms,
            "platforms_refs": platforms_refs,
            "stops": stops,
            "stops_refs": stops_refs,
            "route": transport_name,
            "extra_data": extra,
        }
    )


def geometry_to_graph_edge_node_df(loc: pd.Series, transport_type, loc_id) -> DataFrame | None:

    name = loc.route
    last_projected_stop_id = None

    platforms = list(loc.platforms or [])  # same size ->
    platforms_refs = list(loc.platforms_refs or [])

    stops = list(loc.stops or [])
    stops_refs = list(loc.stops_refs or [])

    path = loc.path
    extra_data = loc.extra_data

    if not path:
        warnings.warn("В одном из маршрутов нет пути, пока не работаю с этим", UserWarning)
        return None

    def _side_left_or_right(point):
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
        return 0

    def _offset_point(point, direction, distance=7) -> tuple[float, float]:
        # для размещения платформы по одну сторону от пути на расстоянии
        dist = path.project(point)
        d1 = dist - 1 if dist - 1 > 0 else 0
        d2 = dist + 1.1 if dist + 1.1 < path.length else path.length
        nearest_pt_on_line = path.interpolate(dist)
        line = substring(path, d1, d2)

        x1, y1 = line.coords[0]
        x2, y2 = line.coords[-1]

        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        dx, dy = dx / length, dy / length

        if direction == 0:  # Вправо
            nx, ny = dy, -dx
        else:  # Влево
            nx, ny = -dy, dx

        # Смещенная точка
        offset_x = nearest_pt_on_line.x + nx * distance
        offset_y = nearest_pt_on_line.y + ny * distance
        return offset_x, offset_y

    path = LineString(path)

    items = []  # [(platform_coords, platform_ref, stop_coords, stop_ref), на выходе не может быть пары без платформы

    platform_len, stops_len = len(platforms), len(stops)
    matched_p, matched_s = set(), set()
    pairs = {}  # p_idx -> s_idx

    if platform_len and stops_len:
        stop_tree = cKDTree(stops)
        dists, idxs = stop_tree.query(platforms, k=2)  # Смотрим 2 ближайших соседа

        if stops_len == 1:
            dists = np.array(dists).reshape(platform_len, 1)
            idxs = np.array(idxs, dtype=int).reshape(platform_len, 1)

        candidates = []
        for p_i in range(platform_len):
            for rank in range(stops_len):
                dist = float(dists[p_i][rank])
                s_i = int(idxs[p_i][rank])
                if dist <= THRESHOLD_METERS:
                    candidates.append((dist, p_i, s_i))

        candidates.sort(key=lambda t: t[0])
        for dist, p_i, s_i in candidates:
            if p_i not in matched_p and s_i not in matched_s:
                pairs[p_i] = s_i
                matched_p.add(p_i)
                matched_s.add(s_i)

    for p_i, s_i in pairs.items():
        items.append(
            {
                "p": platforms[p_i],
                "pref": platforms_refs[p_i],
                "s": stops[s_i],
                "sref": stops_refs[s_i],
            }
        )

    if stops_len:
        base_dir = _side_left_or_right(Point(platforms[platform_len // 2])) if platform_len else 1
        for s_i in range(stops_len):
            if s_i in matched_s:
                continue
            s_pt = Point(stops[s_i])
            items.append(
                {
                    "p": _offset_point(s_pt, base_dir),
                    "pref": None,  # new platform
                    "s": s_pt.xy,
                    "sref": stops_refs[s_i],
                }
            )

    for p_i in range(platform_len):
        if p_i in matched_p:
            continue
        items.append(
            {
                "p": platforms[p_i],
                "pref": platforms_refs[p_i],
                "s": None,
                "sref": None,
            }
        )

    if not items:
        items.append({"p": _offset_point(path.interpolate(0), 1, 7), "pref": None, "s": None, "sref": None})
        items.append({"p": _offset_point(path.interpolate(path.length), 1, 7), "pref": None, "s": None, "sref": None})

    items.sort(key=lambda d: path.project(Point(d["p"])))

    graph_data: list[dict] = []
    node_id = 0
    last_dist = None

    def add_node(desc, x, y, transport=None, ref_id=None):
        x = round(x, 5)
        y = round(y, 5)
        graph_data.append(
            {
                "node_id": (loc_id, node_id),
                "point": (x, y),
                "route": name,
                "type": (transport_type if transport else desc),
                "ref_id": ref_id,
            }
        )

    def add_edge(u, v, geometry=None, transport=None):

        payload = {
            "u": (loc_id, u),
            "v": (loc_id, v),
            "route": name,
            "type": (transport_type if transport else "boarding"),
            "extra_data": extra_data,
        }
        if geometry is not None:
            payload["geometry"] = LineString([(round(x, 5), round(y, 5)) for x, y in geometry.coords])
        graph_data.append(payload)

    def process_platform(platform, idx):
        nonlocal node_id, last_dist, last_projected_stop_id

        dist = path.project(platform)
        projected_stop = path.interpolate(dist)
        add_node("stop", projected_stop.x, projected_stop.y, transport=True, ref_id=(stops_refs[idx]))

        if last_dist is not None:
            cur_path = substring(path, last_dist, dist)
            if isinstance(cur_path, Point):
                cur_path = LineString((cur_path, cur_path))
            add_edge(last_projected_stop_id, node_id, geometry=cur_path, transport=True)
        last_projected_stop_id = node_id

        node_id += 1
        add_node("platform", platform.x, platform.y, ref_id=(platforms_refs[idx]))
        add_edge(node_id - 1, node_id)
        add_edge(node_id, node_id - 1)
        node_id += 1
        return dist, last_projected_stop_id, node_id

    if len(platforms) >= len(stops):
        for idx, platform in enumerate(platforms):
            if not last_dist:
                last_dist, last_projected_stop_id, node_id = process_platform(platform, idx)
                if last_dist > path.length / 2:
                    path = path.reverse()
                    last_dist = path.project(platform)
            else:
                last_dist, last_projected_stop_id, node_id = process_platform(platform, idx)

    to_return = pd.DataFrame(graph_data)
    return to_return


def parse_overpass_to_edgenode(loc, crs, needed_tags) -> pd.DataFrame | None:
    loc_id = loc.name
    transport_type = loc.transport_type
    parsed_geometry = parse_overpass_route_response(loc, crs, needed_tags)
    edgenode = geometry_to_graph_edge_node_df(parsed_geometry, transport_type, loc_id)
    return edgenode
