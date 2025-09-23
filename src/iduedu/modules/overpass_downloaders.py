import datetime as dt
import math
import time
from typing import Literal

import pandas as pd
import geopandas as gpd
import requests
from shapely import LineString, MultiPolygon, Polygon, unary_union
from shapely.ops import polygonize

from iduedu import config

logger = config.logger


class RequestError(RuntimeError):
    """
    Basic error for eny requests problems
    """

    def __init__(self, message, status_code=None, reason=None, response_text=None, response_content=None):
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.response_text = response_text
        self.response_content = response_content

    def __str__(self):
        if self.status_code == 400:
            return (
                f"{super().__str__()} (status: {self.status_code}, reason: {self.reason}). "
                f"Make sure provided polygon is in CRS 4326."
            )
        return f"{super().__str__()} (status: {self.status_code}, reason: {self.reason})."


def _get_overpass_pause(
    base_endpoint: str,
    recursion_pause: float = 5,
) -> float:
    """
    Retrieve a pause duration from the Overpass API status endpoint.

    Parameters:
        base_endpoint (str):
            Base Overpass API URL (without "/status" at the end).
        recursion_pause (float):
            How long to wait between recursive calls if the server is currently running a query.
    Returns:
        (float):
            Pause duration in seconds.
    """
    url = base_endpoint[: -len("/interpreter")] + "/status"
    try:
        response = requests.get(url, timeout=config.timeout)
        response_text = response.text
    except requests.RequestException as e:
        raise RequestError(f"Unable to reach {url}, {e}") from e

    try:
        status = response_text.split("\n")[4]
        status_first_part = status.split(" ")[0]
    except (AttributeError, IndexError, ValueError):
        raise RequestError(f"Unable to parse {url} response: {response_text}")

    try:
        _ = int(status_first_part)  # number of available slots
        pause: float = 0
    except ValueError:
        if status_first_part == "Slot":
            utc_time_str = status.split(" ")[3]
            pattern = "%Y-%m-%dT%H:%M:%SZ,"
            utc_time = dt.datetime.strptime(utc_time_str, pattern).replace(tzinfo=dt.timezone.utc)
            utc_now = dt.datetime.now(tz=dt.timezone.utc)
            seconds = int(math.ceil((utc_time - utc_now).total_seconds()))
            pause = max(seconds, 1)
        elif status_first_part == "Currently":
            time.sleep(recursion_pause)
            pause = _get_overpass_pause(
                base_endpoint,
                recursion_pause=recursion_pause,
            )
        else:
            raise RequestError(f"Unrecognized server status: {status!r}")

    return pause


def _overpass_request(
    method: Literal["GET", "POST"],
    overpass_url: str,
    params: dict | None = None,
    data: dict | None = None,
    timeout: float | None = None,
) -> requests.Response:
    """Single Overpass request with status-aware pausing (no retries)."""
    pause = _get_overpass_pause(overpass_url)
    if pause > 0:
        logger.warning(f"Waiting {pause} seconds for available Overpass API slot")
        time.sleep(pause)

    try:
        if method == "GET":
            resp = requests.get(overpass_url, params=params, timeout=timeout or config.timeout)
        else:
            resp = requests.post(overpass_url, data=data, timeout=timeout or config.timeout)
    except requests.RequestException as e:
        raise RequestError(f"Request error: {e}") from e

    if resp.status_code == 200:
        return resp

    raise RequestError(
        message=f"Request failed with status code {resp.status_code}, reason: {resp.reason}",
        status_code=resp.status_code,
        reason=resp.reason,
        response_text=resp.text,
        response_content=resp.content,
    )


def get_boundary_by_osm_id(osm_id) -> MultiPolygon | Polygon:
    overpass_query = f"""
                [out:json];
                        (
                            relation({osm_id});
                        );
                out geom;
                """
    logger.debug(f"Downloading territory bounds with osm_id <{osm_id}> ...")
    resp = _overpass_request(
        method="GET",
        overpass_url=config.overpass_url,
        params={"data": overpass_query},
    )
    json_result = resp.json()
    geoms = []
    for element in json_result["elements"]:
        geometries_inners = []
        geometries_outers = []
        for member in element["members"]:
            if "geometry" in member:
                if member["role"] == "outer":
                    geometries_outers.append(
                        LineString([(coords["lon"], coords["lat"]) for coords in member["geometry"]])
                    )
                if member["role"] == "inner":
                    geometries_inners.append(
                        LineString([(coords["lon"], coords["lat"]) for coords in member["geometry"]])
                    )
        outer_poly = unary_union(list(polygonize(geometries_outers)))
        inner_poly = unary_union(list(polygonize(geometries_inners)))
        geoms.append(outer_poly.difference(inner_poly))
    return unary_union(geoms)


def get_4326_boundary(
    osm_id: int | None = None, territory: Polygon | MultiPolygon | gpd.GeoDataFrame | None = None
) -> Polygon:

    if osm_id:
        territory = get_boundary_by_osm_id(osm_id)

    if isinstance(territory, Polygon):
        return territory

    if isinstance(territory, MultiPolygon):
        return Polygon(territory.convex_hull)

    if isinstance(territory, gpd.GeoDataFrame):
        return get_4326_boundary(territory=territory.to_crs(4326).union_all())

    raise ValueError("Either osm_id or polygon must be specified")


def _poly_to_overpass(poly: Polygon) -> str:
    return " ".join(f"{y} {x}" for x, y in poly.exterior.coords[:-1])


def get_routes_by_poly(polygon: Polygon, public_transport_type: str) -> pd.DataFrame:
    polygon_coords = _poly_to_overpass(polygon)
    overpass_query = f"""
        [out:json][timeout:{config.timeout}];
                (
                    relation(poly:\"{polygon_coords}\")[ 'route' = '{public_transport_type}' ];
                );
        out geom;
        """
    logger.debug(f"Downloading routes from OSM with type <{public_transport_type}> ...")
    resp = _overpass_request(
        method="POST",
        overpass_url=config.overpass_url,
        data={"data": overpass_query},
    )
    json_result = resp.json()["elements"]
    data = pd.DataFrame(json_result)
    data["transport_type"] = public_transport_type
    return data


def get_network_by_filters(polygon: Polygon, way_filter: str) -> pd.DataFrame:
    polygon_coords = _poly_to_overpass(polygon)
    overpass_query = f"""
        [out:json][timeout:{config.timeout}];
            (way{way_filter}(poly:\"{polygon_coords}\"););
        out geom;
        """
    logger.debug(f"Downloading network from OSM with filters <{way_filter}> ...")
    resp = _overpass_request(
        method="POST",
        overpass_url=config.overpass_url,
        data={"data": overpass_query},
    )

    json_result = resp.json()["elements"]
    return pd.DataFrame(json_result)
