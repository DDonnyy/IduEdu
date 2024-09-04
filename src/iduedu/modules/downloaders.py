import geopandas as gpd
import osm2geojson
import osmnx as ox
import pandas as pd
import requests
from loguru import logger
from shapely import MultiPolygon, Polygon, unary_union

OVERPASS_URL = "http://lz4.overpass-api.de/api/interpreter"


def get_boundary_by_osm_id(osm_id) -> MultiPolygon | Polygon:
    overpass_query = f"""
            [out:json];
                    (
                        relation({osm_id});
                    );
            out geom;
            """
    logger.debug(f"Downloading territory bounds with osm_id <{osm_id}> ...")
    result = requests.get(OVERPASS_URL, params={"data": overpass_query})
    if result.status_code == 200:
        json_result = result.json()
        boundary = osm2geojson.json2geojson(json_result)
        boundary = gpd.GeoDataFrame.from_features(boundary["features"]).set_crs(4326)
        poly = unary_union(boundary.geometry)
        return poly
    else:
        raise RuntimeError(f"Request failed with status code {result.status_code}, reason: {result.reason}")


def get_boundary_by_name(territory_name: str) -> Polygon | MultiPolygon:
    # logger.info(f"Retrieving polygon geometry for '{territory_name}'")
    logger.debug(f"Downloading territory bounds with name <{territory_name}> ...")
    place = ox.geocode_to_gdf(territory_name)
    return unary_union(place.geometry)


def get_boundary(
    osm_id: int | None = None, territory_name: str | None = None, polygon: Polygon | MultiPolygon | None = None
) -> Polygon:
    """
    Retrieve the boundary polygon for a given territory, either by OSM ID, territory name, or an existing polygon.
    If a MultiPolygon is provided, it will be converted to a convex hull.

    Parameters
    ----------
    osm_id : int, optional
        OpenStreetMap ID of the territory to retrieve the boundary for. Either this or `territory_name` must be provided.
    territory_name : str, optional
        Name of the territory to retrieve the boundary for. Either this or `osm_id` must be provided.
    polygon : Polygon | MultiPolygon, optional
        A custom polygon or MultiPolygon to use instead of querying by `osm_id` or `territory_name`.

    Returns
    -------
    Polygon
        The boundary polygon for the specified territory. If a MultiPolygon was provided, it will return the convex hull.

    Raises
    ------
    ValueError
        If neither `osm_id`, `territory_name`, nor `polygon` are provided.

    Examples
    --------
    >>> boundary = get_boundary(osm_id=123456)
    >>> boundary = get_boundary(territory_name="New York")
    >>> boundary = get_boundary(polygon=some_polygon)
    """

    if osm_id is None and territory_name is None and polygon is None:
        raise ValueError("Either osm_id or name or polygon must be specified")
    if osm_id:
        polygon: Polygon = get_boundary_by_osm_id(osm_id)
    elif territory_name:
        polygon: Polygon = get_boundary_by_name(territory_name)

    if isinstance(polygon, MultiPolygon):
        polygon: Polygon = polygon.convex_hull

    return polygon


def get_routes_by_poly(polygon: Polygon, public_transport_type: str) -> pd.DataFrame:
    polygon_coords = " ".join(f"{y} {x}" for x, y in polygon.exterior.coords[:-1])
    overpass_query = f"""
        [out:json];
                (
                    relation(poly:"{polygon_coords}")['route'='{public_transport_type}'];
                );
        out geom;
        """
    logger.debug(f"Downloading routes from OSM with type <{public_transport_type}> ...")
    result = requests.post(OVERPASS_URL, data={"data": overpass_query})
    if result.status_code == 200:
        json_result = result.json()["elements"]
        data = pd.DataFrame(json_result)
        data["transport_type"] = public_transport_type
        return data
    else:
        raise RuntimeError(f"Request failed with status code {result.status_code}, reason: {result.reason}")


def get_routes_by_osm_id(osm_id, public_transport_type: str) -> pd.DataFrame:
    boundary = get_boundary_by_osm_id(osm_id)
    if isinstance(boundary, MultiPolygon):
        boundary = boundary.convex_hull
    return get_routes_by_poly(boundary, public_transport_type)


def get_routes_by_terr_name(terr_name: str, public_transport_type: str) -> pd.DataFrame:
    boundary = get_boundary_by_name(terr_name)

    if isinstance(boundary, MultiPolygon):
        boundary = boundary.convex_hull
    return get_routes_by_poly(boundary, public_transport_type)
