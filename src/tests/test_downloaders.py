# pylint: disable=redefined-outer-name
import time

import geopandas as gpd
import pytest

from iduedu import (
    config,
    get_4326_boundary,
)
from iduedu.modules.overpass_downloaders import RequestError, get_routes_by_poly

config.configure_logging("DEBUG")


def test_get_boundary_by_osm_id(bounds):
    assert bounds is not None

