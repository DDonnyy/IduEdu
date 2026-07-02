import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse

from iduedu._version import VERSION
from iduedu.graph.urban_graph import UrbanGraph

URBANGRAPH_FORMAT = "iduedu.urbangraph"
URBANGRAPH_FORMAT_VERSION = 1
URBANGRAPH_SUFFIX = ".urbangraph"

METADATA_FILE = "metadata.json"
NODES_FILE = "nodes.parquet"
EDGES_FILE = "edges.parquet"
ADJACENCY_FILE = "adjacency.npz"
ADJACENCY_NODELIST_FILE = "adjacency_nodelist.parquet"


def write_urban_graph(
    graph: UrbanGraph,
    path: str | Path,
    *,
    include_adjacency: bool = False,
) -> Path:
    """Write an ``UrbanGraph`` to an ``.urbangraph`` archive.

    The archive contains ``metadata.json``, ``nodes.parquet`` and
    ``edges.parquet``. If ``include_adjacency`` is true and the graph has a
    cached adjacency matrix, the cache is stored as ``adjacency.npz`` together
    with its nodelist.

    Args:
        graph: Graph to serialize.
        path: Destination path with the ``.urbangraph`` suffix.
        include_adjacency: Whether to persist the cached adjacency matrix.

    Returns:
        Path to the written archive.

    Raises:
        TypeError: If ``graph`` is not an ``UrbanGraph``.
        ValueError: If ``path`` does not use the ``.urbangraph`` suffix.
        ImportError: If parquet support is not installed.
    """

    if not isinstance(graph, UrbanGraph):
        raise TypeError(f"graph must be UrbanGraph, got {type(graph).__name__}")

    archive_path = _normalize_urbangraph_path(path)
    graph.validate()
    metadata = _build_metadata(graph, include_adjacency=include_adjacency)

    try:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            graph.nodes_gdf.to_parquet(tmp_path / NODES_FILE, index=True)
            graph.edges_gdf.to_parquet(tmp_path / EDGES_FILE, index=True)

            if metadata["has_adjacency"]:
                sparse.save_npz(tmp_path / ADJACENCY_FILE, graph.adjacency_matrix)
                pd.DataFrame({"node": graph.adjacency_nodelist}).to_parquet(
                    tmp_path / ADJACENCY_NODELIST_FILE,
                    index=False,
                )

            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr(
                    METADATA_FILE, json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default)
                )
                archive.write(tmp_path / NODES_FILE, NODES_FILE)
                archive.write(tmp_path / EDGES_FILE, EDGES_FILE)
                if metadata["has_adjacency"]:
                    archive.write(tmp_path / ADJACENCY_FILE, ADJACENCY_FILE)
                    archive.write(tmp_path / ADJACENCY_NODELIST_FILE, ADJACENCY_NODELIST_FILE)
    except ImportError as exc:
        raise ImportError(".urbangraph IO requires parquet support. Install iduedu[io].") from exc

    return archive_path


def read_urban_graph(path: str | Path, *, validate: bool = True) -> UrbanGraph:
    """Read an ``UrbanGraph`` from an ``.urbangraph`` archive.

    Args:
        path: Source path with the ``.urbangraph`` suffix.
        validate: Whether to validate the graph after reading.

    Returns:
        Restored ``UrbanGraph`` instance.

    Raises:
        ValueError: If the archive is missing required members or uses an
            unsupported format version.
        ImportError: If parquet support is not installed.
    """

    archive_path = _normalize_urbangraph_path(path)

    try:
        with ZipFile(archive_path) as archive:
            _validate_archive_members(archive)
            metadata = json.loads(archive.read(METADATA_FILE).decode("utf-8"))
            _validate_metadata(metadata)

            nodes_gdf = _read_frame(archive, NODES_FILE, metadata["nodes_frame"])
            edges_gdf = _read_frame(archive, EDGES_FILE, metadata["edges_frame"])
            graph = UrbanGraph(
                nodes_gdf=nodes_gdf,
                edges_gdf=edges_gdf,
                is_multigraph=metadata["is_multigraph"],
                is_directed=metadata["is_directed"],
                edge_direction_column=metadata["edge_direction_column"],
                adjacency_weight=metadata["adjacency_weight"],
                crs=metadata["crs"],
                graph_type=metadata["graph_type"],
            )

            if metadata.get("has_adjacency") and {ADJACENCY_FILE, ADJACENCY_NODELIST_FILE} <= set(archive.namelist()):
                graph.adjacency_matrix = sparse.load_npz(BytesIO(archive.read(ADJACENCY_FILE)))
                adjacency_nodelist = pd.read_parquet(BytesIO(archive.read(ADJACENCY_NODELIST_FILE)))
                graph.adjacency_nodelist = adjacency_nodelist["node"].tolist()
                graph.node_to_adjacency_pos = {node: pos for pos, node in enumerate(graph.adjacency_nodelist)}

    except ImportError as exc:
        raise ImportError(".urbangraph IO requires parquet support. Install iduedu[io].") from exc

    if validate:
        graph.validate()

    return graph


def _normalize_urbangraph_path(path: str | Path) -> Path:
    archive_path = Path(path)
    if archive_path.suffix != URBANGRAPH_SUFFIX:
        raise ValueError(f"path must use the {URBANGRAPH_SUFFIX!r} suffix")
    return archive_path


def _build_metadata(graph: UrbanGraph, *, include_adjacency: bool) -> dict:
    has_adjacency = include_adjacency and graph.adjacency_matrix is not None
    return {
        "format": URBANGRAPH_FORMAT,
        "format_version": URBANGRAPH_FORMAT_VERSION,
        "iduedu_version": VERSION,
        "nodes_frame": _frame_kind(graph.nodes_gdf),
        "edges_frame": _frame_kind(graph.edges_gdf),
        "crs": str(graph.crs) if graph.crs is not None else None,
        "graph_type": graph.type,
        "is_multigraph": graph.is_multigraph,
        "is_directed": graph.is_directed,
        "edge_direction_column": graph.edge_direction_column,
        "adjacency_weight": graph.adjacency_weight,
        "has_adjacency": has_adjacency,
    }


def _frame_kind(frame: pd.DataFrame | gpd.GeoDataFrame) -> str:
    if isinstance(frame, gpd.GeoDataFrame):
        return "geodataframe"
    return "dataframe"


def _validate_archive_members(archive: ZipFile) -> None:
    missing = {METADATA_FILE, NODES_FILE, EDGES_FILE} - set(archive.namelist())
    if missing:
        raise ValueError(f".urbangraph archive is missing required files: {sorted(missing)}")


def _validate_metadata(metadata: dict) -> None:
    if metadata.get("format") != URBANGRAPH_FORMAT:
        raise ValueError(f"Unsupported graph archive format: {metadata.get('format')!r}")
    if metadata.get("format_version") != URBANGRAPH_FORMAT_VERSION:
        raise ValueError(f"Unsupported .urbangraph format version: {metadata.get('format_version')!r}")


def _read_frame(archive: ZipFile, member: str, frame_kind: str) -> pd.DataFrame | gpd.GeoDataFrame:
    buffer = BytesIO(archive.read(member))
    if frame_kind == "geodataframe":
        return gpd.read_parquet(buffer)
    if frame_kind == "dataframe":
        return pd.read_parquet(buffer)
    raise ValueError(f"Unsupported frame kind: {frame_kind!r}")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
