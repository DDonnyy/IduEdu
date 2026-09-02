"""Лежат ли две сети в разных местах или одна внутри другой.

Четыре случая низкого покрытия различаются по расстоянию от остановки до
ближайшей остановки другого источника, и на этом расстоянии два разных явления
неразличимы. Если OpenStreetMap описывает центр города, а фид — весь ареал, то
периферийные остановки фида тоже оказываются дальше километра от любой
остановки OpenStreetMap, и город попадает в группу «источники описывают разные
части города», хотя описывают они одну и ту же часть, просто с разным охватом.

Пуна — наглядный пример: 6648 остановок фида, 262 в OpenStreetMap, медианное
расстояние 8.7 км, доля удалённых 0.89 — по всем признакам смещение, а на карте
облако OpenStreetMap лежит ровно в середине облака фида.

Здесь считается признак, который эти случаи разводит: расстояние между
центроидами двух облаков остановок, делённое на их собственный разброс.
Отношение меньше единицы означает, что центры совпадают в пределах размера
самих облаков, то есть одно вложено в другое; больше единицы — что облака
разъехались.

    python wide_displacement.py --run
"""

import argparse
import logging

import numpy as np
import pandas as pd
from bench_common import sweep
from wide_cohort import measurable, restrict, with_gtfs_graph
from wide_paths import ALL_OSM_MODES, WIDE_DIR, gtfs_graph_path, osm_graph_path

from iduedu import read_urban_graph

logger = logging.getLogger(__name__)

REPORT_CSV = WIDE_DIR / "displacement.csv"

#: Ниже этого числа остановок центроид и разброс считать не по чему.
MIN_STOPS = 3

#: Граница между «вложены» и «разъехались». Единица означает, что центры облаков
#: разнесены ровно на характерный размер облака; всё, что больше, — это уже два
#: отдельных скопления.
DISPLACED_RATIO = 1.0


def _cloud(nodes, mode: str = "bus"):
    """Точки остановок одного вида транспорта в проекции графа."""
    if "type" in nodes.columns:
        nodes = nodes.loc[nodes["type"].astype(str).eq(mode)]
    if len(nodes) < MIN_STOPS:
        return None
    return np.column_stack([nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()])


def measure_city(city_key: str) -> dict:
    row = {"city_key": city_key}
    osm_path = osm_graph_path(city_key, "pt", ALL_OSM_MODES)
    gtfs_path = gtfs_graph_path(city_key)
    if not (osm_path.exists() and gtfs_path.exists()):
        row["status"] = "missing_graph"
        return row

    osm = _cloud(read_urban_graph(osm_path).nodes_gdf)
    gtfs = _cloud(read_urban_graph(gtfs_path).nodes_gdf)
    if osm is None or gtfs is None:
        row["status"] = "too_few_stops"
        return row

    osm_centre, gtfs_centre = osm.mean(axis=0), gtfs.mean(axis=0)
    separation = float(np.linalg.norm(osm_centre - gtfs_centre))
    # Среднеквадратичный радиус облака вокруг своего центра, усреднённый по двум
    # источникам: масштаб, с которым осмысленно сравнивать разнос центров.
    spread = float(
        np.mean(
            [
                np.sqrt(np.mean(np.sum((osm - osm_centre) ** 2, axis=1))),
                np.sqrt(np.mean(np.sum((gtfs - gtfs_centre) ** 2, axis=1))),
            ]
        )
    )

    row.update(
        {
            "n_osm_bus": len(osm),
            "n_gtfs_bus": len(gtfs),
            "separation_m": round(separation, 1),
            "spread_m": round(spread, 1),
            "ratio": round(separation / spread, 3) if spread else None,
            "status": "ok",
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="*")
    parser.add_argument("--run", action="store_true", help="без него печатается только план")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logger.setLevel(logging.INFO)

    keys = restrict(sorted(set(measurable()) & with_gtfs_graph(include_untimed=True)), arguments.cities)
    if not arguments.run:
        print(f"{len(keys)} cities would be measured into {REPORT_CSV}")
        print("Nothing was computed. Re-run with --run.")
        return

    sweep(
        keys,
        measure_city,
        REPORT_CSV,
        force=arguments.force,
        logger=logger,
        describe=lambda rows: f"{rows[0].get('status')} ratio {rows[0].get('ratio')}",
    )

    frame = pd.read_csv(REPORT_CSV).drop_duplicates("city_key", keep="last")
    frame = frame.loc[frame["status"].eq("ok")]
    displaced = frame.loc[frame["ratio"] > DISPLACED_RATIO]
    print(f"\n{len(frame)} cities measured, {len(displaced)} with clouds further apart than their own size")


if __name__ == "__main__":
    main()
