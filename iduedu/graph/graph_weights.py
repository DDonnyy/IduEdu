import numpy as np

WEIGHT_SCALE = 100.0


def int_weight_to_float(value) -> np.ndarray | float:
    """Возвращает вес из целочисленного формата numba в исходные единицы."""

    return value.astype(float) / WEIGHT_SCALE if hasattr(value, "astype") else float(value) / WEIGHT_SCALE


def cutoff_to_int(weight_value_cutoff: float | None) -> np.int32:
    """Переводит ограничение поиска в формат numba; ``None`` означает без ограничения."""

    if weight_value_cutoff is None:
        return np.int32(np.iinfo(np.int32).max)
    return np.int32(round(float(weight_value_cutoff) * WEIGHT_SCALE))
