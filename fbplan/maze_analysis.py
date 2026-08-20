"""Геометрия лабиринта — ТОЛЬКО для анализа и графиков.

ВАЖНО. Ни одна функция отсюда не вызывается из метода: ни при отборе узлов, ни
при построении рёбер, ни при выборе подцелей, ни в критерии переключения.
Планировщик знает о мире ровно то, что закодировано в замороженных FB. Здесь же
мы пользуемся привилегированной информацией (координаты и карта стен), чтобы
ПРОВЕРИТЬ, что он выучил, и честно измерить, где он ошибается.

Разметка координат в OGBench antmaze:
    x = (j - 1) * maze_unit,  y = (i - 1) * maze_unit
где (i, j) — индексы в `maze_map` (1 — стена, 0 — свободно).
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np


def maze_geometry(env) -> Tuple[np.ndarray, float]:
    """Возвращает (карта стен, размер клетки)."""
    unwrapped = env.unwrapped
    return np.asarray(unwrapped.maze_map), float(unwrapped._maze_unit)


def xy_to_ij(xy: np.ndarray, unit: float) -> np.ndarray:
    """Мировые координаты -> индексы клетки. -> (..., 2) целых."""
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    j = np.round(xy[:, 0] / unit).astype(int) + 1
    i = np.round(xy[:, 1] / unit).astype(int) + 1
    return np.stack([i, j], axis=-1)


def ij_to_xy(ij: np.ndarray, unit: float) -> np.ndarray:
    """Индексы клетки -> мировые координаты центра клетки."""
    ij = np.atleast_2d(np.asarray(ij))
    return np.stack([(ij[:, 1] - 1) * unit, (ij[:, 0] - 1) * unit], axis=-1).astype(np.float64)


def geodesic_field(grid: np.ndarray, source_ij: Tuple[int, int]) -> np.ndarray:
    """BFS по свободным клеткам от источника. Недостижимые — inf. -> (H, W) в клетках."""
    dist = np.full(grid.shape, np.inf)
    if grid[source_ij] != 0:
        return dist

    dist[source_ij] = 0.0
    queue = deque([source_ij])
    while queue:
        i, j = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (i + di, j + dj)
            if (
                0 <= nxt[0] < grid.shape[0]
                and 0 <= nxt[1] < grid.shape[1]
                and grid[nxt] == 0
                and not np.isfinite(dist[nxt])
            ):
                dist[nxt] = dist[i, j] + 1.0
                queue.append(nxt)
    return dist


def all_geodesic_fields(grid: np.ndarray) -> dict:
    """Поля расстояний от каждой свободной клетки (кэш для массовых запросов)."""
    return {
        (int(i), int(j)): geodesic_field(grid, (int(i), int(j)))
        for i, j in np.argwhere(grid == 0)
    }


def geodesic_distances(
    grid: np.ndarray,
    unit: float,
    xy_from: np.ndarray,
    xy_to: np.ndarray,
    fields: Optional[dict] = None,
) -> np.ndarray:
    """Геодезические расстояния между наборами точек, в мировых единицах.

    Считается по клеткам лабиринта, поэтому это грубая оценка: клетка 4x4
    мировых единиц. Для оси абсцисс на графиках «ошибка против дальности»
    точности хватает с запасом.
    """
    fields = fields if fields is not None else all_geodesic_fields(grid)

    ij_from = xy_to_ij(xy_from, unit)
    ij_to = xy_to_ij(xy_to, unit)

    out = np.full(len(ij_from), np.inf)
    for k, (src, dst) in enumerate(zip(ij_from, ij_to)):
        field = fields.get((int(src[0]), int(src[1])))
        if field is None:
            continue
        i, j = int(dst[0]), int(dst[1])
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            out[k] = field[i, j] * unit
    return out


def crosses_wall(
    grid: np.ndarray, unit: float, xy_from: np.ndarray, xy_to: np.ndarray, samples: int = 24
) -> np.ndarray:
    """Пересекает ли отрезок между точками стену. -> (N,) bool.

    Нужно для честного подсчёта доли «галлюцинированных» рёбер графа: ребро,
    прошивающее стену насквозь, — это ошибка представления, которую метод обязан
    отсекать прунингом.
    """
    xy_from = np.atleast_2d(np.asarray(xy_from, dtype=np.float64))
    xy_to = np.atleast_2d(np.asarray(xy_to, dtype=np.float64))

    ts = np.linspace(0.0, 1.0, samples)[None, :, None]
    points = xy_from[:, None, :] * (1 - ts) + xy_to[:, None, :] * ts  # точки вдоль отрезка: (N, samples, 2)

    flat_ij = xy_to_ij(points.reshape(-1, 2), unit)
    inside = (
        (flat_ij[:, 0] >= 0) & (flat_ij[:, 0] < grid.shape[0])
        & (flat_ij[:, 1] >= 0) & (flat_ij[:, 1] < grid.shape[1])
    )

    is_wall = np.ones(len(flat_ij), dtype=bool)
    is_wall[inside] = grid[flat_ij[inside, 0], flat_ij[inside, 1]] != 0

    return is_wall.reshape(len(xy_from), samples).any(axis=1)
