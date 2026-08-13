"""Отбор узлов графа подцелей из оффлайн-датасета.

Узел — это НАБОР состояний одной локации, а не одно состояние. Причина
измерена: successor measure одиночной позы муравья почти не несёт информации о
достижимости (корреляция с истинным расстоянием 0.18), а агрегация по набору
поднимает её до 0.70 — шум конфигурации суставов гасится, остаётся сигнал о
месте. Подробности в docstring `fb_api`.

Набор берётся окном подряд идущих состояний одной траектории: за несколько
десятков шагов агент почти не смещается, зато поза меняется полностью. Это
единственный источник «нескольких взглядов на одну локацию», доступный без
привилегированных координат.

Отбор самих локаций идёт farthest point sampling в пространстве B — ради
покрытия лабиринта, и снова без координат.
"""

from typing import Tuple

import numpy as np

from .fb_api import FBOracle


def episode_bounds(terminals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Начало и конец эпизода для каждого перехода датасета.

    Нужно, чтобы окно узла не пересекало границу эпизода: состояния из разных
    эпизодов не связаны ни во времени, ни в пространстве.
    """
    terminals = np.asarray(terminals).astype(bool)
    episode_id = np.concatenate([[0], np.cumsum(terminals)[:-1]])

    starts = np.zeros(len(terminals), dtype=np.int64)
    ends = np.zeros(len(terminals), dtype=np.int64)
    boundaries = np.concatenate([[0], np.nonzero(terminals)[0] + 1, [len(terminals)]])
    for begin, finish in zip(boundaries[:-1], boundaries[1:]):
        starts[begin:finish] = begin
        ends[begin:finish] = finish - 1
    return starts, ends


def window_indices(
    centers: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    num_members: int,
    stride: int,
) -> np.ndarray:
    """Индексы членов окна вокруг каждого центра. -> (K, num_members)

    Окно подрезается границами своего эпизода; при подрезке индексы
    повторяются, что для агрегации максимумом безвредно.
    """
    offsets = (np.arange(num_members) - num_members // 2) * stride
    idxs = centers[:, None] + offsets[None, :]
    return np.clip(idxs, starts[centers][:, None], ends[centers][:, None])


def select_nodes(
    oracle: FBOracle,
    observations: np.ndarray,
    terminals: np.ndarray,
    num_nodes: int,
    num_members: int = 16,
    stride: int = 4,
    method: str = 'fps',
    candidate_pool: int = 20_000,
    seed: int = 0,
) -> np.ndarray:
    """Отбирает `num_nodes` узлов-наборов.

    Args:
        method: 'fps' — farthest point sampling в пространстве B по агрегиро-
            ванному представлению узла (покрытие); 'random' — равномерно
            случайно (абляция, показывает вклад покрытия).
        num_members: сколько состояний в наборе одного узла.
        stride: шаг между членами окна. `num_members * stride` — охват окна в
            шагах среды; при 16 x 4 = 64 шага агент смещается примерно на
            полторы мировых единицы.

    Returns:
        (num_nodes, num_members) — индексы состояний датасета.
    """
    rng = np.random.default_rng(seed)
    starts, ends = episode_bounds(terminals)

    pool = rng.choice(len(observations), size=min(candidate_pool, len(observations)), replace=False)
    members = window_indices(pool, starts, ends, num_members, stride)

    if num_nodes >= len(pool):
        return members

    if method == 'random':
        return members[rng.choice(len(pool), size=num_nodes, replace=False)]
    if method != 'fps':
        raise ValueError(f"method должен быть 'fps' или 'random', получено {method!r}")

    # Представление узла для FPS — среднее B по членам: оно и есть та самая
    # агрегация, которая гасит шум позы.
    node_repr = oracle.normalize_z(oracle.backward(observations[members]).mean(axis=1))
    return members[_farthest_point_sampling(node_repr, num_nodes, rng)]


def _farthest_point_sampling(points: np.ndarray, num: int, rng: np.random.Generator) -> np.ndarray:
    """Жадный farthest point sampling. -> (num,) индексы в `points`."""
    selected = np.empty(num, dtype=np.int64)
    selected[0] = rng.integers(len(points))

    # min_dist[i] — расстояние от точки i до ближайшей уже выбранной.
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)
    for k in range(1, num):
        selected[k] = int(np.argmax(min_dist))
        np.minimum(min_dist, np.linalg.norm(points - points[selected[k]], axis=1), out=min_dist)
    return selected
