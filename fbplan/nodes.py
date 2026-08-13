"""Выбор состояний-узлов графа подцелей из оффлайн-датасета.

Узлы отбираются в пространстве backward-репрезентаций B(s), а не по
привилегированным координатам (x, y): планировщик не должен знать о геометрии
лабиринта ничего сверх того, что уже выучено в FB.
"""

from typing import Optional

import numpy as np

from .fb_api import FBOracle


def select_nodes(
    oracle: FBOracle,
    observations: np.ndarray,
    num_nodes: int,
    method: str = 'fps',
    candidate_pool: int = 50_000,
    seed: int = 0,
) -> np.ndarray:
    """Отбирает `num_nodes` состояний-узлов из датасета.

    Args:
        oracle: оракул поверх замороженных FB.
        observations: (N, obs_dim) — все состояния датасета.
        num_nodes: сколько узлов вернуть.
        method: 'fps' — farthest point sampling в B-пространстве (покрытие),
            'random' — равномерно случайно (абляция: показывает вклад покрытия).
        candidate_pool: размер случайной подвыборки, из которой идёт отбор.
            Прогонять B по всему миллиону состояний незачем — покрытие даёт уже
            несколько десятков тысяч кандидатов.
        seed: сид отбора кандидатов.

    Returns:
        (num_nodes,) — индексы выбранных состояний в исходном массиве.
    """
    rng = np.random.default_rng(seed)
    n_total = len(observations)

    pool_size = min(candidate_pool, n_total)
    pool_idxs = rng.choice(n_total, size=pool_size, replace=False)

    if num_nodes >= pool_size:
        return np.sort(pool_idxs)

    if method == 'random':
        return np.sort(rng.choice(pool_idxs, size=num_nodes, replace=False))
    if method != 'fps':
        raise ValueError(f"method должен быть 'fps' или 'random', получено {method!r}")

    # FPS в пространстве нормированных интенций: именно эта геометрия
    # используется дальше при построении рёбер.
    z_pool = oracle.normalize_z(oracle.backward(observations[pool_idxs]))
    selected_local = _farthest_point_sampling(z_pool, num_nodes, rng)
    return np.sort(pool_idxs[selected_local])


def _farthest_point_sampling(points: np.ndarray, num: int, rng: np.random.Generator) -> np.ndarray:
    """Жадный farthest point sampling. -> (num,) локальные индексы.

    Сложность O(pool * num); при pool=50k и num=1000 это ~5e7 операций — секунды.
    """
    n = len(points)
    selected = np.empty(num, dtype=np.int64)
    selected[0] = rng.integers(n)

    # min_dist[i] — расстояние от точки i до ближайшей уже выбранной.
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)
    for k in range(1, num):
        selected[k] = int(np.argmax(min_dist))
        dist = np.linalg.norm(points - points[selected[k]], axis=1)
        np.minimum(min_dist, dist, out=min_dist)
    return selected


def subsample_states(
    observations: np.ndarray,
    num: int,
    seed: int = 0,
    extra: Optional[dict] = None,
):
    """Равномерная подвыборка состояний (плюс синхронно — сопутствующих массивов).

    Используется в скриптах анализа, где нужны и наблюдения, и ground-truth
    координаты из `qpos`.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(observations), size=min(num, len(observations)), replace=False)
    if extra is None:
        return observations[idxs], idxs
    return observations[idxs], idxs, {k: v[idxs] for k, v in extra.items()}
