"""Синтетический лабиринт и подставной FB-оракул для тестов.

Зачем это нужно. Логику планировщика надо проверять отдельно от нейросетей:
если тест гоняется на настоящем чекпоинте, любая ошибка в Дейкстре или в
критерии переключения подцелей замаскируется качеством представлений. Здесь
геодезические расстояния известны точно, поэтому правильный ответ известен.

Подставной оракул воспроизводит и главную ПАТОЛОГИЮ настоящего FB. Модель такая:

    p(s -> w) = max( gamma^d_geo ,  gamma^h * gamma^d_euclid )

Первое слагаемое — честная достижимость. Второе — «дальнее поле»: как только
истинный сигнал gamma^d_geo падает ниже разрешения представления (порядка
gamma^h), от него остаётся лишь гладкий артефакт сети, а гладкость MLP по входу
живёт в евклидовой метрике состояний, не в геодезической.

Отсюда ровно два наблюдаемых эффекта, оба есть у настоящего FB:

1. контраст на длинном горизонте пропадает — оценка упирается в пол;
2. пары, близкие по прямой, но разделённые стеной, выглядят достижимыми.

Честная зона — там, где d_geo - d_euclid <= h. Именно её вырезает прунинг рёбер,
и именно в ней граф собирает длинный маршрут из коротких честных переходов.
"""

from collections import deque
from typing import List, Tuple

import numpy as np

from fbplan.fb_api import MIN_REACH_PROB, TargetSet

# '#' — стена, '.' — свободная клетка.
# Проём в перегородке намеренно сдвинут в левый край: тогда обход из правого
# нижнего угла в правый верхний длиннее прямой линии в четыре с лишним раза, и
# разрыв между честной оценкой и артефактом дальнего поля становится большим.
DEFAULT_MAZE = [
    '#############',
    '#...........#',
    '#...........#',
    '#...........#',
    '#.###########',
    '#...........#',
    '#...........#',
    '#...........#',
    '#############',
]

# Единственный проход между нижней и верхней половиной.
GAP_CELL = (4, 1)


class GridMaze:
    """Сетчатый лабиринт с точными геодезическими расстояниями."""

    def __init__(self, layout: List[str] = None):
        self.layout = layout if layout is not None else DEFAULT_MAZE
        self.grid = np.array([[c == '.' for c in row] for row in self.layout], dtype=bool)
        self.free_cells = np.argwhere(self.grid)  # (N, 2) в координатах (строка, столбец)

    def is_free(self, row: int, col: int) -> bool:
        if not (0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]):
            return False
        return bool(self.grid[row, col])

    def geodesic_from(self, cell: Tuple[int, int]) -> np.ndarray:
        """BFS-расстояния от клетки до всех остальных. Недостижимые — inf."""
        dist = np.full(self.grid.shape, np.inf)
        dist[cell] = 0.0
        queue = deque([cell])
        while queue:
            row, col = queue.popleft()
            for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (row + drow, col + dcol)
                if self.is_free(*nxt) and not np.isfinite(dist[nxt]):
                    dist[nxt] = dist[row, col] + 1.0
                    queue.append(nxt)
        return dist

    def all_geodesics(self) -> dict:
        return {tuple(cell): self.geodesic_from(tuple(cell)) for cell in self.free_cells}

    def sample_states(self, num: int, seed: int = 0) -> np.ndarray:
        """Состояния = координаты клеток (как float). -> (num, 2)"""
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self.free_cells), size=num, replace=True)
        return self.free_cells[idxs].astype(np.float32)

    def all_states(self) -> np.ndarray:
        return self.free_cells.astype(np.float32)


class SyntheticOracle:
    """Подставной FB-оракул с контролируемой деградацией на длинном горизонте.

    Args:
        maze: лабиринт.
        discount: gamma. Взят заметно меньше 0.99, чтобы эффекты были видны на
            лабиринте из десятка клеток, а не из сотен.
        honest_radius: h из формулы выше — горизонт, на котором представление
            ещё разрешает истинную достижимость.
    """

    latent_dim = 2
    ensemble_reduce = 'min'

    def __init__(self, maze: GridMaze, discount: float = 0.9, honest_radius: float = 6.0):
        self.maze = maze
        self.discount = discount
        self.honest_radius = honest_radius
        # Атрибуты настоящего оракула, на которые смотрит код построения графа.
        # Пессимистичный отбор в синтетике моделировать нечем: ансамбля нет,
        # оракул возвращает одно детерминированное число.
        self.disagreement_penalty = 0.0
        self.num_heads = 1
        self._geodesics = maze.all_geodesics()

    # -- интерфейс, которым пользуются graph.py и planner.py ------------- #

    def normalize_z(self, z: np.ndarray) -> np.ndarray:
        # В синтетике латент — это сами координаты цели, нормировать нечего.
        return np.asarray(z, dtype=np.float32)

    def backward(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray(observations, dtype=np.float32)

    def make_targets(self, member_observations: np.ndarray, reference_observations=None,
                     per_head: bool = False) -> TargetSet:
        """Узел — набор состояний, как в настоящем оракуле.

        В синтетике клетка лабиринта полностью задаёт состояние, поэтому набор
        вырождается в одного члена: моделировать шум позы здесь нечем и незачем,
        тесты проверяют логику планирования, а не качество представлений.

        `per_head` принимается ради совместимости сигнатуры: ансамбля здесь нет,
        и знаменатель у единственной «головы» тот же самый.
        """
        members = np.asarray(member_observations, dtype=np.float32)
        if members.ndim == 2:
            members = members[:, None, :]  # один член на узел: (K, 1, dim)
        ones = np.ones(len(members), dtype=np.float32)
        return TargetSet(
            observations=members,
            b=members.copy(),
            z=members.copy(),
            normalizer=ones,
            normalizer_heads=ones[None] if per_head else None,
        )

    def cost(self, src_observations: np.ndarray, targets: TargetSet) -> np.ndarray:
        src = np.atleast_2d(np.asarray(src_observations, dtype=np.float32))
        # Стоимость до набора — минимум по его членам (max по достижимости).
        return np.stack(
            [np.stack([self._cost_row(s, m) for m in targets.observations], axis=0).min(axis=1)
             for s in src],
            axis=0,
        )

    def cost_from_state(self, observation: np.ndarray, targets: TargetSet) -> np.ndarray:
        return self.cost(np.asarray(observation, dtype=np.float32).reshape(1, -1), targets)[0]

    def steps_from_cost(self, cost: np.ndarray) -> np.ndarray:
        return cost / (-np.log(self.discount))

    def low_action(self, observation, z, seed=None, temperature=0.0) -> np.ndarray:
        """«Политика»: единичный шаг в сторону подцели, закодированной в z."""
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        target = np.asarray(z, dtype=np.float32).reshape(-1)
        direction = target - observation
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 1e-6 else np.zeros_like(direction)

    def high_intent(self, observation, z_reward, seed=None, temperature=0.0) -> np.ndarray:
        return np.asarray(z_reward, dtype=np.float32).reshape(1, -1)

    # -- внутреннее ------------------------------------------------------ #

    def _cost_row(self, src: np.ndarray, targets: np.ndarray) -> np.ndarray:
        p = np.array([self._reach_prob(src, t) for t in targets], dtype=np.float32)
        return -np.log(np.clip(p, MIN_REACH_PROB, 1.0))

    def _reach_prob(self, src: np.ndarray, tgt: np.ndarray) -> float:
        """p = max(честная достижимость, гладкий артефакт дальнего поля)."""
        src_cell = tuple(np.round(src).astype(int))
        tgt_cell = tuple(np.round(tgt).astype(int))

        geodesic = self._geodesics.get(src_cell)
        d_geo = np.inf if geodesic is None else float(geodesic[tgt_cell])
        d_euclid = float(np.linalg.norm(np.asarray(src) - np.asarray(tgt)))

        honest = self.discount**d_geo if np.isfinite(d_geo) else 0.0
        far_field = self.discount**self.honest_radius * self.discount**d_euclid
        return max(honest, far_field)

    def true_cost(self, src: np.ndarray, tgt: np.ndarray) -> float:
        """Истинная стоимость по геодезическому расстоянию — эталон для тестов."""
        geodesic = self._geodesics.get(tuple(np.round(src).astype(int)))
        d_geo = np.inf if geodesic is None else float(geodesic[tuple(np.round(tgt).astype(int))])
        return -np.log(np.clip(self.discount**d_geo, MIN_REACH_PROB, 1.0))


def simulate(
    maze: GridMaze,
    controller,
    start: np.ndarray,
    goal_cell: Tuple[int, int],
    max_steps: int = 300,
    step_size: float = 0.5,
    goal_tol: float = 1.0,
) -> Tuple[bool, int, np.ndarray]:
    """Гоняет контроллер по лабиринту точечной массой.

    Движение со скольжением вдоль стен: если полный шаг упирается, пробуем
    отдельно по каждой оси. Это важно для корректности теста. Настоящая
    low-level политика локально компетентна — она обходит угол, если подцель за
    ним. Точечная масса, жёстко идущая по прямой, застревала бы на любом угле, и
    тест мерил бы не качество плана, а примитивность симулятора.

    Сквозь стену пройти по-прежнему нельзя, поэтому контроллер, поверивший в
    несуществующий проход, честно упирается и никуда не приходит.

    Returns:
        (успех, число шагов, траектория).
    """
    position = np.asarray(start, dtype=np.float32).copy()
    trajectory = [position.copy()]

    for step in range(max_steps):
        action, _ = controller.act(position, seed=None)
        delta = np.asarray(action, dtype=np.float32) * step_size

        for attempt in (delta, np.array([delta[0], 0.0]), np.array([0.0, delta[1]])):
            candidate = position + attempt
            if maze.is_free(*np.round(candidate).astype(int)):
                position = candidate
                break
        trajectory.append(position.copy())

        if np.linalg.norm(position - np.asarray(goal_cell, dtype=np.float32)) <= goal_tol:
            return True, step + 1, np.array(trajectory)

    return False, max_steps, np.array(trajectory)
