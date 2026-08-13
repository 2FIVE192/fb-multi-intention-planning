"""Граф подцелей поверх FB и динамическое программирование на нём.

Рёбра графа — стоимости `c(w_i -> w_j) = -log p(w_i -> w_j)` из `fb_api`. Они
неотрицательны и аддитивны вдоль цепочки, поэтому оптимальная последовательность
интенций — это буквально кратчайший путь (Дейкстра).

Смысл прунинга рёбер
--------------------
FB-оценка `p` тем точнее, чем короче переход: при gamma = 0.99 значение
gamma^500 ~ 6.6e-3 уже тонет в шуме низкорангового (d = 128) приближения, а
gamma^50 ~ 0.6 — вполне разрешимо. Поэтому мы оставляем только «короткие»
рёбра, а длинные маршруты собираем из них композицией. Это же отсекает
галлюцинированные проходы сквозь стены: они почти всегда выглядят как одно
длинное ребро.
"""

import dataclasses
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .fb_api import FBOracle, TargetSet

# Порог, выше которого стоимость считается бесконечной (ребра нет).
UNREACHABLE = np.inf


def cost_from_steps(oracle: FBOracle, steps: float) -> float:
    """Переводит «горизонт в шагах среды» в порог стоимости: c = -H * log(gamma)."""
    return float(steps * -np.log(oracle.discount))


@dataclasses.dataclass
class SubgoalGraph:
    """Направленный граф подцелей.

    Attributes:
        targets: `TargetSet` узлов (B, нормированные интенции, M(w -> w)).
        dataset_idxs: (K,) индексы узлов в исходном датасете — только для анализа.
        cost_matrix: (K, K) плотные стоимости до прунинга; нужны для абляций и
            графиков, на планирование не влияют.
        edges: (K, K) разреженная матрица после прунинга.
        max_edge_cost: использованный порог.
    """

    targets: TargetSet
    dataset_idxs: np.ndarray
    cost_matrix: np.ndarray
    edges: csr_matrix
    max_edge_cost: float

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def num_edges(self) -> int:
        return int(self.edges.nnz)


def build_subgoal_graph(
    oracle: FBOracle,
    node_observations: np.ndarray,
    max_edge_cost: float,
    k_neighbors: int = 16,
    dataset_idxs: Optional[np.ndarray] = None,
) -> SubgoalGraph:
    """Строит граф подцелей: K^2 запросов к F, затем прунинг рёбер.

    Args:
        node_observations: (K, obs_dim) состояния-узлы.
        max_edge_cost: рёбра дороже отбрасываются (см. `cost_from_steps`).
        k_neighbors: сколько ближайших исходящих рёбер оставить у каждого узла.
    """
    targets = oracle.make_targets(node_observations)
    if dataset_idxs is None:
        dataset_idxs = np.arange(len(targets))

    targets, dataset_idxs = _drop_degenerate_nodes(targets, np.asarray(dataset_idxs))

    cost_matrix = oracle.cost(targets.observations, targets)  # (K, K)
    # p(w -> w) = 1 по определению; численно оно может слегка отличаться.
    np.fill_diagonal(cost_matrix, 0.0)

    edges = _prune_edges(cost_matrix, max_edge_cost=max_edge_cost, k_neighbors=k_neighbors)

    return SubgoalGraph(
        targets=targets,
        dataset_idxs=np.asarray(dataset_idxs),
        cost_matrix=cost_matrix,
        edges=edges,
        max_edge_cost=float(max_edge_cost),
    )


def _drop_degenerate_nodes(targets: TargetSet, dataset_idxs: np.ndarray):
    """Выбрасывает узлы с неположительной самомерой M(w -> w).

    M(w -> w) стоит в знаменателе формулы p(s -> w) = M(s -> w) / M(w -> w).
    Если он не строго положителен, нормировка теряет смысл, и такая вершина не
    годится в подцели. У обученного FB это пик successor measure, так что доля
    таких узлов должна быть близка к нулю — заметная доля означает, что с
    представлением что-то не так, поэтому мы её печатаем, а не глушим.
    """
    valid = (targets.self_measure > 0.0).all(axis=0)  # строго во всех головах ансамбля
    dropped = int((~valid).sum())

    if dropped:
        print(
            f'[graph] отброшено узлов с M(w -> w) <= 0: {dropped} из {len(valid)} '
            f'({dropped / len(valid):.1%})'
        )
    if not valid.any():
        raise RuntimeError(
            'Ни у одного узла нет положительной самомеры M(w -> w). '
            'Похоже, чекпоинт не обучен или загрузился неверно.'
        )

    idxs = np.nonzero(valid)[0]
    return targets.subset(idxs), dataset_idxs[idxs]


def _prune_edges(cost_matrix: np.ndarray, max_edge_cost: float, k_neighbors: int) -> csr_matrix:
    """Оставляет у каждого узла k самых дешёвых исходящих рёбер дешевле порога."""
    n = cost_matrix.shape[0]
    costs = cost_matrix.copy()
    np.fill_diagonal(costs, UNREACHABLE)  # петли не нужны

    k = min(k_neighbors, n - 1)
    nearest = np.argpartition(costs, kth=k - 1, axis=1)[:, :k]  # (n, k)

    rows = np.repeat(np.arange(n), k)
    cols = nearest.reshape(-1)
    values = costs[rows, cols]

    keep = np.isfinite(values) & (values <= max_edge_cost)

    # Дейкстра в scipy трактует явный ноль как «ребра нет», поэтому
    # нулевые стоимости подменяем на минимальный положительный эпсилон.
    weights = np.maximum(values[keep], 1e-12)
    return csr_matrix((weights, (rows[keep], cols[keep])), shape=(n, n))


@dataclasses.dataclass
class GoalPlan:
    """Результат DP для конкретной цели.

    Attributes:
        cost_to_goal: (K,) стоимость пути от каждого узла до цели (inf — нет пути).
        next_hop: (K,) индекс следующего узла на оптимальном пути; -1 — переход
            напрямую в цель; -9999 — пути нет.
        direct_cost: (K,) стоимость прямого перехода узел -> цель, без графа.
        goal_targets: `TargetSet` состояний-целей (их может быть несколько).
        z_goal: (d,) нормированная латентная задача — среднее B по целевым
            состояниям. Ровно то, что даёт авторский `infer_latent`, и ровно то,
            чем кондиционируется бейзлайн.
    """

    cost_to_goal: np.ndarray
    next_hop: np.ndarray
    direct_cost: np.ndarray
    goal_targets: TargetSet
    z_goal: np.ndarray


NO_PATH = -9999
DIRECT_TO_GOAL = -1


def solve_goal(
    oracle: FBOracle,
    graph: SubgoalGraph,
    goal_observations: np.ndarray,
    z_goal: Optional[np.ndarray] = None,
) -> GoalPlan:
    """Один прогон Дейкстры: стоимость до цели из каждого узла графа.

    Args:
        goal_observations: (M, obs_dim) — состояния датасета, попадающие в
            целевую область (reward = 1). Их может быть много; стоимость до
            «цели» берётся как минимум по ним. Использование состояний
            ДАТАСЕТА, а не привилегированного `info['goal']`, принципиально:
            наш метод должен получать ровно тот же вход, что и бейзлайн, —
            оффлайн-данные плюс reward-функция задачи.

    Цель добавляется временным узлом с индексом K. Поиск идёт по
    транспонированному графу от цели — так за один вызов получаем расстояния
    «до цели» сразу для всех узлов.
    """
    goal_observations = np.atleast_2d(np.asarray(goal_observations, dtype=np.float32))
    goal_targets = oracle.make_targets(goal_observations)

    usable_goals = (goal_targets.self_measure > 0.0).all(axis=0)
    if not usable_goals.any():
        print('[graph] ВНИМАНИЕ: ни одно целевое состояние не имеет M(g -> g) > 0; '
              'планировщик будет откатываться на поведение бейзлайна')
    elif not usable_goals.all():
        goal_targets = goal_targets.subset(np.nonzero(usable_goals)[0])

    # min по целевым состояниям: попасть в любое из них — значит решить задачу.
    direct_cost = oracle.cost(graph.targets.observations, goal_targets).min(axis=1)  # (K,)

    # По умолчанию z_r — среднее B по целевым состояниям (то же, что даёт
    # `infer_latent`). Скрипты оценки передают сюда латент бейзлайна явно,
    # чтобы «прямой бросок в цель» был у обоих методов побитово одинаковым.
    if z_goal is None:
        z_goal = oracle.normalize_z(goal_targets.b.mean(axis=0))
    else:
        z_goal = oracle.normalize_z(z_goal)

    n = len(graph)
    augmented = _augment_with_goal(graph, direct_cost)

    # dijkstra по транспонированному графу из узла-цели == расстояния до цели.
    dist, predecessors = dijkstra(
        augmented.T.tocsr(), directed=True, indices=n, return_predecessors=True
    )

    cost_to_goal = dist[:n]
    # predecessors[i] на транспонированном графе = следующий узел на пути i -> цель.
    next_hop = predecessors[:n].astype(np.int64)
    next_hop[next_hop == n] = DIRECT_TO_GOAL
    next_hop[~np.isfinite(cost_to_goal)] = NO_PATH

    return GoalPlan(
        cost_to_goal=cost_to_goal,
        next_hop=next_hop,
        direct_cost=direct_cost,
        goal_targets=goal_targets,
        z_goal=z_goal,
    )


def _augment_with_goal(graph: SubgoalGraph, direct_cost: np.ndarray) -> csr_matrix:
    """Добавляет к графу узел-цель с рёбрами (узел -> цель) дешевле порога."""
    n = len(graph)
    keep = np.isfinite(direct_cost) & (direct_cost <= graph.max_edge_cost)

    rows = np.concatenate([graph.edges.tocoo().row, np.nonzero(keep)[0]])
    cols = np.concatenate([graph.edges.tocoo().col, np.full(int(keep.sum()), n)])
    vals = np.concatenate([graph.edges.tocoo().data, np.maximum(direct_cost[keep], 1e-12)])

    return csr_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))


def extract_path(plan: GoalPlan, start_node: int, max_len: int = 128) -> list:
    """Восстанавливает последовательность узлов от `start_node` до цели.

    Возвращает список индексов узлов (без самой цели). Нужен для визуализации
    планов в отчёте и для проверки композициональности.
    """
    if plan.next_hop[start_node] == NO_PATH:
        return []

    path, node = [start_node], start_node
    while len(path) <= max_len:
        nxt = plan.next_hop[node]
        if nxt in (DIRECT_TO_GOAL, NO_PATH):
            break
        path.append(int(nxt))
        node = int(nxt)
    return path
