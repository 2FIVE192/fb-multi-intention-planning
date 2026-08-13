"""Тесты логики планирования на синтетическом лабиринте.

Проверяем именно планировщик, отдельно от нейросетей: здесь известны точные
геодезические расстояния, поэтому известен и правильный ответ. Если тест
падает — виновата Дейкстра, прунинг или критерий переключения подцелей, а не
качество представлений.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from fbplan.graph import (
    DIRECT_TO_GOAL,
    build_subgoal_graph,
    cost_from_steps,
    extract_path,
    solve_goal,
)
from fbplan.planner import GraphPlanner, PlannerConfig
from fbplan.stats import bootstrap_ci, paired_comparison
from fbplan.task_setup import TaskSpec

from tests.synthetic import GAP_CELL, GridMaze, SyntheticOracle, simulate

START_CELL = (7, 11)  # нижний правый угол
GOAL_CELL = (1, 11)  # верхний правый угол: по прямой — через стену


def make_setup(honest_radius: float = 6.0):
    """Лабиринт, оракул и граф по всем свободным клеткам."""
    maze = GridMaze()
    oracle = SyntheticOracle(maze, discount=0.9, honest_radius=honest_radius)

    node_observations = maze.all_states()
    graph = build_subgoal_graph(
        oracle,
        node_observations,
        max_edge_cost=cost_from_steps(oracle, honest_radius),
        k_neighbors=8,
    )
    return maze, oracle, graph


def make_task(goal_cell=GOAL_CELL) -> TaskSpec:
    goal = np.array([goal_cell], dtype=np.float32)
    return TaskSpec(task_id=1, z_reward=goal[0], goal_observations=goal, num_goal_states=1)


class GreedyController:
    """Аналог бейзлайна для синтетики: всегда целится прямо в далёкую цель."""

    name = 'greedy'

    def __init__(self, oracle, goal):
        self.oracle = oracle
        self.goal = np.asarray(goal, dtype=np.float32)

    def reset(self, task):
        self.goal = np.asarray(task.goal_observations[0], dtype=np.float32)

    def act(self, observation, seed=None):
        return self.oracle.low_action(observation, self.goal), {}


# --------------------------------------------------------------------------- #


def test_pruning_respects_threshold():
    """После прунинга не должно остаться рёбер дороже порога."""
    _, oracle, graph = make_setup()
    threshold = cost_from_steps(oracle, 6.0)

    weights = graph.edges.tocoo().data
    assert graph.num_edges > 0, 'граф остался без рёбер'
    assert weights.max() <= threshold + 1e-9, f'ребро дороже порога: {weights.max()} > {threshold}'


def test_far_field_hallucination_exists():
    """Санити: подставной оракул действительно «протекает» сквозь стену.

    Если этот тест не проходит, остальные ничего не проверяют — они бы работали
    на оракуле без патологии, где и жадный контроллер справляется.
    """
    _, oracle, _ = make_setup()
    targets = oracle.make_targets(np.array([GOAL_CELL], dtype=np.float32))

    naive = float(oracle.cost_from_state(np.array(START_CELL, dtype=np.float32), targets)[0])
    truth = oracle.true_cost(np.array(START_CELL), np.array(GOAL_CELL))

    assert naive < truth * 0.8, (
        f'оракул должен ЗАНИЖАТЬ стоимость сквозь стену: naive={naive:.3f}, true={truth:.3f}'
    )


def test_dijkstra_recovers_true_distance():
    """Стоимость пути по графу должна совпадать с истинной геодезической."""
    maze, oracle, graph = make_setup()
    plan = solve_goal(oracle, graph, np.array([GOAL_CELL], dtype=np.float32))

    start_node = int(np.argmin(np.linalg.norm(graph.targets.observations - np.array(START_CELL), axis=1)))
    planned = float(plan.cost_to_goal[start_node])
    truth = oracle.true_cost(np.array(START_CELL), np.array(GOAL_CELL))

    assert np.isfinite(planned), 'путь до цели не найден'
    # Композиция коротких честных рёбер должна давать истинную стоимость с
    # точностью до дискретизации узлов.
    assert abs(planned - truth) < 0.25 * truth, (
        f'путь по графу {planned:.3f} против истины {truth:.3f}'
    )
    # И, в отличие от прямой оценки, он НЕ занижает.
    assert planned > float(plan.direct_cost[start_node]), (
        'путь по графу должен быть дороже галлюцинированной прямой оценки'
    )


def test_extracted_path_goes_through_the_gap():
    """Восстановленный маршрут обязан пройти через единственный проём в стене."""
    maze, oracle, graph = make_setup()
    plan = solve_goal(oracle, graph, np.array([GOAL_CELL], dtype=np.float32))

    start_node = int(np.argmin(np.linalg.norm(graph.targets.observations - np.array(START_CELL), axis=1)))
    path = extract_path(plan, start_node)

    assert len(path) > 1, f'маршрут из одной подцели: {path}'
    cells = [tuple(c) for c in graph.targets.observations[path].round().astype(int)]

    # Рёбра перешагивают через несколько клеток, поэтому сама клетка проёма в
    # списке узлов может и не оказаться. Проверяем переход между половинами:
    # он обязан произойти на колонке проёма.
    wall_row, gap_col = GAP_CELL
    crossings = [
        (a, b) for a, b in zip(cells, cells[1:])
        if (a[0] < wall_row) != (b[0] < wall_row)
    ]
    assert crossings, f'маршрут не пересёк перегородку: {cells}'
    for a, b in crossings:
        assert a[1] == gap_col and b[1] == gap_col, (
            f'перегородка пересечена не через проём (колонка {gap_col}): {a} -> {b}'
        )


def test_planner_solves_maze_where_greedy_fails():
    """Главный тест: планировщик доходит, жадный контроллер застревает."""
    maze, oracle, graph = make_setup()
    task = make_task()

    greedy = GreedyController(oracle, np.array(GOAL_CELL))
    greedy.reset(task)
    greedy_ok, _, _ = simulate(maze, greedy, np.array(START_CELL, dtype=np.float32), GOAL_CELL)

    planner = GraphPlanner(
        oracle,
        graph,
        PlannerConfig(replan_every=1, reach_steps=1.5, max_subgoal_steps=40,
                      switch_margin_steps=0.5, execution='low'),
    )
    planner.reset(task)
    planner_ok, steps, _ = simulate(maze, planner, np.array(START_CELL, dtype=np.float32), GOAL_CELL)

    assert not greedy_ok, 'жадный контроллер неожиданно решил задачу — тест не различает методы'
    assert planner_ok, f'планировщик не дошёл до цели за {steps} шагов'


def test_planner_falls_back_when_graph_is_useless():
    """Без пригодных узлов планировщик обязан вести себя как бейзлайн.

    Это гарантия отсутствия регрессии: наш метод не может быть ХУЖЕ бейзлайна
    из-за самого факта наличия графа.
    """
    maze = GridMaze()
    oracle = SyntheticOracle(maze, discount=0.9, honest_radius=6.0)

    # Порог настолько жёсткий, что не выживает ни одно ребро — включая рёбра в
    # цель. Значит, цель из графа недостижима и планировать нечего.
    graph = build_subgoal_graph(
        oracle, maze.all_states(), max_edge_cost=1e-9, k_neighbors=8
    )
    planner = GraphPlanner(oracle, graph, PlannerConfig(replan_every=1))
    planner.reset(make_task())
    planner.act(np.array(START_CELL, dtype=np.float32), seed=None)

    assert planner._current_node == DIRECT_TO_GOAL, 'ожидался откат на прямое целеуказание'
    assert planner._num_fallbacks > 0, 'откат не был зафиксирован в диагностике'


def test_paired_comparison_and_bootstrap():
    """Санити статистики: парная разность и бутстрэп считаются как задумано."""
    records = []
    for seed in range(4):
        for episode in range(10):
            records.append({'method': 'graph', 'run_seed': seed, 'task_id': 1,
                            'episode': episode, 'success': 1.0})
            records.append({'method': 'baseline', 'run_seed': seed, 'task_id': 1,
                            'episode': episode, 'success': 1.0 if episode < 5 else 0.0})

    cmp = paired_comparison(pd.DataFrame(records), 'graph', 'baseline')
    assert abs(cmp['delta'] - 0.5) < 1e-9, cmp
    assert cmp['num_pairs'] == 40
    assert cmp['wins_a'] == 20 and cmp['wins_b'] == 0

    lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert lo <= 0.25 <= hi, (lo, hi)


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for test in tests:
        try:
            test()
            print(f'  OK   {test.__name__}')
        except AssertionError as exc:
            failed += 1
            print(f'  FAIL {test.__name__}: {exc}')
    print(f'\n{len(tests) - failed}/{len(tests)} тестов прошло')
    sys.exit(1 if failed else 0)
