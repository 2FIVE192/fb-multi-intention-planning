"""E2 и E5: карты ценности и качество графа.

Строит четыре картинки для отчёта:

1. Прямая FB-оценка `p(s -> цель)` по лабиринту. Ожидаем увидеть патологию:
   поле «плоское» вдали от цели и протекает сквозь стены.
2. Плановая стоимость до цели `min_w [ c(s -> w) + d*(w -> цель) ]`. Ожидаем
   аккуратные «слои» вдоль коридоров.
3. Рёбра графа поверх карты стен; рёбра, прошивающие стену, выделены. Это
   честная мера того, сколько галлюцинаций пережило прунинг.
4. Пример маршрута из подцелей для дальней задачи.

Запуск:
    python scripts/analysis_value_maps.py --checkpoint_dir /path/to/antmaze-medium --task_id 1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fbplan.experiment import Experiment, GraphSpec
from fbplan.graph import extract_path, solve_goal
from fbplan.maze_analysis import crosses_wall, maze_geometry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    p.add_argument('--task_id', type=int, default=1, help='для каких целей рисовать карты')
    p.add_argument('--num_nodes', type=int, default=1000)
    p.add_argument('--k_neighbors', type=int, default=16)
    p.add_argument('--max_edge_steps', type=float, default=75.0)
    p.add_argument('--ensemble_reduce', default='min', choices=('min', 'mean'))
    p.add_argument('--grid_resolution', type=float, default=0.8,
                   help='шаг сетки запросов в мировых единицах')
    p.add_argument('--output_dir', default='results/figures')
    p.add_argument('--graph_cache_dir', default='results/graph_cache')
    return p.parse_args()


def representative_states(dataset, resolution: float):
    """По одному реальному состоянию датасета на ячейку сетки (x, y).

    Синтезировать наблюдения нельзя: это 29-мерная поза муравья, а не точка на
    плоскости. Поэтому берём настоящие состояния и раскладываем их по ячейкам.
    """
    xy = np.asarray(dataset['qpos'])[:, :2]
    cell = np.floor(xy / resolution).astype(np.int64)

    # np.unique по строкам даёт по одному индексу на уникальную ячейку.
    _, first_idx = np.unique(cell, axis=0, return_index=True)
    return np.asarray(dataset['observations'][first_idx], dtype=np.float32), xy[first_idx]


def draw_walls(ax, grid, unit):
    """Рисует стены лабиринта поверх осей."""
    for i, j in np.argwhere(grid != 0):
        ax.add_patch(
            plt.Rectangle(
                ((j - 1.5) * unit, (i - 1.5) * unit), unit, unit,
                facecolor='0.85', edgecolor='0.6', linewidth=0.4, zorder=0,
            )
        )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    exp = Experiment(args.checkpoint_dir, args.env_name, ensemble_reduce=args.ensemble_reduce)
    graph = exp.build_graph(
        GraphSpec(
            num_nodes=args.num_nodes,
            k_neighbors=args.k_neighbors,
            max_edge_steps=args.max_edge_steps,
            ensemble_reduce=args.ensemble_reduce,
        ),
        cache_dir=args.graph_cache_dir,
    )

    task = next(t for t in exp.tasks() if t.task_id == args.task_id)
    plan = solve_goal(exp.oracle, graph, task.goal_observations, z_goal=task.z_reward)

    grid, unit = maze_geometry(exp.env)
    query_obs, query_xy = representative_states(exp.train_dataset, args.grid_resolution)
    print(f'[E2] запросов по сетке: {len(query_obs)}')

    direct_cost = exp.oracle.cost(query_obs, plan.goal_targets)[:, 0]
    planned_cost = _planned_cost(exp, graph, plan, query_obs)

    _plot_value_maps(args, exp, grid, unit, query_xy, direct_cost, planned_cost, task)
    _plot_graph_quality(args, exp, graph, grid, unit)
    _plot_example_route(args, exp, graph, plan, grid, unit, task)

    print(f'\nкартинки: {args.output_dir}')


def _planned_cost(exp, graph, plan, query_obs, chunk=256):
    """min_w [ c(s -> w) + d*(w -> цель) ] по всем состояниям сетки."""
    out = np.empty(len(query_obs), dtype=np.float64)
    for i in range(0, len(query_obs), chunk):
        cost_to_nodes = exp.oracle.cost(query_obs[i : i + chunk], graph.targets)
        total = cost_to_nodes + plan.cost_to_goal[None, :]
        # Тот же порог доверия, что и в планировщике.
        total = np.where(cost_to_nodes <= graph.max_edge_cost, total, np.inf)
        out[i : i + chunk] = total.min(axis=1)
    return out


def _plot_value_maps(args, exp, grid, unit, xy, direct_cost, planned_cost, task):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    goal_xy = exp.env.unwrapped.task_infos[task.task_id - 1]['goal_xy']

    panels = [
        (axes[0], exp.oracle.steps_from_cost(direct_cost),
         'Прямая FB-оценка: c(s → цель)\nодин запрос, как у бейзлайна'),
        (axes[1], exp.oracle.steps_from_cost(planned_cost),
         'Плановая стоимость: min_w [ c(s → w) + d*(w → цель) ]\nкомпозиция коротких переходов'),
    ]

    # Общая шкала: иначе визуальное сравнение вводит в заблуждение.
    finite = np.concatenate([v[np.isfinite(v)] for _, v, _ in panels])
    vmin, vmax = np.percentile(finite, [1, 99])

    for ax, values, title in panels:
        draw_walls(ax, grid, unit)
        mask = np.isfinite(values)
        sc = ax.scatter(xy[mask, 0], xy[mask, 1], c=values[mask], s=6,
                        cmap='viridis_r', vmin=vmin, vmax=vmax, zorder=1)
        ax.scatter(xy[~mask, 0], xy[~mask, 1], c='crimson', s=6, marker='x',
                   label='пути нет', zorder=2)
        ax.scatter(*goal_xy, marker='*', s=320, c='gold', edgecolor='black',
                   linewidth=0.8, label='цель', zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=8)
        fig.colorbar(sc, ax=ax, label='оценка расстояния, шагов среды')

    fig.suptitle(f'Задача {task.task_id}: одношаговая оценка против плановой', fontsize=12)
    path = os.path.join(args.output_dir, f'value_maps_task{task.task_id}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[E2] {path}')


def _plot_graph_quality(args, exp, graph, grid, unit):
    """E5: сколько рёбер прошивает стены, и как они распределены по длине."""
    node_xy = np.asarray(exp.train_dataset['qpos'])[graph.representative_idxs][:, :2]
    coo = graph.edges.tocoo()
    bad = crosses_wall(grid, unit, node_xy[coo.row], node_xy[coo.col])

    # Для сравнения: сколько было бы галлюцинаций без прунинга по порогу.
    all_costs = graph.cost_matrix.copy()
    np.fill_diagonal(all_costs, np.inf)
    dense_src, dense_dst = np.nonzero(np.isfinite(all_costs))
    sample = np.random.default_rng(0).choice(len(dense_src), size=min(20000, len(dense_src)),
                                             replace=False)
    bad_dense = crosses_wall(grid, unit, node_xy[dense_src[sample]], node_xy[dense_dst[sample]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    draw_walls(axes[0], grid, unit)
    for keep, color, label, width in (
        (~bad, 'tab:blue', 'корректные', 0.25),
        (bad, 'crimson', 'сквозь стену', 0.8),
    ):
        rows, cols = coo.row[keep], coo.col[keep]
        segments_x = np.stack([node_xy[rows, 0], node_xy[cols, 0]])
        segments_y = np.stack([node_xy[rows, 1], node_xy[cols, 1]])
        axes[0].plot(segments_x, segments_y, color=color, linewidth=width, alpha=0.5, zorder=1)
        axes[0].plot([], [], color=color, label=f'{label} ({keep.sum()})')
    axes[0].scatter(node_xy[:, 0], node_xy[:, 1], s=3, c='black', zorder=2)
    axes[0].set_title(f'Рёбра графа после прунинга\nсквозь стену: {bad.mean():.1%}', fontsize=10)
    axes[0].set_aspect('equal')
    axes[0].legend(loc='upper left', fontsize=8)

    steps = exp.oracle.steps_from_cost(coo.data)
    bins = np.linspace(0, args.max_edge_steps, 25)
    axes[1].hist([steps[~bad], steps[bad]], bins=bins, stacked=True,
                 color=['tab:blue', 'crimson'], label=['корректные', 'сквозь стену'])
    axes[1].set_xlabel('длина ребра по оценке FB, шагов среды')
    axes[1].set_ylabel('число рёбер')
    axes[1].set_title(
        f'Длина рёбер и доля галлюцинаций\n'
        f'без порога было бы {bad_dense.mean():.1%}, после прунинга {bad.mean():.1%}',
        fontsize=10,
    )
    axes[1].legend(fontsize=8)

    path = os.path.join(args.output_dir, 'graph_quality.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[E5] {path}  (сквозь стену: {bad.mean():.1%}, без прунинга: {bad_dense.mean():.1%})')


def _plot_example_route(args, exp, graph, plan, grid, unit, task):
    """Пример последовательности подцелей от стартовой клетки задачи."""
    node_xy = np.asarray(exp.train_dataset['qpos'])[graph.representative_idxs][:, :2]
    info = exp.env.unwrapped.task_infos[task.task_id - 1]

    start_node = int(np.argmin(np.linalg.norm(node_xy - np.asarray(info['init_xy']), axis=1)))
    route = extract_path(plan, start_node)

    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    draw_walls(ax, grid, unit)
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=3, c='0.6', zorder=1)

    if route:
        route_xy = node_xy[route]
        ax.plot(route_xy[:, 0], route_xy[:, 1], '-o', color='tab:blue', markersize=5,
                linewidth=1.6, label=f'подцелей: {len(route)}', zorder=2)
    ax.scatter(*info['init_xy'], marker='s', s=140, c='tab:green', edgecolor='black',
               label='старт', zorder=3)
    ax.scatter(*info['goal_xy'], marker='*', s=320, c='gold', edgecolor='black',
               label='цель', zorder=3)

    ax.set_title(f'Задача {task.task_id}: спланированная последовательность интенций', fontsize=10)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)

    path = os.path.join(args.output_dir, f'route_task{task.task_id}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[E2] {path}  (подцелей в маршруте: {len(route)})')


if __name__ == '__main__':
    main()
