"""Замер смещения отбора: чем ошибается argmin по сравнению со средним узлом.

Центральный замер отчёта (раздел 5.10). Идея простая: сравнить ошибку оценки
`c(s -> w)` у ТИПИЧНОГО узла с ошибкой у того узла, который реально выбирает
планировщик. Если вторая систематически оптимистичнее, значит `argmin` выбирает
не ближайший узел, а тот, чья ошибка оказалась самой выгодной, — и случайный
шум превращается в систематическое смещение просто фактом выбора минимума.

Заодно меряется, как смещение зависит от числа членов в наборе узла: агрегация
по набору гасит шум позы, и вопрос в том, достаточно ли этого.

Привилегированные координаты используются ТОЛЬКО здесь, как эталон истины;
в самом методе их нет.

Запуск:
    python scripts/analysis_selection_bias.py --checkpoint_dir checkpoints/medium
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from fbplan.experiment import Experiment, GraphSpec
from fbplan.maze_analysis import all_geodesic_fields, geodesic_distances, maze_geometry

#: Сколько шагов среды приходится на одну мировую единицу (замер calibrate.py).
STEPS_PER_UNIT = 11.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    p.add_argument('--num_nodes', type=int, default=500)
    p.add_argument('--num_members', type=int, default=16, help='максимум; меньшие — подмножества')
    p.add_argument('--member_stride', type=int, default=4)
    p.add_argument('--num_sources', type=int, default=200, help='состояний, из которых меряем')
    p.add_argument('--num_references', type=int, default=1500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='results/raw/selection_bias.csv')
    p.add_argument('--graph_cache_dir', default='results/graph_cache')
    return p.parse_args()


def main():
    args = parse_args()
    exp = Experiment(args.checkpoint_dir, args.env_name)
    graph = exp.build_graph(
        GraphSpec(num_nodes=args.num_nodes, num_members=args.num_members,
                  member_stride=args.member_stride),
        cache_dir=args.graph_cache_dir,
    )

    qpos = np.asarray(exp.train_dataset['qpos'])[:, :2]
    grid, unit = maze_geometry(exp.env)
    fields = all_geodesic_fields(grid)
    node_xy = qpos[graph.representative_idxs]

    rng = np.random.default_rng(args.seed)
    source_idxs = rng.choice(len(qpos), args.num_sources, replace=False)
    source_obs = np.asarray(exp.train_dataset['observations'][source_idxs], dtype=np.float32)
    references = np.asarray(
        exp.train_dataset['observations'][rng.choice(len(qpos), args.num_references, replace=False)],
        dtype=np.float32,
    )

    # Эталон: истинное геодезическое расстояние от каждого источника до каждого узла.
    print(f'[bias] считаю истинные расстояния для {args.num_sources} состояний...')
    true_steps = np.stack([
        geodesic_distances(grid, unit, np.repeat(qpos[i][None], len(node_xy), axis=0),
                           node_xy, fields=fields)
        for i in source_idxs
    ]) * STEPS_PER_UNIT
    reachable = np.isfinite(true_steps)
    rows_idx = np.arange(len(source_obs))

    members = graph.targets.observations
    widths = [w for w in (1, 2, 4, 8, 16, 32) if w <= members.shape[1]]

    records = []
    for width in widths:
        targets = exp.oracle.make_targets(members[:, :width], reference_observations=references)
        estimated = exp.oracle.steps_from_cost(exp.oracle.cost(source_obs, targets))

        # Кого выберет планировщик и насколько ошибётся именно на нём.
        picked = np.argmin(np.where(reachable, estimated, np.inf), axis=1)
        error_picked = estimated[rows_idx, picked] - true_steps[rows_idx, picked]
        error_mean = np.nanmean(np.where(reachable, estimated - true_steps, np.nan))

        records.append({
            'num_members': width,
            'error_average_node': float(error_mean),
            'error_selected_node': float(np.mean(error_picked)),
            'selection_bias': float(np.mean(error_picked) - error_mean),
            'true_distance_selected': float(np.mean(true_steps[rows_idx, picked])),
        })
        print(f'  членов {width:>3}: у среднего {error_mean:+8.1f}ш, '
              f'у выбранного {np.mean(error_picked):+8.1f}ш, '
              f'смещение {records[-1]["selection_bias"]:+8.1f}ш')

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f'\nсохранено: {args.output}')
    print('Отрицательная ошибка означает, что FB считает узел БЛИЖЕ, чем он есть.')


if __name__ == '__main__':
    main()
