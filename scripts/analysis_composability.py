"""E1: проверка композициональности — центральный эксперимент отчёта.

Гипотеза. Одношаговый контроллер ограничен не архитектурой, а численным
разрешением FB на длинном горизонте. При gamma = 0.99 значение gamma^500 ~ 6.6e-3
для задач antmaze-medium тонет в шуме низкорангового (d = 128) приближения,
тогда как gamma^50 ~ 0.6 разрешается уверенно.

Что меряем. Для пар состояний с известным геодезическим расстоянием сравниваем
две оценки стоимости:
    прямую     c(s -> g) = -log p(s -> g), один запрос к FB;
    составную  кратчайший путь по графу из коротких рёбер.
и смотрим, как каждая ведёт себя с ростом истинной дальности.

Предсказание, которое можно опровергнуть: прямая оценка выходит на плато
(теряет контраст) и систематически ЗАНИЖАЕТ дальние расстояния, а составная
остаётся примерно линейной по истинной дальности.

Запуск:
    python scripts/analysis_composability.py --checkpoint_dir /path/to/antmaze-medium
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from fbplan.experiment import Experiment, GraphSpec
from fbplan.maze_analysis import all_geodesic_fields, geodesic_distances, maze_geometry


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    p.add_argument('--num_nodes', type=int, default=1000)
    p.add_argument('--k_neighbors', type=int, default=16)
    p.add_argument('--max_edge_steps', type=float, default=75.0)
    p.add_argument('--ensemble_reduce', default='min', choices=('min', 'mean'))
    p.add_argument('--num_pairs', type=int, default=4000, help='сколько пар узлов измерять')
    p.add_argument('--output_dir', default='results/raw')
    p.add_argument('--tag', default='composability')
    p.add_argument('--graph_cache_dir', default='results/graph_cache')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    exp = Experiment(args.checkpoint_dir, args.env_name, ensemble_reduce=args.ensemble_reduce)
    spec = GraphSpec(
        num_nodes=args.num_nodes,
        k_neighbors=args.k_neighbors,
        max_edge_steps=args.max_edge_steps,
        ensemble_reduce=args.ensemble_reduce,
    )
    graph = exp.build_graph(spec, cache_dir=args.graph_cache_dir)

    # Стоимости всех кратчайших путей по графу: композиция коротких рёбер.
    print('[E1] Дейкстра для всех пар узлов...')
    path_cost = dijkstra(graph.edges, directed=True)

    # Привилегированные координаты — только здесь, только для оси абсцисс.
    grid, unit = maze_geometry(exp.env)
    node_xy = np.asarray(exp.train_dataset['qpos'])[graph.representative_idxs][:, :2]
    fields = all_geodesic_fields(grid)

    rng = np.random.default_rng(0)
    n = len(graph)
    src = rng.integers(0, n, size=args.num_pairs)
    dst = rng.integers(0, n, size=args.num_pairs)
    keep = src != dst
    src, dst = src[keep], dst[keep]

    print(f'[E1] измеряю {len(src)} пар...')
    true_distance = geodesic_distances(grid, unit, node_xy[src], node_xy[dst], fields=fields)

    df = pd.DataFrame(
        {
            'src': src,
            'dst': dst,
            'true_geodesic': true_distance,
            'direct_cost': graph.cost_matrix[src, dst],
            'path_cost': path_cost[src, dst],
            'direct_steps': exp.oracle.steps_from_cost(graph.cost_matrix[src, dst]),
            'path_steps': exp.oracle.steps_from_cost(path_cost[src, dst]),
            'src_x': node_xy[src, 0], 'src_y': node_xy[src, 1],
            'dst_x': node_xy[dst, 0], 'dst_y': node_xy[dst, 1],
        }
    )
    df = df[np.isfinite(df['true_geodesic'])]

    out_path = os.path.join(args.output_dir, f'{args.tag}_pairs.csv')
    df.to_csv(out_path, index=False)

    _print_summary(df)
    print(f'\nсырые данные: {out_path}')


def _print_summary(df: pd.DataFrame) -> None:
    """Сводка по бинам дальности: где именно прямая оценка теряет контраст."""
    bins = [0, 8, 16, 24, 32, 40, 1e9]
    labels = ['0-8', '8-16', '16-24', '24-32', '32-40', '40+']
    df = df.assign(bin=pd.cut(df['true_geodesic'], bins=bins, labels=labels, right=False))

    print(f'\n{"дальность":>10} {"пар":>6} {"прямая":>10} {"по графу":>10} '
          f'{"недостижим":>11} {"corr(прямая)":>13}')
    print('-' * 66)
    for label, group in df.groupby('bin', observed=True):
        if len(group) == 0:
            continue
        finite = group[np.isfinite(group['path_cost'])]
        corr = (
            np.corrcoef(group['true_geodesic'], group['direct_cost'])[0, 1]
            if len(group) > 2 else np.nan
        )
        print(
            f'{label:>10} {len(group):>6} {group["direct_cost"].mean():>10.3f} '
            f'{finite["path_cost"].mean() if len(finite) else float("nan"):>10.3f} '
            f'{1 - len(finite) / len(group):>10.1%} {corr:>13.3f}'
        )

    overall_direct = np.corrcoef(df['true_geodesic'], df['direct_cost'])[0, 1]
    finite = df[np.isfinite(df['path_cost'])]
    overall_path = np.corrcoef(finite['true_geodesic'], finite['path_cost'])[0, 1]
    print(f'\nкорреляция с истинной дальностью:  прямая {overall_direct:.3f}, '
          f'по графу {overall_path:.3f}')
    print('Гипотеза подтверждается, если корреляция у прямой оценки заметно ниже '
          'и падает с ростом дальности.')


if __name__ == '__main__':
    main()
