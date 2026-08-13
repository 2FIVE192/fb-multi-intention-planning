"""E4: абляции. Что именно даёт выигрыш.

Среда, агент и оракул загружаются ОДИН раз и переиспользуются всеми вариантами,
а графы кэшируются на диске — иначе перебор десятка конфигураций упирается в
повторное построение графа, а не в прогон эпизодов.

Разбираемые оси:
    max_edge_steps     порог доверия к ребру — главный гиперпараметр метода;
    num_nodes          плотность графа;
    node_method        покрытие (farthest point sampling) против случайного отбора;
    ensemble_reduce    пессимизм по ансамблю против усреднения;
    execution          подцель прямо в pi_l или сначала через замороженный pi_h;
    replan_every       частота пересмотра плана.

Запуск:
    python scripts/run_ablations.py --checkpoint_dir /path/to/antmaze-medium --seeds 0,1,2
"""

import argparse
import dataclasses
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fbplan.controllers import SingleIntentionController
from fbplan.experiment import Experiment, GraphSpec
from fbplan.planner import GraphPlanner, PlannerConfig
from fbplan.rollout import rollout_all_tasks
from fbplan.stats import summarize

# Ось абляции -> перебираемые значения. Дефолт всегда первый в списке.
SWEEPS = {
    'max_edge_steps': [75.0, 30.0, 50.0, 100.0, 150.0],
    'num_nodes': [1000, 250, 500, 2000],
    'node_method': ['fps', 'random'],
    'ensemble_reduce': ['min', 'mean'],
    'execution': ['low', 'high'],
    'replan_every': [10, 1, 25],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--num_episodes', type=int, default=20)
    p.add_argument('--axes', default=','.join(SWEEPS),
                   help=f'какие оси перебирать, через запятую; доступны {list(SWEEPS)}')
    p.add_argument('--include_baseline', action='store_true',
                   help='добавить бейзлайн как точку отсчёта')
    p.add_argument('--output_dir', default='results/raw')
    p.add_argument('--tag', default='ablations')
    p.add_argument('--graph_cache_dir', default='results/graph_cache')
    return p.parse_args()


def variants(axes):
    """Конфигурации абляции: дефолт плюс по одному отклонению вдоль каждой оси.

    Полный факторный перебор не нужен и слишком дорог; нас интересует
    чувствительность к каждому параметру по отдельности.
    """
    defaults = {axis: values[0] for axis, values in SWEEPS.items()}
    yield 'default', dict(defaults)

    for axis in axes:
        for value in SWEEPS[axis][1:]:
            yield f'{axis}={value}', {**defaults, axis: value}


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    axes = [a.strip() for a in args.axes.split(',') if a.strip()]

    unknown = set(axes) - set(SWEEPS)
    if unknown:
        raise SystemExit(f'неизвестные оси: {sorted(unknown)}; доступны {list(SWEEPS)}')

    os.makedirs(args.output_dir, exist_ok=True)

    # ensemble_reduce влияет и на оракул, поэтому оракул пересобирается внутри
    # цикла; среда и агент грузятся один раз.
    exp = Experiment(args.checkpoint_dir, args.env_name, seed=seeds[0])
    tasks = exp.tasks()

    records, started = [], time.time()

    if args.include_baseline:
        for seed in seeds:
            records.extend(
                _tag(rollout_all_tasks(SingleIntentionController(exp.agent), exp.env, tasks,
                                       num_episodes=args.num_episodes, run_seed=seed),
                     variant='baseline')
            )

    for name, cfg in variants(axes):
        exp.oracle.ensemble_reduce = cfg['ensemble_reduce']

        graph = exp.build_graph(
            GraphSpec(
                num_nodes=cfg['num_nodes'],
                node_method=cfg['node_method'],
                max_edge_steps=cfg['max_edge_steps'],
                ensemble_reduce=cfg['ensemble_reduce'],
            ),
            cache_dir=args.graph_cache_dir,
        )
        planner_config = PlannerConfig(
            replan_every=cfg['replan_every'], execution=cfg['execution']
        )

        for seed in seeds:
            planner = GraphPlanner(exp.oracle, graph, planner_config)
            records.extend(
                _tag(rollout_all_tasks(planner, exp.env, tasks,
                                       num_episodes=args.num_episodes, run_seed=seed),
                     variant=name, **cfg)
            )
        print(f'[abl] {name}: готово, {time.time() - started:.0f} с суммарно')

    df = pd.DataFrame.from_records(records)
    raw_path = os.path.join(args.output_dir, f'{args.tag}_episodes.csv')
    df.to_csv(raw_path, index=False)

    summary = _summarize_variants(df)
    summary.to_csv(os.path.join(args.output_dir, f'{args.tag}_summary.csv'), index=False)

    print('\n' + summary.to_string(index=False))
    print(f'\nсырые данные: {raw_path}')


def _tag(records, variant, **extra):
    for record in records:
        record['variant'] = variant
        record.update(extra)
    return records


def _summarize_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по вариантам: `summarize` работает по колонке method, здесь — variant."""
    renamed = df.rename(columns={'method': 'controller', 'variant': 'method'})
    return summarize(renamed).rename(columns={'method': 'variant'})


if __name__ == '__main__':
    main()
