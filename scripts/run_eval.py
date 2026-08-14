"""Основной прогон: бейзлайн против планировщика на задачах OGBench.

Пример:
    python scripts/run_eval.py \
        --checkpoint_dir /path/to/antmaze-medium \
        --env_name ogbench-antmaze-medium-navigate-v0 \
        --methods baseline,graph \
        --seeds 0,1,2,3,4 \
        --num_episodes 20

Все методы в одном запуске делят среду, агента и граф, а эпизоды стартуют из
одинаковых состояний (см. `fbplan/rollout.py`), поэтому сравнение парное.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fbplan.controllers import FlatController, SingleIntentionController
from fbplan.experiment import Experiment, GraphSpec
from fbplan.planner import GraphPlanner, PlannerConfig
from fbplan.rollout import rollout_all_tasks
from fbplan.stats import format_summary, paired_comparison, summarize, summarize_by_task

METHODS = ('baseline', 'flat', 'graph')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--checkpoint_dir', required=True, help='директория с params_*.pkl и flags.json')
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    p.add_argument('--methods', default='baseline,graph', help=f'через запятую из {METHODS}')
    p.add_argument('--seeds', default='0,1,2,3,4', help='сиды прогона через запятую')
    p.add_argument('--num_episodes', type=int, default=20, help='эпизодов на задачу на сид')
    p.add_argument('--temperature', type=float, default=0.0, help='температура политик при оценке')

    g = p.add_argument_group('граф подцелей')
    g.add_argument('--num_nodes', type=int, default=500)
    g.add_argument('--node_method', default='fps', choices=('fps', 'random'))
    g.add_argument('--k_neighbors', type=int, default=16)
    g.add_argument('--num_members', type=int, default=16,
                   help='состояний в наборе одного узла; успех метода упирается именно в это')
    g.add_argument('--member_stride', type=int, default=4,
                   help='шаг между членами набора; num_members * stride — охват окна в шагах')
    g.add_argument('--normalizer_references', type=int, default=2000,
                   help='опорных состояний для оценки знаменателя max_s M(s -> узел)')
    g.add_argument('--max_edge_steps', type=float, default=75.0,
                   help='максимальная длина ребра в шагах среды: FB надёжен только на коротких переходах')
    g.add_argument('--ensemble_reduce', default='min', choices=('min', 'mean'))
    g.add_argument('--node_seed', type=int, default=0,
                   help='-1 — свой граф на каждый сид прогона (проверка устойчивости к отбору узлов)')

    c = p.add_argument_group('онлайн-контроллер')
    c.add_argument('--replan_every', type=int, default=10)
    c.add_argument('--reach_steps', type=float, default=20.0)
    c.add_argument('--max_subgoal_steps', type=int, default=120)
    c.add_argument('--switch_margin_steps', type=float, default=5.0)
    c.add_argument('--plan_advantage_steps', type=float, default=0.0,
                   help='насколько маршрут по графу должен быть дешевле прямого броска, '
                        'чтобы его предпочли: оценка до цели надёжнее оценки до узла')
    c.add_argument('--min_commit_steps', type=int, default=40,
                   help='минимальное удержание подцели; защита от дребезга выбора')
    c.add_argument('--tail_estimate', default='dijkstra', choices=('dijkstra', 'direct'),
                   help="чем оценивать хвост до цели: 'dijkstra' — многошаговая "
                        "композиция, 'direct' — план ровно из одной подцели")
    c.add_argument('--execution', default='high', choices=('low', 'high'),
                   help="'high' — подцель идёт через замороженный pi_h, как у бейзлайна")

    o = p.add_argument_group('вывод')
    o.add_argument('--output_dir', default='results/raw')
    o.add_argument('--tag', default='main', help='префикс имён файлов результата')
    o.add_argument('--graph_cache_dir', default='results/graph_cache')
    o.add_argument('--no_progress', action='store_true')

    return p.parse_args()


def build_controller(name, exp, args, run_seed):
    """Создаёт контроллер по имени метода."""
    if name == 'baseline':
        return SingleIntentionController(exp.agent, temperature=args.temperature)
    if name == 'flat':
        return FlatController(exp.oracle, temperature=args.temperature)
    if name == 'graph':
        node_seed = run_seed if args.node_seed < 0 else args.node_seed
        spec = GraphSpec(
            num_nodes=args.num_nodes,
            node_method=args.node_method,
            k_neighbors=args.k_neighbors,
            max_edge_steps=args.max_edge_steps,
            ensemble_reduce=args.ensemble_reduce,
            node_seed=node_seed,
            num_members=args.num_members,
            member_stride=args.member_stride,
            normalizer_references=args.normalizer_references,
        )
        graph = exp.build_graph(spec, cache_dir=args.graph_cache_dir)
        config = PlannerConfig(
            replan_every=args.replan_every,
            reach_steps=args.reach_steps,
            max_subgoal_steps=args.max_subgoal_steps,
            switch_margin_steps=args.switch_margin_steps,
            min_commit_steps=args.min_commit_steps,
            plan_advantage_steps=args.plan_advantage_steps,
            tail_estimate=args.tail_estimate,
            execution=args.execution,
            temperature=args.temperature,
        )
        return GraphPlanner(exp.oracle, graph, config)
    raise ValueError(f'неизвестный метод {name!r}, доступны {METHODS}')


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f'неизвестные методы: {sorted(unknown)}; доступны {METHODS}')

    os.makedirs(args.output_dir, exist_ok=True)

    exp = Experiment(
        args.checkpoint_dir,
        args.env_name,
        ensemble_reduce=args.ensemble_reduce,
        seed=seeds[0],
    )
    tasks = exp.tasks()

    started = time.time()
    records = []
    for run_seed in seeds:
        for method in methods:
            controller = build_controller(method, exp, args, run_seed)
            records.extend(
                rollout_all_tasks(
                    controller,
                    exp.env,
                    tasks,
                    num_episodes=args.num_episodes,
                    run_seed=run_seed,
                    progress=not args.no_progress,
                )
            )
            done = sum(r['method'] == method and r['run_seed'] == run_seed for r in records)
            print(f'[eval] сид {run_seed}, метод {method}: {done} эпизодов, '
                  f'{time.time() - started:.0f} с суммарно')

    df = pd.DataFrame.from_records(records)

    raw_path = os.path.join(args.output_dir, f'{args.tag}_episodes.csv')
    df.to_csv(raw_path, index=False)

    meta = {**exp.metadata(), 'args': vars(args)}
    with open(os.path.join(args.output_dir, f'{args.tag}_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    summary = summarize(df)
    summary.to_csv(os.path.join(args.output_dir, f'{args.tag}_summary.csv'), index=False)
    summarize_by_task(df).to_csv(
        os.path.join(args.output_dir, f'{args.tag}_by_task.csv'), index=False
    )

    print('\n' + format_summary(summary))

    if 'graph' in methods and 'baseline' in methods:
        cmp = paired_comparison(df, 'graph', 'baseline')
        print(
            f'\nпарное сравнение graph - baseline: {cmp["delta"]:+.3f} '
            f'[{cmp["ci_low"]:+.3f}, {cmp["ci_high"]:+.3f}], '
            f'пар {cmp["num_pairs"]}, разошлись на {cmp["frac_disagree"]:.1%} '
            f'(побед graph {cmp["wins_a"]}, baseline {cmp["wins_b"]})'
        )
        with open(os.path.join(args.output_dir, f'{args.tag}_paired.json'), 'w', encoding='utf-8') as f:
            json.dump(cmp, f, ensure_ascii=False, indent=2)

    print(f'\nрезультаты: {raw_path}')


if __name__ == '__main__':
    main()
