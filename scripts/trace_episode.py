"""Трассировка одного эпизода планировщика: что он выбирает и почему.

Агрегированные метрики говорят, что метод работает хуже, но не говорят почему.
Здесь печатается по шагам: какая подцель выбрана, насколько она далеко по оценке
FB и насколько на самом деле, менялась ли она и по какой причине.

Привилегированные координаты используются только для печати.

Запуск:
    python scripts/trace_episode.py --checkpoint_dir checkpoints/medium --task_id 1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import numpy as np

from fbplan.experiment import Experiment, GraphSpec
from fbplan.graph import DIRECT_TO_GOAL, extract_path, solve_goal
from fbplan.maze_analysis import all_geodesic_fields, geodesic_distances, maze_geometry
from fbplan.planner import GraphPlanner, PlannerConfig
from fbplan.rollout import episode_seed, reset_episode


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    p.add_argument('--task_id', type=int, default=1)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--run_seed', type=int, default=0)
    p.add_argument('--num_nodes', type=int, default=1000)
    p.add_argument('--max_edge_steps', type=float, default=75.0)
    p.add_argument('--replan_every', type=int, default=10)
    p.add_argument('--reach_steps', type=float, default=20.0)
    p.add_argument('--switch_margin_steps', type=float, default=5.0)
    p.add_argument('--max_subgoal_steps', type=int, default=120)
    p.add_argument('--execution', default='low', choices=('low', 'high'))
    p.add_argument('--graph_cache_dir', default='results/graph_cache')
    return p.parse_args()


def main():
    args = parse_args()

    exp = Experiment(args.checkpoint_dir, args.env_name)
    graph = exp.build_graph(
        GraphSpec(num_nodes=args.num_nodes, max_edge_steps=args.max_edge_steps),
        cache_dir=args.graph_cache_dir,
    )
    task = next(t for t in exp.tasks() if t.task_id == args.task_id)

    grid, unit = maze_geometry(exp.env)
    fields = all_geodesic_fields(grid)
    node_xy = np.asarray(exp.train_dataset['qpos'])[graph.representative_idxs][:, :2]
    goal_xy = np.asarray(exp.env.unwrapped.task_infos[args.task_id - 1]['goal_xy'])

    plan = solve_goal(exp.oracle, graph, task.goal_observations, z_goal=task.z_reward)
    _report_plan(exp, graph, plan, node_xy, goal_xy, grid, unit, fields)

    planner = GraphPlanner(
        exp.oracle,
        graph,
        PlannerConfig(
            replan_every=args.replan_every,
            reach_steps=args.reach_steps,
            switch_margin_steps=args.switch_margin_steps,
            max_subgoal_steps=args.max_subgoal_steps,
            execution=args.execution,
        ),
    )
    _trace(args, exp, planner, task, node_xy, goal_xy, grid, unit, fields)


def _report_plan(exp, graph, plan, node_xy, goal_xy, grid, unit, fields):
    """Качество самого плана, до всякого исполнения."""
    reachable = np.isfinite(plan.cost_to_goal)
    print(f'\n=== ПЛАН ===')
    print(f'узлов с найденным маршрутом до цели: {reachable.sum()} из {len(graph)} '
          f'({reachable.mean():.1%})')

    true_to_goal = geodesic_distances(
        grid, unit, node_xy, np.repeat(goal_xy[None], len(node_xy), axis=0), fields=fields
    )
    ok = reachable & np.isfinite(true_to_goal)
    planned_steps = exp.oracle.steps_from_cost(plan.cost_to_goal[ok])
    print(f'корреляция плановой стоимости с истинной геодезической: '
          f'{np.corrcoef(planned_steps, true_to_goal[ok])[0, 1]:.3f}')
    print(f'плановая стоимость до цели: медиана {np.median(planned_steps):.0f} шагов, '
          f'макс {planned_steps.max():.0f}')


def _trace(args, exp, planner, task, node_xy, goal_xy, grid, unit, fields):
    planner.reset(task)
    # Через reset_episode, а не напрямую: иначе старт эпизода невоспроизводим —
    # OGBench берёт часть случайности мимо аргумента seed (см. rollout.py).
    observation, _ = reset_episode(
        exp.env, episode_seed(args.run_seed, task.task_id, args.episode), task.task_id
    )

    rng = jax.random.PRNGKey(args.run_seed)
    print(f'\n=== ЭПИЗОД (задача {task.task_id}, эпизод {args.episode}) ===')
    print(f'{"шаг":>5} {"позиция":>16} {"подцель":>16} {"до подцели":>11} '
          f'{"истинно":>9} {"до цели":>9} {"смен":>6}')
    print('-' * 82)

    prev_node, done, step = object(), False, 0
    while not done:
        rng, key = jax.random.split(rng)
        action, info = planner.act(observation, seed=key)

        if info['node'] != prev_node:
            xy = np.asarray(observation[:2])
            node = info['node']
            if node == DIRECT_TO_GOAL:
                subgoal_xy, label = goal_xy, 'ЦЕЛЬ'
            else:
                subgoal_xy = node_xy[node]
                label = f'#{node}'

            fb_steps = exp.oracle.steps_from_cost(
                float(planner._current_cost(
                    exp.oracle.cost_from_state(xy_obs(observation), planner.graph.targets),
                    float(exp.oracle.cost_from_state(
                        xy_obs(observation), planner._plan.goal_targets)[0]),
                ))
            )
            true_steps = geodesic_distances(
                grid, unit, xy[None], subgoal_xy[None], fields=fields
            )[0] * 11.0  # ~11 шагов среды на мировую единицу
            to_goal = geodesic_distances(grid, unit, xy[None], goal_xy[None], fields=fields)[0]

            print(f'{step:>5} {np.array2string(xy, precision=1):>16} '
                  f'{label:>16} {fb_steps:>10.0f} {true_steps:>9.0f} '
                  f'{to_goal:>8.1f}е {info["num_switches"]:>6}')
            prev_node = info['node']

        observation, _, terminated, truncated, env_info = exp.env.step(np.clip(action, -1, 1))
        step += 1
        done = terminated or truncated

    print(f'\nитог: успех={env_info.get("success", 0)}, шагов={step}, '
          f'смен подцели={info["num_switches"]}, откатов={info["num_fallbacks"]}')


def xy_obs(observation):
    """Наблюдение в форме, которую ждёт оракул."""
    return np.asarray(observation, dtype=np.float32).reshape(1, -1)


if __name__ == '__main__':
    main()
