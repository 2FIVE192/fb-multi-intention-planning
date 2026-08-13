"""Калибровка масштабов среды: откуда взяты дефолтные пороги планировщика.

Пороги планировщика задаются в шагах среды и переводятся в стоимости через
c = −H·log γ. Чтобы эти числа не были взяты с потолка, здесь измеряется, как
шаги среды соотносятся с расстоянием в лабиринте, и как длина задач
соотносится с эффективным горизонтом дисконтирования.

Чекпоинт не нужен — считается только по датасету и геометрии среды.

Запуск:
    python scripts/calibrate.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings('ignore')

import numpy as np

np.in1d = np.isin

from fbplan import _upstream  # noqa: F401
from fbplan.maze_analysis import all_geodesic_fields, geodesic_distances, maze_geometry

from utils.env_utils import make_env_and_datasets


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    p.add_argument('--discount', type=float, default=0.99, help='γ, с которым обучался FB')
    return p.parse_args()


def main():
    args = parse_args()
    env, train_dataset, _ = make_env_and_datasets(args.env_name, add_info=True)
    grid, unit = maze_geometry(env)

    print(f'\nсреда: {args.env_name}')
    print(f'карта {grid.shape}, свободных клеток {(grid == 0).sum()}, '
          f'размер клетки {unit} мировых единиц')

    steps_per_unit = _report_speed(train_dataset)
    _report_tasks(env, grid, unit, steps_per_unit)
    _report_horizon(args.discount, unit, steps_per_unit)


def _report_speed(dataset) -> float:
    """Сколько шагов среды нужно, чтобы сместиться на одну мировую единицу."""
    xy = np.asarray(dataset['qpos'])[:, :2]
    (ends,) = np.nonzero(np.asarray(dataset['terminals']))
    episode_len = int(ends[0]) + 1
    num_episodes = len(xy) // episode_len
    xy = xy[: num_episodes * episode_len].reshape(num_episodes, episode_len, 2)

    print(f'\nдатасет: {num_episodes} эпизодов по {episode_len} шагов')
    print(f'\n{"шагов":>8} {"медианное смещение":>20} {"90-й перцентиль":>18}')
    print('-' * 48)

    reference = None
    for k in (10, 25, 50, 100, 200):
        displacement = np.linalg.norm(xy[:, k:] - xy[:, :-k], axis=-1)
        median = float(np.median(displacement))
        print(f'{k:>8} {median:>17.2f} ед. {np.percentile(displacement, 90):>15.2f} ед.')
        if k == 50:
            reference = k / median

    print(f'\nмасштаб: ~{reference:.1f} шагов среды на одну мировую единицу '
          f'(~{reference * 4:.0f} шагов на клетку)')
    return reference


def _report_tasks(env, grid, unit, steps_per_unit):
    """Геодезическая длина каждой задачи — она же ожидаемая сложность."""
    fields = all_geodesic_fields(grid)
    print(f'\n{"задача":>8} {"старт":>14} {"цель":>14} {"геодезич.":>11} '
          f'{"по прямой":>11} {"~шагов":>9}')
    print('-' * 72)

    for task_id, info in enumerate(env.unwrapped.task_infos, start=1):
        start = np.array([info['init_xy']], dtype=np.float64)
        goal = np.array([info['goal_xy']], dtype=np.float64)
        geodesic = float(geodesic_distances(grid, unit, start, goal, fields=fields)[0])
        straight = float(np.linalg.norm(goal - start))
        print(
            f'{task_id:>8} {str(info["init_xy"]):>14} {str(info["goal_xy"]):>14} '
            f'{geodesic:>9.1f} ед. {straight:>9.1f} ед. {geodesic * steps_per_unit:>9.0f}'
        )
    print('\nЗадачи с большим отношением геодезической длины к прямой требуют обхода —\n'
          'именно на них одношаговый контроллер должен проигрывать сильнее всего.')


def _report_horizon(discount, unit, steps_per_unit):
    """Сопоставление горизонта дисконтирования с длиной задач."""
    horizon = 1.0 / (1.0 - discount)
    print(f'\nγ = {discount}: эффективный горизонт 1/(1−γ) = {horizon:.0f} шагов '
          f'≈ {horizon / steps_per_unit:.1f} мировых единиц ≈ '
          f'{horizon / steps_per_unit / unit:.1f} клетки')

    print(f'\n{"дистанция, шагов":>18} {"γ^H":>12}')
    print('-' * 32)
    for steps in (50, 100, 200, 400, 800):
        print(f'{steps:>18} {discount ** steps:>12.4f}')

    print('\nВывод: на дистанции всей задачи величина γ^H падает на порядки ниже\n'
          'значений вблизи единицы, которыми оперирует представление вблизи цели.\n'
          'Контраста на длинном горизонте нет — отсюда и весь метод.')


if __name__ == '__main__':
    main()
