"""Проверка, что развёртка воспроизводима при одном сиде.

Написан после того, как обнаружилось: два одинаковых прогона бейзлайна с одним
`run_seed` расходились в 8 эпизодах из 25. Причина была в OGBench —
`MazeEnv.reset` берёт случайность из глобального генератора numpy (`add_noise`)
и из генератора пространства действий (пять стабилизирующих шагов), а до обоих
аргумент `seed` не достаёт. Значит спаренность эпизодов, на которой держится всё
сравнение методов, была фиктивной.

Тест требует чекпоинт и среду, поэтому он отдельный от `test_planning.py`: тот
работает на синтетике и проходит за секунды.

Запуск:
    python tests/test_determinism.py --checkpoint_dir checkpoints/medium
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings('ignore')

import numpy as np

np.in1d = np.isin

from fbplan.experiment import Experiment
from fbplan.rollout import episode_seed, reset_episode


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', default='checkpoints/medium')
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    return p.parse_args()


def test_reset_is_reproducible(env):
    """Один и тот же сид обязан давать одно и то же стартовое состояние."""
    seed = episode_seed(run_seed=0, task_id=1, episode=0)

    first, _ = reset_episode(env, seed, task_id=1)
    # Между сбросами намеренно «пачкаем» состояние среды случайными шагами:
    # именно так и выглядит реальная развёртка.
    for _ in range(50):
        env.step(env.action_space.sample())
    second, _ = reset_episode(env, seed, task_id=1)

    difference = float(np.abs(np.asarray(first) - np.asarray(second)).max())
    assert difference == 0.0, (
        f'сброс с одним сидом дал разные состояния (max разница {difference:.4f}). '
        'Скорее всего, потеряно засеивание np.random или action_space в reset_episode.'
    )


def test_different_seeds_differ(env):
    """Обратная проверка: разные сиды обязаны давать разные старты.

    Без неё тест выше проходил бы и на среде, которая просто игнорирует сид.
    """
    first, _ = reset_episode(env, episode_seed(0, 1, 0), task_id=1)
    second, _ = reset_episode(env, episode_seed(0, 1, 1), task_id=1)

    assert not np.allclose(first, second), 'разные сиды дали одинаковый старт'


def main():
    args = parse_args()
    if not os.path.isdir(args.checkpoint_dir):
        print(f'пропущено: нет чекпоинта {args.checkpoint_dir}')
        return 0

    experiment = Experiment(args.checkpoint_dir, args.env_name)

    failed = 0
    for test in (test_reset_is_reproducible, test_different_seeds_differ):
        try:
            test(experiment.env)
            print(f'  OK   {test.__name__}')
        except AssertionError as exc:
            failed += 1
            print(f'  FAIL {test.__name__}: {exc}')

    print(f'\n{2 - failed}/2 тестов прошло')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
