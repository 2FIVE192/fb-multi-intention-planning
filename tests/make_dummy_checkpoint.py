"""Фиктивный чекпоинт со случайными весами — для smoke-теста пайплайна.

Позволяет прогнать весь путь (загрузка -> оракул -> граф -> Дейкстра ->
контроллер -> статистика) не дожидаясь настоящих чекпоинтов. Метрики при этом
бессмысленны: сети не обучены. Проверяется только то, что код сходится по
формам, не падает и выдаёт корректные по структуре результаты.

Запуск:
    python tests/make_dummy_checkpoint.py --output_dir results/dummy_checkpoint
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings

warnings.filterwarnings('ignore')

import numpy as np

np.in1d = np.isin

import flax

from fbplan import _upstream  # noqa: F401
from fbplan.checkpoint import make_example_batch

from agents.fbpiswitch import FBpiSwitchAgent, get_config
from utils.env_utils import make_env_and_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    parser.add_argument('--output_dir', default='results/dummy_checkpoint')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1_000_000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Ветка `dataset_only=True` в upstream сломана (обращается к неинициализи-
    # рованной переменной `env`), поэтому грузим полным путём.
    _, train_dataset, _ = make_env_and_datasets(args.env_name)
    example_batch = make_example_batch(train_dataset['observations'], train_dataset['actions'])

    config = dict(get_config())
    agent = FBpiSwitchAgent.create(args.seed, example_batch, config)

    params_path = os.path.join(args.output_dir, f'params_{args.epoch}.pkl')
    with open(params_path, 'wb') as f:
        pickle.dump({'agent': flax.serialization.to_state_dict(agent)}, f)

    with open(os.path.join(args.output_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump({'agent': config, 'env_name': args.env_name, 'seed': args.seed}, f,
                  ensure_ascii=False, indent=2)

    print(f'фиктивный чекпоинт: {params_path}')
    print('ВНИМАНИЕ: веса случайные, метрики бессмысленны — это только smoke-тест.')


if __name__ == '__main__':
    main()
