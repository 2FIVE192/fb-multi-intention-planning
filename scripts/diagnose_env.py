"""Локализация падения при создании среды MuJoCo.

Зачем. На Colab создание среды падало по SIGSEGV, причём одинаково при egl и при
osmesa. Osmesa — программный рендер и GPU не трогает вообще, поэтому дело не в
графическом backend'е, и подбирать его бесполезно. При этом прямой вызов
`gymnasium.make` в том же окружении отрабатывал, а путь через upstream — нет.
Отличаются они одним: upstream по цепочке импортов тянет jax до создания среды.

Скрипт разделяет эти случаи. Каждая проба запускается ОТДЕЛЬНЫМ процессом с
включённым faulthandler, поэтому падение печатает стек Python и не роняет
остальные пробы. Датасет не нужен — только создание среды.

Запуск:
    python scripts/diagnose_env.py                    # все пробы, таблицей
    python scripts/diagnose_env.py --backends egl,osmesa
    python scripts/diagnose_env.py --probe jax_then_env   # одна проба в этом же процессе
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Пробы упорядочены по нарастанию: первая же упавшая называет виновника.
PROBES = {
    'env_only': 'создание среды без jax',
    'jax_import_then_env': 'импорт jax (без обращения к устройствам), затем среда',
    'jax_devices_then_env': 'импорт jax + jax.devices(), затем среда',
    'env_then_jax': 'сначала среда, потом импорт jax',
    'upstream_path': 'полный путь через utils.env_utils, как в рабочем коде',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--probe', choices=sorted(PROBES), help='выполнить одну пробу в этом процессе')
    p.add_argument('--backends', default='auto,egl,osmesa',
                   help="значения MUJOCO_GL через запятую; 'auto' — не задавать "
                        'переменную и оставить выбор самому mujoco')
    p.add_argument('--env_name', default='ogbench-antmaze-medium-navigate-v0')
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Сами пробы. Импорты намеренно внутри функций: важен ПОРЯДОК загрузки библиотек.
# --------------------------------------------------------------------------- #


def _make_env_directly(env_name: str):
    """Создание среды в обход upstream — тем же идентификатором, что и он.

    Для `antmaze-medium-navigate-v0` ogbench строит `antmaze-medium-v0` и
    никаких дополнительных аргументов не передаёт, поэтому вызов эквивалентен.
    """
    import gymnasium
    import ogbench  # noqa: F401  (импорт регистрирует среды)

    plain = '-'.join(env_name.replace('ogbench-', '').split('-')[:-2] + ['v0'])
    print(f'    gymnasium.make({plain!r})', flush=True)
    gymnasium.make(plain).close()


def probe_env_only(env_name):
    _make_env_directly(env_name)


def probe_jax_import_then_env(env_name):
    import jax  # noqa: F401
    print('    jax импортирован (устройства не запрашивались)', flush=True)
    _make_env_directly(env_name)


def probe_jax_devices_then_env(env_name):
    import jax
    print(f'    jax.devices() -> {jax.devices()}', flush=True)
    _make_env_directly(env_name)


def probe_env_then_jax(env_name):
    _make_env_directly(env_name)
    import jax
    print(f'    jax.devices() -> {jax.devices()}', flush=True)


def probe_upstream_path(env_name):
    from fbplan import _upstream  # noqa: F401
    from utils.env_utils import make_env_and_datasets

    print('    make_env_and_datasets(env_only=True)', flush=True)
    make_env_and_datasets(env_name, env_only=True)


# --------------------------------------------------------------------------- #


def run_single(name: str, env_name: str) -> None:
    """Выполняет одну пробу в текущем процессе."""
    import faulthandler
    import warnings

    faulthandler.enable()  # печатает стек Python при SIGSEGV
    warnings.filterwarnings('ignore')

    import numpy as np
    np.in1d = np.isin  # numpy 2 убрал in1d, а ogbench 1.1.4 его ещё зовёт

    globals()[f'probe_{name}'](env_name)
    print('    успех', flush=True)


def run_all(args) -> int:
    backends = [b.strip() for b in args.backends.split(',') if b.strip()]
    results = {}

    for backend in backends:
        print(f'\n=== MUJOCO_GL={backend} ===')
        for name, description in PROBES.items():
            print(f'  [{name}] {description}')
            child_env = dict(os.environ)
            if backend == 'auto':
                child_env.pop('MUJOCO_GL', None)
            else:
                child_env['MUJOCO_GL'] = backend
            completed = subprocess.run(
                [sys.executable, '-X', 'faulthandler', '-u', __file__,
                 '--probe', name, '--env_name', args.env_name],
                env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
            )
            print(completed.stdout.rstrip())
            results[(backend, name)] = completed.returncode
            print(f'  -> код {completed.returncode}\n')

    print('=' * 62)
    print(f'{"проба":<24}' + ''.join(f'{b:>12}' for b in backends))
    print('-' * 62)
    for name in PROBES:
        row = ''.join(
            f'{("ok" if results[(b, name)] == 0 else str(results[(b, name)])):>12}'
            for b in backends
        )
        print(f'{name:<24}{row}')

    print('\nПервая упавшая проба и называет виновника: пробы отличаются друг от '
          'друга ровно одним шагом.')
    return 0 if all(code == 0 for code in results.values()) else 1


def main() -> int:
    args = parse_args()
    if args.probe:
        run_single(args.probe, args.env_name)
        return 0
    return run_all(args)


if __name__ == '__main__':
    sys.exit(main())
