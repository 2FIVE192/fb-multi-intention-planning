"""Загрузка замороженного FB pi-Switch агента из авторского чекпоинта.

Чекпоинт содержит четыре модуля:
    modules_forward_repr   — forward-репрезентация F(s, z)
    modules_backward_repr  — backward-репрезентация B(s)
    modules_actor          — low-level политика pi_l(a | s, z)
    modules_high_actor     — high-level контроллер pi_h(z_w | s, z_r), это бейзлайн

Мы не создаём собственных сетей: агент инстанцируется авторским
`FBpiSwitchAgent.create`, а веса грузятся штатным flax-десериализатором.
"""

import glob
import json
import os
import pickle
import re
from typing import Any, Dict, Optional, Tuple

from . import _upstream  # noqa: F401  (побочный эффект: sys.path)

import flax
import numpy as np

from agents.fbpiswitch import FBpiSwitchAgent, get_config as get_default_config


def find_params_file(checkpoint_dir: str, restore_epoch: Optional[int] = None) -> Tuple[str, int]:
    """Находит файл весов `params_{epoch}.pkl` в директории чекпоинта.

    Если `restore_epoch` не задан, берётся самая поздняя эпоха.
    """
    candidates = sorted(glob.glob(os.path.join(checkpoint_dir, 'params_*.pkl')))
    if not candidates:
        raise FileNotFoundError(
            f'В {checkpoint_dir} нет файлов params_*.pkl. '
            'Укажите директорию с чекпоинтом (в ней же должен лежать flags.json).'
        )

    epochs = {}
    for path in candidates:
        match = re.search(r'params_(\d+)\.pkl$', os.path.basename(path))
        if match is not None:
            epochs[int(match.group(1))] = path

    if restore_epoch is None:
        restore_epoch = max(epochs)
    if restore_epoch not in epochs:
        raise FileNotFoundError(
            f'Нет эпохи {restore_epoch} в {checkpoint_dir}. Доступны: {sorted(epochs)}'
        )
    return epochs[restore_epoch], restore_epoch


def load_agent_config(checkpoint_dir: str) -> Dict[str, Any]:
    """Читает конфиг агента из `flags.json`, дополняя его дефолтами upstream.

    Дефолты нужны потому, что в старых чекпоинтах могут отсутствовать поля,
    добавленные позже, — без них `FBpiSwitchAgent.create` упадёт по KeyError.
    """
    config = dict(get_default_config())

    flags_path = os.path.join(checkpoint_dir, 'flags.json')
    if os.path.exists(flags_path):
        with open(flags_path, 'r', encoding='utf-8') as f:
            saved_flags = json.load(f)
        config.update(saved_flags.get('agent', {}))
    else:
        print(f'[checkpoint] flags.json не найден в {checkpoint_dir}; беру дефолтный конфиг.')

    return config


def read_env_name(checkpoint_dir: str) -> Optional[str]:
    """Возвращает `env_name`, с которым обучался чекпоинт (или None)."""
    flags_path = os.path.join(checkpoint_dir, 'flags.json')
    if not os.path.exists(flags_path):
        return None
    with open(flags_path, 'r', encoding='utf-8') as f:
        return json.load(f).get('env_name')


def make_example_batch(observations: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
    """Минимальный батч для инициализации формы сетей."""
    return {
        'observations': np.asarray(observations[:1], dtype=np.float32),
        'actions': np.asarray(actions[:1], dtype=np.float32),
    }


def load_agent(
    checkpoint_dir: str,
    example_batch: Dict[str, np.ndarray],
    seed: int = 0,
    restore_epoch: Optional[int] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[FBpiSwitchAgent, Dict[str, Any]]:
    """Создаёт агента и восстанавливает в него веса чекпоинта.

    Returns:
        (agent, config) — агент со всеми четырьмя модулями и его конфиг.
    """
    config = load_agent_config(checkpoint_dir)
    if config_overrides:
        config.update(config_overrides)

    agent = FBpiSwitchAgent.create(seed, example_batch, config)

    params_path, epoch = find_params_file(checkpoint_dir, restore_epoch)
    with open(params_path, 'rb') as f:
        state_dict = pickle.load(f)['agent']

    _check_checkpoint_modules(state_dict, params_path)

    # То же, что делает upstream `utils.flax_utils.restore_agent`; вынесено сюда
    # ради вменяемого разрешения пути (авторская версия использует glob с
    # assert len(candidates) == 1, что ломается на путях с пробелами).
    agent = flax.serialization.from_state_dict(agent, state_dict)

    print(f'[checkpoint] загружено: {params_path} (epoch={epoch})')
    return agent, config


_REQUIRED_MODULES = (
    'modules_forward_repr',
    'modules_backward_repr',
    'modules_actor',
    'modules_high_actor',
)


def _check_checkpoint_modules(state_dict: Dict[str, Any], params_path: str) -> None:
    """Проверяет, что в чекпоинте лежат все четыре нужных модуля.

    Делается ДО десериализации: иначе несовпадение структуры даст невнятную
    ошибку flax где-то в глубине дерева параметров.
    """
    try:
        present = set(state_dict['network']['params'].keys())
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f'Неожиданная структура чекпоинта {params_path}: '
            f'ожидался ключ ["network"]["params"], получено {list(state_dict)}'
        ) from exc

    missing = [m for m in _REQUIRED_MODULES if m not in present]
    if missing:
        raise RuntimeError(
            f'В чекпоинте {params_path} отсутствуют модули: {missing}. '
            f'Найдены: {sorted(present)}'
        )
