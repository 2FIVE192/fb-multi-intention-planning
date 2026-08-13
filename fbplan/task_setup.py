"""Подготовка задачи: латентная задача z_r и целевые состояния.

Оба метода — и бейзлайн, и планировщик — получают на вход ровно одно и то же:
оффлайн-датасет, переразмеченный reward-функцией задачи. Никакого доступа к
`info['goal']` или к координатам цели. Это ключ к честности сравнения.
"""

import dataclasses
from typing import Any, Dict, Optional

from . import _upstream  # noqa: F401

import numpy as np

from utils.datasets import Dataset
from utils.env_utils import relabel_dataset


@dataclasses.dataclass
class TaskSpec:
    """Всё, что нужно знать про одну задачу (task_id).

    Attributes:
        task_id: номер задачи в OGBench (1..5).
        z_reward: (d,) латентная задача из `agent.infer_latent` — вход бейзлайна.
        goal_observations: (M, obs_dim) состояния датасета с reward = 1 — вход
            планировщика.
        num_goal_states: сколько таких состояний нашлось всего в датасете.
    """

    task_id: int
    z_reward: np.ndarray
    goal_observations: np.ndarray
    num_goal_states: int


def prepare_task(
    agent: Any,
    env: Any,
    env_name: str,
    dataset: Dataset,
    task_id: int,
    num_zero_shot_samples: int = 100_000,
    max_goal_states: int = 64,
    seed: int = 0,
) -> TaskSpec:
    """Повторяет протокол zero-shot из авторского `main.py` и добавляет цели.

    Args:
        dataset: датасет для zero-shot вывода латента (у авторов — валидационный,
            при его отсутствии — тренировочный). Требует поля `qpos`, то есть
            загрузку с `add_info=True`.
        num_zero_shot_samples: сколько первых переходов идёт в `infer_latent`.
        max_goal_states: сколько целевых состояний оставить (их бывают тысячи,
            а нужны они только чтобы задать целевой узел графа).
    """
    env.reset(options=dict(task_id=task_id))
    relabeled = relabel_dataset(env_name, env, dataset)

    n = min(num_zero_shot_samples, relabeled.size)
    if n < num_zero_shot_samples:
        print(f'[task] в датасете только {n} переходов < {num_zero_shot_samples} запрошенных')

    # Эквивалентно авторскому `zero_shot_dataset.sample(n, idxs=arange(n),
    # relabeling=False, augmentation=False)`: обёртка HGCDataset при этих флагах
    # просто берёт срез, а `infer_latent` смотрит лишь на observations и rewards.
    zero_shot_batch = {
        'observations': np.asarray(relabeled['observations'][:n]),
        'rewards': np.asarray(relabeled['rewards'][:n]),
    }
    z_reward = np.asarray(agent.infer_latent(zero_shot_batch))

    goal_observations, num_goal_states = _extract_goal_states(
        relabeled, max_goal_states=max_goal_states, seed=seed
    )

    return TaskSpec(
        task_id=task_id,
        z_reward=z_reward,
        goal_observations=goal_observations,
        num_goal_states=num_goal_states,
    )


def _extract_goal_states(
    relabeled: Dataset, max_goal_states: int, seed: int
) -> tuple:
    """Состояния датасета, попадающие в целевую область (reward = 1)."""
    rewards = np.asarray(relabeled['rewards'])
    (goal_idxs,) = np.nonzero(rewards > 0)

    if len(goal_idxs) == 0:
        raise RuntimeError(
            'В датасете нет ни одного состояния в целевой области. '
            'Планировщику не из чего построить целевой узел — проверьте '
            'task_id и то, что датасет соответствует среде.'
        )

    rng = np.random.default_rng(seed)
    if len(goal_idxs) > max_goal_states:
        chosen = rng.choice(goal_idxs, size=max_goal_states, replace=False)
    else:
        chosen = goal_idxs

    observations = np.asarray(relabeled['observations'][np.sort(chosen)], dtype=np.float32)
    return observations, int(len(goal_idxs))


def dataset_xy(dataset: Dataset) -> Optional[np.ndarray]:
    """Ground-truth координаты (x, y) из `qpos` — ТОЛЬКО для анализа и графиков.

    В методе не используются нигде: ни в отборе узлов, ни в рёбрах, ни в
    критерии переключения подцелей.
    """
    if 'qpos' not in dataset:
        return None
    return np.asarray(dataset['qpos'])[:, :2]
