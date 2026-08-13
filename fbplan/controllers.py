"""Контроллеры с общим интерфейсом `reset(task) -> act(observation, seed)`.

Общий интерфейс нужен, чтобы цикл прогона (`rollout.py`) был буквально одним и
тем же кодом для всех методов: никакой разницы в бюджете шагов, обработке
терминации или сидировании между бейзлайном и нашим планировщиком быть не должно.

Методы
------
`baseline`  — авторский FB pi-Switch: pi_h выбирает интенцию заново каждый шаг,
              глядя на далёкую цель. Вызывается авторский `agent.sample_actions`,
              строка в строку.
`flat`      — pi_l(a | s, z_r) без иерархии вообще. Референс, показывающий,
              сколько даёт сам high-level уровень.
`graph`     — наш планировщик (см. `planner.GraphPlanner`).
"""

from typing import Any, Dict, Tuple

from . import _upstream  # noqa: F401

import numpy as np

from .fb_api import FBOracle
from .task_setup import TaskSpec


class SingleIntentionController:
    """Бейзлайн: одна интенция за раз, перевыбор на каждом шаге.

    Внутри — авторский `agent.sample_actions`, чтобы результат совпадал с
    upstream-оценкой без оговорок.
    """

    name = 'baseline'

    def __init__(self, agent: Any, temperature: float = 0.0):
        self.agent = agent
        self.temperature = temperature
        self._z_reward = None

    def reset(self, task: TaskSpec) -> None:
        self._z_reward = np.asarray(task.z_reward, dtype=np.float32)

    def act(self, observation: np.ndarray, seed) -> Tuple[np.ndarray, Dict[str, Any]]:
        action = self.agent.sample_actions(
            observation, self._z_reward, seed=seed, temperature=self.temperature
        )
        return np.asarray(action), {}


class FlatController:
    """Неиерархический референс: pi_l(a | s, z_r), high-level не участвует."""

    name = 'flat'

    def __init__(self, oracle: FBOracle, temperature: float = 0.0):
        self.oracle = oracle
        self.temperature = temperature
        self._z_reward = None

    def reset(self, task: TaskSpec) -> None:
        self._z_reward = self.oracle.normalize_z(task.z_reward).reshape(1, -1)

    def act(self, observation: np.ndarray, seed) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        action = self.oracle.low_action(
            observation, self._z_reward, seed=seed, temperature=self.temperature
        )
        return action.reshape(-1), {}
