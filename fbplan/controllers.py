"""Контроллеры с общим интерфейсом `reset(task) -> act(observation, seed)`.

Общий интерфейс нужен, чтобы цикл прогона (`rollout.py`) был буквально одним и
тем же кодом для всех методов: разницы в бюджете шагов, обработке терминации или
сидировании между бейзлайном и планировщиком быть не должно. Иначе сравнение
методов измеряло бы в том числе и различия в обвязке.

Методы
------
`baseline`  — авторский FB pi-Switch: pi_h выбирает интенцию заново каждый шаг,
              глядя на далёкую цель. Вызывается авторский `agent.sample_actions`,
              строка в строку.
`flat`      — pi_l(a | s, z_r) без иерархии вообще. Референс, показывающий,
              сколько даёт сам по себе high-level уровень: замер по отчёту —
              0.65 против 0.75 у бейзлайна, то есть иерархия стоит +10 п.п.
`graph`     — планировщик последовательности интенций (`planner.GraphPlanner`).
"""

from typing import Any, Dict, Tuple

from . import _upstream  # noqa: F401

import numpy as np

from .fb_api import FBOracle
from .task_setup import TaskSpec


class SingleIntentionController:
    """Бейзлайн: одна интенция за раз, перевыбор на каждом шаге.

    Реализации здесь намеренно нет: вызывается авторский
    `agent.sample_actions`. Это принципиально для чистоты сравнения — если бы я
    переписал бейзлайн сам, любое расхождение с числами статьи пришлось бы
    объяснять, и было бы невозможно отличить ошибку переписывания от свойств
    метода.
    """

    name = 'baseline'

    def __init__(self, agent: Any, temperature: float = 0.0):
        """Args:
            agent: загруженный `FBpiSwitchAgent` из upstream.
            temperature: температура сэмплирования политик; 0 — детерминированно.
        """
        self.agent = agent
        self.temperature = temperature
        self._z_reward = None

    def reset(self, task: TaskSpec) -> None:
        """Запоминает латент задачи перед эпизодом.

        `z_reward` приходит из авторского `infer_latent` и НЕ нормируется здесь:
        `sample_actions` делает это сам. Планировщик получает ровно тот же
        латент, поэтому «прямой бросок в цель» у обоих методов побитово
        одинаков.
        """
        self._z_reward = np.asarray(task.z_reward, dtype=np.float32)

    def act(self, observation: np.ndarray, seed) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Действие бейзлайна. Второй элемент — пустая диагностика.

        Диагностика возвращается ради единообразия с планировщиком, который
        отдаёт в ней число смен подцели и откатов: `rollout.py` пишет её в csv
        одинаково для всех методов.
        """
        action = self.agent.sample_actions(
            observation, self._z_reward, seed=seed, temperature=self.temperature
        )
        return np.asarray(action), {}


class FlatController:
    """Неиерархический референс: pi_l(a | s, z_r), high-level не участвует.

    Нужен, чтобы отделить вклад иерархии от вклада представлений. Без него
    невозможно сказать, что именно улучшает результат: наличие high-level
    контроллера или качество FB само по себе.
    """

    name = 'flat'

    def __init__(self, oracle: FBOracle, temperature: float = 0.0):
        self.oracle = oracle
        self.temperature = temperature
        self._z_reward = None

    def reset(self, task: TaskSpec) -> None:
        """Готовит латент задачи в форме, которую ждёт `pi_l`.

        В отличие от бейзлайна нормировка нужна здесь явно: `low_action`
        обращается к политике напрямую, минуя `sample_actions`, который делал бы
        это сам.
        """
        self._z_reward = self.oracle.normalize_z(task.z_reward).reshape(1, -1)

    def act(self, observation: np.ndarray, seed) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Действие low-level политики, кондиционированной прямо на задаче."""
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        action = self.oracle.low_action(
            observation, self._z_reward, seed=seed, temperature=self.temperature
        )
        return action.reshape(-1), {}
