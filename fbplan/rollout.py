"""Единый цикл прогона для всех методов.

Один и тот же код гоняет бейзлайн и планировщик: одинаковый бюджет шагов,
одинаковая обработка терминации, одинаковые начальные состояния.

Про сидирование
---------------
Эпизод i задачи t стартует через `reset_episode(env, f(run_seed, t, i), t)`. Значит при
одном и том же `run_seed` бейзлайн и наш метод видят ПОБИТОВО одинаковые
начальные состояния. Сравнение становится парным, и разброс, вызванный
случайностью стартов, из разницы методов уходит. Авторский `evaluate`
сидируется от `np.random.randint`, то есть даёт несопоставимые между запусками
старты; для нашей задачи это лишний шум.
"""

from typing import Any, Dict, List

import jax
import numpy as np
from tqdm import trange

from .task_setup import TaskSpec


def episode_seed(run_seed: int, task_id: int, episode: int) -> int:
    """Детерминированный сид старта эпизода."""
    return int((run_seed * 1_000_003 + task_id * 10_007 + episode) % (2**31 - 1))


def reset_episode(env: Any, seed: int, task_id: int):
    """Сброс среды с ПОЛНОСТЬЮ воспроизводимым стартовым состоянием.

    Одного `env.reset(seed=...)` для этого не хватает, и это не наша ошибка, а
    поведение OGBench. `MazeEnv.reset` берёт случайность из двух источников, до
    которых аргумент `seed` не достаёт:

    * `add_noise` сдвигает стартовую позицию через `np.random.uniform`, то есть
      через ГЛОБАЛЬНЫЙ генератор numpy;
    * затем делается пять стабилизирующих шагов `action_space.sample()`, а
      `reset(seed=...)` в gymnasium засеивает `env.np_random`, но не генератор
      пространства действий.

    Замер до исправления: два подряд `reset` с одним и тем же сидом давали
    наблюдения, различающиеся на 0.53, а два одинаковых прогона бейзлайна
    расходились в 8 эпизодах из 25. Спаренность эпизодов, на которой держится
    сравнение методов, при этом была фиктивной.
    """
    np.random.seed(seed)
    env.action_space.seed(seed)
    return env.reset(seed=seed, options=dict(task_id=task_id, render_goal=False))


def rollout_task(
    controller: Any,
    env: Any,
    task: TaskSpec,
    num_episodes: int = 20,
    run_seed: int = 0,
    max_steps: int = None,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Прогоняет контроллер по одной задаче и возвращает записи по эпизодам."""
    controller.reset(task)

    rng = jax.random.PRNGKey(run_seed)
    records = []

    iterator = trange(num_episodes, desc=f'{controller.name} task {task.task_id}', leave=False)
    for episode in iterator if progress else range(num_episodes):
        # reset контроллера перед каждым эпизодом: он держит состояние (текущая
        # подцель, счётчики). DP внутри кэшируется по task_id и не пересчитывается.
        controller.reset(task)

        observation, info = reset_episode(
            env, episode_seed(run_seed, task.task_id, episode), task.task_id
        )

        done, step, last_info = False, 0, {}
        while not done:
            rng, key = jax.random.split(rng)
            action, ctrl_info = controller.act(observation, seed=key)
            action = np.clip(np.asarray(action), -1.0, 1.0)

            observation, _, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated
            if max_steps is not None and step >= max_steps:
                done = True
            last_info = ctrl_info

        records.append(
            {
                'method': controller.name,
                'task_id': task.task_id,
                'episode': episode,
                'run_seed': run_seed,
                'success': float(info.get('success', 0.0)),
                'length': step,
                **{f'ctrl_{k}': v for k, v in last_info.items()},
            }
        )

    return records


def rollout_all_tasks(
    controller: Any,
    env: Any,
    tasks: List[TaskSpec],
    num_episodes: int = 20,
    run_seed: int = 0,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Прогоняет контроллер по всем задачам среды."""
    records = []
    for task in tasks:
        records.extend(
            rollout_task(
                controller,
                env,
                task,
                num_episodes=num_episodes,
                run_seed=run_seed,
                progress=progress,
            )
        )
    return records
