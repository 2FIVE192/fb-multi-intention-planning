"""Планирование последовательностей интенций поверх замороженных FB-представлений.

Публичный интерфейс пакета — то, чем пользуются скрипты в `scripts/`.

Импорты здесь ленивые (PEP 562). Причина практическая: `fbplan.stats` — это
чистые numpy и pandas, и считать по csv сводку хочется без запуска JAX. При
жадных импортах `from fbplan.stats import paired_comparison` тянул за собой
`fb_api`, а с ним jax и весь upstream — в ноутбуке это и падало, и вдобавок
занимало видеопамять в процессе, которому она не нужна.
"""

import importlib
from typing import Any

#: Имя объекта -> модуль, в котором он определён.
_EXPORTS = {
    'FBOracle': 'fb_api',
    'FlatController': 'controllers',
    'GraphPlanner': 'planner',
    'PlannerConfig': 'planner',
    'SingleIntentionController': 'controllers',
    'SubgoalGraph': 'graph',
    'TargetSet': 'fb_api',
    'TaskSpec': 'task_setup',
    'build_subgoal_graph': 'graph',
    'cost_from_steps': 'graph',
    'load_agent': 'checkpoint',
    'make_example_batch': 'checkpoint',
    'prepare_task': 'task_setup',
    'read_env_name': 'checkpoint',
    'rollout_all_tasks': 'rollout',
    'rollout_task': 'rollout',
    'select_nodes': 'nodes',
    'solve_goal': 'graph',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Подгружает модуль только когда из него что-то действительно берут."""
    if name not in _EXPORTS:
        raise AttributeError(f'модуль {__name__!r} не экспортирует {name!r}')
    module = importlib.import_module(f'.{_EXPORTS[name]}', __name__)
    value = getattr(module, name)
    globals()[name] = value  # повторные обращения идут уже напрямую
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
