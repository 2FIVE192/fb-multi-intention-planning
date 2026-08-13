"""Онлайн-контроллер: исполнение последовательности интенций.

Отличие от бейзлайна в одной строчке: бейзлайн на каждом шаге спрашивает
`pi_h(z_w | s, z_goal)` — то есть жадно выбирает ОДНУ следующую интенцию,
глядя сразу на далёкую цель. Здесь мы вместо этого спрашиваем у графа
кратчайший путь до цели и отдаём low-level политике ближайшую подцель на этом
пути, перевыбирая её по мере продвижения.

Критерий переключения подцели — FB-овский, а не геометрический: подцель
считается достигнутой, когда `c(s -> w)` опускается ниже порога. Это прямой
аналог hitting-time-переключения из статьи и не требует привилегированных
координат.
"""

import dataclasses
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .fb_api import FBOracle
from . import graph as graph_mod
from .graph import GoalPlan, SubgoalGraph
from .task_setup import TaskSpec


@dataclasses.dataclass
class PlannerConfig:
    """Гиперпараметры онлайн-контроллера.

    Пороги задаются в «шагах среды» — так их можно осмысленно обсуждать; внутрь
    они переводятся в стоимости через c = -H * log(gamma).

    Attributes:
        replan_every: раз в сколько шагов пересматривать выбор подцели.
        reach_steps: подцель считается достигнутой, если до неё осталось
            меньше стольких шагов по оценке FB.
        max_subgoal_steps: сколько шагов максимум держаться за одну подцель,
            прежде чем принудительно перевыбрать (защита от застревания).
        switch_margin_steps: гистерезис. Менять подцель на новую, только если
            она лучше текущей более чем на столько шагов. Без него выбор
            дребезжит между двумя соседними узлами.
        execution: 'low' — интенция подцели идёт прямо в pi_l;
            'high' — сначала через замороженный pi_h (тогда наш планировщик
            отвечает за дальний горизонт, а бейзлайн-контроллер за локальный).
        temperature: температура сэмплирования политик (0 — детерминированно).
    """

    replan_every: int = 10
    reach_steps: float = 20.0
    max_subgoal_steps: int = 120
    switch_margin_steps: float = 5.0
    execution: str = 'low'
    temperature: float = 0.0

    def __post_init__(self):
        if self.execution not in ('low', 'high'):
            raise ValueError(f"execution должен быть 'low' или 'high', получено {self.execution!r}")


class GraphPlanner:
    """Планировщик последовательности интенций поверх графа подцелей."""

    name = 'graph'

    def __init__(self, oracle: FBOracle, graph: SubgoalGraph, config: PlannerConfig):
        self.oracle = oracle
        self.graph = graph
        self.config = config

        self.reach_cost = graph_mod.cost_from_steps(oracle, config.reach_steps)
        self.switch_margin = graph_mod.cost_from_steps(oracle, config.switch_margin_steps)

        self._plan: Optional[GoalPlan] = None
        self._cached_task_id: Optional[int] = None
        self._reset_episode_state()

    # ------------------------------------------------------------------ #

    def reset(self, task: TaskSpec) -> None:
        """Пересчитывает DP под новую задачу.

        DP зависит только от задачи, не от эпизода, поэтому результат кэшируется
        между эпизодами одного task_id — Дейкстра по 1000 узлам дешёвая, но
        `solve_goal` ещё и считает K оценок FB до цели.
        """
        if self._cached_task_id != task.task_id:
            self._plan = graph_mod.solve_goal(
                self.oracle, self.graph, task.goal_observations, z_goal=task.z_reward
            )
            self._cached_task_id = task.task_id
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._step = 0
        self._subgoal_age = 0
        self._current_node = graph_mod.DIRECT_TO_GOAL  # по умолчанию целимся прямо в цель
        self._current_z = None
        self._num_switches = 0
        self._num_fallbacks = 0

    # ------------------------------------------------------------------ #

    def act(self, observation: np.ndarray, seed) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Возвращает действие и диагностику шага."""
        if self._plan is None:
            raise RuntimeError('Перед act() нужно вызвать reset(goal_observation).')

        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)

        if self._should_replan():
            self._select_subgoal(observation)

        z_subgoal = self._current_z
        info = {
            'node': self._current_node,
            'subgoal_age': self._subgoal_age,
            'num_switches': self._num_switches,
            'num_fallbacks': self._num_fallbacks,
        }

        if self.config.execution == 'high':
            # Подцель играет роль «задачи» для замороженного pi_h.
            z_exec = self.oracle.high_intent(
                observation, z_subgoal, seed=seed, temperature=self.config.temperature
            )
        else:
            z_exec = z_subgoal.reshape(1, -1)

        action = self.oracle.low_action(
            observation, z_exec, seed=seed, temperature=self.config.temperature
        )

        self._step += 1
        self._subgoal_age += 1
        return action.reshape(-1), info

    # ------------------------------------------------------------------ #

    def _should_replan(self) -> bool:
        """Пересматривать выбор раз в `replan_every` шагов, но всегда на первом."""
        return self._step == 0 or self._step % self.config.replan_every == 0

    def _select_subgoal(self, observation: np.ndarray) -> None:
        """Выбирает следующую подцель: argmin [ c(s -> w) + d*(w -> цель) ].

        Порог доверия `max_edge_cost` применяется ОДИНАКОВО ко всем вариантам —
        и к переходам в узлы графа, и к прямому броску в цель. Это не деталь, а
        суть метода. Если разрешить прямому броску обходить порог, планировщик
        почти всегда будет выбирать именно его: на длинной дистанции FB
        систематически ЗАНИЖАЕТ стоимость (значение упирается в пол разрешения,
        и остаётся гладкий артефакт сети), так что галлюцинированное «до цели
        рукой подать» окажется дешевле честного составного маршрута. Мы
        сравнивали бы честную композицию с испорченным числом.

        Прямой бросок остаётся вариантом только когда цель реально близко, плюс
        как аварийный откат, если граф не предлагает вообще ничего.
        """
        plan = self._plan
        cost_to_nodes = self.oracle.cost_from_state(observation, self.graph.targets)  # (K,)
        # Цель — один узел, поэтому просто первый (и единственный) элемент.
        cost_direct = float(self.oracle.cost_from_state(observation, plan.goal_targets)[0])

        total = cost_to_nodes + plan.cost_to_goal
        admissible = (
            (cost_to_nodes <= self.graph.max_edge_cost)
            & np.isfinite(total)
            # Узел, до которого уже дошли, — не подцель. Без этого фильтра
            # планировщик залипает: стоя на подцели w, он видит total[w] равным
            # total[следующего узла] (это и есть оптимальность пути) и может
            # снова выбрать w, никуда не двигаясь.
            & (cost_to_nodes > self.reach_cost)
        )

        # Прямой бросок — «план из нуля подцелей», но только если цель попадает
        # в зону доверия. Иначе его стоимость нам просто неизвестна.
        direct_trusted = cost_direct <= self.graph.max_edge_cost

        best_node = graph_mod.DIRECT_TO_GOAL if direct_trusted else None
        best_cost = cost_direct if direct_trusted else np.inf

        if admissible.any():
            node = self._argmin_with_tiebreak(total, admissible, plan.cost_to_goal)
            if total[node] < best_cost:
                best_node, best_cost = node, float(total[node])

        if best_node is None:
            # Ни доверенного прямого броска, ни маршрута по графу: делаем то же,
            # что бейзлайн, и честно считаем это откатом.
            best_node, best_cost = graph_mod.DIRECT_TO_GOAL, cost_direct
            self._num_fallbacks += 1

        self._commit(best_node, best_cost, cost_to_nodes, cost_direct, direct_trusted)

    def _argmin_with_tiebreak(
        self, total: np.ndarray, admissible: np.ndarray, cost_to_goal: np.ndarray
    ) -> int:
        """argmin по полной стоимости, ничьи разрешаются в пользу узла ближе к цели.

        Ничьи здесь не редкость, а норма: на оптимальном пути все узлы имеют
        одинаковую полную стоимость. Без явного правила выбор среди них
        определялся бы порядком в массиве, и планировщик предпочитал бы
        ближний к себе конец пути — то есть топтался на месте.
        """
        masked_total = np.where(admissible, total, np.inf)
        best = float(masked_total.min())

        # Ничья — расхождение меньше одного шага среды.
        tie_eps = graph_mod.cost_from_steps(self.oracle, 1.0)
        tied = admissible & (total <= best + tie_eps)
        return int(np.argmin(np.where(tied, cost_to_goal, np.inf)))

    def _commit(
        self,
        best_node: int,
        best_cost: float,
        cost_to_nodes: np.ndarray,
        cost_direct: float,
        direct_trusted: bool,
    ) -> None:
        """Меняет текущую подцель с учётом гистерезиса и таймаута."""
        if self._current_z is None:
            self._set_subgoal(best_node)
            return

        current_cost = self._current_cost(cost_to_nodes, cost_direct)
        current_total = self._current_total(cost_to_nodes, cost_direct, direct_trusted)

        reached = current_cost <= self.reach_cost
        stale = self._subgoal_age >= self.config.max_subgoal_steps
        clearly_better = best_cost + self.switch_margin < current_total

        if reached or stale or clearly_better:
            if best_node != self._current_node:
                self._num_switches += 1
            self._set_subgoal(best_node)

    def _set_subgoal(self, node: int) -> None:
        self._current_node = node
        # Целясь «прямо в цель», используем ровно ту же латентную задачу z_r,
        # которой кондиционируется бейзлайн, — иначе сравнение было бы нечестным.
        self._current_z = (
            self._plan.z_goal
            if node == graph_mod.DIRECT_TO_GOAL
            else self.graph.targets.node_z[node]
        )
        self._subgoal_age = 0

    def _current_cost(self, cost_to_nodes: np.ndarray, cost_direct: float) -> float:
        """Стоимость от текущего состояния до ТЕКУЩЕЙ подцели."""
        if self._current_node == graph_mod.DIRECT_TO_GOAL:
            return cost_direct
        return float(cost_to_nodes[self._current_node])

    def _current_total(
        self, cost_to_nodes: np.ndarray, cost_direct: float, direct_trusted: bool
    ) -> float:
        """Полная стоимость до цели через ТЕКУЩУЮ подцель.

        Если мы сейчас идём напрямую в цель, а цель вне зоны доверия (то есть
        мы в аварийном откате), стоимость считается бесконечной: любой маршрут
        по графу должен немедленно её перебить.
        """
        if self._current_node == graph_mod.DIRECT_TO_GOAL:
            return cost_direct if direct_trusted else np.inf
        return float(cost_to_nodes[self._current_node] + self._plan.cost_to_goal[self._current_node])
