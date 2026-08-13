"""Сборка эксперимента: среда, агент, оракул, граф, задачи.

Вынесено отдельно, потому что этим пользуются и `run_eval.py`, и скрипты
анализа, и ноутбук. Дублировать двадцать строк загрузки в каждом — верный
способ получить незаметно расходящиеся конфигурации.
"""

import dataclasses
import hashlib
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional

from . import _upstream  # noqa: F401

import numpy as np

from utils.env_utils import make_env_and_datasets

from .checkpoint import load_agent, make_example_batch
from .fb_api import FBOracle
from .graph import SubgoalGraph, build_subgoal_graph, cost_from_steps
from .nodes import select_nodes
from .task_setup import TaskSpec, prepare_task


@dataclasses.dataclass
class GraphSpec:
    """Параметры построения графа подцелей."""

    num_nodes: int = 1000
    node_method: str = 'fps'
    k_neighbors: int = 16
    max_edge_steps: float = 75.0
    ensemble_reduce: str = 'min'
    node_seed: int = 0
    candidate_pool: int = 50_000

    def cache_key(self, checkpoint_dir: str, env_name: str) -> str:
        payload = json.dumps(
            {**dataclasses.asdict(self), 'checkpoint': os.path.abspath(checkpoint_dir),
             'env': env_name},
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


class Experiment:
    """Загруженное окружение эксперимента.

    Attributes:
        env: среда OGBench (обёрнутая EpisodeMonitor).
        train_dataset / val_dataset: оффлайн-данные.
        agent: замороженный FB pi-Switch агент.
        oracle: `FBOracle` поверх агента.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        env_name: str,
        ensemble_reduce: str = 'min',
        seed: int = 0,
    ):
        print(f'[exp] среда {env_name}')
        self.env_name = env_name
        self.env, self.train_dataset, self.val_dataset = make_env_and_datasets(
            env_name, add_info=True
        )
        # Авторский main.py отключает шум в цели при оценке — повторяем.
        self.env.unwrapped._add_noise_to_goal = False

        example_batch = make_example_batch(
            self.train_dataset['observations'], self.train_dataset['actions']
        )
        print(f'[exp] чекпоинт {checkpoint_dir}')
        self.agent, self.config = load_agent(checkpoint_dir, example_batch, seed=seed)
        self.checkpoint_dir = checkpoint_dir

        self.oracle = FBOracle(self.agent, self.config, ensemble_reduce=ensemble_reduce)

        self._tasks: Optional[List[TaskSpec]] = None

    # ------------------------------------------------------------------ #

    @property
    def task_ids(self) -> List[int]:
        task_infos = getattr(self.env.unwrapped, 'task_infos', None)
        if task_infos is None:
            task_infos = self.env.task_infos
        return list(range(1, len(task_infos) + 1))

    def tasks(self, num_zero_shot_samples: int = 100_000) -> List[TaskSpec]:
        """Готовит все задачи среды (кэшируется)."""
        if self._tasks is not None:
            return self._tasks

        # Авторский протокол: латент выводится по валидационному датасету.
        dataset = self.val_dataset if self.val_dataset is not None else self.train_dataset

        self._tasks = []
        for task_id in self.task_ids:
            task = prepare_task(
                self.agent,
                self.env,
                self.env_name,
                dataset,
                task_id=task_id,
                num_zero_shot_samples=num_zero_shot_samples,
            )
            print(
                f'[exp] задача {task_id}: целевых состояний в датасете '
                f'{task.num_goal_states}, взято {len(task.goal_observations)}'
            )
            self._tasks.append(task)
        return self._tasks

    # ------------------------------------------------------------------ #

    def build_graph(self, spec: GraphSpec, cache_dir: Optional[str] = None) -> SubgoalGraph:
        """Строит (или поднимает из кэша) граф подцелей.

        Построение — это K^2 запросов к forward-репрезентации, самая дорогая
        часть пайплайна. Абляции по гиперпараметрам ПЛАНИРОВЩИКА граф не меняют,
        поэтому кэш экономит заметно.
        """
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = spec.cache_key(self.checkpoint_dir, self.env_name)
            cache_path = os.path.join(cache_dir, f'graph_{key}.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    print(f'[exp] граф из кэша: {cache_path}')
                    return pickle.load(f)

        started = time.time()
        node_idxs = select_nodes(
            self.oracle,
            self.train_dataset['observations'],
            num_nodes=spec.num_nodes,
            method=spec.node_method,
            candidate_pool=spec.candidate_pool,
            seed=spec.node_seed,
        )
        node_observations = np.asarray(
            self.train_dataset['observations'][node_idxs], dtype=np.float32
        )

        graph = build_subgoal_graph(
            self.oracle,
            node_observations,
            max_edge_cost=cost_from_steps(self.oracle, spec.max_edge_steps),
            k_neighbors=spec.k_neighbors,
            dataset_idxs=node_idxs,
        )
        print(
            f'[exp] граф: {len(graph)} узлов, {graph.num_edges} рёбер, '
            f'{time.time() - started:.1f} с'
        )

        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(graph, f)
        return graph

    # ------------------------------------------------------------------ #

    def metadata(self) -> Dict[str, Any]:
        """Метаданные прогона — уходят рядом с результатами."""
        return {
            'env_name': self.env_name,
            'checkpoint_dir': os.path.abspath(self.checkpoint_dir),
            'latent_dim': self.oracle.latent_dim,
            'discount': self.oracle.discount,
            'ensemble_reduce': self.oracle.ensemble_reduce,
            'train_size': int(self.train_dataset.size),
            'obs_dim': int(self.train_dataset['observations'].shape[-1]),
        }
