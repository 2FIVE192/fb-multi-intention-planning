"""Оракул поверх замороженных FB-представлений.

Здесь собрана вся математика, которой пользуются планировщик и анализ. Ничего не
обучается — только форвард-проходы замороженных сетей.

Центральная величина
--------------------
Для состояния `s` и состояния-подцели `w`:

    z_w = normalize(B(w))                        латентная интенция «дойти до w»
    M(s -> w) = F(s, z_w)^T B(w)                 successor measure состояния w
    p(s -> w) = M(s -> w) / M(w -> w) ≈ E[γ^H]                          (*)

Тождество (*) — множитель `Msww / Mwww` из Теоремы 1 статьи (он же стоит в
`high_actor_loss` авторского кода). Нормировка сокращает плотность данных и
масштаб `B`, оставляя безразмерную величину в (0, 1]. Отсюда главное:
`c(s -> w) := -log p(s -> w)` АДДИТИВНА вдоль цепочки подцелей, и выбор
последовательности интенций сводится к кратчайшему пути.

Почему подцель — это НАБОР состояний
------------------------------------
Наблюдение antmaze 29-мерно: это конкретная поза муравья, а не точка на плоскости.
Successor measure одиночной позы оказался почти неинформативным — замеренная
корреляция `c(s -> w)` с истинным геодезическим расстоянием составила 0.18.
Агрегация по набору состояний одной локации,

    M(s -> W) := max_{w in W} M(s -> w),

поднимает её до 0.70: шум позы гасится, остаётся сигнал о месте. Поэтому узел
здесь — всегда набор, а не состояние. Цель задачи ложится в ту же абстракцию:
это один узел, членами которого являются состояния датасета с reward = 1 (ровно
тот набор, по которому авторский `infer_latent` усредняет `B`).

Порядок операций важен: сначала агрегируем набор, потом нормируем. Обратный
порядок (нормировать каждого члена, потом агрегировать) даёт 0.22 вместо 0.58 —
нормировочные константы у членов разные, и агрегация их перемешивает.
"""

import dataclasses
from typing import Any, Optional

from . import _upstream  # noqa: F401  (побочный эффект: sys.path)

import jax
import jax.numpy as jnp
import numpy as np

# Ниже этого значения `p` считается численным нулём: переход непроходим.
MIN_REACH_PROB = 1e-8


# --------------------------------------------------------------------------- #
# Джиттед-ядра. Вынесены на уровень модуля, чтобы jax кэшировал компиляцию
# между экземплярами оракула. `network` — flax TrainState (pytree), его
# не-pytree поля (model_def, apply_fn) jax трактует как статические.
# --------------------------------------------------------------------------- #


@jax.jit
def _forward_measures(network, observations, z_intents, z_targets):
    """M_e = F(s, z_intent)^T z_target по головам ансамбля. -> (E, N)

    `z_intents` должен быть уже нормирован.
    """
    forward = network.select('forward_repr')(observations, z_intents, goal_encoded=True)
    return jnp.sum(forward * z_targets[None], axis=-1)


@jax.jit
def _backward_repr(network, observations):
    """Сырое B(s), без нормировки. -> (N, d)"""
    return network.select('backward_repr')(observations)


@jax.jit
def _pairwise_measures_per_head(network, src_observations, tgt_z, tgt_b):
    """То же, что `_pairwise_min_measures`, но БЕЗ сведения ансамбля. -> (E, S, T)

    Нужно для пессимистичного отбора: штраф за расхождение голов невозможно
    посчитать после того, как головы уже сведены в одно число.
    """
    n_src, n_tgt = src_observations.shape[0], tgt_z.shape[0]
    obs = jnp.repeat(src_observations, n_tgt, axis=0)
    z = jnp.tile(tgt_z, (n_src, 1))
    b = jnp.tile(tgt_b, (n_src, 1))
    measures = _forward_measures(network, obs, z, b)  # (E, S*T)
    return measures.reshape(-1, n_src, n_tgt)


@jax.jit
def _pairwise_min_measures(network, src_observations, tgt_z, tgt_b):
    """M(s_i -> m_j) для всех пар, с min по головам ансамбля.

    Пессимизм по ансамблю применяется здесь, до агрегации набора: чтобы переход
    считался дешёвым, в него должны «поверить» обе головы сразу.

    Args:
        src_observations: (S, obs_dim)
        tgt_z: (T, d) нормированные интенции членов наборов.
        tgt_b: (T, d) сырые B членов.

    Returns:
        (S, T)
    """
    n_src, n_tgt = src_observations.shape[0], tgt_z.shape[0]
    obs = jnp.repeat(src_observations, n_tgt, axis=0)
    z = jnp.tile(tgt_z, (n_src, 1))
    b = jnp.tile(tgt_b, (n_src, 1))
    measures = _forward_measures(network, obs, z, b)  # (E, S*T)
    return measures.min(axis=0).reshape(n_src, n_tgt)


@jax.jit
def _sample_low_action(network, observations, z, seed, temperature):
    """Действие low-level политики pi_l(a | s, z). `z` уже нормирован."""
    dist = network.select('actor')(observations, z, goal_encoded=True, temperature=temperature)
    return jnp.clip(dist.sample(seed=seed), -1.0, 1.0)


@jax.jit
def _sample_high_intent(network, observations, z_reward, seed, temperature):
    """Интенция pi_h(z_w | s, z_r) — сырая, без нормировки."""
    dist = network.select('high_actor')(
        observations, z_reward, goal_encoded=True, temperature=temperature
    )
    return dist.sample(seed=seed)


# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class TargetSet:
    """Набор узлов-подцелей; каждый узел — набор состояний одной локации.

    Attributes:
        observations: (K, W, obs_dim) состояния-члены.
        b: (K, W, d) сырое B членов.
        z: (K, W, d) нормированные интенции членов.
        normalizer: (K,) знаменатель в (*), оценка max_s M(s -> узел).
    """

    observations: np.ndarray
    b: np.ndarray
    z: np.ndarray
    normalizer: np.ndarray
    #: (E, K) знаменатель по каждой голове ансамбля отдельно. Нужен только
    #: пессимистичному отбору: там стоимость считается поголовно, и делить на
    #: общий, уже сведённый знаменатель было бы неверно.
    normalizer_heads: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.observations.shape[0]

    @property
    def num_members(self) -> int:
        return self.observations.shape[1]

    @property
    def representatives(self) -> np.ndarray:
        """(K, obs_dim) — по одному представителю на узел: центральный член окна.

        Используется там, где нужен именно источник: в матрице стоимостей между
        узлами и в диагностике. Центр окна ближе всего к «месту» узла.
        """
        return self.observations[:, self.num_members // 2]

    @property
    def node_z(self) -> np.ndarray:
        """(K, d) — одна интенция на узел: нормированное среднее B по членам.

        Именно так авторский `infer_latent` строит латент задачи из целевой
        области, поэтому low-level политика получает привычный ей вход.
        Нормировку делает `FBOracle.low_action`, здесь достаточно среднего.
        """
        return self.b.mean(axis=1)

    @property
    def flat_z(self) -> np.ndarray:
        return self.z.reshape(-1, self.z.shape[-1])

    @property
    def flat_b(self) -> np.ndarray:
        return self.b.reshape(-1, self.b.shape[-1])

    def subset(self, idxs: np.ndarray) -> 'TargetSet':
        return TargetSet(
            observations=self.observations[idxs],
            b=self.b[idxs],
            z=self.z[idxs],
            normalizer=self.normalizer[idxs],
            normalizer_heads=(None if self.normalizer_heads is None
                              else self.normalizer_heads[:, idxs]),
        )

    def with_normalizer(self, normalizer: np.ndarray) -> 'TargetSet':
        return dataclasses.replace(self, normalizer=normalizer)


class FBOracle:
    """Тонкая обёртка над замороженным агентом: всё, что нужно планировщику.

    Args:
        agent: загруженный `FBpiSwitchAgent` со всеми четырьмя модулями.
        config: конфиг агента (нужен `latent_dim`).
        ensemble_reduce: 'min' — пессимизм по головам forward-репрезентации,
            'mean' — абляция.
        disagreement_penalty: коэффициент штрафа за расхождение голов ансамбля.
            0 — поведение по умолчанию (стоимость = худшая из голов).

            Зачем это нужно. Замер (REPORT 5.10) показал смещение отбора: узел,
            который выбирает argmin, оценён на 27 шагов ОПТИМИСТИЧНЕЕ истины,
            тогда как типичный узел — на 39 шагов пессимистичнее. argmin
            выбирает не ближайший узел, а тот, чья ошибка оказалась самой
            выгодной. Сведение ансамбля минимумом меры уже даёт пессимизм, но
            его явно мало: две головы часто ошибаются согласованно.

            Штраф добавляет к стоимости разброс между головами, поэтому узлы, в
            достижимости которых головы не сходятся, перестают выигрывать отбор
            за счёт удачного выброса.
        max_pairs: верхняя граница на число пар в одном форвард-проходе;
            ограничивает пиковую память.
    """

    def __init__(
        self,
        agent: Any,
        config: Optional[dict] = None,
        ensemble_reduce: str = 'min',
        disagreement_penalty: float = 0.0,
        max_pairs: int = 1 << 16,
        tgt_chunk: int = 128,
    ):
        if ensemble_reduce not in ('min', 'mean'):
            raise ValueError(f'ensemble_reduce должен быть min|mean, получено {ensemble_reduce!r}')

        self.agent = agent
        self.network = agent.network
        self.config = config if config is not None else dict(agent.config)
        self.latent_dim = int(self.config['latent_dim'])
        # Число голов ансамбля берём из самих весов, а не из конфига: так
        # исключена рассинхронизация с чекпоинтом.
        self.num_heads = _count_ensemble_heads(agent)
        self.discount = float(self.config['discount'])
        self.ensemble_reduce = ensemble_reduce
        self.disagreement_penalty = float(disagreement_penalty)
        self.tgt_chunk = int(tgt_chunk)
        self.src_chunk = max(1, int(max_pairs) // self.tgt_chunk)

    # ------------------------------------------------------------------ #
    # Базовые операции
    # ------------------------------------------------------------------ #

    def normalize_z(self, z: np.ndarray) -> np.ndarray:
        """Нормировка латента до длины sqrt(d) — как в авторском коде."""
        z = np.asarray(z, dtype=np.float32)
        norm = np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8
        return z / norm * np.sqrt(self.latent_dim)

    def backward(self, observations: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Сырое B(s). Форма входа сохраняется, добавляется ось латента."""
        observations = np.asarray(observations, dtype=np.float32)
        shape = observations.shape[:-1]
        flat = observations.reshape(-1, observations.shape[-1])

        out = [
            np.asarray(_backward_repr(self.network, flat[i : i + batch_size]))
            for i in range(0, len(flat), batch_size)
        ]
        return np.concatenate(out, axis=0).reshape(*shape, self.latent_dim)

    def make_targets(
        self, member_observations: np.ndarray, reference_observations: Optional[np.ndarray] = None,
        per_head: bool = False
    ) -> TargetSet:
        """Готовит узлы из наборов состояний.

        Args:
            member_observations: (K, W, obs_dim). Форма (K, obs_dim)
                трактуется как K узлов по одному члену; чтобы собрать ОДИН узел
                из набора состояний, передавайте явную форму (1, W, obs_dim).
            reference_observations: (R, obs_dim) опорные состояния для оценки
                знаменателя max_s M(s -> узел). Без них знаменателем станет
                самомера узла, и отношение начнёт превышать единицу — см.
                предупреждение `_warn_if_edges_degenerate` в graph.py.
        """
        members = np.asarray(member_observations, dtype=np.float32)
        if members.ndim == 2:
            members = members[:, None, :]

        b = self.backward(members)
        z = self.normalize_z(b)

        targets = TargetSet(
            observations=members, b=b, z=z, normalizer=np.ones(len(members), dtype=np.float32)
        )
        references = (
            targets.representatives if reference_observations is None else reference_observations
        )
        targets = targets.with_normalizer(self.estimate_normalizer(targets, references))

        # Поголовный знаменатель нужен только пессимистичному отбору, а стоит
        # ещё один проход по опорным состояниям — считаем по требованию.
        if per_head or self.disagreement_penalty > 0.0:
            targets = dataclasses.replace(
                targets,
                normalizer_heads=self.estimate_normalizer(targets, references, per_head=True),
            )
        return targets

    def estimate_normalizer(
        self, targets: TargetSet, reference_observations: np.ndarray,
        per_head: bool = False
    ) -> np.ndarray:
        """max_s M(s -> узел) по опорным состояниям. -> (K,), либо (E, K)

        Теория говорит, что мера максимальна, когда стартуешь в самой цели, то
        есть max_s M(s -> W) — состоятельная оценка знаменателя в (*), и притом
        куда устойчивее одиночной самомеры.
        """
        references = np.asarray(reference_observations, dtype=np.float32)
        shape = (self.num_heads, len(targets)) if per_head else (len(targets),)
        best = np.full(shape, -np.inf, dtype=np.float32)

        for source_chunk in self._iter_chunks(references, self.src_chunk):
            measures = self._raw_measures(source_chunk.data, targets, per_head)
            measures = measures[..., : source_chunk.size, :]
            np.maximum(best, measures.max(axis=-2), out=best)

        # Самомеру тоже учитываем: узел достижим как минимум из самого себя.
        self_measure = self._raw_measures(targets.representatives, targets, per_head)
        diagonal = np.diagonal(self_measure, axis1=-2, axis2=-1)
        return np.maximum(best, diagonal)

    # ------------------------------------------------------------------ #
    # Достижимость и стоимость
    # ------------------------------------------------------------------ #

    def reach_prob(self, src_observations: np.ndarray, targets: TargetSet) -> np.ndarray:
        """p(s_i -> узел_j) — дисконтированная вероятность достижения. -> (S, K)"""
        measures = self._raw_measures(np.asarray(src_observations, dtype=np.float32), targets)

        denom = targets.normalizer[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.where(denom > 0.0, measures / denom, 0.0)
        return np.clip(p, MIN_REACH_PROB, 1.0)

    def cost(self, src_observations: np.ndarray, targets: TargetSet) -> np.ndarray:
        """c(s -> узел) = -log p >= 0. Аддитивна вдоль цепочки подцелей. -> (S, K)

        При `disagreement_penalty > 0` возвращается пессимистичная оценка
        `c_худшая + lambda * (c_худшая - c_лучшая)`: к худшей из голов
        добавляется их расхождение. См. пояснение в `__init__`.
        """
        if self.disagreement_penalty <= 0.0:
            return -np.log(self.reach_prob(src_observations, targets))

        per_head = self._cost_per_head(src_observations, targets)  # (E, S, K)
        worst, best = per_head.max(axis=0), per_head.min(axis=0)
        return worst + self.disagreement_penalty * (worst - best)

    def _cost_per_head(self, src_observations: np.ndarray, targets: TargetSet) -> np.ndarray:
        """Стоимость по каждой голове ансамбля отдельно. -> (E, S, K)"""
        if targets.normalizer_heads is None:
            raise ValueError(
                'Для пессимистичного отбора нужен поголовный знаменатель. '
                'Соберите цели через make_targets(..., per_head=True).'
            )
        measures = self._raw_measures(
            np.asarray(src_observations, dtype=np.float32), targets, per_head=True
        )
        denom = targets.normalizer_heads[:, None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            p = np.where(denom > 0.0, measures / denom, 0.0)
        return -np.log(np.clip(p, MIN_REACH_PROB, 1.0))

    def cost_from_state(self, observation: np.ndarray, targets: TargetSet) -> np.ndarray:
        """c(s -> узел) из ОДНОГО состояния ко всем узлам. -> (K,)"""
        return self.cost(np.asarray(observation, dtype=np.float32).reshape(1, -1), targets)[0]

    def steps_from_cost(self, cost: np.ndarray) -> np.ndarray:
        """Перевод стоимости в шаги среды: c = -log p ~= -H*log(gamma).

        Только для интерпретации графиков; в методе не используется.
        """
        return cost / (-np.log(self.discount))

    # ------------------------------------------------------------------ #
    # Ценность для произвольной reward-функции
    # ------------------------------------------------------------------ #

    def reward_value(self, observations: np.ndarray, z_reward: np.ndarray) -> np.ndarray:
        """V(s) = F(s, z_r)^T z_r — ценность состояния под задачей z_r. -> (N,)"""
        observations = np.asarray(observations, dtype=np.float32)
        z_reward = np.asarray(z_reward, dtype=np.float32).reshape(-1)
        z_norm = self.normalize_z(z_reward)

        out = []
        for i in range(0, len(observations), self.src_chunk):
            batch = observations[i : i + self.src_chunk]
            per_head = np.asarray(
                _forward_measures(
                    self.network,
                    batch,
                    np.broadcast_to(z_norm, (len(batch), self.latent_dim)).copy(),
                    np.broadcast_to(z_reward, (len(batch), self.latent_dim)).copy(),
                )
            )
            out.append(per_head.min(axis=0) if self.ensemble_reduce == 'min' else per_head.mean(axis=0))
        return np.concatenate(out)

    # ------------------------------------------------------------------ #
    # Политики
    # ------------------------------------------------------------------ #

    def low_action(self, observation: np.ndarray, z: np.ndarray, seed, temperature: float = 0.0):
        """Действие pi_l(a | s, z). `z` нормируется и приводится к форме здесь же."""
        observation = np.atleast_2d(np.asarray(observation, dtype=np.float32))
        z = self._match_batch(z, len(observation))
        return np.asarray(
            _sample_low_action(self.network, observation, z, seed, temperature)
        )

    def high_intent(
        self, observation: np.ndarray, z_reward: np.ndarray, seed, temperature: float = 0.0
    ) -> np.ndarray:
        """Интенция pi_h(z_w | s, z_r) бейзлайна, уже нормированная."""
        observation = np.atleast_2d(np.asarray(observation, dtype=np.float32))
        z_reward = self._match_batch(z_reward, len(observation))
        raw = np.asarray(
            _sample_high_intent(self.network, observation, z_reward, seed, temperature)
        )
        return self.normalize_z(raw)

    def _match_batch(self, z: np.ndarray, batch_size: int) -> np.ndarray:
        """Нормирует латент и разворачивает его до (batch_size, d).

        Латент приходит и одномерным (одна интенция узла), и уже батчем;
        сети же требуют, чтобы форма совпадала с наблюдениями.
        """
        z = self.normalize_z(np.asarray(z, dtype=np.float32).reshape(-1, self.latent_dim))
        if len(z) == batch_size:
            return z
        return np.ascontiguousarray(np.broadcast_to(z, (batch_size, self.latent_dim)))

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #

    def _raw_measures(self, src_observations: np.ndarray, targets: TargetSet,
                      per_head: bool = False) -> np.ndarray:
        """M(s -> узел) = агрегация по членам набора. -> (S, K), либо (E, S, K)

        Агрегация — максимум по членам (то же самое, что минимум стоимости):
        попасть в любого члена набора значит оказаться в нужной локации.
        Порядок «сначала агрегировать, потом нормировать» принципиален, см.
        docstring модуля.
        """
        src_observations = np.atleast_2d(np.asarray(src_observations, dtype=np.float32))
        n_src, n_nodes = len(src_observations), len(targets)
        width = targets.num_members

        # Столько узлов помещается в один блок фиксированной ширины.
        nodes_per_chunk = max(1, self.tgt_chunk // width)
        flat_z, flat_b = targets.flat_z, targets.flat_b

        n_heads = self.num_heads if per_head else 1
        out = np.empty((n_heads, n_src, n_nodes) if per_head else (n_src, n_nodes),
                       dtype=np.float32)
        for node_chunk in self._iter_chunks(np.arange(n_nodes), nodes_per_chunk):
            j0, j1 = node_chunk.start, node_chunk.start + node_chunk.size
            columns = slice(j0 * width, j1 * width)
            z_block = _pad_to(flat_z[columns], nodes_per_chunk * width)
            b_block = _pad_to(flat_b[columns], nodes_per_chunk * width)

            for source_chunk in self._iter_chunks(src_observations, self.src_chunk, pad=False):
                kernel = _pairwise_measures_per_head if per_head else _pairwise_min_measures
                measures = np.asarray(
                    kernel(self.network, source_chunk.data, z_block, b_block)
                )[..., : node_chunk.size * width]

                # Агрегация по членам делается ПОГОЛОВНО: смешивать головы до
                # неё значило бы сравнивать разные оценки одного и того же узла.
                aggregated = measures.reshape(
                    *measures.shape[:-1], node_chunk.size, width
                ).max(axis=-1)

                i0 = source_chunk.start
                out[..., i0 : i0 + source_chunk.size, j0:j1] = aggregated

        return out

    def _iter_chunks(self, array: np.ndarray, chunk: int, pad: bool = True):
        """Разбивает массив на блоки.

        `pad=True` (цели) выравнивает блок до фиксированного размера, чтобы jax
        не перекомпилировал ядро на каждом хвостовом блоке другой формы.

        `pad=False` (источники) выравнивания НЕ делает, и это важно: онлайн-цикл
        планировщика спрашивает стоимость ровно из одного состояния, а паддинг
        раздувал бы такой запрос до полного блока в сотни строк. Разных размеров
        блока источников на практике два-три, так что лишние компиляции jax
        обходятся куда дешевле постоянного перерасхода.
        """
        for start in range(0, len(array), chunk):
            size = min(chunk, len(array) - start)
            data = array[start : start + size]
            yield _Chunk(start=start, size=size, data=_pad_to(data, chunk) if pad else data)


@dataclasses.dataclass(frozen=True)
class _Chunk:
    start: int
    size: int
    data: np.ndarray


def _pad_to(arr: np.ndarray, size: int) -> np.ndarray:
    """Дополняет массив повторением последней строки до длины `size`."""
    if len(arr) == size:
        return arr
    pad = np.repeat(arr[-1:], size - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)


def _count_ensemble_heads(agent) -> int:
    """Число голов forward-репрезентации по форме её весов.

    У ансамблевых модулей flax добавляет ведущую ось к каждому тензору, поэтому
    достаточно посмотреть на любой из них: у обычного слоя ядро двумерно, у
    ансамбля из E голов — трёхмерно с E впереди.
    """
    params = agent.network.params['modules_forward_repr']

    def first_kernel(tree):
        for key, value in tree.items():
            if isinstance(value, dict):
                found = first_kernel(value)
                if found is not None:
                    return found
            elif key == 'kernel':
                return np.asarray(value)
        return None

    kernel = first_kernel(params)
    return int(kernel.shape[0]) if kernel is not None and kernel.ndim == 3 else 1
