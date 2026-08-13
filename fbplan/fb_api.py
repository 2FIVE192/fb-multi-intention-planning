"""Оракул поверх замороженных FB-представлений.

Здесь собрана вся математика, которой пользуются планировщик и анализ. Ничего не
обучается — только форвард-проходы замороженных сетей.

Центральная величина
--------------------
Для состояния `s` и состояния-подцели `w` определим

    z_w  = normalize(B(w))                       — латентная интенция «дойти до w»
    M(s -> w) = F(s, z_w)^T B(w)                 — successor measure состояния w
                                                   при следовании политике pi_{z_w}

    p(s -> w) = M(s -> w) / M(w -> w)   ≈   E[ gamma^{H(s -> w)} ]                 (*)

где H — время достижения w. Тождество (*) — это ровно множитель `Msww / Mwww` из
Теоремы 1 статьи (он же стоит в `high_actor_loss` авторского кода). Его ценность
в том, что нормировка на `M(w -> w)` сокращает и плотность данных `rho(w)`, и
масштаб `B(w)`: остаётся безразмерная величина в (0, 1].

Из (*) следует главное свойство, на котором построен метод: по строго
марковскому свойству дисконт-множители перемножаются вдоль цепочки подцелей,

    p(s -> w2)  ~=  p(s -> w1) * p(w1 -> w2),

то есть  c(s -> w) := -log p(s -> w)  АДДИТИВНА. Поиск лучшей последовательности
интенций превращается в задачу кратчайшего пути с неотрицательными весами.
"""

import dataclasses
from typing import Any, Optional

from . import _upstream  # noqa: F401  (побочный эффект: sys.path)

import jax
import jax.numpy as jnp
import numpy as np

# Ниже этого значения `p` считается численным нулём: ребро непроходимо.
MIN_REACH_PROB = 1e-8


# --------------------------------------------------------------------------- #
# Джиттед-ядра. Вынесены на уровень модуля, чтобы jax кэшировал компиляцию
# между экземплярами оракула. `network` — flax TrainState (pytree), его
# не-pytree поля (model_def, apply_fn) jax трактует как статические.
# --------------------------------------------------------------------------- #


@jax.jit
def _forward_measures(network, observations, z_intents, z_targets):
    """M_e = F(s, z_intent)^T z_target для каждой головы ансамбля.

    `z_intents` должен быть УЖЕ нормирован (см. `FBOracle.normalize_z`).

    Returns:
        (E, N) — по одному значению на голову ансамбля.
    """
    forward = network.select('forward_repr')(observations, z_intents, goal_encoded=True)
    return jnp.sum(forward * z_targets[None], axis=-1)


@jax.jit
def _backward_repr(network, observations):
    """Сырое B(s), без нормировки. -> (N, d)"""
    return network.select('backward_repr')(observations)


@jax.jit
def _pairwise_measures(network, src_observations, tgt_z, tgt_b):
    """M_e(s_i -> w_j) для всех пар из блока.

    Args:
        src_observations: (S, obs_dim)
        tgt_z: (T, d) — нормированные интенции целей.
        tgt_b: (T, d) — сырые B(w) целей.

    Returns:
        (E, S, T)
    """
    n_src, n_tgt = src_observations.shape[0], tgt_z.shape[0]
    # repeat по источникам + tile по целям даёт раскладку [s0w0, s0w1, ..., s1w0, ...]
    obs = jnp.repeat(src_observations, n_tgt, axis=0)
    z = jnp.tile(tgt_z, (n_src, 1))
    b = jnp.tile(tgt_b, (n_src, 1))
    measures = _forward_measures(network, obs, z, b)
    return measures.reshape(measures.shape[0], n_src, n_tgt)


@jax.jit
def _sample_low_action(network, observations, z, seed, temperature):
    """Действие low-level политики pi_l(a | s, z). `z` уже нормирован."""
    dist = network.select('actor')(observations, z, goal_encoded=True, temperature=temperature)
    return jnp.clip(dist.sample(seed=seed), -1.0, 1.0)


@jax.jit
def _sample_high_intent(network, observations, z_reward, seed, temperature):
    """Интенция high-level контроллера pi_h(z_w | s, z_r) — СЫРАЯ, без нормировки."""
    dist = network.select('high_actor')(
        observations, z_reward, goal_encoded=True, temperature=temperature
    )
    return dist.sample(seed=seed)


# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class TargetSet:
    """Предпосчитанные величины для набора состояний-подцелей.

    Держим их вместе, потому что `self_measure` (знаменатель в (*)) считается
    один раз и переиспользуется во всех запросах.

    Attributes:
        observations: (K, obs_dim) — сами состояния.
        b: (K, d) — сырое B(w).
        z: (K, d) — нормированная интенция normalize(B(w)).
        self_measure: (E, K) — M(w -> w) по головам ансамбля.
    """

    observations: np.ndarray
    b: np.ndarray
    z: np.ndarray
    self_measure: np.ndarray

    def __len__(self) -> int:
        return self.observations.shape[0]

    def subset(self, idxs: np.ndarray) -> 'TargetSet':
        return TargetSet(
            observations=self.observations[idxs],
            b=self.b[idxs],
            z=self.z[idxs],
            self_measure=self.self_measure[:, idxs],
        )


class FBOracle:
    """Тонкая обёртка над замороженным агентом: всё, что нужно планировщику.

    Args:
        agent: загруженный `FBpiSwitchAgent` со всеми четырьмя модулями.
        config: конфиг агента (нужен `latent_dim`).
        ensemble_reduce: как сводить головы ансамбля forward-репрезентации.
            'min' — пессимизм: галлюцинированный проход сквозь стену должен
            «поверить» обеим головам сразу, что заметно реже. 'mean' — абляция.
        max_pairs: верхняя граница на число пар в одном форвард-проходе.
            Ограничивает пиковую память; на форму jit-компиляции не влияет,
            так как блоки паддятся до фиксированного размера.
    """

    def __init__(
        self,
        agent: Any,
        config: Optional[dict] = None,
        ensemble_reduce: str = 'min',
        max_pairs: int = 1 << 16,
        tgt_chunk: int = 64,
    ):
        if ensemble_reduce not in ('min', 'mean'):
            raise ValueError(f'ensemble_reduce должен быть min|mean, получено {ensemble_reduce!r}')

        self.agent = agent
        self.network = agent.network
        self.config = config if config is not None else dict(agent.config)
        self.latent_dim = int(self.config['latent_dim'])
        self.discount = float(self.config['discount'])
        self.ensemble_reduce = ensemble_reduce
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
        """Сырое B(s) для набора состояний. -> (N, d)"""
        observations = np.asarray(observations, dtype=np.float32)
        out = [
            np.asarray(_backward_repr(self.network, observations[i : i + batch_size]))
            for i in range(0, len(observations), batch_size)
        ]
        return np.concatenate(out, axis=0)

    def make_targets(self, observations: np.ndarray) -> TargetSet:
        """Готовит `TargetSet`: B, нормированные интенции и M(w -> w)."""
        observations = np.asarray(observations, dtype=np.float32)
        b = self.backward(observations)
        z = self.normalize_z(b)

        self_measure = []
        for i in range(0, len(observations), self.src_chunk):
            sl = slice(i, i + self.src_chunk)
            self_measure.append(
                np.asarray(_forward_measures(self.network, observations[sl], z[sl], b[sl]))
            )
        return TargetSet(
            observations=observations,
            b=b,
            z=z,
            self_measure=np.concatenate(self_measure, axis=1),
        )

    # ------------------------------------------------------------------ #
    # Достижимость и стоимость
    # ------------------------------------------------------------------ #

    def reach_prob(self, src_observations: np.ndarray, targets: TargetSet) -> np.ndarray:
        """p(s_i -> w_j) — дисконтированная вероятность достижения, формула (*).

        Отношение считается ПОГОЛОВНО (M_e / M_e), и только потом головы
        сводятся. Так пессимизм остаётся корректным: смешивать числитель одной
        головы со знаменателем другой смысла не имеет.

        Returns:
            (S, K) в диапазоне [MIN_REACH_PROB, 1].
        """
        src_observations = np.asarray(src_observations, dtype=np.float32)
        n_src, n_tgt = len(src_observations), len(targets)

        # M(w -> w) <= 0 означает, что FB не считает w достижимым даже из него
        # самого; такая цель непригодна — зануляем её вклад ниже.
        denom = targets.self_measure  # (E, K)
        denom_ok = denom > 0.0

        out = np.empty((n_src, n_tgt), dtype=np.float32)
        for j0, j1, tgt_pad, n_valid_t in self._chunks(targets, self.tgt_chunk):
            for i0 in range(0, n_src, self.src_chunk):
                i1 = min(i0 + self.src_chunk, n_src)
                src = _pad_to(src_observations[i0:i1], self.src_chunk)

                measures = np.asarray(  # (E, src_chunk, tgt_chunk)
                    _pairwise_measures(self.network, src, tgt_pad.z, tgt_pad.b)
                )[:, : i1 - i0, :n_valid_t]

                d = denom[:, j0:j1][:, None, :]  # (E, 1, T)
                with np.errstate(divide='ignore', invalid='ignore'):
                    p_per_head = np.where(denom_ok[:, None, j0:j1], measures / d, 0.0)

                p = p_per_head.min(axis=0) if self.ensemble_reduce == 'min' else p_per_head.mean(axis=0)
                out[i0:i1, j0:j1] = p

        return np.clip(out, MIN_REACH_PROB, 1.0)

    def cost(self, src_observations: np.ndarray, targets: TargetSet) -> np.ndarray:
        """c(s -> w) = -log p(s -> w) >= 0. Аддитивна вдоль цепочки подцелей.

        Returns:
            (S, K); недостижимые пары дают -log(MIN_REACH_PROB) ~ 18.4.
        """
        return -np.log(self.reach_prob(src_observations, targets))

    def cost_from_state(self, observation: np.ndarray, targets: TargetSet) -> np.ndarray:
        """c(s -> w_j) из ОДНОГО состояния ко всем целям, за один форвард.

        Отдельный путь нужен для онлайн-цикла: общий `reach_prob` паддит
        источники до `src_chunk`, и на батче из одного состояния это давало бы
        ~1000-кратный перерасход. Здесь форма (K, ...) постоянна на всём
        прогоне, так что jax компилирует ядро ровно один раз.

        Returns:
            (K,)
        """
        obs = np.broadcast_to(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            (len(targets), self._obs_dim(observation)),
        )
        measures = np.asarray(  # (E, K)
            _forward_measures(self.network, np.ascontiguousarray(obs), targets.z, targets.b)
        )

        denom = targets.self_measure
        with np.errstate(divide='ignore', invalid='ignore'):
            p_per_head = np.where(denom > 0.0, measures / denom, 0.0)

        p = p_per_head.min(axis=0) if self.ensemble_reduce == 'min' else p_per_head.mean(axis=0)
        return -np.log(np.clip(p, MIN_REACH_PROB, 1.0))

    @staticmethod
    def _obs_dim(observation: np.ndarray) -> int:
        return int(np.asarray(observation).reshape(-1).shape[0])

    def steps_from_cost(self, cost: np.ndarray) -> np.ndarray:
        """Перевод стоимости в «шаги среды»: c = -log p ~= -H*log(gamma).

        Нужен только для интерпретации графиков в отчёте, в самом методе не
        используется.
        """
        return cost / (-np.log(self.discount))

    # ------------------------------------------------------------------ #
    # Ценность для произвольной reward-функции
    # ------------------------------------------------------------------ #

    def reward_value(self, observations: np.ndarray, z_reward: np.ndarray) -> np.ndarray:
        """V(s) = F(s, z_r)^T z_r — ценность состояния под задачей z_r.

        Это `Vrstar` из авторского лосса: ценность следования собственной
        оптимальной для z_r политике.
        """
        observations = np.asarray(observations, dtype=np.float32)
        z_reward = np.asarray(z_reward, dtype=np.float32)
        z_norm = self.normalize_z(z_reward)

        z_batch = np.broadcast_to(z_norm, (len(observations), self.latent_dim))
        r_batch = np.broadcast_to(z_reward, (len(observations), self.latent_dim))

        out = []
        for i in range(0, len(observations), self.src_chunk):
            sl = slice(i, i + self.src_chunk)
            out.append(
                np.asarray(
                    _forward_measures(
                        self.network,
                        observations[sl],
                        np.ascontiguousarray(z_batch[sl]),
                        np.ascontiguousarray(r_batch[sl]),
                    )
                )
            )
        per_head = np.concatenate(out, axis=1)  # (E, N)
        return per_head.min(axis=0) if self.ensemble_reduce == 'min' else per_head.mean(axis=0)

    # ------------------------------------------------------------------ #
    # Политики
    # ------------------------------------------------------------------ #

    def low_action(self, observation: np.ndarray, z: np.ndarray, seed, temperature: float = 0.0):
        """Действие pi_l(a | s, z). `z` нормируется здесь же."""
        z = self.normalize_z(z)
        return np.asarray(
            _sample_low_action(self.network, observation, z, seed, temperature)
        )

    def high_intent(
        self, observation: np.ndarray, z_reward: np.ndarray, seed, temperature: float = 0.0
    ) -> np.ndarray:
        """Интенция pi_h(z_w | s, z_r) бейзлайна, уже нормированная."""
        z_reward = self.normalize_z(z_reward)
        raw = np.asarray(
            _sample_high_intent(self.network, observation, z_reward, seed, temperature)
        )
        return self.normalize_z(raw)

    # ------------------------------------------------------------------ #

    def _chunks(self, targets: TargetSet, chunk: int):
        """Разбивает цели на блоки фиксированного размера (с паддингом).

        Фиксированный размер важен: иначе jax перекомпилирует ядро на каждом
        «хвостовом» блоке другой формы.
        """
        n = len(targets)
        for j0 in range(0, n, chunk):
            j1 = min(j0 + chunk, n)
            sub = targets.subset(np.arange(j0, j1))
            padded = TargetSet(
                observations=_pad_to(sub.observations, chunk),
                b=_pad_to(sub.b, chunk),
                z=_pad_to(sub.z, chunk),
                self_measure=sub.self_measure,
            )
            yield j0, j1, padded, j1 - j0


def _pad_to(arr: np.ndarray, size: int) -> np.ndarray:
    """Дополняет массив повторением последней строки до длины `size`."""
    if len(arr) == size:
        return arr
    pad = np.repeat(arr[-1:], size - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)
