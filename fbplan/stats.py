"""Агрегация результатов: доверительные интервалы и парные сравнения.

Успех эпизода — бернуллиевская величина, и на 20 эпизодах × 5 задач обычная
«средняя ± std по сидам» вводит в заблуждение. Поэтому:

* бутстрэп по СИДАМ (а не по эпизодам) — сиды независимы, эпизоды внутри сида
  делят граф и латент задачи;
* парное сравнение методов — эпизоды с одинаковым (сид, задача, номер) стартуют
  из одинакового состояния, так что разность успехов на паре несёт куда меньше
  дисперсии, чем разность средних.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def to_frame(records: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def success_by_seed(df: pd.DataFrame, method: str) -> pd.Series:
    """Средний успех по каждому сиду (усреднение по задачам и эпизодам)."""
    return df[df['method'] == method].groupby('run_seed')['success'].mean()


def bootstrap_ci(
    values: Sequence[float],
    num_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Перцентильный бутстрэп-интервал для среднего."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float('nan'), float('nan'))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(num_resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(df: pd.DataFrame, num_resamples: int = 10_000) -> pd.DataFrame:
    """Сводка по методам: успех, std по сидам, бутстрэп-CI."""
    rows = []
    for method in df['method'].unique():
        per_seed = success_by_seed(df, method)
        lo, hi = bootstrap_ci(per_seed.values, num_resamples=num_resamples)
        rows.append(
            {
                'method': method,
                'success': float(per_seed.mean()),
                'std_over_seeds': float(per_seed.std(ddof=1)) if len(per_seed) > 1 else float('nan'),
                'ci_low': lo,
                'ci_high': hi,
                'num_seeds': int(len(per_seed)),
                'num_episodes': int((df['method'] == method).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values('success', ascending=False).reset_index(drop=True)


def summarize_by_task(df: pd.DataFrame) -> pd.DataFrame:
    """Разбивка успеха по задачам — здесь и видно эффект длины горизонта."""
    grouped = (
        df.groupby(['method', 'task_id', 'run_seed'])['success'].mean().reset_index()
    )
    return (
        grouped.groupby(['method', 'task_id'])['success']
        .agg(['mean', 'std'])
        .reset_index()
        .rename(columns={'mean': 'success', 'std': 'std_over_seeds'})
    )


def paired_comparison(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    num_resamples: int = 10_000,
    seed: int = 0,
) -> Dict[str, float]:
    """Парное сравнение A - B по эпизодам с одинаковым стартовым состоянием.

    Возвращает среднюю разность, бутстрэп-CI по сидам и долю пар, где методы
    разошлись (полезно понимать, насколько сравнение вообще информативно).
    """
    keys = ['run_seed', 'task_id', 'episode']
    a = df[df['method'] == method_a].set_index(keys)['success']
    b = df[df['method'] == method_b].set_index(keys)['success']

    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError(f'Нет общих эпизодов между {method_a} и {method_b}')

    diff = (a.loc[common] - b.loc[common]).reset_index()
    diff.columns = list(keys) + ['delta']

    per_seed = diff.groupby('run_seed')['delta'].mean()
    lo, hi = bootstrap_ci(per_seed.values, num_resamples=num_resamples, seed=seed)

    return {
        'method_a': method_a,
        'method_b': method_b,
        'delta': float(per_seed.mean()),
        'ci_low': lo,
        'ci_high': hi,
        'num_pairs': int(len(common)),
        'frac_disagree': float((diff['delta'] != 0).mean()),
        'wins_a': int((diff['delta'] > 0).sum()),
        'wins_b': int((diff['delta'] < 0).sum()),
    }


def format_summary(summary: pd.DataFrame) -> str:
    """Человекочитаемая таблица для консоли и отчёта."""
    lines = [f'{"метод":<12} {"успех":>8}  {"95% CI":>18}  {"сидов":>6}']
    lines.append('-' * 50)
    for _, row in summary.iterrows():
        ci = f'[{row["ci_low"]:.3f}, {row["ci_high"]:.3f}]'
        lines.append(
            f'{row["method"]:<12} {row["success"]:>8.3f}  {ci:>18}  {row["num_seeds"]:>6d}'
        )
    return '\n'.join(lines)
