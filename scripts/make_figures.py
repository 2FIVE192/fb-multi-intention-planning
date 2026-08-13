"""Графики для отчёта из сырых csv в results/raw.

Ни одно число в отчёте не вписывается руками: всё, что попадает в REPORT.md,
порождается этим скриптом из файлов, сохранённых прогонами.

Запуск (после run_eval.py / analysis_composability.py / run_ablations.py):
    python scripts/make_figures.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fbplan.stats import bootstrap_ci, paired_comparison, summarize_by_task

METHOD_LABELS = {'baseline': 'бейзлайн (одна интенция)', 'graph': 'планировщик (наш)',
                 'flat': 'без иерархии'}
METHOD_COLORS = {'baseline': 'tab:orange', 'graph': 'tab:blue', 'flat': '0.6'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_dir', default='results/raw')
    p.add_argument('--output_dir', default='results/figures')
    p.add_argument('--main_tag', default='main')
    p.add_argument('--composability_tag', default='composability')
    p.add_argument('--ablations_tag', default='ablations')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    made = []
    made += _figure_main(args)
    made += _figure_composability(args)
    made += _figure_ablations(args)

    if not made:
        print('Не найдено ни одного csv — сначала запустите прогоны (см. README).')
    for path in made:
        print(f'[fig] {path}')


# --------------------------------------------------------------------------- #


def _load(args, tag, suffix):
    path = os.path.join(args.raw_dir, f'{tag}_{suffix}.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


def _figure_main(args):
    """E3: успех по задачам и парная разность с бейзлайном."""
    df = _load(args, args.main_tag, 'episodes')
    if df is None:
        return []

    by_task = summarize_by_task(df)
    methods = [m for m in ('baseline', 'graph', 'flat') if m in set(df['method'])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    # Слева: успех по задачам.
    task_ids = sorted(by_task['task_id'].unique())
    width = 0.8 / len(methods)
    for k, method in enumerate(methods):
        sub = by_task[by_task['method'] == method].set_index('task_id').reindex(task_ids)
        offset = (k - (len(methods) - 1) / 2) * width
        axes[0].bar(np.arange(len(task_ids)) + offset, sub['success'], width,
                    yerr=sub['std_over_seeds'], capsize=3,
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method))
    axes[0].set_xticks(np.arange(len(task_ids)), [f'задача {t}' for t in task_ids])
    axes[0].set_ylabel('success rate')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Успех по задачам (усы — std по сидам)', fontsize=10)
    axes[0].legend(fontsize=8)

    # Справа: парная разность по задачам с бутстрэп-CI.
    if 'graph' in methods and 'baseline' in methods:
        deltas, los, his = [], [], []
        for task_id in task_ids:
            sub = df[df['task_id'] == task_id]
            cmp = paired_comparison(sub, 'graph', 'baseline')
            deltas.append(cmp['delta'])
            los.append(cmp['delta'] - cmp['ci_low'])
            his.append(cmp['ci_high'] - cmp['delta'])

        colors = ['tab:blue' if d >= 0 else 'crimson' for d in deltas]
        axes[1].bar(np.arange(len(task_ids)), deltas, 0.6,
                    yerr=[los, his], capsize=4, color=colors)
        axes[1].axhline(0, color='black', linewidth=0.8)
        axes[1].set_xticks(np.arange(len(task_ids)), [f'задача {t}' for t in task_ids])
        axes[1].set_ylabel('успех: планировщик − бейзлайн')
        axes[1].set_title('Парная разность, 95% бутстрэп-CI по сидам', fontsize=10)

    path = os.path.join(args.output_dir, 'main_results.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _figure_composability(args):
    """E1: обе оценки против истинной геодезической дальности."""
    df = _load(args, args.composability_tag, 'pairs')
    if df is None:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    finite = df[np.isfinite(df['path_cost'])]

    # Слева: облака точек.
    axes[0].scatter(df['true_geodesic'], df['direct_steps'], s=3, alpha=0.2,
                    color='tab:orange', label='прямая оценка (как у бейзлайна)')
    axes[0].scatter(finite['true_geodesic'], finite['path_steps'], s=3, alpha=0.2,
                    color='tab:blue', label='композиция по графу (наш метод)')

    # Идеал: оценка растёт пропорционально истинной дальности. Наклон берём из
    # ближней зоны, где FB заведомо надёжен, и продлеваем.
    near = df[df['true_geodesic'] <= 8]
    if len(near) > 10 and near['true_geodesic'].sum() > 0:
        slope = float(np.sum(near['direct_steps'] * near['true_geodesic'])
                      / np.sum(near['true_geodesic'] ** 2))
        xs = np.linspace(0, df['true_geodesic'].max(), 50)
        axes[0].plot(xs, slope * xs, 'k--', linewidth=1,
                     label='экстраполяция ближней зоны')

    axes[0].set_xlabel('истинное геодезическое расстояние, мировых единиц')
    axes[0].set_ylabel('оценка расстояния, шагов среды')
    axes[0].set_title('Оценка против истины', fontsize=10)
    axes[0].legend(fontsize=8, markerscale=3)

    # Справа: среднее по бинам — видно плато прямой оценки.
    bins = np.arange(0, df['true_geodesic'].max() + 4, 4)
    centers = (bins[:-1] + bins[1:]) / 2
    for column, color, label in (
        ('direct_steps', 'tab:orange', 'прямая оценка'),
        ('path_steps', 'tab:blue', 'композиция по графу'),
    ):
        source = df if column == 'direct_steps' else finite
        grouped = source.groupby(pd.cut(source['true_geodesic'], bins),
                                 observed=True)[column]
        axes[1].errorbar(centers[: len(grouped.mean())], grouped.mean(),
                         yerr=grouped.std(), color=color, marker='o',
                         capsize=3, label=label)
    axes[1].set_xlabel('истинное геодезическое расстояние, мировых единиц')
    axes[1].set_ylabel('оценка расстояния, шагов среды')
    axes[1].set_title('Среднее по бинам: где теряется контраст', fontsize=10)
    axes[1].legend(fontsize=8)

    path = os.path.join(args.output_dir, 'composability.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _figure_ablations(args):
    """E4: чувствительность к гиперпараметрам."""
    df = _load(args, args.ablations_tag, 'episodes')
    if df is None:
        return []

    per_seed = df.groupby(['variant', 'run_seed'])['success'].mean().reset_index()
    stats = []
    for variant, group in per_seed.groupby('variant'):
        lo, hi = bootstrap_ci(group['success'].values)
        stats.append({'variant': variant, 'success': group['success'].mean(),
                      'lo': lo, 'hi': hi})
    stats = pd.DataFrame(stats).sort_values('success')

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(stats) + 2), constrained_layout=True)
    ypos = np.arange(len(stats))
    colors = ['tab:blue' if v == 'default' else 'tab:orange' if v == 'baseline' else '0.55'
              for v in stats['variant']]
    ax.barh(ypos, stats['success'],
            xerr=[stats['success'] - stats['lo'], stats['hi'] - stats['success']],
            capsize=3, color=colors)
    ax.set_yticks(ypos, stats['variant'])
    ax.set_xlabel('success rate (95% бутстрэп-CI по сидам)')
    ax.set_title('Абляции: отклонение по одной оси от дефолта', fontsize=10)

    path = os.path.join(args.output_dir, 'ablations.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


if __name__ == '__main__':
    main()
