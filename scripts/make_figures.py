"""Два графика для отчёта: главный результат и объясняющий его механизм.

Оба строятся из уже закоммиченных данных, GPU и чекпоинт не нужны.

1. Абляция глубины плана. Единственное различие между ветками — чем оценивается
   хвост маршрута до цели: цепочкой рёбер по Дейкстре или одним запросом FB.
   Показано на двух конфигурациях сразу, чтобы видеть, что вывод не держится на
   одной точке.

2. Смещение отбора. Ошибка оценки у типичного узла против ошибки у того узла,
   который выбирает argmin. Объясняет, почему композиция проигрывает.

Запуск:
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

# Единая палитра: тёплый — композиция (то, что проверялось), холодный — её
# отсутствие, серый — бейзлайн как точка отсчёта.
COLOR_DIJKSTRA = '#c1442e'
COLOR_DIRECT = '#2c6e9b'
COLOR_BASELINE = '#8a8a8a'
COLOR_BIAS = '#c1442e'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw_dir', default='results/raw')
    p.add_argument('--gpu_summary', default='results/colab_gpu_summary.csv')
    p.add_argument('--output_dir', default='results/figures')
    return p.parse_args()


def success_of(raw_dir: str, tag: str, method: str = None) -> float:
    """Средний успех прогона из поэпизодного csv."""
    df = pd.read_csv(os.path.join(raw_dir, f'{tag}_episodes.csv'))
    if method is not None:
        df = df[df.method == method]
    return float(df.success.mean())


def figure_ablation(args) -> str:
    """Главный результат: композиция по Дейкстре проигрывает плану из одной подцели."""
    gpu = pd.read_csv(args.gpu_summary).set_index(['tag', 'method'])

    # Две конфигурации: слабые рёбра на CPU и лучшие доступные на GPU.
    groups = [
        {
            'name': 'CPU\n300 узлов, 8 членов\n1 сид, 100 эпизодов',
            'dijkstra': (success_of(args.raw_dir, 'commit_dijkstra'), None, None),
            'direct': (success_of(args.raw_dir, 'commit_direct'), None, None),
            # Бейзлайн того же сида и того же числа эпизодов, что и обе ветки.
            'baseline': success_of(args.raw_dir, 'baseline_check', method='baseline'),
        },
        {
            'name': 'GPU\n1000 узлов, 32 члена\n3 сида, 300 эпизодов',
            'dijkstra': (gpu.loc[('gpu_tail_dijkstra', 'graph'), 'success'],
                         gpu.loc[('gpu_tail_dijkstra', 'graph'), 'ci_low'],
                         gpu.loc[('gpu_tail_dijkstra', 'graph'), 'ci_high']),
            'direct': (gpu.loc[('gpu_tail_direct', 'graph'), 'success'],
                       gpu.loc[('gpu_tail_direct', 'graph'), 'ci_low'],
                       gpu.loc[('gpu_tail_direct', 'graph'), 'ci_high']),
            'baseline': float(gpu.loc[('gpu_full', 'baseline'), 'success']),
        },
    ]

    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    positions = np.arange(len(groups))
    height = 0.34

    for offset, key, color, label in (
        (+height / 2, 'dijkstra', COLOR_DIJKSTRA, 'хвост по Дейкстре (много подцелей)'),
        (-height / 2, 'direct', COLOR_DIRECT, 'хвост одним запросом (одна подцель)'),
    ):
        values = [g[key][0] for g in groups]
        # Ошибки задаются как расстояния от точки до концов интервала.
        errors = np.array([
            [v - (g[key][1] if g[key][1] is not None else v),
             (g[key][2] if g[key][2] is not None else v) - v]
            for g, v in zip(groups, values)
        ]).T

        bars = ax.barh(positions + offset, values, height=height, color=color, label=label,
                       xerr=errors, error_kw=dict(ecolor='0.25', capsize=3, lw=1.2))
        # Подпись ставится ЗА правым усом, иначе налезает на него.
        for bar, value, right in zip(bars, values, errors[1]):
            ax.text(value + right + 0.018, bar.get_y() + bar.get_height() / 2, f'{value:.3f}',
                    va='center', fontsize=10, color=color, fontweight='bold')

    for position, group in zip(positions, groups):
        ax.plot([group['baseline']] * 2, [position - 0.45, position + 0.45],
                color=COLOR_BASELINE, ls='--', lw=1.8, zorder=3)
        ax.text(group['baseline'], position + 0.47, f'бейзлайн {group["baseline"]:.3f}',
                color='0.35', fontsize=9, va='bottom', ha='center')

    ax.set_yticks(positions)
    ax.set_yticklabels([g['name'] for g in groups], fontsize=9)
    ax.set_xlabel('доля успешных эпизодов')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.62, len(groups) - 0.38)
    ax.set_title('Вклад многошаговой композиции отрицателен\n'
                 'ветки различаются ровно одним: чем оценивается хвост маршрута до цели',
                 fontsize=11)
    # Легенда под осями: внутри поля она перекрывала столбцы.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9,
              frameon=False)
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)

    path = os.path.join(args.output_dir, 'ablation_plan_depth.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_selection_bias(args) -> str:
    """Механизм: argmin выбирает узел с самой оптимистичной ошибкой."""
    df = pd.read_csv(os.path.join(args.raw_dir, 'selection_bias.csv'))

    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    x = np.arange(len(df))

    # Вертикальный разрыв между двумя кривыми и есть смещение отбора.
    ax.vlines(x, df.error_selected_node, df.error_average_node,
              color=COLOR_BIAS, alpha=0.28, lw=7, zorder=1)
    for xi, low, high, bias in zip(x, df.error_selected_node, df.error_average_node,
                                   df.selection_bias):
        # Последнюю подпись уводим влево, иначе она вылезает за поле.
        last = xi == x[-1]
        ax.text(xi + (-0.11 if last else 0.11), (low + high) / 2, f'{abs(bias):.0f} шагов',
                fontsize=9, color=COLOR_BIAS, va='center',
                ha='right' if last else 'left')

    ax.plot(x, df.error_average_node, 'o-', color=COLOR_BASELINE, lw=2, ms=7,
            label='ошибка у типичного узла', zorder=2)
    ax.plot(x, df.error_selected_node, 'o-', color=COLOR_DIRECT, lw=2, ms=7,
            label='ошибка у узла, выбранного argmin', zorder=2)
    ax.axhline(0, color='0.3', lw=1)
    # Всё, что ниже нуля, — оптимистичная ошибка: именно она и вредна.
    ax.axhspan(ax.get_ylim()[0], 0, color=COLOR_DIRECT, alpha=0.05, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(df.num_members)
    ax.set_xlabel('состояний в наборе узла')
    ax.set_ylabel('оценка минус истина, шагов среды')
    ax.set_title('Смещение отбора: argmin выбирает узел с самой оптимистичной ошибкой\n'
                 'выше нуля — FB считает узел дальше, чем он есть; ниже — ближе',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    path = os.path.join(args.output_dir, 'selection_bias.png')
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for build in (figure_ablation, figure_selection_bias):
        try:
            print(f'[figures] {build(args)}')
        except FileNotFoundError as exc:
            print(f'[figures] пропущен {build.__name__}: нет данных ({exc.filename})')


if __name__ == '__main__':
    main()
