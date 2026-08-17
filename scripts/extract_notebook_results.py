"""Извлекает результаты GPU-прогона из выполненного ноутбука в csv.

Зачем это нужно. Прогон на Colab отработал, но сырые `results/raw/*.csv` остались
в рантайме и пропали вместе с ним, когда кончилась квота GPU. Единственный
уцелевший след — вывод ячеек в `notebooks/colab_reproduce_runned.ipynb`.

Вписывать такие числа в отчёт руками нельзя: их нельзя ни проверить, ни
пересчитать. Поэтому они извлекаются отсюда — из самого артефакта прогона, и
результат можно сверить с ноутбуком построчно.

Запуск:
    python scripts/extract_notebook_results.py
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

#: Строка запуска: из неё берём тег прогона.
RUN_RE = re.compile(r'run_eval\.py.*?--tag (\S+)')
#: Строка сводки: метод, успех, доверительный интервал, число сидов.
SUMMARY_RE = re.compile(
    r'^(baseline|graph|flat)\s+([\d.]+)\s+\[\s*([-\d.nan]+),\s*([-\d.nan]+)\]\s+(\d+)', re.M
)
#: Строка парного сравнения.
PAIRED_RE = re.compile(
    r'парное сравнение graph - baseline: ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\], пар (\d+)'
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--notebooks', default='notebooks/*_runned.ipynb',
                   help='glob по выполненным ноутбукам')
    p.add_argument('--output', default='results/colab_gpu_summary.csv')
    return p.parse_args()


def cell_outputs(notebook_path: str):
    """Текст вывода каждой code-ячейки."""
    with open(notebook_path, encoding='utf-8') as f:
        notebook = json.load(f)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            yield ''.join(''.join(o.get('text', '')) for o in cell.get('outputs', []))


def parse_runs(text: str):
    """Разбирает вывод одной ячейки на отдельные прогоны.

    Ячейка может содержать несколько запусков подряд, поэтому текст режется по
    строкам запуска, и каждый кусок разбирается отдельно.
    """
    starts = [(m.start(), m.group(1)) for m in RUN_RE.finditer(text)]
    for i, (position, tag) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        block = text[position:end]

        paired = PAIRED_RE.search(block)
        for method, success, low, high, seeds in SUMMARY_RE.findall(block):
            yield {
                'tag': tag,
                'method': method,
                'success': float(success),
                'ci_low': float(low),
                'ci_high': float(high),
                'num_seeds': int(seeds),
                'paired_delta': float(paired.group(1)) if paired and method == 'graph' else None,
                'paired_ci_low': float(paired.group(2)) if paired and method == 'graph' else None,
                'paired_ci_high': float(paired.group(3)) if paired and method == 'graph' else None,
                'num_pairs': int(paired.group(4)) if paired and method == 'graph' else None,
            }


def main():
    args = parse_args()
    notebooks = sorted(glob.glob(args.notebooks))
    if not notebooks:
        raise SystemExit(f'не нашлось ноутбуков по маске {args.notebooks}')

    rows = []
    for notebook in notebooks:
        found = [row for text in cell_outputs(notebook) for row in parse_runs(text)]
        print(f'{os.path.basename(notebook)}: прогонов {len({r["tag"] for r in found})}')
        rows.extend(found)
    if not rows:
        raise SystemExit('в выводе ноутбука не нашлось ни одной сводки прогона')

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f'извлечено прогонов: {df.tag.nunique()}, строк: {len(df)}\n')
    print(df.to_string(index=False))
    print(f'\nсохранено: {args.output}')
    print('источник: вывод ячеек ноутбука; сырые csv остались в рантайме Colab')


if __name__ == '__main__':
    main()
