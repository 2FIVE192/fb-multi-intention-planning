"""Проверка, что все code-ячейки ноутбуков синтаксически корректны.

Написано после того, как в Colab выложенный ноутбук упал с
`SyntaxError: unterminated f-string literal`: ячейка была собрана скриптом, и
экранирование в ней поехало. Ноутбук нельзя «просто прочитать глазами» — а
запускать его ради проверки синтаксиса дорого, поэтому компилируем ячейки здесь.

Строки с магией IPython (`!pip`, `%cd`) валидным Python не являются, поэтому
прогоняются через штатный трансформер IPython, если он доступен, иначе
отбрасываются.

Запуск:
    python tests/test_notebooks.py
"""

import glob
import json
import os
import sys

NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks')


def to_python(source: str) -> str:
    """Превращает содержимое ячейки в компилируемый Python.

    Если IPython доступен, используется его штатный трансформер. Иначе строки с
    магией заменяются на `pass` с тем же отступом — именно заменяются, а не
    выбрасываются: `!команда` бывает телом цикла или `if`, и удаление строки
    оставило бы пустой блок и ложную синтаксическую ошибку.
    """
    try:
        from IPython.core.inputtransformer2 import TransformerManager
    except ImportError:
        pass
    else:
        return TransformerManager().transform_cell(source)

    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(('!', '%')):
            lines.append(' ' * (len(line) - len(stripped)) + 'pass')
        else:
            lines.append(line)
    return '\n'.join(lines)


def check_notebook(path: str) -> list:
    """Возвращает список ошибок вида (индекс ячейки, сообщение)."""
    with open(path, encoding='utf-8') as f:
        notebook = json.load(f)

    errors = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if not source.strip():
            continue
        try:
            compile(to_python(source), f'{os.path.basename(path)}:cell{index}', 'exec')
        except SyntaxError as exc:
            errors.append((index, f'{exc.msg} (строка {exc.lineno}: {(exc.text or "").strip()})'))
    return errors


def main() -> int:
    notebooks = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, '*.ipynb')))
    if not notebooks:
        print('ноутбуков не найдено')
        return 0

    failed = 0
    for path in notebooks:
        name = os.path.basename(path)
        errors = check_notebook(path)
        if errors:
            failed += 1
            print(f'  FAIL {name}')
            for index, message in errors:
                print(f'       ячейка {index}: {message}')
        else:
            print(f'  OK   {name}')

    print(f'\n{len(notebooks) - failed}/{len(notebooks)} ноутбуков без синтаксических ошибок')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
