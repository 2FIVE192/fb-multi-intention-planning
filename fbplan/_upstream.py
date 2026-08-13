"""Подключение upstream-репозитория `switching-successor-measures` к путям импорта.

Upstream лежит в `third_party/` и используется как есть: мы не переписываем ни
определения сетей, ни логику загрузки чекпоинтов. Это принципиально — иначе
нельзя утверждать, что наш бейзлайн совпадает с авторским.

Импортировать этот модуль нужно ДО любого `import agents...` / `import utils...`.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPSTREAM_DIR = os.path.join(_REPO_ROOT, 'third_party', 'switching-successor-measures')


def ensure_upstream_on_path():
    """Добавляет upstream в `sys.path`. Идемпотентна."""
    if not os.path.isdir(UPSTREAM_DIR):
        raise RuntimeError(
            f'Не найден upstream-репозиторий в {UPSTREAM_DIR}.\n'
            'Выполните: git submodule update --init --recursive\n'
            'или: git clone https://github.com/stestoKTH/switching-successor-measures.git '
            f'{UPSTREAM_DIR}'
        )
    if UPSTREAM_DIR not in sys.path:
        # В конец, а не в начало: свои модули должны иметь приоритет.
        sys.path.append(UPSTREAM_DIR)


ensure_upstream_on_path()
