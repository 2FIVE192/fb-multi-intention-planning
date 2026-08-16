"""Подключение upstream-репозитория `switching-successor-measures` к путям импорта.

Upstream лежит в `third_party/` и используется как есть: мы не переписываем ни
определения сетей, ни логику загрузки чекпоинтов. Это принципиально — иначе
нельзя утверждать, что наш бейзлайн совпадает с авторским.

Импортировать этот модуль нужно ДО любого `import agents...` / `import utils...`.
Здесь же выбирается backend рендера MuJoCo — по той же причине «до первого
импорта».
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPSTREAM_DIR = os.path.join(_REPO_ROOT, 'third_party', 'switching-successor-measures')


def ensure_headless_rendering():
    """На Linux без дисплея выбирает EGL как backend MuJoCo.

    OGBench создаёт `mujoco.Renderer` прямо в `MazeEnv.__init__`, поэтому без
    графического контекста среда не поднимается вообще — даже когда рендер не
    нужен и мы везде передаём `render_goal=False`. На сервере или в Colab это
    выглядит как `mujoco.FatalError: an OpenGL platform library has not been
    loaded`, причём падение происходит уже после загрузки данных и чекпоинта.

    EGL — вариант, рекомендованный README самого OGBench. Значение ставится
    только если пользователь не задал своё: `MUJOCO_GL=osmesa` (программный
    рендер) остаётся рабочей альтернативой там, где EGL недоступен.
    """
    if sys.platform != 'linux':
        return  # На Windows и macOS контекст есть штатно.
    if os.environ.get('MUJOCO_GL') or os.environ.get('DISPLAY'):
        return  # Выбор пользователя или живой дисплей не трогаем.
    os.environ['MUJOCO_GL'] = 'egl'


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


ensure_headless_rendering()
ensure_upstream_on_path()
