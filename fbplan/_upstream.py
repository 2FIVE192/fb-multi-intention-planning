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


#: Установите в '1', чтобы библиотека не трогала MUJOCO_GL вообще. Нужно, когда
#: вызывающий сам подбирает backend и хочет проверить в том числе вариант
#: «не задавать переменную и оставить выбор самому mujoco».
NO_GL_DEFAULT_ENV = 'FBPLAN_NO_GL_DEFAULT'

#: Установите в '1', чтобы оставить настоящий рендерер MuJoCo. Нужно только если
#: вы собираетесь получать картинки; расчётам они не требуются.
KEEP_RENDERER_ENV = 'FBPLAN_KEEP_RENDERER'


class _NullRenderer:
    """Заглушка вместо `mujoco.Renderer`: не создаёт графический контекст.

    Зачем. `MazeEnv.__init__` в OGBench безусловно создаёт `mujoco.Renderer` и
    сразу рендерит кадр — то есть требует рабочего OpenGL даже тогда, когда
    картинки не нужны никому. На headless-машинах это стабильный источник
    падений: замеренный на Colab SIGSEGV приходил именно из
    `mujoco.MjrContext`, причём и на egl, и на osmesa, и воспроизводился только
    когда до создания среды успевал загрузиться jaxlib с CUDA. Подбор backend'а
    такую поломку не лечит — конфликтуют сами нативные библиотеки.

    Мы кадры не используем: `render_goal=False` передаётся везде, а результат
    `env.render()` нигде не читается. Поэтому дешевле убрать источник проблемы,
    чем искать работающую комбинацию драйверов.

    Возвращается чёрный кадр правильной формы — на случай, если чужой код
    всё-таки заглянет в результат.
    """

    def __init__(self, model=None, height=240, width=320, **kwargs):
        self.height, self.width = height, width

    def update_scene(self, *args, **kwargs):
        pass

    def render(self):
        import numpy as np
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
        pass


def install_null_renderer():
    """Подменяет `mujoco.Renderer` заглушкой на headless-машинах.

    Патчится атрибут модуля, а OGBench обращается к нему в момент создания
    среды (`mujoco.Renderer(...)`), поэтому порядок импортов роли не играет.
    """
    if sys.platform != 'linux':
        return  # На Windows и macOS контекст создаётся штатно.
    if os.environ.get(KEEP_RENDERER_ENV) == '1' or os.environ.get('DISPLAY'):
        return

    try:
        import mujoco
    except ImportError:
        return  # mujoco подтянется позже вместе с ogbench — тогда и не понадобится.

    if getattr(mujoco.Renderer, '__name__', '') != '_NullRenderer':
        mujoco.Renderer = _NullRenderer


def ensure_headless_rendering():
    """На Linux без дисплея выбирает EGL как backend MuJoCo.

    OGBench создаёт `mujoco.Renderer` прямо в `MazeEnv.__init__`, поэтому без
    графического контекста среда не поднимается вообще — даже когда рендер не
    нужен и мы везде передаём `render_goal=False`. На сервере это выглядит как
    `mujoco.FatalError: an OpenGL platform library has not been loaded`, причём
    падение происходит уже после загрузки данных и чекпоинта.

    EGL — вариант, рекомендованный README самого OGBench, но подходит он не
    везде: на части образов Colab он роняет процесс по SIGSEGV. Поэтому значение
    здесь — только умолчание. Оно не ставится, если пользователь задал своё,
    если есть живой дисплей или если выставлен `FBPLAN_NO_GL_DEFAULT=1`.
    """
    if sys.platform != 'linux':
        return  # На Windows и macOS контекст есть штатно.
    if os.environ.get(NO_GL_DEFAULT_ENV) == '1':
        return  # Вызывающий подбирает backend сам.
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
install_null_renderer()
