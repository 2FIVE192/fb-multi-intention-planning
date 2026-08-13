"""Скачивание датасетов OGBench с обходом бага на Windows.

`ogbench.download_datasets` пишет во временный файл через `tqdm.wrapattr`, не
закрывает его и сразу вызывает `os.rename`. На Windows это падает с
`PermissionError: [WinError 32]` — файл ещё занят. На Linux/Colab проблемы нет,
но и там этот скрипт работает: он просто скачивает то же самое корректно.

Пример:
    python scripts/download_datasets.py --datasets antmaze-medium-navigate-v0
"""

import argparse
import os
import sys
import urllib.request

import tqdm

DATASET_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'
DEFAULT_DATASET_DIR = os.path.expanduser('~/.ogbench/data')


def download_file(url: str, destination: str) -> None:
    """Скачивает файл во временный путь и переименовывает его после закрытия."""
    if os.path.exists(destination):
        print(f'[skip] уже есть: {destination}')
        return

    tmp_path = f'{destination}.tmp'
    if os.path.exists(tmp_path) and _is_complete(tmp_path, url):
        # Хвост прерванной загрузки: файл целый, упало только переименование.
        print(f'[resume] переименовываю готовый {tmp_path}')
        os.replace(tmp_path, destination)
        return

    print(f'[get] {url}')
    response = urllib.request.urlopen(url)
    total = getattr(response, 'length', None)

    # Ключевое отличие от ogbench: явный `with open(...)`, поэтому дескриптор
    # закрыт до момента переименования.
    with open(tmp_path, 'wb') as raw:
        with tqdm.tqdm.wrapattr(raw, 'write', miniters=1, total=total,
                                desc=os.path.basename(destination)) as wrapped:
            for chunk in response:
                wrapped.write(chunk)

    os.replace(tmp_path, destination)
    print(f'[ok] {destination}')


def _is_complete(tmp_path: str, url: str) -> bool:
    """Проверяет, что временный файл скачан целиком (по Content-Length)."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD')) as response:
            expected = int(response.headers.get('Content-Length', -1))
    except Exception:
        return False
    return expected > 0 and os.path.getsize(tmp_path) == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', default='antmaze-medium-navigate-v0',
                        help='имена датасетов через запятую (без -val)')
    parser.add_argument('--dataset_dir', default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    os.makedirs(args.dataset_dir, exist_ok=True)

    for name in (d.strip() for d in args.datasets.split(',') if d.strip()):
        for file_name in (f'{name}.npz', f'{name}-val.npz'):
            download_file(
                f'{DATASET_URL}/{file_name}',
                os.path.join(args.dataset_dir, file_name),
            )

    print('\nготово:', args.dataset_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
