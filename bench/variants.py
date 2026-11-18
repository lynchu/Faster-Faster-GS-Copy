"""Creates and builds isolated Faster-GS variants.

Each variant gets its own NeRFICG clone and a git worktree of the Faster-GS repository on its own branch,
so variants never share compiled CUDA code. The CUDA backend is built in-place and found through PYTHONPATH.

usage:
  python variants.py create <name> [--base <commit>]
  python variants.py build <name>
"""
import argparse
import subprocess
import sys

from common import BASE_COMMIT, DATASET, FGS_REPO, NERFICG_SRC, PYTHON, backend_dir, git, method_dir, run_env, variant_dir


def create(name: str, base: str) -> None:
    vdir = variant_dir(name)
    if vdir.exists():
        sys.exit(f'variant {name} already exists at {vdir}')
    vdir.mkdir(parents=True)
    subprocess.run(['git', 'clone', '-q', '--local', str(NERFICG_SRC), str(vdir / 'nerficg')], check=True)
    dataset_link = vdir / 'nerficg' / 'dataset'
    if dataset_link.exists():
        subprocess.run(['rm', '-rf', str(dataset_link)], check=True)
    dataset_link.symlink_to(DATASET)
    branch = f'opt/{name}' if name != 'baseline' else 'bench/baseline'
    git(['worktree', 'add', '-b', branch, str(method_dir(name)), base], FGS_REPO)
    print(f'created variant {name} on branch {branch} at {vdir}')


def build(name: str) -> None:
    cmd = [str(PYTHON), 'setup.py', 'build_ext', '--inplace']
    result = subprocess.run(cmd, cwd=backend_dir(name), env=run_env(name))
    if result.returncode != 0:
        sys.exit(f'build failed for variant {name}')
    print(f'built CUDA backend for variant {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['create', 'build'])
    parser.add_argument('name')
    parser.add_argument('--base', default=BASE_COMMIT)
    args = parser.parse_args()
    if args.action == 'create':
        create(args.name, args.base)
    else:
        build(args.name)
