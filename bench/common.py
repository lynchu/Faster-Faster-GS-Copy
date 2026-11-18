"""Shared paths and helpers for the Faster-GS optimization benchmark."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Workspace layout (override with FFGS_ROOT): nerficg/, faster-gaussian-splatting/ (this repository), variants/, dataset/
ROOT = Path(os.environ.get('FFGS_ROOT', Path(__file__).resolve().parents[2]))
NERFICG_SRC = ROOT / 'nerficg'
FGS_REPO = ROOT / 'faster-gaussian-splatting'
VARIANTS = ROOT / 'variants'
DATASET = ROOT / 'dataset'
BENCH = Path(__file__).resolve().parent
REGISTRY = BENCH / 'results' / 'registry.jsonl'
CONDA_ENV = Path(os.environ.get('FFGS_CONDA_ENV', sys.prefix))  # Python environment with PyTorch and NeRFICG requirements
PYTHON = CONDA_ENV / 'bin' / 'python'
BASE_COMMIT = '4f8e186'
GPU_INDEX = int(os.environ.get('FFGS_GPU', 0))  # nvidia-smi index of the benchmark GPU

OUTDOOR = ['bicycle', 'flowers', 'garden', 'stump', 'treehill']
INDOOR = ['room', 'counter', 'kitchen', 'bonsai']
MIPNERF360 = OUTDOOR + INDOOR
TANDT = ['truck', 'train']
DEEPBLENDING = ['drjohnson', 'playroom']
ALL13 = MIPNERF360 + TANDT + DEEPBLENDING
DEV_SCENES = ['garden', 'bonsai']


def variant_dir(name: str) -> Path:
    return VARIANTS / name


def method_dir(name: str) -> Path:
    return variant_dir(name) / 'nerficg' / 'src' / 'Methods' / 'FasterGS'


def backend_dir(name: str) -> Path:
    return method_dir(name) / 'FasterGSCudaBackend'


def run_env(name: str, gpu: int = GPU_INDEX) -> dict:
    env = dict(os.environ)
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # match nvidia-smi numbering (CUDA's default order is fastest-first)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONPATH'] = str(backend_dir(name))  # in-place built extension of this variant
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    env['PATH'] = f'{CONDA_ENV / "bin"}:{cuda_home}/bin:' + env.get('PATH', '')
    env['CUDA_HOME'] = cuda_home
    env['TORCH_CUDA_ARCH_LIST'] = os.environ.get('FGS_ARCH', '12.0')  # e.g. FGS_ARCH='8.6;12.0' for A6000 probe runs
    # optional host compiler override, e.g. FFGS_CC=/usr/bin/gcc-11 FFGS_CXX=/usr/bin/g++-11
    if 'FFGS_CC' in os.environ:
        env['CC'] = os.environ['FFGS_CC']
    if 'FFGS_CXX' in os.environ:
        env['CXX'] = os.environ['FFGS_CXX']
    env.setdefault('MAX_JOBS', '32')
    return env


def git(args: list[str], cwd: Path) -> str:
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def variant_commit(name: str) -> tuple[str, bool]:
    """Returns (commit hash, is_dirty) of the variant's Faster-GS worktree."""
    d = method_dir(name)
    commit = git(['rev-parse', 'HEAD'], d)
    dirty = bool(git(['status', '--porcelain', '--untracked-files=no'], d))
    return commit, dirty


def nerficg_commit(name: str) -> tuple[str, bool]:
    """Returns (commit hash, is_dirty) of the variant's NeRFICG clone (excluding the Faster-GS worktree, configs, outputs)."""
    d = variant_dir(name) / 'nerficg'
    commit = git(['rev-parse', 'HEAD'], d)
    status = git(['status', '--porcelain', '--untracked-files=no'], d)
    dirty = any(not line.split(None, 1)[-1].startswith(('src/Methods/FasterGS', 'configs', 'output', 'dataset')) for line in status.splitlines() if line)
    return commit, dirty


def extension_hash(name: str) -> str:
    """Hash of every compiled extension a run of this variant loads (variant backend + shared NeRFICG extensions)."""
    files = sorted(backend_dir(name).rglob('_C*.so'))
    site = CONDA_ENV / 'lib' / 'python3.11' / 'site-packages'
    for pattern in ['fused_ssim*/*.so', 'MortonEncoding*/*.so', 'simple_knn*/*.so', '*.so']:
        files += sorted(site.glob(pattern))
    h = hashlib.sha256()
    for f in files:
        if f.is_file() and any(k in f.name.lower() for k in ['_c', 'ssim', 'morton', 'knn']):
            h.update(f.name.encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def load_registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]


def append_registry(entry: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY, 'a') as f:
        f.write(json.dumps(entry, sort_keys=True) + '\n')
