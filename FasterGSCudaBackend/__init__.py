from pathlib import Path

import Framework

extension_dir = Path(__file__).parent
__extension_name__ = extension_dir.name
__install_command__ = [
    'pip', 'install',
    str(extension_dir),
    '--no-build-isolation',  # to build the extension using the current environment instead of creating a new one
]

try:
    from .FasterGSCudaBackend.torch_bindings.rasterization import diff_rasterize, rasterize, update_pruning_scores, RasterizerSettings
    from .FasterGSCudaBackend.torch_bindings.adam import FusedAdam, LazyAdam
    from .FasterGSCudaBackend.torch_bindings.filter3d import update_3d_filter
    from .FasterGSCudaBackend.torch_bindings.densification import relocation_adjustment, add_noise
    from .FasterGSCudaBackend.torch_bindings.fused_loss import fused_l1_dssim
    __all__ = ['diff_rasterize', 'rasterize', 'update_pruning_scores', 'RasterizerSettings', 'FusedAdam', 'LazyAdam', 'update_3d_filter', 'relocation_adjustment', 'add_noise', 'fused_l1_dssim']
except ImportError as e:
    raise Framework.ExtensionError(name=__extension_name__, install_command=__install_command__)
