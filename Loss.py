"""FasterGS/Loss.py"""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import fused_dssim
from Methods.FasterGS.Model import FasterGSModel


class FasterGSLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList, model: FasterGSModel) -> None:
        super().__init__()
        self.add_loss_metric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.add_loss_metric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_loss_metric('OPACITY_REGULARIZATION', model.gaussians.opacity_regularization_loss, loss_config.LAMBDA_OPACITY_REGULARIZATION)
        self.add_loss_metric('SCALE_REGULARIZATION', model.gaussians.scale_regularization_loss, loss_config.LAMBDA_SCALE_REGULARIZATION)
        if model.ppisp is None:
            self.add_loss_metric('PPISP_REGULARIZATION', lambda: 0.0, 0.0)
        else:
            self.add_loss_metric('PPISP_REGULARIZATION', model.ppisp.model.get_regularization_loss, 1.0)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.activate_logging:
            # Faster-GS+SyncFree: evaluate only the active loss terms. The generic path creates torch.tensor(0.0) for every
            # zero-weight term, which (with CUDA as default tensor type) is a synchronous host-to-device copy that blocks the
            # CPU until the rasterizer has finished. The result is bitwise identical: adding exact zeros does not change it.
            kwargs = {'L1_Color': {'input': input, 'target': target}, 'DSSIM_Color': {'input': input, 'target': target}}
            loss = None
            for item in self.loss_metrics:
                if item.weight > 0.0:
                    term = item.metric_func(**kwargs.get(item.name, {})) * item.weight
                    loss = term if loss is None else loss + term
            return loss
        return super().forward({
            'L1_Color': {'input': input, 'target': target},
            'DSSIM_Color': {'input': input, 'target': target},
            'OPACITY_REGULARIZATION': {},
            'SCALE_REGULARIZATION': {},
            'PPISP_REGULARIZATION': {},
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })
