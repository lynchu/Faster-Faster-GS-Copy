"""Image quality metrics for rendered test views, computed exactly like NeRFICG (Methods/Base/Renderer.py,
compute_image_metrics): 8-bit PNGs read with torchvision, scaled to [0, 1], torchmetrics PSNR (data_range 1),
SSIM (data_range 1) and LPIPS (VGG, normalize=True so inputs are mapped to [-1, 1]), averaged over images.

Used for the external baselines (3DGS, Taming-3DGS) so that every method in Table 1 is scored by the same code.

usage: python eval_images.py <renders_dir> <gt_dir> [--gpu 0] [--out metrics.json]
"""
import argparse
import json
import os
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('renders')
    parser.add_argument('gt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    import torchmetrics
    from torchvision import io

    def load(path: Path) -> torch.Tensor:
        image = io.read_image(path=str(path), mode=io.ImageReadMode.UNCHANGED)
        image = image.float() / (65535 if image.dtype == torch.uint16 else 255)
        return image[:3]

    names = sorted(n for n in os.listdir(args.renders) if 'png' in n)
    gt_names = sorted(n for n in os.listdir(args.gt) if 'png' in n)
    if len(names) != len(gt_names) or not names:
        raise SystemExit(f'{len(names)} renders vs {len(gt_names)} ground-truth images')
    if 'FFGS_TORCH_HUB' in os.environ:  # e.g. a NeRFICG .cache that already holds the VGG weights
        torch.hub.set_dir(os.environ['FFGS_TORCH_HUB'])
    dev = 'cuda'
    metrics = {
        'psnr': torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(dev),
        'ssim': torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(dev),
        'lpips': torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(dev),
    }
    values = {k: [] for k in metrics}
    with torch.no_grad():
        for r, g in zip(names, gt_names):
            result = load(Path(args.renders) / r).to(dev)[None]
            target = load(Path(args.gt) / g).to(dev)[None]
            for k, m in metrics.items():
                values[k].append(m(result, target).item())
    res = {k: mean(v) for k, v in values.items()}
    res['n_images'] = len(names)
    print(' '.join(f'{k.upper()}:{v}' for k, v in res.items()))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + '\n')


if __name__ == '__main__':
    main()
