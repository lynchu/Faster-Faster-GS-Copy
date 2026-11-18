# Faster-Faster-GS

這是平行程式設計課程的期末專題：在不降低影像品質的前提下，讓 [Faster-GS](https://github.com/nerficg-project/faster-gaussian-splatting) 的訓練再更快。
目前的版本在程式和量測紀錄裡叫做 **Faster-GS+Combo-Full**，它在 Faster-GS 之上結合了十項優化。

## 結果

Mip-NeRF360 全部 9 個場景，NVIDIA RTX PRO 6000 Blackwell，每個方法 3 個 seed。品質指標取 seed 平均，訓練時間和 VRAM 取每個場景的 seed 中位數，表中數值都是 9 個場景的平均。加速倍數和相同 seed 的 Faster-GS 配對計算。

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 訓練時間↓ | VRAM↓ | #Gs | 加速 |
|---|---|---|---|---|---|---|---|
| Faster-GS | 27.535 | 0.8144 | 0.2544 | 231.6 s | 5.96 GiB | 2.70M | 1.00× |
| Faster-GS+Combo-Full | 27.531 | 0.8146 | 0.2543 | 106.8 s | 5.76 GiB | 2.70M | 2.17× |

## 十項優化

| 名稱 | 做了什麼 |
|---|---|
| SyncFree | 拿掉每次迭代裡 loss 記錄和相機上傳造成的 CPU–GPU 同步，CPU 可以一直往前排 kernel。 |
| FusedLoss | L1 和 D-SSIM 合成一個 autograd 節點，loss 的前向和反向從 24 個 kernel 變成 6 個。 |
| LeanLoop | 拿掉訓練迴圈裡每次迭代的 Python 額外開銷，例如模式切換、學習率查詢和 optimizer step 的包裝層。 |
| WarpSync | blend forward 在 block barrier 之前先讓 warp 重新會合，barrier 不再占用整個 SM 的所有 barrier 單元，每個 SM 可以同時放 8 個 block，原本只能放 1 個。 |
| LazyAdam-Exact | Adam 只更新目前視角看得到的高斯。被跳過的步數在高斯再次被渲染之前用封閉解補上，所以參數和一般 Adam 完全一致。 |
| HalfMoments-SR | Adam 的 m 和 v 用 bfloat16 儲存；v 使用隨機捨入，衰減才不會出現偏差。 |
| LeanShuffle | blend backward 只用 warp shuffle 傳遞 lane 之間會改變的 4 個像素值，另外 5 個常數改從 shared memory 的環狀緩衝區讀取。 |
| BwdHoist | 把每個高斯固定不變的係數移出 blend backward 的像素迴圈，迴圈結束後只乘一次。建立在 LeanShuffle 之上。 |
| WarpLazy | lazy Adam 和補更新的 kernel 改成一個 warp 處理 32 個高斯，32 個都看不到時整個 warp 直接跳過。建立在 LazyAdam-Exact 之上。 |
| FusedShAdam | SH 高階係數的 lazy Adam 更新直接在 preprocess backward 裡完成，它們的梯度不必寫回 global memory。建立在 LazyAdam-Exact 和 WarpLazy 之上。 |

LazyAdam-Exact 假設看不到的高斯除了 Adam 本身以外不會收到任何更新，所以開啟 MCMC、3D filter、PPISP，或 opacity / scale 正則化時，訓練會直接報錯。

## Repository 結構

- 方法程式碼維持上游的結構，安裝到 NeRFICG 的 `src/Methods/FasterGS`，安裝方式見下方原本的說明。
- `bench/` 是量測與 profiling 工具：
  - `variants.py`：為每個版本建立獨立的 NeRFICG clone 和 Faster-GS worktree，並編譯它的 CUDA backend；
  - `run_bench.py`、`run_suite.py`：跑完整訓練，多個版本交錯執行，每次結果記錄到 registry；支援 Mip-NeRF360、Tanks and Temples、Deep Blending；
  - `summarize.py`：把 registry 整理成結果表；
  - `eval_images.py`：用和 NeRFICG 相同的算法計算測試影像的 PSNR、SSIM、LPIPS，用來評估外部方法；
  - `prof/`：nsys 和 ncu 的輔助腳本；
  - `tests/`：各項優化的等價性與精度測試；
  - `patches/`：NeRFICG 的一個小修改，記錄訓練迴圈的實際時間。

## 重現量測

1. 依照下方原本的說明安裝 NeRFICG 和這個方法，然後套用計時修改：
   `git -C <nerficg> am <本 repo>/bench/patches/0001-bench-record-wall-clock-time-of-the-training-loop.patch`
2. 準備一個工作目錄，裡面放 `nerficg/`、`faster-gaussian-splatting/`（也就是本 repo）、`dataset/` 和一個空的 `variants/`，並把 `FFGS_ROOT` 指到這個目錄。其他可設定的環境變數：
   - `FFGS_CONDA_ENV`：Python 環境；
   - `FFGS_GPU`：nvidia-smi 的 GPU 編號；
   - `CUDA_HOME`；
   - `FGS_ARCH`：要編譯的 CUDA 架構，預設 `12.0`；
   - `FFGS_CC`、`FFGS_CXX`：主機端編譯器。
3. 建立、編譯並量測一個版本：
   ```shell
   python bench/variants.py create combofull --base main
   python bench/variants.py build combofull
   python bench/run_suite.py baseline,combofull --scenes all --seeds 0,1,2
   python bench/summarize.py --no-dashboard
   ```

## 授權

和上游 Faster-GS 一樣採用 Apache-2.0。本專題修改和新增的檔案列在 [NOTICE](./NOTICE)。

---

# 原本的 Faster-GS README

# Faster Gaussian Splatting
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=PyTorch&logoColor=white)&nbsp;
![CUDA](https://img.shields.io/badge/-CUDA-76B900?logo=NVIDIA&logoColor=white)&nbsp;
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./LICENSE)

<img src="resources/teaser.gif" height="342"/>

__Progress accelerates when feedback cycles shrink.__  
Ever wondered how much faster you could iterate in your [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) research by unifying the most impactful advances from its many follow-up works?  
This repository presents our answer: the official implementation of “_Faster-GS: Analyzing and Improving Gaussian Splatting Optimization_”, an updated baseline for 3DGS optimization.

> __Faster-GS: Analyzing and Improving Gaussian Splatting Optimization__  
> [Florian Hahlbohm](https://fhahlbohm.github.io), [Linus Franke](https://lfranke.github.io/), [Martin Eisemann](https://graphics.tu-bs.de/people/eisemann), [Marcus Magnor](https://graphics.tu-bs.de/people/magnor)  
 _CVPR, June 2026_  
> __[Project page](https://fhahlbohm.github.io/faster-gaussian-splatting)&nbsp;| [Paper](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf)&nbsp;| [BibTeX](https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.bib)__


## Overview

Faster-GS is a highly-optimized yet readable and extensible 3DGS implementation.
Compared to other research codebases that are currently available, it is roughly 2-5x faster and uses less VRAM.
We achieve this by combining ideas from multiple 3DGS follow-up works with our own improvements. If you are interested in the details, we recommend you read our paper.

This repository has different branches that provide variants of Faster-GS. While we intentionally kept these variants separate to improve readability and modularity, there are no fundamental limitations with combining their respective features.

The `main` branch provides our fast 3DGS implementation with many additional features. We hope that their joint availability within an optimized framework will accelerate future research.

- Anti-aliasing improvements based on [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/)
- MCMC densification based on [3DGS-MCMC](https://ubc-vision.github.io/3dgs-mcmc/)
- Different Gaussian truncation modes enabled by a revised opacity interpretation within the rasterizer
- Improved random initialization through visibility and/or mask-based carving
- Random background color training to reduce false transparency
- Even faster training through informed pruning from [Speedy-Splat](https://speedysplat.github.io/)
- Handling of photometric variations in input images through [PPISP](https://research.nvidia.com/labs/sil/projects/ppisp/)
- Warp-level culling for efficient blending from [Warp-Level Culling](https://plse-splats.github.io/faster-gaussian-splatting/)
- Absolute gradient densification from [AbsGS](https://ty424.github.io/AbsGS.github.io/) using the selection strategy of [Gaussian Opacity Fields](https://niujinshuchong.github.io/gaussian-opacity-fields/)

On the `FasterGSFused` branch, we went full nerd-mode and fused the backward pass and optimizer. It provides an additional speedup over our default implementation and also uses less VRAM. And the best part is that the code is still pretty clean. Great if you need every bit of performance or just like to train extra fast.

The `FasterGS4D` branch provides an extension of our implementation to dynamic scene reconstruction with [4D Gaussians](https://fudan-zvg.github.io/4d-gaussian-splatting/). Compared to the original implementation by Yang et al., ours trains significantly faster and uses less VRAM without sacrificing quality.

The `FasterGSTestbed` branch contains the implementation we used to develop, integrate, and evaluate all improvements we discuss in our paper. It allows you to individually enable/disable the components we evaluate in the ablation study of our paper. This branch is a great starting point if, for example, you can not afford the extra VRAM required by the improved backward pass and want to develop a VRAM-optimized implementation.

On the `FasterGSBasis` branch, we provide our basis implementation described in the appendix of our paper. All other variants are built on top of this basis. We used it to develop and test multiple minor algorithmic and cleanup changes to ensure that they do not affect quality.


## Installation

Our implementation is provided as an extension to [NeRFICG](https://github.com/nerficg-project), a radiance field and view synthesis framework actively maintained and developed by our research group.

__Easy integration into existing codebases:__  
We understand that for compatibility with previous or current projects you might not be able to immediately switch from your current 3DGS framework.
Therefore, we tried to make it as easy as possible to integrate the most impactful improvements into existing codebases. Our CUDA backend can be installed into an existing environment as a PyTorch extension through the following command:
```shell
pip install git+https://github.com/nerficg-project/faster-gaussian-splatting/#subdirectory=FasterGSCudaBackend --no-build-isolation
```
Integrating the key features of this extension into the official 3DGS codebase (`graphdeco-inria/gaussian-splatting`) requires only minimal code changes.  
A reference implementation is available [here](https://github.com/fhahlbohm/gaussian-splatting).

__That being said, we encourage you to give the native implementation a shot as it provides the full performance improvements as well as a ton of additional features. We also tried to make the code as readable and extensible as possible by optimizing and refactoring the PyTorch and CUDA code of all integrated methods.__


### Requirements

- An NVIDIA GPU
- Linux (preferred) or Windows
- A recent CUDA SDK ([CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) recommended) and a compatible C++ compiler
- [Anaconda / Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed


### Setup

As a preparatory step, the [NeRFICG framework](https://github.com/nerficg-project/nerficg) needs to be set up.
Please follow the instructions in its README to set up a compatible Conda environment.

Now add Faster-GS as an additional method by cloning this repository into `src/Methods/FasterGS`:
```shell
# HTTPS
git clone https://github.com/nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS
```
or
```shell
# SSH
git clone git@github.com:nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS
```

Next, install all method-specific dependencies and CUDA extensions using:
```shell
python ./scripts/install.py -m FasterGS
```

_Note: The framework determines on-the-fly what extra modules need to be installed. Sometimes this causes unnecessary errors/warnings that can interrupt the installation process. In this case, first try to rerun the command before investigating the error in detail._


### Variants

Installing one of our Faster-GS variants (e.g., `FasterGS4D`) from their respective branch can be achieved by
```shell
# HTTPS
git clone --single-branch -b FasterGS4D https://github.com/nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS4D
```
or
```shell
# SSH
git clone --single-branch -b FasterGS4D git@github.com:nerficg-project/faster-gaussian-splatting.git src/Methods/FasterGS4D
```
and then
```shell
python ./scripts/install.py -m FasterGS4D
```


## Usage

The Faster-GS method is fully compatible with the NeRFICG scripts in the `scripts/` directory.
This includes config file generation via `create_config.py`,
training via `train.py`/`sequential_train.py`/`benchmark.py`,
inference and performance benchmarking via `inference.py`,
exporting trained models to .ply files via `convert_to_ply.py`,
and interactive rendering via `gui.py`.

For detailed instructions, please refer to the [NeRFICG repository](https://github.com/nerficg-project/nerficg).


## Acknowledgements

This work was partially funded by the DFG projects
[Real-Action VR](https://graphics.tu-bs.de/projects/real-action-vr) (ID 523421583) and
[Increasing Realism of Omnidirectional Videos in Virtual Reality](https://graphics.tu-bs.de/projects/increasing-perceived-realism-in-omnidirectional-visual-media) (ID 491805996).  
Linus Franke was supported by the ERC Advanced Grant [NERPHYS](https://project.inria.fr/nerphys) (ID 101141721).

We thank the authors of the following works, whose ideas and open-source implementations form the foundation of this project:
- Kerbl et al. [_3D Gaussian Splatting for Real-Time Radiance Field Rendering._](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) SIGGRAPH 2023.
- Radl et al. [_StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering._](https://r4dl.github.io/StopThePop/) SIGGRAPH 2024.
- Mallick et al. [_Taming 3DGS: High-Quality Radiance Fields with Limited Resources._](https://humansensinglab.github.io/taming-3dgs/) SIGGRAPH Asia 2024.
- Schütz et al. [_Splatshop: Efficiently Editing Large Gaussian Splat Models._](https://publications.graphics.tudelft.nl/papers/822) HPG 2025.
- Yang et al. [_Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting._](https://fudan-zvg.github.io/4d-gaussian-splatting/) ICLR 2024.
- Yu et al. [_Mip-Splatting: Alias-free 3D Gaussian Splatting._](https://niujinshuchong.github.io/mip-splatting/) CVPR 2024.
- Kheradmand et al. [_3D Gaussian Splatting as Markov Chain Monte Carlo._](https://ubc-vision.github.io/3dgs-mcmc/) NeurIPS 2024.
- Hanson et al. [_Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives._](https://speedysplat.github.io/) CVPR 2025.
- Deutsch et al. [_PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction._](https://research.nvidia.com/labs/sil/projects/ppisp/) CVPR 2026.
- Yang et al. [_Warp-Level Culling for Efficient Blending in 3D Gaussian Splatting._](https://plse-splats.github.io/faster-gaussian-splatting/) HPG 2026.
- Ye et al. [_AbsGS: Recovering Fine Details for 3D Gaussian Splatting._](https://ty424.github.io/AbsGS.github.io/) ACM MM 2024.
- Yu et al. [_Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes._](https://niujinshuchong.github.io/gaussian-opacity-fields/) SIGGRAPH Asia 2024.
- Kopanas [_What the 3DGS Literature Left on the Table._](https://grgkopanas.github.io/3dgs_cuda_blog/) Blog post 2026.

We also thank [Janusch Patas (aka. MrNeRF)](https://github.com/MrNeRF) for partially inspiring this project with his open-source bounty for [LichtFeld Studio](https://lichtfeld.io/), where an early version of our work was the [winning submission](https://github.com/MrNeRF/LichtFeld-Studio/pull/245) and has since become one of its core components.


## License and Citation

This project is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

If you use this project in your research, please cite our paper:

```bibtex
@misc{hahlbohm2026fastergs,
  title         = {Faster-GS: Analyzing and Improving Gaussian Splatting Optimization},
  author        = {Florian Hahlbohm and Linus Franke and Martin Eisemann and Marcus Magnor},
  year          = {2026},
  eprint        = {2602.09999},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2602.09999},
}
```
