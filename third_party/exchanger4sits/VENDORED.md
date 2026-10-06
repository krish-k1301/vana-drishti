# Vendored: Exchanger4SITS (Exchanger + U-Net building blocks)

- URL: https://github.com/TotalVariation/Exchanger4SITS
- Commit: 6a5ecf39c356e6ec8c4746cd3cc7936ee610d67c
- License: MIT (`LICENSE`, kept).
- Extracted with `git archive HEAD <paths> | tar -x`, unmodified. Never edit these files; wrap in `src/`.
- Files: `LICENSE`, `lib/layers/`, `lib/modules/`, `lib/ops/`, `lib/utils/`, `lib/losses/` (whole folders).
  Only `modules.Exchanger`, `modules.UNet` and `layers.TemporalPositionalEncoding` are used, but upstream
  `lib/modules/__init__.py` imports every module (FPN, MaskFormer, PaPs, Swin, PVT), which pulls in
  `layers`, `ops`, `utils` and `losses`, so those packages are vendored whole.
- Not vendored: `lib/models/` (its `Segmentor` builds the loss and imports `lib/datasets`, whose PASTIS/MTLCC
  readers need zarr/geopandas readers we do not use), `lib/datasets/`, `lib/apis/`, `lib/config/`, `tools/`,
  `configs/`, `figs/`. The `unet` branch of `lib/models/sem_seg.py::Segmentor` is mirrored in
  `src/models/tier_bc/exchanger_unet.py`.
- Import-time dependencies (installed in /home/user/venv311 on 2026-10-06, one attempt each):
  - `timm==1.0.30` (PyPI)
  - `torch-scatter==2.1.2` (PyPI sdist, built from source, ~5 min on CPU; `data.pyg.org` wheels are blocked here:
    "Failed to fetch https://data.pyg.org/whl/torch-2.5.0+cpu.html ... tunnel error: unsuccessful")
  - `detectron2==0.6` from `git+https://github.com/facebookresearch/detectron2.git@1e3e13bbf607b54f62205c4c33922521822fb298`
    (needs `setuptools` present and `--no-build-isolation`; the first try without setuptools failed with
    "ModuleNotFoundError: No module named 'setuptools'")
  - `opencv-python-headless==5.0.0.93` (detectron2.projects.point_rend imports cv2)
  - plus pandas, tqdm, scipy (already pinned).
  None of these is used at run time by Exchanger + U-Net; they are needed only because upstream's
  `modules/__init__.py` and `losses/__init__.py` import FPN/MaskFormer/PaPs code.
- Pretrained weights (zenodo record 8406435) are not used: zenodo is blocked here and they are PASTIS (optical).
