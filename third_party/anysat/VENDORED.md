# Vendored: AnySat (released feature extractor code only) - STATUS: BLOCKED (weights)

- URL: https://github.com/gastruc/AnySat
- Commit: 5f6f475e1a22ce5e3a56a5b18f4ed6d24eca2a4a
- License: MIT (`LICENSE`, kept).
- Extracted with `git archive HEAD <files> | tar -x`, unmodified. Never edit these files; wrap in `src/`.
- Files:
  - `LICENSE`
  - `hubconf.py` (`AnySat` released model, `get_default_config`)
  - `src/models/__init__.py`
  - `src/models/networks/encoder/Any_multi.py`, `src/models/networks/encoder/Transformer.py`
  - `src/models/networks/encoder/utils/{ltae.py, utils.py, utils_ViT.py, pos_embed.py, irpe.py, patch_embeddings.py}`
- Not vendored: training code, configs, data loaders, `rpe_ops` (optional compiled op; `irpe.py` falls back
  to PyTorch with a warning), `.media/`.
- Code imports and runs cleanly with torch only (`flash_attn=False`).
- Weights: `hubconf.py` downloads https://huggingface.co/g-astruc/AnySat/resolve/main/models/AnySat.pth.
  One attempt with `torch.hub.load_state_dict_from_url` on 2026-10-06 failed with
  `URLError <urlopen error Tunnel connection failed: 403 Forbidden>` (huggingface.co blocked by the proxy).
  The owner must download `AnySat.pth` and set `model.params.checkpoint_path` in `configs/models/anysat.yaml`.
  Without it the encoder is randomly initialised, which is not the PRD's pretrained foundation model.
- Used by: `src/models/tier_bc/anysat.py` (imports through `src/models/tier_bc/upstream.py::isolated_sys_path`,
  because AnySat's top-level package is also called `src`).
