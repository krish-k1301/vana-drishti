# Vendored: DeepSatModels (TSViT only)

- URL: https://github.com/michaeltrs/DeepSatModels
- Commit: 53e655936be088522ec419b3ead38684c20ae1ec
- License: Apache License 2.0 (`LICENSE.txt`, kept). No NOTICE file upstream.
- Extracted with `git archive HEAD <files> | tar -x`, unmodified. Never edit these files; wrap in `src/`.
- Files:
  - `LICENSE.txt`
  - `models/TSViT/TSViTdense.py` (class `TSViT`, main-results model)
  - `models/TSViT/module.py` (`Attention`, `PreNorm`, `FeedForward`)
  - `utils/__init__.py`, `utils/config_files_utils.py` (`get_params_values`, imported by TSViTdense.py)
- Not vendored: `models/__init__.py` (imports every DeepSatModels model), so `models` is a namespace package.
- Imports needed: torch, einops, numpy, pyyaml (all in environment.yml).
- Used by: `src/models/tier_bc/tsvit.py` (imports through `src/models/tier_bc/upstream.py::import_isolated`,
  because the top-level names `models` and `utils` are generic).
- Reference recipe: `configs/PASTIS24/TSViT_fold1.yaml` at the same commit (not vendored; line numbers cited
  in `configs/models/tsvit.yaml`).
