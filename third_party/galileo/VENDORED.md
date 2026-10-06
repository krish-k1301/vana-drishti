# Vendored: Galileo (single-file encoder + nano weights)

- URL: https://github.com/nasaharvest/galileo
- Commit: 0f0b5b95ac81acef4b74cf4686dc877202f4541b
- License: MIT (`LICENSE`, kept). The nano weights are committed in the upstream repository under the same
  license (README: "The nano model weights are available on github (data/models/nano)").
- Extracted with `git archive HEAD <files> | tar -x`, unmodified. Never edit these files; wrap in `src/`.
- Files:
  - `LICENSE`
  - `single_file_galileo.py` (self-contained `Encoder`; needs torch, einops, numpy)
  - `data/models/nano/config.json` (encoder architecture: embedding 128, depth 4, 8 heads, 24 timesteps)
  - `data/models/nano/encoder.pt` (4.2 MB, the encoder state dict, loaded strictly by `Encoder.load_from_folder`)
- Not vendored: `decoder.pt`, `second_decoder.pt`, `target_encoder.pt` (pretraining-only), `src/`, larger models
  (huggingface.co, blocked here).
- Used by: `src/models/tier_bc/galileo.py` (`configs/models/galileo.yaml` points `checkpoint_path` at
  `third_party/galileo/data/models/nano`).
