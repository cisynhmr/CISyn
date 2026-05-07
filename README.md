# CISyn

Physics-based pipeline for generating synthetic multi-person contact/collision sequences. The generated dataset is available at **[HuggingFace: cisyn/cisyn](https://huggingface.co/datasets/cisyn/cisyn)**. Given random character identities, motions, and environments, it:

1. **Defines** sequences (`seq_define`) — samples characters, AMASS motions, HDRI lighting, and camera pose; writes `meta.json` per sequence.
2. **Simulates** them (`simulate_seq`) — runs a Genesis rigid-body physics simulation; outputs per-character `.npz` motion files (AMASS-compatible).
3. **Renders** them (`simulate_render_seq`) — drives Blender via xrfeitoria to produce RGB frames, instance masks, and depth maps.

Three variants cover different scene configurations:

| Suffix | Script set | Scene |
|--------|-----------|-------|
| *(none)* | `*_seq.py` | 2-person, high speed |
| `_2` | `*_seq_2.py` | 2-person, lower speed |
| `_3p` | `*_seq_3p.py` | 3-person |

All variants write output to `cisyn/`.

---

## Environment setup

Two separate conda environments are required — one for simulation, one for rendering.

### Simulation env (`genesis`)

Used for Steps 1 and 2 (`seq_define`, `simulate_seq`).

```bash
conda env create -f env_genesis.yaml
conda activate genesis
```

### Render env (`xrfeitoria`)

Used for Step 3 (`simulate_render_seq`). Requires **Blender** installed and reachable by xrfeitoria, and **ffmpeg** in PATH.

```bash
conda env create -f env_xrfeitoria.yaml
conda activate xrfeitoria
```

Then apply the required patches to the xrfeitoria library:

```bash
bash patches/apply_patches.sh
```

This replaces two files in the installed xrfeitoria package:

| Patch file | Replaces |
|-----------|---------|
| `patches/xrfeitoria_motion.py` | `xrfeitoria/utils/anim/motion.py` |
| `patches/xrfeitoria_render.py` | `xrfeitoria/renderer/renderer_blender.py` |

---

## Data setup

Download the data bundle from HuggingFace and extract it into the repo root as `data/`:

```bash
wget "https://huggingface.co/datasets/cisyn/asset/resolve/main/data.tar?download=true" -O data.tar
tar -xf data.tar          # extracts to data/
```

Expected layout after extraction:

```
data/
  amass.npy                   # AMASS poses array (large)
  asset_list.json             # character asset index
  hdri_list.json              # HDRI environment index
  betas_index.npy             # SMPL body shape indices
  betas_index_smpl.npy
  betas_index_smplx.npy
  shifts_index_smpl.npy
  asset.npy
  assets/                     # per-character dirs (humanoid.xml, SMPL-XL-baked.fbx, meta_smplx.npz, betas-*.npy)
  HDRI/                       # outdoor .HDR environment maps
  HDRI2/                      # indoor .HDR environment maps
```

---

## Usage

### Step 1 — Define sequences

Generates `meta.json` for each sequence in `cisyn/`. Run once per variant; subsequent steps consume these files.

```bash
python seq_define.py      # 2-person, high speed
python seq_define_2.py    # 2-person, lower speed
python seq_define_3p.py   # 3-person
```

### Step 2 — Simulate

Runs Genesis physics for a single sequence folder. Parallelise over many folders with a job scheduler.

```bash
python simulate_seq.py    --seq-dir cisyn/000000   # (or _2 / _3p variant)
```

Output per folder: `{asset_folder}.npz` files containing simulated AMASS-format motion.

### Step 3 — Render

Renders a simulated sequence with Blender/xrfeitoria. Must be run after Step 2.

```bash
python simulate_render_seq.py    --seq-dir cisyn/000000   # (or _2 / _3p variant)
```

Output per folder: `img/`, `mask/`, `depth/` frame directories and `video.mp4`.

`--root-dir` defaults to the repo directory (where `data/` lives); override if running from elsewhere:

```bash
python simulate_render_seq.py --seq-dir /path/to/cisyn/000000 --root-dir /path/to/CISyn
```

---

## File overview

```
seq_define.py / _2 / _3p          sequence definition (meta.json generation)
simulate_seq.py / _2 / _3p        Genesis physics simulation
simulate_render_seq.py / _2 / _3p xrfeitoria/Blender rendering
joint_params_loader.py             loads joint_params.yaml for simulate_seq
joint_params.yaml                  joint limits, stiffness, damping values
data/                              assets and motion data (see Data setup above)
env_genesis.yaml                   conda env for simulation
env_xrfeitoria.yaml                conda env for rendering
patches/                           xrfeitoria library patches (apply once after env install)
CISyn-HMR/                         HMR model code (see below)
```

---

## HMR model

The [`CISyn-HMR/`](CISyn-HMR/) subfolder contains the code for the Human Mesh Recovery (HMR) model trained on CISyn data. See [CISyn-HMR/README.md](CISyn-HMR/README.md) for installation and usage instructions.

---

## Acknowledgements

This project builds on the following open-source works:

- **[Genesis](https://github.com/Genesis-Embodied-AI/genesis-world)** — physics simulation engine used for rigid-body character simulation.
- **[xrfeitoria](https://github.com/openxrlab/xrfeitoria)** — Blender-based rendering framework used for RGB, mask, and depth rendering.
- **[SynBody](https://huggingface.co/datasets/caizhongang/SynBody)** — synthetic human body dataset that informed asset and body shape design.
- **[SMPLSim](https://github.com/ZhengyiLuo/SMPLSim)** — SMPL-based physics simulation reference used for character control.
