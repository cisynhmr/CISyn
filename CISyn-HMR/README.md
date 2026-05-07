# CISyn-HMR


## ⚙️ Installation

We follow [SAT-HMR](https://github.com/ChiSu001/SAT-HMR) and [MA-HMR](https://github.com/gouba2333/MA-HMR), testing with python 3.11, PyTorch 2.4.1 and CUDA 12.1.

1. Clone the repo and create a conda environment.
```bash
conda create -n cisyn python=3.11 -y
conda activate cisyn
```

2. Install [PyTorch](https://pytorch.org/) and [xFormers](https://github.com/facebookresearch/xformers).
```bash
# Install PyTorch. It is recommended that you follow [official instruction](https://pytorch.org/) and adapt the cuda version to yours.
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install xFormers. It is recommended that you follow [official instruction](https://github.com/facebookresearch/xformers) and adapt the cuda version to yours.
pip install -U xformers==0.0.28.post1  --index-url https://download.pytorch.org/whl/cu121
```

3. Install other dependencies.
```bash
pip install -r requirements.txt
```

4. You may need to modify `chumpy` package to avoid errors. For detailed instructions, please check [this guidance](https://github.com/ChiSu001/SAT-HMR/blob/main/docs/fix_chumpy.md).

## 📦 Download Models & Weights

1. Download SMPL-related weights and place them in `weights/smpl_data/smpl/`. Partially Available at [this link](https://drive.google.com/drive/folders/1C8fZNiiZfC1oMUZq7xNilQcGv4LJf5M8?usp=drive_link). You need to register on the [SMPL website](https://smpl.is.tue.mpg.de/) to get other part of them.

```
weights/
└── smpl_data/
    └── smpl/
        ├── body_verts_smpl.npy
        ├── J_regressor_h36m_correct.npy
        ├── J_regressor_extra.npy
        ├── smpl_mean_params.npz
        ├── SMPL_FEMALE.pkl
        ├── SMPL_MALE.pkl
        ├── SMPL_NEUTRAL.pkl
        └── smpl_kid_template.npy
```

2. Download DINOv2 pretrained weights from [their official repository](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models). We use `ViT-B/14 distilled (without registers)`. Please put `dinov2_vitb14_pretrain.pth` to `weights/dinov2`. These weights will be used to initialize our encoder. You can skip this step if you are not going to train MA-HMR.
```
weights/
└── dinov2
    └── dinov2_vitb14_pretrain.pth
```

3. Download pretrained weights of MA-HMR from [Google Drive](https://drive.google.com/drive/folders/1CaQOaQZ94ot91D_kqhvasfRb-utpfKmM?usp=drive_link) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/99755d6e4b7a463fb673/) and put them to `weights/ma_hmr`. You can skip this step if you are not going to train CISyn-HMR.

```
weights
└── ma_hmr
    └── mahmr_stage3.bin
```

4. Download pretrained weights of **CISyn-HMR** from [🤗HuggingFace](https://huggingface.co/cisyn/hmr/tree/main). Please put them to `weights/cisyn_hmr`
```
weights
└── cisyn_hmr
    └── cisyn_hmr.bin
```

## 📦 Data Preparation

Download CISyn dataset from [🤗HuggingFace](https://huggingface.co/datasets/cisyn/cisyn/tree/main).
Please refer to scripts in `datasets/preprocess/` to preprocess Hi4D, ChI3D. 
Please follow [this guidance](https://github.com/ChiSu001/SAT-HMR/blob/main/docs/data_preparation.md) to prepare AGORA, BEDLAM. Download [DTO-Humans](https://github.com/gouba2333/DTO-Humans.git) annotations from [Google Drive](https://drive.google.com/drive/folders/1ddc43P6iYIctAvmuravIxbxZm3F2uB41?usp=drive_link) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/539173c2952b40f5a422/). Placing all datasets in `data/`.
 You can skip this step if you are not going to train or evaluate MA-HMR.

```
data/
├── agora
│   ├── smpl_neutral_annots
│   │   ├── annots_smpl_train_fit.npz
│   │   └── annots_smpl_validation.npz
│   ├── test/
│   ├── train/
│   └── validation/
├── aic/
│   ├── images/
│   └── AIC_CHMR_SMPL_OPT.npz
├── bedlam
│   ├── train/
│   ├── validation/
│   ├── bedlam_smpl_train_1fps.npz
│   ├── bedlam_smpl_train_6fps.npz
│   └── bedlam_smpl_validation_6fps.npz
├── chi3d/
│   ├── train/
│   ├── test/
│   └── chi3d_smpl_train.npz
├── cisyn/
│   ├── **/
│   ├── cisyn_smpl_test.npz
│   └── cisyn_smpl_train.npz
├── coco2014/
│   ├── images/
│   │   └── train2014/
│   └── COCO_CHMR_SMPL_OPT.npz
├── hi4d/
│   ├── pair**/
│   ├── hi4d_smpl_test.npz
│   └── hi4d_smpl_train.npz
├── insta/
│   ├── images/
│   │   └── insta-train/
│   └── INSTA_CHMR_SMPL_OPT.npz
└── mpii/
    ├── images/
    └── MPII_CHMR_SMPL_OPT.npz
```


## ▶️ Inference on Images
<h4> Inference with 1 GPU</h4>

We provide some demo images in `demo/`. You can run MA-HMR on all images on a single GPU via:


```bash
python main.py --mode infer --cfg demo
```

Results with overlayed meshes will be saved in `${Project}/demo_results`.

You can specify your own inference configuration by modifing `configs/run/demo.yaml`:

- `input_dir` specifies the input image folder.
- `output_dir` specifies the output folder.
- `conf_thresh` specifies a list of confidence thresholds used for detection.
- `infer_batch_size` specifies the batch size used for inference (on a single GPU).

<h4> Inference with Multiple GPUs</h4>

You can also try distributed inference on multiple GPUs if your input folder contains a large number of images. 
Since we use [Accelerate](https://huggingface.co/docs/accelerate/index) to launch our distributed configuration, first you may need to configure [Accelerate](https://huggingface.co/docs/accelerate/index) for how the current system is setup for distributed process. To do so run the following command and answer the questions prompted to you:

```bash
accelerate config
```

Then run:
```bash
accelerate launch main.py --mode infer --cfg demo
```

## 🔧 Training

<h4> Training with Multiple GPUs</h4>

We use [Accelerate](https://huggingface.co/docs/accelerate/index) to launch our distributed configuration, first you may need to configure [Accelerate](https://huggingface.co/docs/accelerate/index) for how the current system is setup for distributed process. To do so run the following command and answer the questions prompted to you:

```bash
accelerate config
```

```bash
accelerate launch main.py --mode train --cfg train_cisyn
```

<h4> Monitor Training Progress</h4>

Training logs and checkpoints will be saved in the `${Project}/outputs/logs` and `${Project}/outputs/ckpts` directories, respectively.

You can monitor the training progress using TensorBoard. To start TensorBoard, run:

```bash
tensorboard --logdir=${Project}/outputs/logs
```

## 📊 Evaluation

<h4> Evaluation with 1 GPU</h4>

```bash
# Evaluate CISyn-HMR
python main.py --mode eval --cfg eval_cisyn
```

<h4> Evaluation with Multiple GPUs</h4>

We recommend using a single GPU for evaluation as it provides more accurate results. However, we also provide code for distributed evaluation to obtain results faster.

```bash
# Multi-GPU configuration
accelerate config
# Evaluation
accelerate launch main.py --mode eval --cfg ${cfg_name}
```

## 📜 License

The code and weights are released under the [**Creative Commons Attribution-NonCommercial 4.0 International License**](https://creativecommons.org/licenses/by-nc/4.0/). This means they are available for **non-commercial academic research purposes only**. Please see the [LICENSE](LICENSE) file for the full license text.

## 🙏 Acknowledgements

This project builds upon several amazing open-source projects and datasets. We would like to thank the authors of:
*   [MA-HMR](https://github.com/gouba2333/MA-HMR)
*   [SAT-HMR](https://github.com/ChiSu001/SAT-HMR)
*   [DINOv2](https://github.com/facebookresearch/dinov2)
*   [Accelerate](https://huggingface.co/docs/accelerate/index)