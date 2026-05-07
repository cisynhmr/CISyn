"""
Fast dataset visualization directly from NPZ annotations.
- No BASE/dataset class overhead; no image resize or augmentation pipeline
- Uses render_meshes (single pyrender scene) for correct per-pixel z-buffer depth
"""
import os
import numpy as np
import torch
import cv2
from tqdm import tqdm

from models.human_models import SMPL_Layer, SMPL_Kid_Layer
from utils.render import render_meshes
from utils.visualization import get_colors_rgb
from configs.paths import smpl_model_path, dataset_root
from utils.constants import smpl_root_idx

# ── config ────────────────────────────────────────────────────────────────────
NPZ_PATH   = os.path.join(dataset_root, 'cisyn/cisyn_smpl_train.npz')
IMG_ROOT   = os.path.join(dataset_root, 'cisyn')
OUTPUT_DIR = './datasets_visualization/cisyn'
VIS_FILTER = ['000003', '000121']   # substring match on NPZ img_name keys; [] = show all
USE_KID    = True
# ─────────────────────────────────────────────────────────────────────────────


def main():
    smpl = (SMPL_Kid_Layer if USE_KID else SMPL_Layer)(
        model_path=smpl_model_path, with_genders=True
    )
    faces = smpl.faces  # (F, 3)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    annots = np.load(NPZ_PATH, allow_pickle=True)['annots'].item()

    for img_name, persons in tqdm(annots.items()):
        if VIS_FILTER and not any(f in img_name for f in VIS_FILTER):
            continue

        img_path = os.path.join(IMG_ROOT, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f'Image not found: {img_path}')
            continue

        pnum = len(persons)
        if pnum == 0:
            continue

        poses_list, betas_list, genders_list = [], [], []
        cam_rots, cam_trans_list, transls = [], [], []

        for p in persons:
            pose = np.concatenate([p['global_orient'], p['body_pose']])  # (72,)
            betas = p['betas'].copy()
            if USE_KID and len(betas) == 10:
                betas = np.concatenate([betas, np.zeros(1)])
            poses_list.append(pose)
            betas_list.append(betas)
            genders_list.append(p['genders'])
            cam_rots.append(p['cam_rot'])       # (3,3)
            cam_trans_list.append(p['cam_trans'].reshape(3))
            transls.append(p['transl'])         # (3,)

        poses    = torch.tensor(np.stack(poses_list),    dtype=torch.float32)
        betas    = torch.tensor(np.stack(betas_list),    dtype=torch.float32)
        cam_rot  = torch.tensor(np.stack(cam_rots),      dtype=torch.float32)  # (pnum,3,3)
        cam_trans = torch.tensor(np.stack(cam_trans_list), dtype=torch.float32) # (pnum,3)
        transl   = torch.tensor(np.stack(transls),       dtype=torch.float32)  # (pnum,3)

        # Bake cam_rot into global orient — mirrors BASE.process_smpl
        go = poses[:, :3].numpy()
        cr = cam_rot.numpy()
        for i in range(pnum):
            r, _ = cv2.Rodrigues(go[i].reshape(1, 3))
            r, _ = cv2.Rodrigues(cr[i] @ r)
            go[i] = r.flatten()
        poses[:, :3] = torch.from_numpy(go)

        with torch.no_grad():
            verts, j3ds = smpl(poses=poses, betas=betas, genders=genders_list)

        root = j3ds[:, smpl_root_idx, :].clone()  # (pnum,3)

        # Camera-space translation — same formula as BASE.process_smpl
        transl_cam = (
            torch.bmm((root + transl).reshape(-1, 1, 3), cam_rot.transpose(-1, -2)).reshape(-1, 3)
            + cam_trans - root
        )
        verts_cam = (verts + transl_cam[:, None, :]).numpy()  # (pnum,6890,3)

        # All persons in one image share the same camera
        cam_param = {
            'focal':   np.asarray(persons[0]['focal'],   dtype=np.float64),
            'princpt': np.asarray(persons[0]['princpt'], dtype=np.float64),
        }

        colors   = get_colors_rgb(pnum)
        rendered = render_meshes(
            img=img,
            l_mesh=list(verts_cam),
            l_face=[faces] * pnum,
            cam_param=cam_param,
            color=colors,
        )

        out_name = img_name.replace('/', '_').rsplit('.', 1)[0]
        cv2.imwrite(os.path.join(OUTPUT_DIR, f'{out_name}.jpg'), rendered)


if __name__ == '__main__':
    main()
