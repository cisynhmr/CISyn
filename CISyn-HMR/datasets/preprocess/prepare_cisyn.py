"""
Process data/cisyn dataset into same format as hi4d (cisyn_smpl_{split}.npz).

cisyn structure per seq:
- camera_params/XF-camera-001.json: camera intrinsics and extrinsics (world2cam)
- img/XF-camera-001/0000.png - 0011.png: 12 frames
- *.npz: AMASS/SMPL-X format, one file per actor, (12, 3) trans, (12, 63) pose_body, etc.
- meta.json: asset1, asset2 reference npz IDs
"""
import os
import json
import glob
import numpy as np
from collections import defaultdict
import argparse
from scipy.spatial.transform import Rotation
import torch
import smplx

from tqdm import tqdm


def load_camera_params(cam_json_path):
    """Load camera from XF-camera-001.json format. Returns focal, princpt, R, T."""
    with open(cam_json_path) as f:
        cam = json.load(f)
    intrinsic = cam['intrinsic']
    # intrinsic: [[fx, 0, cx, 0], [0, fy, cy, 0], ...]
    fx = float(intrinsic[0][0])
    fy = float(intrinsic[1][1])
    cx = float(intrinsic[0][2])
    cy = float(intrinsic[1][2])
    focal = np.array([fx, fy], dtype=np.float32)
    princpt = np.array([cx, cy], dtype=np.float32)
    R = np.array(cam['extrinsic_r'], dtype=np.float32)
    T = np.array(cam['extrinsic_t'], dtype=np.float32)
    return focal, princpt, R, T


# Cache SMPL models to avoid repeated loading
_smpl_models = {}

def get_pelvis_from_betas(betas, body_pose=torch.zeros(1, 23, 3), global_orient=torch.zeros(1, 1, 3), gender='neutral', smpl_model_path='./weights/smpl_data'):
    """
    Compute pelvis position from SMPL betas (shape parameters).
    
    Args:
        betas: np.ndarray, shape (10,) or (n,) - SMPL shape parameters
        gender: str, one of ['neutral', 'male', 'female']
        smpl_model_path: str, path to SMPL model files
        
    Returns:
        pelvis_pos: np.ndarray, shape (3,) - pelvis position in T-pose
    """
    global _smpl_models
    
    # Load or get cached SMPL model
    if gender not in _smpl_models:
        _smpl_models[gender] = smplx.create(
            smpl_model_path, 
            'smpl', 
            gender=gender,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
            create_transl=False
        )
    
    smpl_model = _smpl_models[gender]
    
    # Convert betas to torch tensor
    betas_torch = torch.from_numpy(betas).float().unsqueeze(0)  # (1, 10)
    # Zero pose (T-pose)
    # body_pose = torch.zeros(1, 23, 3)  # (1, 23, 3)
    # global_orient = torch.zeros(1, 1, 3)  # (1, 1, 3)
    body_pose = body_pose.reshape(1, 23, 3)
    global_orient = global_orient.reshape(1, 1, 3)
    if type(body_pose) == np.ndarray:
        body_pose = torch.from_numpy(body_pose).float()
    if type(global_orient) == np.ndarray:
        global_orient = torch.from_numpy(global_orient).float()
    
    # Run SMPL forward pass
    with torch.no_grad():
        smpl_output = smpl_model(
            betas=betas_torch,
            body_pose=body_pose,
            global_orient=global_orient
        )
    
    # Get pelvis position (joint 0)
    pelvis_pos = smpl_output.joints[0, 0].numpy()  # (3,)
    
    return pelvis_pos.astype(np.float32)


def smplx_to_smpl(npz_data, frame_idx, npz_path):
    """
    Convert AMASS/SMPL-X pose to MA-HMR format (24 joints, 72 params).
    MA-HMR/Hi4D/Bedlam use: 1 (global_orient) + 23 (body joints) = 24 joints * 3 = 72.
    SMPL-X has: root_orient (3) + pose_body (21 joints, 63) = 66 params.
    We take the first 21 body joints from SMPL-X (identical to SMPL) and pad with 6 zeros
    for the 2 extra joints (wrists/leaf joints) to reach 24-joint format.
    """
    # import pdb; pdb.set_trace()
    betas = npz_data['betas'].astype(np.float32)
    if len(betas.shape) == 2:
        betas = betas[frame_idx]
    if len(betas) == 10:
        pass
    elif len(betas) > 10:
        betas = betas[:10]
    else:
        betas = np.pad(betas, (0, 10 - len(betas)), mode='constant')
    root_orient_aa = npz_data['root_orient'][frame_idx].astype(np.float64)
    # Apply coordinate shift: 90 deg around X (cisyn/SMPL-X to MA-HMR convention)
    # the camera use opencv, which is -y up, original amass data is z up
    R_orig = Rotation.from_rotvec(root_orient_aa)
    R_shift = Rotation.from_euler('xyz', np.deg2rad([90, 0, 0]))
    R_combined = R_shift * R_orig
    root_orient = R_combined.as_rotvec().astype(np.float32)
    pose_body_smplx = npz_data['pose_body'][frame_idx].astype(np.float32)  # (63,) = 21 joints
    trans = npz_data['trans'][frame_idx].astype(np.float32)
    # to (x, -z, y)
    # root_pos = npz_data['root_location_t0']
    # pelvis_pos = npz_data['pelvis_location_t0']
    # pelvis_ori_pos = pelvis_pos - root_pos
    pelvis_pos_smpl = npz_data['pelvis_pos_smpl'][frame_idx]
    pelvis_ori_pos_smpl = get_pelvis_from_betas(betas, pose_body_smplx, root_orient)
    trans = trans + pelvis_pos_smpl
    trans = np.array([trans[0], -trans[2], trans[1]], dtype=np.float32)
    trans = trans - pelvis_ori_pos_smpl
    # if 'pelvis_pos_smpl' in npz_data:
    #     trans = trans - npz_data['pelvis_pos_smpl'][frame_idx]
    
    # SMPL-X pose_body (21 joints, 63) maps 1:1 to SMPL body joints 1-21.
    # MA-HMR expects 24 joints total: body_pose must be 23 joints = 69 params.
    # Pad 6 zeros for the 2 extra leaf joints (SMPL output joints 22,23).
    if pose_body_smplx.shape[0] == 63:
        body_pose = np.concatenate([pose_body_smplx, np.zeros(6, dtype=np.float32)], axis=0)
    else:
        body_pose = pose_body_smplx
    return root_orient, body_pose, trans, betas


def get_gender(npz_data):
    """Get gender string from npz."""
    g = npz_data['gender']
    if hasattr(g, 'item'):
        return g.item()
    return str(g) if g is not None else 'neutral'


def get_npz_files(seq_path):
    """Get list of npz files (actor poses) in seq folder."""
    # Filter out files that have 'smpl' in their filename
    files = sorted(glob.glob(os.path.join(seq_path, '*.npz')))
    return [f for f in files if 'smpl' not in os.path.basename(f)]


def get_camera_folders(seq_path):
    """Get list of camera names (e.g. XF-camera-001) from camera_params."""
    cam_dir = os.path.join(seq_path, 'camera_params')
    if not os.path.isdir(cam_dir):
        return []
    return [f.replace('.json', '') for f in os.listdir(cam_dir) if f.endswith('.json')]

FAIL_LIST = []
def process_sequence(seq_path, seq_id, data_root, annots):
    """Process one sequence folder, append to annots dict."""
    cam_folders = get_camera_folders(seq_path)
    if not cam_folders:
        return 0
    npz_files = get_npz_files(seq_path)
    if not npz_files:
        return 0

    count = 0
    for cam_name in cam_folders:
        cam_json = os.path.join(seq_path, 'camera_params', f'{cam_name}.json')
        img_dir = os.path.join(seq_path, 'img', cam_name)
        if not os.path.isfile(cam_json) or not os.path.isdir(img_dir):
            continue
        focal, princpt, R, T = load_camera_params(cam_json)

        # Load all actor poses
        actor_data = []
        for npz_path in npz_files:
            try:
                data = np.load(npz_path, allow_pickle=True)
                data = dict(data)
                n_frames = data['trans'].shape[0]
                npz_path_smpl = npz_path.replace('.npz', '_smpl.npz')
                if os.path.isfile(npz_path_smpl):
                    data_smpl = np.load(npz_path_smpl, allow_pickle=True)
                    data['root_orient'] = data_smpl['root_orient']
                    data['pose_body'] = data_smpl['body_pose']
                    data['betas'] = data_smpl['betas']
                    data['pelvis_pos_smpl'] = data_smpl['transl']
                actor_data.append((npz_path, data, n_frames))
            except Exception as e:
                FAIL_LIST.append(npz_path)
                continue

        if not actor_data:
            continue

        n_frames = min(ad[2] for ad in actor_data)
        for frame_idx in range(n_frames):
            img_name = f'{frame_idx:04d}.png'
            img_path = os.path.join(img_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            # Annot key: path relative to cisyn/ folder (same as hi4d: relative to dataset_path)
            annot_key = os.path.join(seq_id, 'img', cam_name, img_name)

            person_list = []
            for npz_path, data, _ in actor_data:
                try:
                    global_orient, body_pose, transl, betas = smplx_to_smpl(data, frame_idx, npz_path)
                    gender = get_gender(data)
                    person_list.append({
                        'focal': focal,
                        'princpt': princpt,
                        'cam_rot': R,
                        'cam_trans': T,
                        'global_orient': global_orient,
                        'body_pose': body_pose,
                        'transl': transl,
                        'betas': betas,
                        'genders': gender
                    })
                except Exception:
                    FAIL_LIST.append(npz_path)
                    continue

            if person_list:
                annots[annot_key] = person_list
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='Data root containing cisyn/')
    parser.add_argument('--set', type=str, choices=['train', 'val', 'test', 'all', 'train_all'], default='all',
                        help='Which split(s) to create. "all" creates train/val/test in one run.')
    parser.add_argument('--split_ratio', type=str, default='0.8,0.1,0.1',
                        help='Train/val/test ratio (e.g. 0.8,0.1,0.1)')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--max_seqs', type=int, default=None, help='Max seqs per split (for testing)')
    args = parser.parse_args()

    data_root = args.data_root
    cisyn_root = os.path.join(data_root, 'cisyn')
    if not os.path.isdir(cisyn_root):
        raise FileNotFoundError(f'cisyn root not found: {cisyn_root}')

    # Get all seq folders
    seq_folders = sorted([d for d in os.listdir(cisyn_root)
                          if os.path.isdir(os.path.join(cisyn_root, d))])
    n_seqs = len(seq_folders)
    if n_seqs == 0:
        raise FileNotFoundError(f'No sequence folders in {cisyn_root}')

    if args.set == 'train_all':
        if args.max_seqs is not None:
            seq_folders = seq_folders[:args.max_seqs]
        split_map = {'train_all': seq_folders}
        splits_to_run = ['train_all']
    else:
        # Create split
        ratios = [float(x) for x in args.split_ratio.split(',')]
        assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 1e-6
        rng = np.random.RandomState(args.split_seed)
        indices = rng.permutation(n_seqs)
        n_train = int(n_seqs * ratios[0])
        n_val = int(n_seqs * ratios[1])
        n_test = n_seqs - n_train - n_val
        train_seqs = [seq_folders[i] for i in indices[:n_train]]
        val_seqs = [seq_folders[i] for i in indices[n_train:n_train + n_val]]
        test_seqs = [seq_folders[i] for i in indices[n_train + n_val:]]

        train_seqs = seq_folders[:24000]
        test_seqs = seq_folders[24000:]

        if args.max_seqs is not None:
            train_seqs = train_seqs[:args.max_seqs]
            val_seqs = val_seqs[:args.max_seqs]
            test_seqs = test_seqs[:args.max_seqs]

        split_map = {'train': train_seqs, 'val': val_seqs, 'test': test_seqs}
        splits_to_run = ['train', 'test'] if args.set == 'all' else [args.set]

    for split in splits_to_run:
        seq_ids = split_map[split]
        annots = {}
        for seq_id in tqdm(seq_ids, desc=f'cisyn {split}', ncols=80):
            seq_path = os.path.join(cisyn_root, seq_id)
            process_sequence(seq_path, seq_id, data_root, annots)

        out_path = os.path.join(cisyn_root, f'cisyn_smpl_{split}.npz')
        np.savez(out_path, annots=annots)
        print(f'Saved {len(annots)} image annotations to {out_path}')


if __name__ == '__main__':
    main()
    print(f'Failed {len(FAIL_LIST)} sequences')
    with open('fail_list.txt', 'w') as f:
        for fail_path in FAIL_LIST:
            f.write(fail_path + '\n')
