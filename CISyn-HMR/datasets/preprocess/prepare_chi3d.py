"""
Process data/chi3d/train into the same annotation format as Hi4D/cisyn.

CHI3D train structure (three scenes: s02, s03, s04):
- camera_parameters/{camera_id}/{seq_name}.json  -- per-sequence camera params (one per seq per cam)
- images/{camera_id}/{seq_name}/{frame:06d}.png   -- frames, 1-indexed (000001.png = SMPL frame 0)
- smplx_amass/{seq_name}/personXX.npz             -- raw SMPL-X AMASS params
    keys: trans (N,3), poses (N,165), betas (10,), betas_all (N,10), gender
- smplx_amass/{seq_name}/personXX_smpl.npz        -- pre-processed SMPL override (optional)
    keys: root_orient (N,3), body_pose (N,69), betas (N,10), transl (N,3), gender

Coordinate system note:
  CHI3D camera extrinsics are calibrated in z-up AMASS world space (NOT OpenCV y-down).
  The camera JSON stores R and T such that:  v_cam = R @ (v_world_zamass - T)
  where T is the camera center in z-up world.
  Therefore SMPL params are kept in z-up AMASS space (no axis swap applied), and
  cam_trans is stored as -R @ T so that base.py's convention v_cam = R @ v_world + cam_trans holds.

  This differs from cisyn/Hi4D where the world is already OpenCV y-down and a 90-deg X
  rotation + axis swap is applied to SMPL params during preprocessing.

Note: sequences without _smpl.npz (all of s04, most of s03) fall back to raw AMASS fields only
but will fail at pelvis_pos_smpl lookup and be silently skipped (same behaviour as cisyn).

Output: data/chi3d/chi3d_smpl_train.npz
  annots dict: {rel_img_path -> [person_dict, ...]}
  where rel_img_path is relative to data/chi3d/
"""
import os
import json
import glob
import argparse
import numpy as np
import torch
import smplx
from tqdm import tqdm

FAIL_LIST = []

# Cache SMPL models to avoid repeated loading
_smpl_models = {}


def get_pelvis_from_betas(betas, body_pose=torch.zeros(1, 23, 3), global_orient=torch.zeros(1, 1, 3),
                          gender='neutral', smpl_model_path='./weights/smpl_data'):
    """Compute pelvis position from SMPL betas and current pose via a forward pass."""
    global _smpl_models
    if gender not in _smpl_models:
        _smpl_models[gender] = smplx.create(
            smpl_model_path, 'smpl', gender=gender,
            create_global_orient=False, create_body_pose=False,
            create_betas=False, create_transl=False,
        )
    smpl_model = _smpl_models[gender]
    betas_torch = torch.from_numpy(betas).float().unsqueeze(0)  # (1, 10)
    body_pose = body_pose.reshape(1, 23, 3)
    global_orient = global_orient.reshape(1, 1, 3)
    if isinstance(body_pose, np.ndarray):
        body_pose = torch.from_numpy(body_pose).float()
    if isinstance(global_orient, np.ndarray):
        global_orient = torch.from_numpy(global_orient).float()
    with torch.no_grad():
        out = smpl_model(betas=betas_torch, body_pose=body_pose, global_orient=global_orient)
    return out.joints[0, 0].numpy().astype(np.float32)  # pelvis = joint 0, shape (3,)


def smplx_to_smpl(npz_data, frame_idx, npz_path):
    """
    Convert AMASS/SMPL-X pose to MA-HMR SMPL format.

    MA-HMR uses: global_orient (3,) + body_pose (69,) = 24 joints × 3.
    SMPL-X pose_body has 21 joints (63,); padded with 6 zeros to reach 69.

    CHI3D camera extrinsics are defined in z-up AMASS world space, so SMPL params
    are kept in z-up AMASS world space with no coordinate swap.  base.py applies
    cam_rot to bring them into camera space at training time.

    Translation pipeline:
      transl = (raw_trans + pelvis_pos_smpl) - pelvis_in_current_pose
    where pelvis_pos_smpl comes from _smpl.transl (set during loading).
    """
    betas = npz_data['betas'].astype(np.float32)
    if len(betas.shape) == 2:
        betas = betas[frame_idx]
    if len(betas) > 10:
        betas = betas[:10]
    elif len(betas) < 10:
        betas = np.pad(betas, (0, 10 - len(betas)), mode='constant')

    # Keep root_orient in z-up AMASS world space; base.py rotates it via cam_rot.
    root_orient = npz_data['root_orient'][frame_idx].astype(np.float32)

    pose_body = npz_data['pose_body'][frame_idx].astype(np.float32)  # (63,) or (69,)
    if pose_body.shape[0] == 63:
        pose_body = np.concatenate([pose_body, np.zeros(6, dtype=np.float32)])

    trans = npz_data['trans'][frame_idx].astype(np.float32)
    pelvis_pos_smpl = npz_data['pelvis_pos_smpl'][frame_idx]          # from _smpl.transl
    pelvis_ori_pos_smpl = get_pelvis_from_betas(betas, pose_body, root_orient)
    # Stay in z-up AMASS world space; no axis swap.
    trans = trans + pelvis_pos_smpl - pelvis_ori_pos_smpl

    return root_orient, pose_body, trans, betas


def get_gender(npz_data):
    g = npz_data['gender']
    if hasattr(g, 'item'):
        return g.item()
    return str(g) if g is not None else 'neutral'


def load_camera_params(cam_json_path):
    """Load camera intrinsics and extrinsics from CHI3D per-sequence camera JSON.

    The JSON stores T as the camera center in z-up AMASS world space, so the
    world-to-camera transform is:  v_cam = R @ (v_world - T)
    base.py uses:                  v_cam = R @ v_world + cam_trans
    Therefore cam_trans = -R @ T.
    """
    with open(cam_json_path) as f:
        cam = json.load(f)
    intr = cam['intrinsics_wo_distortion']
    focal = np.array([float(intr['f'][0]), float(intr['f'][1])], dtype=np.float32)
    princpt = np.array([float(intr['c'][0]), float(intr['c'][1])], dtype=np.float32)
    R = np.array(cam['extrinsics']['R'], dtype=np.float32)              # (3, 3)
    T = np.array(cam['extrinsics']['T'], dtype=np.float32).reshape(3)  # camera center in world
    cam_trans = -(R @ T)                                                # additive translation
    return focal, princpt, R, cam_trans


def load_actor_data(seq_amass_dir):
    """
    Load per-actor data for one sequence, mirroring cisyn's process_sequence logic.

    Steps for each personXX.npz:
      - Load raw AMASS npz; normalise field names to cisyn convention:
          root_orient  <- poses[:, :3]
          pose_body    <- poses[:, 3:66]   (21 SMPL-X body joints)
          betas        <- betas (10,) or betas_all (N,10)
      - If personXX_smpl.npz exists, overwrite root_orient / pose_body / betas and
        store _smpl.transl as pelvis_pos_smpl (required by smplx_to_smpl).
      - n_frames from raw trans.shape[0].

    Returns list of (npz_path, data_dict, n_frames).
    """
    # Raw AMASS files: personXX.npz, exclude _smpl variants
    raw_files = sorted(
        f for f in glob.glob(os.path.join(seq_amass_dir, 'person*.npz'))
        if '_smpl' not in os.path.basename(f)
    )

    actor_data = []
    for npz_path in raw_files:
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
            n_frames = data['trans'].shape[0]

            # Normalise raw AMASS fields to cisyn-style keys
            data['root_orient'] = data['poses'][:, :3]          # (N, 3)
            data['pose_body'] = data['poses'][:, 3:66]          # (N, 63) — 21 SMPL-X joints
            # Prefer per-frame betas_all; fall back to global betas broadcast
            if 'betas_all' in data:
                data['betas'] = data['betas_all']               # (N, 10)
            # else data['betas'] is already (10,) — smplx_to_smpl handles both shapes

            # Merge _smpl.npz overrides if available
            smpl_path = npz_path.replace('.npz', '_smpl.npz')
            if os.path.isfile(smpl_path):
                ds = dict(np.load(smpl_path, allow_pickle=True))
                data['root_orient'] = ds['root_orient']         # better root orient
                data['pose_body'] = ds['body_pose']             # better body pose (may be 69-dim)
                data['betas'] = ds['betas']                     # better betas
                data['pelvis_pos_smpl'] = ds['transl']          # _smpl.transl = pelvis offset

            actor_data.append((npz_path, data, n_frames))
        except Exception as e:
            print(f'Warning: could not load {npz_path}: {e}')
            FAIL_LIST.append(npz_path)
            continue

    return actor_data


def process_scene(scene_path, scene_name, annots, max_seqs=None):
    """Process one scene (s02/s03/s04) and populate annots."""
    amass_root = os.path.join(scene_path, 'smplx_amass')
    if not os.path.isdir(amass_root):
        return 0

    seq_names = sorted(os.listdir(amass_root))
    if max_seqs is not None:
        seq_names = seq_names[:max_seqs]
    cam_ids = sorted(os.listdir(os.path.join(scene_path, 'camera_parameters')))

    count = 0
    for seq_name in tqdm(seq_names, desc=scene_name, ncols=80):
        seq_amass_dir = os.path.join(amass_root, seq_name)
        actor_data = load_actor_data(seq_amass_dir)
        if not actor_data:
            continue

        n_frames = min(ad[2] for ad in actor_data)

        for cam_id in cam_ids:
            cam_json = os.path.join(scene_path, 'camera_parameters', cam_id, f'{seq_name}.json')
            img_dir = os.path.join(scene_path, 'images', cam_id, seq_name)
            if not os.path.isfile(cam_json) or not os.path.isdir(img_dir):
                continue

            focal, princpt, R, T = load_camera_params(cam_json)

            for frame_idx in range(n_frames):
                # Images are 1-indexed: SMPL frame 0 -> 000001.png
                img_name = f'{frame_idx + 1:06d}.png'
                img_path = os.path.join(img_dir, img_name)
                if not os.path.isfile(img_path):
                    continue

                # Key relative to data/chi3d/
                annot_key = os.path.join('train', scene_name, 'images', cam_id, seq_name, img_name)

                person_list = []
                for npz_path, data, _ in actor_data:
                    try:
                        global_orient, body_pose, transl, betas = smplx_to_smpl(
                            data, frame_idx, npz_path)
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
                            'genders': gender,
                        })
                    except Exception as e:
                        FAIL_LIST.append(npz_path)
                        continue

                if person_list:
                    annots[annot_key] = person_list
                    count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data',
                        help='Data root containing chi3d/')
    parser.add_argument('--max_seqs', type=int, default=None,
                        help='Max sequences per scene to process (for testing)')
    args = parser.parse_args()

    chi3d_root = os.path.join(args.data_root, 'chi3d')
    train_root = os.path.join(chi3d_root, 'train')
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f'CHI3D train root not found: {train_root}')

    scenes = sorted([d for d in os.listdir(train_root)
                     if os.path.isdir(os.path.join(train_root, d))])
    if args.max_seqs is not None:
        scenes = scenes[:1]
    print(f'Found scenes: {scenes}')

    annots = {}
    total = 0
    for scene_name in scenes:
        scene_path = os.path.join(train_root, scene_name)
        n = process_scene(scene_path, scene_name, annots, max_seqs=args.max_seqs)
        print(f'  {scene_name}: {n} annotated frames')
        total += n

    out_path = os.path.join(chi3d_root, 'chi3d_smpl_train.npz')
    np.savez(out_path, annots=annots)
    print(f'Saved {total} image annotations to {out_path}')
    if FAIL_LIST:
        print(f'Failed {len(FAIL_LIST)} actor loads:')
        for p in FAIL_LIST:
            print(f'  {p}')


if __name__ == '__main__':
    main()
