import genesis as gs
from genesis.options.solvers import RigidOptions, SimOptions
import numpy as np
import os
import glob
import json
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from tqdm import tqdm
import mediapy as media
from joint_params_loader import (
    SIM_JOINT_ORDER,
    SMPLX_BODY_ORDER,
    SMPLX_LEFT_HAND_ORDER,
    SMPLX_RIGHT_HAND_ORDER,
    JOINT_LIMITS_DEG,
)

# Joint order mapping: Simulation -> SMPLX (from joint_params.yaml)
SMPLX_BODY_HAND_ORDER = SMPLX_BODY_ORDER + SMPLX_LEFT_HAND_ORDER + SMPLX_RIGHT_HAND_ORDER


def create_joint_mapping(joint_list):
    """Create mapping from simulation joint order to SMPLX order"""
    mapping = []
    for smplx_joint in joint_list:
        if smplx_joint in SIM_JOINT_ORDER:
            sim_idx = SIM_JOINT_ORDER.index(smplx_joint)
            joint_rots_idx = sim_idx - 1
            mapping.append(joint_rots_idx)
        else:
            raise ValueError(f"Joint {smplx_joint} not found in simulation order")
    return np.array(mapping)


BODY_MAPPING = create_joint_mapping(SMPLX_BODY_ORDER)
LEFT_HAND_MAPPING = create_joint_mapping(SMPLX_LEFT_HAND_ORDER)
RIGHT_HAND_MAPPING = create_joint_mapping(SMPLX_RIGHT_HAND_ORDER)


def euler_to_axis_angle(euler_xyz):
    """Convert euler angles (xyz order) to axis-angle representation."""
    rot = R.from_euler('XYZ', euler_xyz, degrees=False)
    return rot.as_rotvec()


def axis_angle_to_euler(axis_angle):
    """Convert axis-angle (rotvec) to euler angles (xyz order, radians)."""
    rot = R.from_rotvec(np.asarray(axis_angle, dtype=np.float64))
    return rot.as_euler('XYZ', degrees=False)


def _euler_rotation_error(euler_xyz, R_target):
    """Frobenius norm squared of (R_from_euler - R_target). Zero iff same rotation."""
    R_from_euler = R.from_euler("XYZ", euler_xyz, degrees=False).as_matrix()
    return np.sum((R_from_euler - R_target) ** 2)


def axis_angle_to_euler_in_range(axis_angle, limits_deg, n_restarts=4):
    """
    Find Euler XYZ (radians) within joint limits that best represents the same rotation.
    Conversion is not unique; we minimize rotation error over (rx, ry, rz) in the limit box
    so that small-range axes are used less when another equivalent euler uses them more.
    limits_deg: list of 3 ranges [[x_lo, x_hi], [y_lo, y_hi], [z_lo, z_hi]] in degrees.
    """
    R_target = R.from_rotvec(np.asarray(axis_angle, dtype=np.float64)).as_matrix()
    bounds_rad = [
        (np.deg2rad(limits_deg[i][0]), np.deg2rad(limits_deg[i][1]))
        for i in range(3)
    ]
    lo = np.array([b[0] for b in bounds_rad])
    hi = np.array([b[1] for b in bounds_rad])

    def run(euler0):
        return minimize(
            _euler_rotation_error,
            euler0,
            args=(R_target,),
            bounds=bounds_rad,
            method="L-BFGS-B",
        )

    # Start from clamped standard conversion
    euler0 = np.clip(axis_angle_to_euler(axis_angle), lo, hi)
    best = run(euler0)
    # Random restarts to escape local minima (different equivalent eulers)
    for _ in range(n_restarts):
        euler_rand = lo + np.random.default_rng().uniform(size=3) * (hi - lo)
        r = run(euler_rand)
        if r.fun < best.fun:
            best = r
    return best.x.astype(np.float64)


def smplx_pose_to_sim_dofs(pose_smplx, trans=None, ori=None):
    """
    Convert SMPLX-order pose (165 dim) to simulation DOFs: trans(3) + root_euler(3) + joint_eulers.
    SMPLX layout: root_orient(3) + pose_body(63) + pose_jaw(3) + pose_eye(6) + pose_hand(90).
    """
    pose_smplx = np.asarray(pose_smplx, dtype=np.float64).flatten()
    if pose_smplx.size != 165:
        raise ValueError(f"pose_smplx must have 165 elements, got {pose_smplx.size}")
    # root_orient = pose_smplx[0:3]
    pose_body = pose_smplx[3:66]   # 21 * 3
    pose_hand = pose_smplx[75:165] # 45 + 45
    left_hand = pose_hand[:45]
    right_hand = pose_hand[45:90]

    n_sim_joints = len(SIM_JOINT_ORDER) - 1  # exclude Pelvis
    joint_eulers = np.zeros(n_sim_joints * 3, dtype=np.float64)

    for j in range(n_sim_joints):
        name = SIM_JOINT_ORDER[j + 1]
        if name in SMPLX_BODY_ORDER:
            idx = SMPLX_BODY_ORDER.index(name)
            aa = pose_body[idx * 3:(idx + 1) * 3]
        elif name in SMPLX_LEFT_HAND_ORDER:
            idx = SMPLX_LEFT_HAND_ORDER.index(name)
            aa = left_hand[idx * 3:(idx + 1) * 3]
        elif name in SMPLX_RIGHT_HAND_ORDER:
            idx = SMPLX_RIGHT_HAND_ORDER.index(name)
            aa = right_hand[idx * 3:(idx + 1) * 3]
        else:
            raise ValueError(f"Joint {name} not in SMPLX body/hand orders")
        limits_deg = JOINT_LIMITS_DEG[name]
        joint_eulers[j * 3:(j + 1) * 3] = axis_angle_to_euler_in_range(aa, limits_deg)

    root_euler = np.array(ori if ori is not None else [0, 0, 0], dtype=np.float64)
    root_euler[0] += 90
    root_euler = root_euler * np.pi / 180.0
    trans = np.zeros(3, dtype=np.float64) if trans is None else np.asarray(trans, dtype=np.float64)
    return np.concatenate([trans, root_euler, joint_eulers])


def convert_and_reorder_joints(joint_rots_euler, body_mapping, left_hand_mapping, right_hand_mapping):
    """Convert euler angles to axis-angle and reorder according to SMPLX order."""
    n_joints = len(joint_rots_euler) // 3
    joint_axis_angles = []
    for i in range(n_joints):
        euler = joint_rots_euler[i * 3:(i + 1) * 3]
        joint_axis_angles.append(euler_to_axis_angle(euler))
    joint_axis_angles = np.array(joint_axis_angles).flatten()

    body_rots = np.zeros(21 * 3)
    for smplx_idx, sim_idx in enumerate(body_mapping):
        if sim_idx < n_joints:
            body_rots[smplx_idx * 3:(smplx_idx + 1) * 3] = joint_axis_angles[sim_idx * 3:(sim_idx + 1) * 3]
    left_hand_rots = np.zeros(15 * 3)
    for smplx_idx, sim_idx in enumerate(left_hand_mapping):
        if sim_idx < n_joints:
            left_hand_rots[smplx_idx * 3:(smplx_idx + 1) * 3] = joint_axis_angles[sim_idx * 3:(sim_idx + 1) * 3]
    right_hand_rots = np.zeros(15 * 3)
    for smplx_idx, sim_idx in enumerate(right_hand_mapping):
        if sim_idx < n_joints:
            right_hand_rots[smplx_idx * 3:(smplx_idx + 1) * 3] = joint_axis_angles[sim_idx * 3:(sim_idx + 1) * 3]
    return body_rots, left_hand_rots, right_hand_rots


def process(meta, seq_dir, num_steps=480, fps_target=120.0, slowdown=4, render=True, video_fps=30):
    """
    Run Genesis simulation from meta (from seq_define_3p) and save three .npz files to seq_dir.
    If render=True, add one camera and save video at video_fps (one frame per sim step).
    """
    gs.init(backend=gs.cpu)
    sim_options = SimOptions(dt=0.002083333333, gravity=(0.0, 0.0, -9.81), requires_grad=False)
    rigid_opts = RigidOptions(
        enable_collision=True,
        enable_self_collision=True,
        max_collision_pairs=2000,
    )

    scene = gs.Scene(sim_options=sim_options, rigid_options=rigid_opts, show_viewer=False)

    humanoid1 = scene.add_entity(
        gs.morphs.MJCF(file=meta["asset1"], scale=1.0)
    )
    humanoid2 = scene.add_entity(
        gs.morphs.MJCF(file=meta["asset2"], scale=1.0)
    )
    humanoid3 = scene.add_entity(
        gs.morphs.MJCF(file=meta["asset3"], scale=1.0)
    )
    scene.add_entity(gs.morphs.Plane())

    if render:
        cam_pos = meta.get("camera", (0, -5, 2))
        camera = scene.add_camera(
            res=(1280, 720),
            pos=tuple(cam_pos),
            lookat=(0, 0, 1.5),
            fov=90,
            GUI=False,
        )

    scene.build()

    # Pelvis position when loaded at origin (for AMASS trans relative to initial)
    initial_pelvis_1 = np.array(humanoid1.get_dofs_position(), dtype=np.float64)[:3].copy()
    initial_pelvis_2 = np.array(humanoid2.get_dofs_position(), dtype=np.float64)[:3].copy()
    initial_pelvis_3 = np.array(humanoid3.get_dofs_position(), dtype=np.float64)[:3].copy()

    # Set DOFs from meta: use pose1/pose2/pose3 (SMPLX) if present, else pos/ori only
    if "pose1" in meta and "pose2" in meta and "pose3" in meta:
        dofs1 = smplx_pose_to_sim_dofs(meta["pose1"], trans=meta["pos1"], ori=meta["ori1"])
        dofs2 = smplx_pose_to_sim_dofs(meta["pose2"], trans=meta["pos2"], ori=meta["ori2"])
        dofs3 = smplx_pose_to_sim_dofs(meta["pose3"], trans=meta["pos3"], ori=meta["ori3"])
        n1, n2, n3 = humanoid1.n_dofs, humanoid2.n_dofs, humanoid3.n_dofs
        if len(dofs1) != n1:
            dofs1 = np.resize(dofs1, n1) if len(dofs1) < n1 else dofs1[:n1].copy()
        if len(dofs2) != n2:
            dofs2 = np.resize(dofs2, n2) if len(dofs2) < n2 else dofs2[:n2].copy()
        if len(dofs3) != n3:
            dofs3 = np.resize(dofs3, n3) if len(dofs3) < n3 else dofs3[:n3].copy()
        humanoid1.set_dofs_position(dofs1)
        humanoid2.set_dofs_position(dofs2)
        humanoid3.set_dofs_position(dofs3)
    else:
        dofs1 = np.array(humanoid1.get_dofs_position())
        dofs2 = np.array(humanoid2.get_dofs_position())
        dofs3 = np.array(humanoid3.get_dofs_position())
        ori1 = np.array(meta["ori1"], dtype=np.float64)
        ori2 = np.array(meta["ori2"], dtype=np.float64)
        ori3 = np.array(meta["ori3"], dtype=np.float64)
        dofs1[3:6] = ori1
        dofs2[3:6] = ori2
        dofs3[3:6] = ori3
        humanoid1.set_dofs_position(dofs1)
        humanoid2.set_dofs_position(dofs2)
        humanoid3.set_dofs_position(dofs3)

    vel1 = np.array(meta["vel1"], dtype=np.float64)
    vel2 = np.array(meta["vel2"], dtype=np.float64)
    vel3 = np.array(meta["vel3"], dtype=np.float64)
    angvel1 = np.array(meta["angvel1"], dtype=np.float64)
    angvel2 = np.array(meta["angvel2"], dtype=np.float64)
    angvel3 = np.array(meta["angvel3"], dtype=np.float64)
    target_vel1 = np.zeros(humanoid1.n_dofs)
    target_vel2 = np.zeros(humanoid2.n_dofs)
    target_vel3 = np.zeros(humanoid3.n_dofs)
    target_vel1[0:3] = vel1
    target_vel1[3:6] = angvel1
    target_vel2[0:3] = vel2
    target_vel2[3:6] = angvel2
    target_vel3[0:3] = vel3
    target_vel3[3:6] = angvel3
    humanoid1.set_dofs_velocity(target_vel1)
    humanoid2.set_dofs_velocity(target_vel2)
    humanoid3.set_dofs_velocity(target_vel3)

    dt = sim_options.dt
    steps_per_frame = max(1, int(1.0 / (fps_target * dt)))

    # Clipping params (read here so early-stop can use them)
    amass_start_frame = int(meta.get("amass_start_frame", 26))
    amass_step = int(meta.get("amass_step", 4))
    amass_num_frames = int(meta.get("amass_num_frames", 12))
    # Last frame index we actually need (0-based in the sampled-frame array)
    max_needed_frame_idx = amass_start_frame + (amass_num_frames - 1) * amass_step

    poses_h1, poses_h2, poses_h3 = [], [], []
    trans_h1, trans_h2, trans_h3 = [], [], []

    # Initial frame (frame index 0)
    dofs_h1 = np.array(humanoid1.get_dofs_position())
    dofs_h2 = np.array(humanoid2.get_dofs_position())
    dofs_h3 = np.array(humanoid3.get_dofs_position())
    trans_h1.append((dofs_h1[:3] - initial_pelvis_1).copy())
    trans_h2.append((dofs_h2[:3] - initial_pelvis_2).copy())
    trans_h3.append((dofs_h3[:3] - initial_pelvis_3).copy())
    root_rot_h1 = euler_to_axis_angle(dofs_h1[3:6])
    root_rot_h2 = euler_to_axis_angle(dofs_h2[3:6])
    root_rot_h3 = euler_to_axis_angle(dofs_h3[3:6])
    body_rots_h1, lh_h1, rh_h1 = convert_and_reorder_joints(dofs_h1[6:], BODY_MAPPING, LEFT_HAND_MAPPING, RIGHT_HAND_MAPPING)
    body_rots_h2, lh_h2, rh_h2 = convert_and_reorder_joints(dofs_h2[6:], BODY_MAPPING, LEFT_HAND_MAPPING, RIGHT_HAND_MAPPING)
    body_rots_h3, lh_h3, rh_h3 = convert_and_reorder_joints(dofs_h3[6:], BODY_MAPPING, LEFT_HAND_MAPPING, RIGHT_HAND_MAPPING)
    jaw_eyes = np.zeros(9)
    poses_h1.append(np.concatenate([root_rot_h1, body_rots_h1, jaw_eyes, lh_h1, rh_h1]))
    poses_h2.append(np.concatenate([root_rot_h2, body_rots_h2, jaw_eyes, lh_h2, rh_h2]))
    poses_h3.append(np.concatenate([root_rot_h3, body_rots_h3, jaw_eyes, lh_h3, rh_h3]))

    if render:
        all_frames = [camera.render()[0]]

    frame_idx = 0  # tracks sampled-frame index (initial frame = 0)
    for i in range(num_steps):
        scene.step()
        if render:
            all_frames.append(camera.render()[0])
        if (i + 1) % steps_per_frame == 0:
            frame_idx += 1
            dofs_h1 = np.array(humanoid1.get_dofs_position())
            dofs_h2 = np.array(humanoid2.get_dofs_position())
            dofs_h3 = np.array(humanoid3.get_dofs_position())
            trans_h1.append((dofs_h1[:3] - initial_pelvis_1).copy())
            trans_h2.append((dofs_h2[:3] - initial_pelvis_2).copy())
            trans_h3.append((dofs_h3[:3] - initial_pelvis_3).copy())
            root_rot_h1 = euler_to_axis_angle(dofs_h1[3:6])
            root_rot_h2 = euler_to_axis_angle(dofs_h2[3:6])
            root_rot_h3 = euler_to_axis_angle(dofs_h3[3:6])
            body_rots_h1, lh_h1, rh_h1 = convert_and_reorder_joints(dofs_h1[6:], BODY_MAPPING, LEFT_HAND_MAPPING, RIGHT_HAND_MAPPING)
            body_rots_h2, lh_h2, rh_h2 = convert_and_reorder_joints(dofs_h2[6:], BODY_MAPPING, LEFT_HAND_MAPPING, RIGHT_HAND_MAPPING)
            body_rots_h3, lh_h3, rh_h3 = convert_and_reorder_joints(dofs_h3[6:], BODY_MAPPING, LEFT_HAND_MAPPING, RIGHT_HAND_MAPPING)
            poses_h1.append(np.concatenate([root_rot_h1, body_rots_h1, jaw_eyes, lh_h1, rh_h1]))
            poses_h2.append(np.concatenate([root_rot_h2, body_rots_h2, jaw_eyes, lh_h2, rh_h2]))
            poses_h3.append(np.concatenate([root_rot_h3, body_rots_h3, jaw_eyes, lh_h3, rh_h3]))
            # Early stop: all needed frames collected
            if frame_idx >= max_needed_frame_idx:
                break

    poses_h1 = np.array(poses_h1)
    poses_h2 = np.array(poses_h2)
    poses_h3 = np.array(poses_h3)
    trans_h1 = np.array(trans_h1)
    trans_h2 = np.array(trans_h2)
    trans_h3 = np.array(trans_h3)

    # Downsample: slice the needed window from collected frames
    frame_indices = np.arange(amass_start_frame, amass_start_frame + amass_num_frames * amass_step, amass_step)
    poses_h1 = poses_h1[frame_indices]
    poses_h2 = poses_h2[frame_indices]
    poses_h3 = poses_h3[frame_indices]
    trans_h1 = trans_h1[frame_indices]
    trans_h2 = trans_h2[frame_indices]
    trans_h3 = trans_h3[frame_indices]

    mocap_time_length = len(poses_h1) / fps_target * slowdown
    frame_rate = fps_target / slowdown

    betas_paths = glob.glob(os.path.join(meta["asset1"].replace('humanoid.xml', ''), "betas-*.npy"))
    betas1 = np.load(betas_paths[0])
    gender1 = os.path.basename(betas_paths[0]).replace("betas-", "").replace(".npy", "")
    betas_paths = glob.glob(os.path.join(meta["asset2"].replace('humanoid.xml', ''), "betas-*.npy"))
    betas2 = np.load(betas_paths[0])
    gender2 = os.path.basename(betas_paths[0]).replace("betas-", "").replace(".npy", "")
    betas_paths = glob.glob(os.path.join(meta["asset3"].replace('humanoid.xml', ''), "betas-*.npy"))
    betas3 = np.load(betas_paths[0])
    gender3 = os.path.basename(betas_paths[0]).replace("betas-", "").replace(".npy", "")

    asset_meta_path1 = os.path.join(meta["asset1"].replace('humanoid.xml', ''), "meta_smplx.npz")
    asset_meta_path2 = os.path.join(meta["asset2"].replace('humanoid.xml', ''), "meta_smplx.npz")
    asset_meta_path3 = os.path.join(meta["asset3"].replace('humanoid.xml', ''), "meta_smplx.npz")
    asset_meta1 = np.load(asset_meta_path1, allow_pickle=True)
    asset_meta2 = np.load(asset_meta_path2, allow_pickle=True)
    asset_meta3 = np.load(asset_meta_path3, allow_pickle=True)
    root_pos1 = asset_meta1["smplx"].item()["root_location_t0"]
    root_pos2 = asset_meta2["smplx"].item()["root_location_t0"]
    root_pos3 = asset_meta3["smplx"].item()["root_location_t0"]
    pelvis_pos1 = asset_meta1["smplx"].item()["pelvis_location_t0"]
    pelvis_pos2 = asset_meta2["smplx"].item()["pelvis_location_t0"]
    pelvis_pos3 = asset_meta3["smplx"].item()["pelvis_location_t0"]

    def make_amass(poses, trans, gender_str, betas_arr, root_pos, pelvis_pos):
        root_orient = poses[:, 0:3].astype(np.float64)
        pose_body = poses[:, 3:66].astype(np.float64)
        pose_jaw = poses[:, 66:69].astype(np.float64)
        pose_eye = poses[:, 69:75].astype(np.float64)
        pose_hand = poses[:, 75:165].astype(np.float64)
        return {
            "gender": np.array(gender_str, dtype="<U7"),
            "surface_model_type": np.array("smplx", dtype="<U5"),
            "mocap_frame_rate": np.array(frame_rate, dtype=np.float64),
            "mocap_time_length": np.array(mocap_time_length, dtype=np.float64),
            "trans": trans.astype(np.float64),
            "poses": poses.astype(np.float64),
            "root_orient": root_orient,
            "pose_body": pose_body,
            "pose_hand": pose_hand,
            "pose_jaw": pose_jaw,
            "pose_eye": pose_eye,
            "betas": betas_arr.astype(np.float32),
            "num_betas": np.array(betas_arr.shape[0], dtype=np.int64),
            "root_location_t0": root_pos,
            "pelvis_location_t0": pelvis_pos,
        }

    np.savez(os.path.join(seq_dir, f"{meta['asset1'].split('/')[-2]}.npz"), **make_amass(poses_h1, trans_h1, gender1, betas1, root_pos1, pelvis_pos1))
    np.savez(os.path.join(seq_dir, f"{meta['asset2'].split('/')[-2]}.npz"), **make_amass(poses_h2, trans_h2, gender2, betas2, root_pos2, pelvis_pos2))
    np.savez(os.path.join(seq_dir, f"{meta['asset3'].split('/')[-2]}.npz"), **make_amass(poses_h3, trans_h3, gender3, betas3, root_pos3, pelvis_pos3))

    if render:
        os.makedirs(seq_dir, exist_ok=True)
        media.write_video(os.path.join(seq_dir, "video.mp4"), all_frames, fps=video_fps)

    gs.destroy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Genesis simulation for a single sequence folder (3-person).")
    parser.add_argument("--seq-dir", required=True, help="Path to sequence folder containing meta.json")
    args = parser.parse_args()

    seq_dir = args.seq_dir
    meta = json.load(open(os.path.join(seq_dir, "meta.json")))
    process(meta, seq_dir, render=False)
