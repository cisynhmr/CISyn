"""
Render a simulated sequence with xrfeitoria.
Expects simulate.py to have run first; meta (from seq_define) and results live in seq_dir.
"""
import os
import subprocess
from pathlib import Path
from typing import Union

import xrfeitoria as xf
from xrfeitoria.data_structure.models import RenderPass
from xrfeitoria.utils.anim import load_amass_motion


def _asset_folder(asset_path: str) -> str:
    """From meta['asset1'] / meta['asset2'] (e.g. 'data/0000001/humanoid.xml') return folder name (e.g. '0000001')."""
    return asset_path.rstrip("/").split("/")[-2]


def _npz_names(meta: dict) -> tuple:
    a1 = _asset_folder(meta["asset1"])
    a2 = _asset_folder(meta["asset2"])
    if a1 == a2:
        return f"{a1}_1.npz", f"{a2}_2.npz"
    return f"{a1}.npz", f"{a2}.npz"


def _actor_fbx_path(asset_path: str, root_dir: str, fbx_name: str = "SMPL-XL-baked.fbx") -> Path:
    """Resolve actor FBX path: root_dir / asset_path with humanoid.xml replaced by fbx_name."""
    base = asset_path.replace("humanoid.xml", "").rstrip("/")
    return Path(root_dir) / base / fbx_name


def _hdri_path(hdri_rel: str, root_dir: str) -> Path:
    """Resolve HDR path: root_dir / hdri_rel (e.g. 'data/.../foo.HDR')."""
    return Path(root_dir) / hdri_rel


def process(
    meta: dict,
    seq_dir: Union[str, Path],
    *,
    root_dir: Union[str, Path] = ".",
    resolution: tuple[int, int] = (1280, 720),
    video_fps: int = 30,
    seq_name: str = "seq",
    background: bool = True,
    encode_video: bool = True,
):
    """
    Render the sequence for one seq folder using meta (from seq_define) and motion npz from simulate.py.

    meta: dict from seq_define (camera, asset1, asset2, hdri, ...). simulate.py must have been run
          so that seq_dir contains {asset_folder1}.npz and {asset_folder2}.npz (and optionally meta.json).
    seq_dir: folder where simulate.py wrote meta.json and the two .npz motion files.
    root_dir: project root to resolve meta['asset1'], meta['asset2'], meta['hdri'] into absolute paths.
    resolution: render resolution (width, height).
    video_fps: fps for the output video.
    seq_name: internal sequence name for xrfeitoria.
    background: run Blender in background.
    encode_video: run ffmpeg to produce video from rendered frames.
    """
    seq_dir = Path(seq_dir)
    root_dir = Path(root_dir)

    if "resolution" in meta:
        w, h = meta["resolution"]
        resolution = (w, h)

    npz_name1, npz_name2 = _npz_names(meta)
    npz_path1 = seq_dir / npz_name1
    npz_path2 = seq_dir / npz_name2
    if not npz_path1.exists():
        raise FileNotFoundError(f"Motion npz not found: {npz_path1}. Run simulate.py first.")
    if not npz_path2.exists():
        raise FileNotFoundError(f"Motion npz not found: {npz_path2}. Run simulate.py first.")

    actor_path1 = _actor_fbx_path(meta["asset1"], root_dir)
    actor_path2 = _actor_fbx_path(meta["asset2"], root_dir)
    if not actor_path1.exists():
        raise FileNotFoundError(f"Actor FBX not found: {actor_path1}")
    if not actor_path2.exists():
        raise FileNotFoundError(f"Actor FBX not found: {actor_path2}")

    hdr_path = _hdri_path(meta["hdri"], root_dir)
    if not hdr_path.exists():
        raise FileNotFoundError(f"HDR map not found: {hdr_path}")

    xf_runner = xf.init_blender(new_process=True, background=background)
    xf_runner.utils.set_hdr_map(hdr_map_path=str(hdr_path))

    DEBUG = False
    insert_rest_pose = False
    if DEBUG:
        insert_rest_pose = True
    motion1 = load_amass_motion(str(npz_path1), insert_rest_pose=insert_rest_pose)
    motion2 = load_amass_motion(str(npz_path2), insert_rest_pose=insert_rest_pose)
    motion_data1 = motion1.get_motion_data()
    motion_data2 = motion2.get_motion_data()
    n_frames = len(motion_data1)
    if len(motion_data2) != n_frames:
        n_frames = min(len(motion_data1), len(motion_data2))
        motion_data1 = motion_data1[:n_frames]
        motion_data2 = motion_data2[:n_frames]

    with xf_runner.sequence(seq_name=seq_name, seq_length=n_frames) as seq:
        actor1 = xf_runner.Actor.import_from_file(
            file_path=str(actor_path1),
            stencil_value=1,
        )
        actor2 = xf_runner.Actor.import_from_file(
            file_path=str(actor_path2),
            stencil_value=2,
        )

        xf_runner.utils.apply_motion_data_to_actor(
            motion_data=motion_data1,
            actor_name=actor1.name,
            is_first_frame_as_origin=False,
        )
        xf_runner.utils.apply_motion_data_to_actor(
            motion_data=motion_data2,
            actor_name=actor2.name,
            is_first_frame_as_origin=False,
        )

        cam_pos = meta.get("camera", (0, -5, 1.5))
        fov = meta.get("fov", 90)
        lookat = meta.get("lookat", (0, 0, 1.5))
        camera = xf_runner.Camera.spawn(location=tuple(cam_pos), fov=fov)
        camera.look_at(target=tuple(lookat))

        if DEBUG:
            xf_runner.utils.save_blend(save_path=f'{seq_dir}/test.blend')
            xf_runner.close()
            return

        output_path = str(seq_dir.parent)

        seq.add_to_renderer(
            output_path=output_path,
            resolution=resolution,
            render_passes=[
                RenderPass("img", "png"),
                RenderPass("mask", "png"),
                RenderPass("depth", "exr"),
            ],
        )

    xf_runner.render()
    xf_runner.close()

    if encode_video:
        # xrfeitoria writes to output_path / seq_name / img / XF-camera-001 / %04d.png
        img_dir = seq_dir / "img" / "XF-camera-001"
        video_out = seq_dir / "video.mp4"
        if img_dir.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(video_fps // 6),
                    "-i",
                    str(img_dir / "%04d.png"),
                    "-c:v",
                    "libx264",
                    str(video_out),
                ],
                check=False,
            )

    return str(output_path)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Render a single sequence folder with xrfeitoria.")
    parser.add_argument("--seq-dir", required=True, help="Path to sequence folder containing meta.json and motion npz files")
    parser.add_argument("--root-dir", default=str(Path(__file__).parent),
                        help="Project root for resolving asset/HDR paths from meta.json")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    meta = json.loads((seq_dir / "meta.json").read_text())
    process(meta, seq_dir, root_dir=args.root_dir, seq_name=seq_dir.name)
