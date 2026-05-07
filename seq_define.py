import random
import math
import json
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Angular velocity range: ±ANGULAR_VEL_RANGE rad/s per axis
ANGULAR_VEL_RANGE = 3

SEED = 1225
random.seed(SEED)
np.random.seed(SEED)  # Rotation.random() uses NumPy's RNG

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
amass = np.load(os.path.join(_DATA_DIR, "amass.npy"))
with open(os.path.join(_DATA_DIR, "asset_list.json"), "r") as f:
    asset_list = json.load(f)
with open(os.path.join(_DATA_DIR, "hdri_list.json"), "r") as f:
    hdri_list = json.load(f)

"""
Random camera position around the collision
look at around (0, 0, 1.5), x y z both +- 0.4 meters
position range is 4-5 meters from origin for x and y, 1-2 meters for z 
random fov between 70-90 degrees
return position, lookat, fov
"""
def random_camera(indoor=False):
    if indoor:
        r = random.uniform(2.0, 3.0)
        angle = random.uniform(0, 2 * math.pi)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = 1.4
        lookat_x = random.uniform(-0.3, 0.3)
        lookat_y = random.uniform(-0.3, 0.3)
        lookat_z = random.uniform(1.3, 1.7)
        if r<2.5:
            fov = random.uniform(92, 110)
        else:
            fov = random.uniform(84, 100)
    else:
        r = random.uniform(2.7, 5.0)
        angle = random.uniform(0, 2 * math.pi)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        z = 1.4
        lookat_x = random.uniform(-0.3, 0.3)
        lookat_y = random.uniform(-0.3, 0.3)
        lookat_z = random.uniform(1.3, 1.7)
        if r<3.5:
            fov = random.uniform(72, 90)
        elif r<4.0:
            fov = random.uniform(66, 78)
        elif r<4.5:
            fov = random.uniform(62, 70)
        else:
            fov = random.uniform(60, 64)
    return (x, y, z), (lookat_x, lookat_y, lookat_z), fov




"""
Random human initial position and orientation
initial orientation: ry ~ uniform(-180, 180), rz, rx ~ randn()*20 (degrees)
initial position is randomize in the range of 2.4 meters from origin for x and y (z is not needed because it's fixed at 1.5 meters)
for human 1, random a degree "angle1" on that circle to get initial position
for human 2, the "angle2" cannot be in angle1 +- 48 degrees
return pos1, pos2, ori1, ori2
"""
def random_human():
    radius = 2.4
    z = 1.5
    angle1 = random.uniform(0, 2 * math.pi)
    while True:
        angle2 = random.uniform(0, 2 * math.pi)
        diff = abs((angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi)
        if math.degrees(diff) >= 48:
            break
    pos1 = (radius * math.cos(angle1), radius * math.sin(angle1), z)
    pos2 = (radius * math.cos(angle2), radius * math.sin(angle2), z)
    def random_euler_deg():
        ry = random.uniform(-180, 180)
        rz = np.random.randn() * 20
        rx = np.random.randn() * 20
        return (rx, ry, rz)
    ori1 = random_euler_deg()
    ori2 = random_euler_deg()
    return pos1, pos2, ori1, ori2


"""
Random velocity for the two humans.
For each human: direction from pos toward origin in xy, add random -10 to 10 deg shift (xy only);
magnitude in xy random 7.5-8.5; z velocity random 1.5-2.5.
return vel1, vel2
"""
def random_vel(pos1, pos2):
    def vel_for_pos(pos):
        px, py, _ = pos
        # direction to origin in xy
        base_angle = math.atan2(-py, -px)
        shift_deg = random.uniform(-6, 6)
        angle = base_angle + math.radians(shift_deg)
        mag_xy = random.uniform(7.5, 8.5)
        vx = mag_xy * math.cos(angle)
        vy = mag_xy * math.sin(angle)
        vz = random.uniform(1.5, 2.5)
        return (vx, vy, vz)
    return vel_for_pos(pos1), vel_for_pos(pos2)


"""
Random angular velocity (rad/s) for the two humans.
Range ±1.0 rad/s per axis (~57 deg/s), plausible for tumbling during collision.
return (wx, wy, wz) for each human.
"""
def random_angular_vel():
    # rad/s, world frame
    def one():
        return tuple(np.random.randn() * ANGULAR_VEL_RANGE for _ in range(3))
    return one(), one()


def main():

    hdri_idx = random.randint(0, len(hdri_list) - 1)
    if "HDRI2" in hdri_list[hdri_idx]:
        indoor = True
    else:
        indoor = False

    cam_pos, lookat, fov = random_camera(indoor)
    pos1, pos2, ori1, ori2 = random_human()
    vel1, vel2 = random_vel(pos1, pos2)
    angvel1, angvel2 = random_angular_vel()
    
    asset1_idx = random.randint(0, len(asset_list) - 1)
    asset2_idx = random.randint(0, len(asset_list) - 1)
    while asset1_idx == asset2_idx:
        asset2_idx = random.randint(0, len(asset_list) - 1)
    pose1_idx = random.randint(0, len(amass) - 1)
    pose2_idx = random.randint(0, len(amass) - 1)
    while pose1_idx == pose2_idx:
        pose2_idx = random.randint(0, len(amass) - 1)
    meta = {
        "camera": cam_pos,
        "lookat": lookat,
        "fov": fov,
        "pos1": pos1,
        "ori1": ori1,
        "vel1": vel1,
        "angvel1": angvel1,
        "pos2": pos2,
        "ori2": ori2,
        "vel2": vel2,
        "angvel2": angvel2,
        "asset1": f'data/{asset_list[asset1_idx]}',
        "asset2": f'data/{asset_list[asset2_idx]}',
        "hdri": f'data/{hdri_list[hdri_idx]}',
        "pose1": amass[pose1_idx].tolist(),
        "pose2": amass[pose2_idx].tolist(),
    }
    return meta


SAVE_DIR = 'cisyn'
SAVE_RANGE = [24000, 25000]

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    for i in tqdm(range(SAVE_RANGE[1])):
        meta = main()
        if i < SAVE_RANGE[0]:
            continue
        os.makedirs(os.path.join(SAVE_DIR, f"{i:06d}"), exist_ok=True)
        with open(os.path.join(SAVE_DIR, f"{i:06d}", "meta.json"), "w") as f:
            json.dump(meta, f)