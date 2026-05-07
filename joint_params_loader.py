"""
Load joint parameters from joint_params.yaml.
Per-joint data lives under "joints"; each joint has limits_deg, stiffness, damping.
"""
import yaml
from pathlib import Path

_PARAMS_FILE = Path(__file__).parent / 'joint_params.yaml'


def load_joint_params():
    with open(_PARAMS_FILE, 'r') as f:
        return yaml.safe_load(f)


def get_joint_orders():
    p = load_joint_params()
    joints = p['joints']
    return {
        'sim': list(joints.keys()),
        'smplx_body': p['smplx_body_order'],
        'smplx_left_hand': p['smplx_left_hand_order'],
        'smplx_right_hand': p['smplx_right_hand_order'],
    }


def get_joints():
    """Return the per-joint dict: { joint_name: { limits_deg, stiffness, damping }, ... }"""
    return load_joint_params()['joints'].copy()


def get_joint_limits_deg():
    """JOINT_LIMITS_DEG keyed by full joint name (e.g. L_Hip, R_Hip)."""
    params = load_joint_params()
    return {
        name: [p['limits_deg']['x'], p['limits_deg']['y'], p['limits_deg']['z']]
        for name, p in params['joints'].items()
    }


def get_joint_stiffness_damping():
    """JOINT_STIFFNESS_DAMPING keyed by full joint name: [stiffness, damping]."""
    params = load_joint_params()
    return {
        name: [p['stiffness'], p['damping']]
        for name, p in params['joints'].items()
    }


# Backward-compatible names keyed by full joint name
_params = load_joint_params()

SIM_JOINT_ORDER = list(_params['joints'].keys())
SMPLX_BODY_ORDER = _params['smplx_body_order']
SMPLX_LEFT_HAND_ORDER = _params['smplx_left_hand_order']
SMPLX_RIGHT_HAND_ORDER = _params['smplx_right_hand_order']

JOINT_LIMITS_DEG = {
    name: [p['limits_deg']['x'], p['limits_deg']['y'], p['limits_deg']['z']]
    for name, p in _params['joints'].items()
}
JOINT_STIFFNESS_DAMPING = {
    name: [p['stiffness'], p['damping']]
    for name, p in _params['joints'].items()
}
