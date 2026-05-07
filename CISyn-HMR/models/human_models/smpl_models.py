import torch
from torch import nn
import smplx
import numpy as np
import pickle
import os.path as osp
from configs.paths import smpl_model_path


class SMPL_Layer(nn.Module):
    def __init__(self, model_path, with_genders = True, **kwargs):
        """
        Extension of the SMPL Layer with gendered inputs.
        """
        super().__init__()
        smpl_kwargs = {'create_global_orient': False, 'create_body_pose': False, 
                        'create_betas': False, 'create_transl': False}
        smpl_kwargs.update(kwargs)
        self.with_genders = with_genders
        if self.with_genders:
            self.layer_n = smplx.create(model_path, 'smpl', gender='neutral', **smpl_kwargs)
            self.layer_m = smplx.create(model_path, 'smpl', gender='male', **smpl_kwargs)
            self.layer_f = smplx.create(model_path, 'smpl', gender='female', **smpl_kwargs) 
            self.layers = {'neutral': self.layer_n, 'male': self.layer_m, 'female': self.layer_f}
        else:
            self.layer_n = smplx.create(model_path, 'smpl', gender='neutral', **smpl_kwargs)
            self.layers = {'neutral': self.layer_n}
        
        self.vertex_num = 6890
        self.faces = self.layer_n.faces

        self.body_vertex_idx = np.load(osp.join(model_path, 'smpl', 'body_verts_smpl.npy'))
        self.smpl2h36m_regressor = np.load(osp.join(model_path, 'smpl', 'J_regressor_h36m_correct.npy'))
        self.J_regressor_extra = np.load(osp.join(model_path, 'smpl', 'J_regressor_extra.npy'))


    def forward_single_gender(self, poses, betas, gender='neutral'):
        bs = poses.shape[0]
        if poses.ndim == 2:
            poses = poses.view(bs, -1, 3)

        assert poses.shape[1] == 24
        pose_params = {'global_orient': poses[:, :1, :],
                    'body_pose': poses[:, 1:, :]}
    
        smpl_output = self.layers[gender](betas=betas, **pose_params)
        return smpl_output.vertices, smpl_output.joints

    def forward(self, poses, betas, genders = None):
        bs = poses.shape[0]
        assert poses.shape[0] == betas.shape[0]
        if genders is None:
            return self.forward_single_gender(poses, betas)
        else:
            assert len(genders) == bs
            assert set(genders) <= {'male', 'female', 'neutral'}
            assert self.with_genders

            gender_groups = {'male': [], 'female': [], 'neutral': []}
            for i, g in enumerate(genders):
                gender_groups[g].append(i)

            non_empty = [g for g, idx in gender_groups.items() if idx]
            if len(non_empty) == 1:
                return self.forward_single_gender(poses, betas, gender=non_empty[0])

            vertices, joints = self.forward_single_gender(poses, betas, gender='neutral')
            for g, idx in gender_groups.items():
                if g != 'neutral' and idx:
                    vertices[idx], joints[idx] = \
                        self.forward_single_gender(poses[idx], betas[idx], gender=g)
            return vertices, joints

class SMPL_Kid_Layer(SMPL_Layer):
    def __init__(self, model_path, with_genders=True, **kwargs):
        kwargs['age'] = 'kid'
        kwargs['kid_template_path'] = osp.join(model_path, 'smpl', 'smpl_kid_template.npy')
        super().__init__(model_path, with_genders, **kwargs)

smpl_gendered = SMPL_Layer(smpl_model_path, with_genders = True)
