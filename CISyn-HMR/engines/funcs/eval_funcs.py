import os
import cv2
import torch
import pickle
import zipfile
import datetime
import time
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from utils.transforms import unNormalize
from utils.constants import H36M_EVAL_JOINTS
from utils.box_ops import box_cxcywh_to_xyxy
from utils.visualization import tensor_to_BGR, pad_img
from utils.visualization import vis_meshes_img, vis_boxes, vis_sat, vis_scale_img, get_colors_rgb, BASE_COLORS
from utils.render import render_side_views, render_meshes
from utils.evaluation import cal_3d_position_error, match_2d_greedy, get_matching_dict, compute_prf1, select_and_align, vectorize_distance
from models.criterion import compute_interpen_loss, _save_debug_ply


def optimize_interpen(smpl_layer, pred_poses, pred_betas, pred_transl, faces,
                      n_iters=20, lr=1e-3):
    """Test-time optimization: minimize interpenetration by refining poses + transl.

    Args:
        smpl_layer: frozen SMPL nn.Module (gradients flow through its forward).
        pred_poses:  (n, 72) axis-angle poses on device.
        pred_betas:  (n, 10) shape params, held fixed.
        pred_transl: (n, 3) 3-D translation on device.
        faces:       (F, 3) long tensor on same device.
        n_iters:     max Adam steps.
        lr:          Adam learning rate.
    Returns:
        Tuple of optimized (poses, transl) as detached tensors, same shapes as input.
    """
    poses_opt  = pred_poses.clone().detach().requires_grad_(True)
    transl_opt = pred_transl.clone().detach().requires_grad_(True)
    betas      = pred_betas.clone().detach()

    optimizer = torch.optim.Adam([poses_opt, transl_opt], lr=lr)
    for _ in range(n_iters):
        optimizer.zero_grad()
        verts, _ = smpl_layer(poses=poses_opt, betas=betas)   # (n, V, 3) local
        verts_cam = verts + transl_opt[:, None, :]
        loss = compute_interpen_loss(verts_cam, faces)
        # print(loss.item(), end=' ')
        if loss.item() < 1e-6:
            break
        loss.backward()
        nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in [poses_opt, transl_opt]
        )
        if nan_grad:
            print("[WARNING] NaN/Inf gradient detected — skipping optimizer step")
            optimizer.zero_grad()
        else:
            optimizer.step()
    return poses_opt.detach(), transl_opt.detach()

# Modified from agora_evaluation
def evaluate_agora(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None, **kwargs):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes

    has_kid = ('train' in eval_dataloader.dataset.split and eval_dataloader.dataset.ds_name == 'agora')
    
    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0
    total_miss_count = 0
    total_count = 0
    total_fp = 0
    mve, mpjpe = [0.], [0.]
    # inference timing (ms) for the model forward call in agora
    inference_times = []

    if has_kid:
        kid_total_miss_count = 0
        kid_total_count = 0
        kid_mve, kid_mpjpe = [0.], [0.]

    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    body_verts_ind = smpl_layer.body_vertex_idx
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        # time only the model forward call
        if cur_device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(samples, targets)
        if cur_device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        # record milliseconds
        inference_times.append((t1 - t0) * 1000.0)
        bs = len(targets)
        for idx in range(bs):
            #gt
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]
            gt_j3ds = targets[idx]['j3ds'].cpu().numpy()[:,:24,:]
            gt_verts = targets[idx]['verts'].cpu().numpy()

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_j3ds = outputs['pred_j3ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach().cpu().numpy()


            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            #matching
            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds) # tuples are (idx_pred_kps, idx_gt_kps)
            matchDict, falsePositive_count = get_matching_dict(greedy_match)

            #align with matching result
            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            gtIdxs = np.arange(len(gt_j3ds))
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(
                        gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                else:
                    miss_flag.append(0)
                    pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                    pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                    matched_verts_idx.append(matchDict[str(gtIdx)])

            if has_kid:
                gt_kid_list = targets[idx]['kid']

            #calculating 3d errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                total_count += 1
                if has_kid and gt_kid_list[i]:
                    kid_total_count += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    total_miss_count += 1
                    if has_kid and gt_kid_list[i]:
                        kid_total_miss_count += 1
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)
                
                gt3d, gt3d_verts = select_and_align(gt3d, gt3d_verts, body_verts_ind)
                pred3d, pred3d_verts = select_and_align(pred3d, pred3d_verts, body_verts_ind)

                #joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                mpjpe.append(error_j)
                if has_kid and gt_kid_list[i]:
                    kid_mpjpe.append(error_j)
                #vertices
                error_v,pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                mve.append(error_v)
                if has_kid and gt_kid_list[i]:
                    kid_mve.append(error_v)


            #counting
            step += 1
            total_fp += falsePositive_count

            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            
            if vis and (img_idx%vis_step == 0):
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())

                # render mesh
                colors = [(1.0, 1.0, 0.9)] * len(gt_verts)
                gt_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = gt_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = targets[idx]['cam_intrinsics'].reshape(3,3).detach().cpu(),
                                            colors = colors)

                colors = [(1.0, 0.6, 0.6)] * len(pred_verts)   
                for i in matched_verts_idx:
                    colors[i] = (0.7, 1.0, 0.4)

                # colors = get_colors_rgb(len(pred_verts))
                pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                                            verts = pred_verts,
                                            smpl_faces = smpl_layer.faces,
                                            cam_intrinsics = outputs['pred_intrinsics'][idx].reshape(3,3).detach().cpu(),
                                            colors = colors,
                                            )


                if 'enc_outputs' not in outputs:
                    pred_scale_img = np.zeros_like(pred_mesh_img)
                else:
                    enc_out = outputs['enc_outputs']
                    h, w = enc_out['hw'][idx]
                    flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    scale_map = torch.zeros((h,w,2))
                    scale_map[ys,xs] = flatten_map

                    pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                   scale_map = scale_map,
                                                   conf_thresh = model.sat_cfg['conf_thresh'],
                                                   patch_size=28)

                pred_boxes = outputs['pred_boxes'][idx][select_queries_idx].detach().cpu()
                pred_boxes = box_cxcywh_to_xyxy(pred_boxes) * model.input_size
                pred_box_img = vis_boxes(ori_img.copy(), pred_boxes, color = (255,0,255))

                # sat
                sat_img = vis_sat(ori_img.copy(),
                                    input_size = model.input_size,
                                    patch_size = 14,
                                    sat_dict = outputs['sat'],
                                    bid = idx)

                ori_img = pad_img(ori_img, model.input_size)

                full_img = np.vstack([np.hstack([ori_img, sat_img]),
                                      np.hstack([pred_scale_img, pred_box_img]),
                                      np.hstack([gt_mesh_img, pred_mesh_img])])

                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)
                
        progress_bar.update(1)

    if distributed:
        mve = accelerator.gather_for_metrics(mve)
        mpjpe = accelerator.gather_for_metrics(mpjpe)


        total_miss_count = sum(accelerator.gather_for_metrics([total_miss_count]))
        total_count = sum(accelerator.gather_for_metrics([total_count]))
        total_fp = sum(accelerator.gather_for_metrics([total_fp]))

        if has_kid:
            kid_mve = accelerator.gather_for_metrics(kid_mve)
            kid_mpjpe = accelerator.gather_for_metrics(kid_mpjpe)
            kid_total_miss_count = sum(accelerator.gather_for_metrics([kid_total_miss_count]))
            kid_total_count = sum(accelerator.gather_for_metrics([kid_total_count]))

    if len(mpjpe) <= num_processes:
        return "Failed to evaluate. Keep training!"
    if has_kid and len(kid_mpjpe) <= num_processes:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['total_miss_count'] = total_miss_count
    error_dict['precision'] = precision
    error_dict['recall'] = recall
    error_dict['f1'] = f1

    error_dict['MPJPE'] = round(float(sum(mpjpe)/(len(mpjpe)-num_processes)), 1)
    error_dict['NMJE'] = round(error_dict['MPJPE'] / (f1), 1)
    error_dict['MVE'] = round(float(sum(mve)/(len(mve)-num_processes)), 1)
    error_dict['NMVE'] = round(error_dict['MVE'] / (f1), 1)

    if has_kid:
        kid_precision, kid_recall, kid_f1 = compute_prf1(kid_total_count,kid_total_miss_count,total_fp)
        error_dict['kid_precision'] = kid_precision
        error_dict['kid_recall'] = kid_recall
        error_dict['kid_f1'] = kid_f1

        error_dict['kid-MPJPE'] = round(float(sum(kid_mpjpe)/(len(kid_mpjpe)-num_processes)), 1)
        error_dict['kid-NMJE'] = round(error_dict['kid-MPJPE'] / (kid_f1), 1)
        error_dict['kid-MVE'] = round(float(sum(kid_mve)/(len(kid_mve)-num_processes)), 1)
        error_dict['kid-NMVE'] = round(error_dict['kid-MVE'] / (kid_f1), 1)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')

    return error_dict


def test_agora(model, eval_dataloader, conf_thresh, 
                vis = True, vis_step = 400, results_save_path = None,
                distributed = False, accelerator = None, **kwargs):
    assert results_save_path is not None
    assert accelerator is not None

    os.makedirs(os.path.join(results_save_path,'predictions'),exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    step = 0
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    
    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=100)
    progress_bar.set_description('testing')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]
        with torch.no_grad():    
           outputs = model(samples, targets)
        bs = len(targets)
        for idx in range(bs):
            #gt
            img_name = targets[idx]['img_name'].split('.')[0]
            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_j2ds = np.array(outputs['pred_j2ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]*(3840/model.input_size)
            pred_j3ds = np.array(outputs['pred_j3ds'][idx][select_queries_idx].detach().to('cpu'))[:,:24,:]
            pred_verts = np.array(outputs['pred_verts'][idx][select_queries_idx].detach().to('cpu'))
            pred_poses = np.array(outputs['pred_poses'][idx][select_queries_idx].detach().to('cpu'))
            pred_betas = np.array(outputs['pred_betas'][idx][select_queries_idx].detach().to('cpu'))

            #visualization
            step+=1
            img_idx = step + accelerator.process_index*len(eval_dataloader)*bs
            if vis and (img_idx%vis_step == 0):
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)

                sat_img = vis_sat(ori_img.copy(),
                                    input_size = model.input_size,
                                    patch_size = 14,
                                    sat_dict = outputs['sat'],
                                    bid = idx)
                
                colors = get_colors_rgb(len(pred_verts))
                mesh_img = vis_meshes_img(img = ori_img.copy(),
                                          verts = pred_verts,
                                          smpl_faces = smpl_layer.faces,
                                          colors = colors,
                                          cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                
                if 'enc_outputs' not in outputs:
                    pred_scale_img = np.zeros_like(ori_img)
                else:
                    enc_out = outputs['enc_outputs']
                    h, w = enc_out['hw'][idx]
                    flatten_map = enc_out['scale_map'].split(enc_out['lens'])[idx].detach().cpu()

                    ys = enc_out['pos_y'].split(enc_out['lens'])[idx]
                    xs = enc_out['pos_x'].split(enc_out['lens'])[idx]
                    scale_map = torch.zeros((h,w,2))
                    scale_map[ys,xs] = flatten_map
                    pred_scale_img = vis_scale_img(img = ori_img.copy(),
                                                   scale_map = scale_map,
                                                   conf_thresh = model.sat_cfg['conf_thresh'],
                                                   patch_size=28)

                full_img = np.vstack([np.hstack([ori_img, mesh_img]),
                                      np.hstack([pred_scale_img, sat_img])])
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)

            
            # submit
            for pnum in range(len(pred_j2ds)):
                smpl_dict = {}
                # smpl_dict['age'] = 'kid'
                smpl_dict['joints'] = pred_j2ds[pnum].reshape(24,2)
                smpl_dict['params'] = {'transl': np.zeros((1,3)),
                                        'betas': pred_betas[pnum].reshape(1,10),
                                        'global_orient': pred_poses[pnum][:3].reshape(1,1,3),
                                        'body_pose': pred_poses[pnum][3:].reshape(1,23,3)}
                # smpl_dict['verts'] = pred_verts[pnum].reshape(6890,3)
                # smpl_dict['allSmplJoints3d'] = pred_j3ds[pnum].reshape(24,3)
                with open(os.path.join(results_save_path,'predictions',f'{img_name}_personId_{pnum}.pkl'), 'wb') as f:
                    pickle.dump(smpl_dict, f)
 
        progress_bar.update(1)

    accelerator.print('Packing...')

    folder_path = os.path.join(results_save_path,'predictions')
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_save_path,f'pred_{timestamp}.zip')
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)


    return 'Results saved at: ' + os.path.join(results_save_path,'predictions')

def evaluate_3dpw(model, eval_dataloader, conf_thresh,
                        vis = True, vis_step = 40, results_save_path = None,
                        distributed = False, accelerator = None,
                        do_tto = False, tto_iters = 20, tto_lr = 3e-4):
    assert results_save_path is not None
    assert accelerator is not None
    num_processes = accelerator.num_processes
    
    os.makedirs(results_save_path,exist_ok=True)
    if vis:
        imgs_save_dir = os.path.join(results_save_path, 'imgs')
        os.makedirs(imgs_save_dir, exist_ok = True)
    
    step = 0
    total_miss_count = 0
    miss_ids = ['dummy']
    total_count = 0
    total_fp = 0

    mve, mpjpe, pa_mpjpe, pa_mve = [0.], [0.], [0.], [0.]
    # focal statistics: collect per-image predicted focal (first person), gt focal (first person), and absolute error
    pred_focals = []
    gt_focals = []
    focal_errors = []
    # height statistics: collect predicted and gt heights and absolute errors (per matched person)
    pred_heights_vals = []
    gt_heights_vals = []
    height_errors = []
    # inference timing (ms) for the model forward call: outputs = model(samples, targets)
    inference_times = []
    # per-sample quality metric: mean MPJPE (mm) per image, with a miss penalty for undetected persons.
    # lower is better.
    MISS_PENALTY_MM = 300.0

    # load select.json for targeted visualization (if present)
    select_json_path = os.path.join(results_save_path, 'select.json')
    if os.path.exists(select_json_path):
        import json as _json
        with open(select_json_path) as _f:
            select_idxs = set(_json.load(_f))
    else:
        select_idxs = None

    per_person_img_paths = []
    per_person_img_idxs = []
    per_person_mpjpes = []
    per_person_pa_mpjpes = []
    per_person_uncertainties = []  # pose_layer_std for each matched/missed person
    per_person_confs = []  # detection confidence for each matched/missed person
    per_person_betas = []  # predicted betas for each matched/missed person
    per_person_poses = []  # predicted smpl poses (axis-angle, 72-d incl. root orient) per matched person
    per_person_trans = []  # predicted smpl translation per matched person
    per_person_intrinsics = []  # predicted camera intrinsics (3x3) per matched person (same for all persons in an image)
    cur_device = next(model.parameters()).device
    smpl_layer = model.human_model
    smpl2h36m_regressor = torch.from_numpy(smpl_layer.smpl2h36m_regressor).float().to(cur_device)
    smpl_faces_t = torch.from_numpy(smpl_layer.faces.astype('int64')).to(cur_device)
    interpen_before = []  # mean penetration depth (m) per image before TTO
    interpen_after  = []  # mean penetration depth (m) per image after  TTO  

    progress_bar = tqdm(total=len(eval_dataloader), disable=not accelerator.is_local_main_process, ncols=80)
    progress_bar.set_description('evaluate')
    for itr, (samples, targets) in enumerate(eval_dataloader):
        samples=[sample.to(device = cur_device, non_blocking = True) for sample in samples]

        bs = len(targets)

        # batch-level skip when select.json is active: avoid model forward entirely
        if select_idxs is not None:
            offset = accelerator.process_index * len(eval_dataloader) * bs
            if not any((step + i + 1 + offset) in select_idxs for i in range(bs)):
                step += bs
                progress_bar.update(1)
                continue

        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model(samples, targets)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            # record milliseconds
            # print((t1 - t0) * 1000.0)
            inference_times.append((t1 - t0) * 1000.0)

        for idx in range(bs):
            # maintain step for all samples so img_idx stays consistent with select.json
            _img_idx = (step + 1) + accelerator.process_index * len(eval_dataloader) * bs
            step += 1
            if select_idxs is not None and _img_idx not in select_idxs:
                continue

            if targets[idx]['pnum'] == 0:
                continue

            #gt
            gt_verts = targets[idx]['verts']
            gt_transl = targets[idx]['transl']
            gt_j3ds = torch.einsum('bik,ji->bjk', [gt_verts - gt_transl[:,None,:], smpl2h36m_regressor]) + gt_transl[:,None,:]

            gt_verts = gt_verts.cpu().numpy()
            gt_heights = targets[idx]['heights'].cpu().numpy()
            gt_j3ds = gt_j3ds.cpu().numpy()
            gt_j2ds = targets[idx]['j2ds'].cpu().numpy()[:,:24,:]
            gt_focal = targets[idx]['focals'][0].cpu().numpy()

            #pred
            select_queries_idx = torch.where(outputs['pred_confs'][idx] > conf_thresh)[0]
            pred_uncertainty = outputs['pose_layer_std'][idx][select_queries_idx].detach().cpu().numpy()
            pred_confs_filtered = outputs['pred_confs'][idx][select_queries_idx].detach().cpu().numpy().squeeze(-1)

            pred_verts = outputs['pred_verts'][idx][select_queries_idx].detach()
            pred_poses = outputs['pred_poses'][idx][select_queries_idx].detach()
            pred_betas = outputs['pred_betas'][idx][select_queries_idx].detach()
            try:
                pred_verts_t = smpl_layer(betas=pred_betas, poses=pred_poses*0)[0]
                pred_heights = torch.max(pred_verts_t[:, :, 1], dim=1).values - torch.min(pred_verts_t[:, :, 1], dim=1).values
            except Exception:
                pred_heights = torch.zeros(len(pred_betas)).to(pred_betas.device)

            pred_transl = outputs['pred_transl'][idx][select_queries_idx].detach()
            pred_poses_filtered = outputs['pred_poses'][idx][select_queries_idx].detach()
            pred_intrinsics_img = outputs['pred_intrinsics'][idx][0].detach().cpu().numpy()  # (3,3)

            # save pre-TTO vertices tensor for interpenetration metric (computed after GT matching)
            pred_verts_pre_tto = pred_verts.detach().clone()

            # --- optional test-time optimisation ---
            if do_tto and len(select_queries_idx) >= 2:
                poses_tto, transl_tto = optimize_interpen(
                    smpl_layer, pred_poses_filtered, pred_betas, pred_transl,
                    smpl_faces_t, n_iters=tto_iters, lr=tto_lr)
                with torch.no_grad():
                    verts_tto, _ = smpl_layer(poses=poses_tto, betas=pred_betas)
                    pred_verts   = verts_tto + transl_tto[:, None, :]
                pred_transl         = transl_tto
                pred_poses_filtered = poses_tto

            # save post-TTO vertices tensor for interpenetration metric (computed after GT matching)
            pred_verts_post_tto = pred_verts.detach().clone()

            pred_j3ds = torch.einsum('bik,ji->bjk', [pred_verts - pred_transl[:,None,:], smpl2h36m_regressor]) + pred_transl[:,None,:]

            pred_verts = pred_verts.cpu().numpy()
            pred_j3ds = pred_j3ds.cpu().numpy()
            pred_j2ds = outputs['pred_j2ds'][idx][select_queries_idx].detach().cpu().numpy()[:,:24,:]
            pred_focal = outputs['pred_intrinsics'][idx][0, 0, 0].detach().cpu().numpy()

            pred_focals.append(pred_focal)
            gt_focals.append(gt_focal)
            focal_errors.append(abs(pred_focal - gt_focal))

            matched_verts_idx = []
            assert len(gt_j2ds.shape) == 3 and len(pred_j2ds.shape) == 3
            #matching
            greedy_match = match_2d_greedy(pred_j2ds, gt_j2ds) # tuples are (idx_pred_kps, idx_gt_kps)
            matchDict, falsePositive_count = get_matching_dict(greedy_match)

            #align with matching result
            gt_verts_list, pred_verts_list, gt_joints_list, pred_joints_list = [], [], [], []
            gt_heights_list, pred_heights_list = [], []
            gtIdxs = np.arange(len(gt_j3ds))
            miss_flag = []
            for gtIdx in gtIdxs:
                gt_verts_list.append(gt_verts[gtIdx])
                gt_joints_list.append(gt_j3ds[gtIdx])
                gt_heights_list.append(gt_heights[gtIdx])
                if matchDict[str(gtIdx)] == 'miss' or matchDict[str(
                        gtIdx)] == 'invalid':
                    miss_flag.append(1)
                    pred_verts_list.append([])
                    pred_joints_list.append([])
                    pred_heights_list.append([])
                else:
                    miss_flag.append(0)
                    pred_joints_list.append(pred_j3ds[matchDict[str(gtIdx)]])
                    pred_verts_list.append(pred_verts[matchDict[str(gtIdx)]])
                    pred_heights_list.append(pred_heights[matchDict[str(gtIdx)]].cpu().numpy())
                    matched_verts_idx.append(matchDict[str(gtIdx)])

            # --- interpenetration metric (after GT alignment) ---
            if len(matched_verts_idx) >= 2:
                matched_idx_t = torch.tensor(matched_verts_idx, device=cur_device)
                with torch.no_grad():
                    pen_b = compute_interpen_loss(pred_verts_pre_tto[matched_idx_t], smpl_faces_t)
                interpen_before.append(pen_b.item())
            else:
                interpen_before.append(0.)
            if do_tto:
                if len(matched_verts_idx) >= 2:
                    matched_idx_t = torch.tensor(matched_verts_idx, device=cur_device)
                    with torch.no_grad():
                        pen_a = compute_interpen_loss(pred_verts_post_tto[matched_idx_t], smpl_faces_t)
                    interpen_after.append(pen_a.item())
                else:
                    interpen_after.append(0.)

            #counting
            total_fp += falsePositive_count
            img_idx = _img_idx

            #calculating 3d errors
            for i, (gt3d, pred) in enumerate(zip(gt_joints_list, pred_joints_list)):
                total_count += 1

                # Get corresponding ground truth and predicted 3d joints and verts
                if miss_flag[i] == 1:
                    total_miss_count += 1
                    miss_ids.append(targets[idx]['img_path'])
                    per_person_img_paths.append(targets[idx]['img_path'])
                    per_person_img_idxs.append(img_idx)
                    per_person_mpjpes.append(MISS_PENALTY_MM)
                    per_person_pa_mpjpes.append(MISS_PENALTY_MM)
                    per_person_uncertainties.append(float('nan'))
                    per_person_confs.append(float('nan'))
                    per_person_betas.append(np.full(pred_betas.shape[-1], float('nan'), dtype=np.float32))
                    per_person_poses.append(np.full(pred_poses_filtered.shape[-1], float('nan'), dtype=np.float32))
                    per_person_trans.append(np.full(3, float('nan'), dtype=np.float32))
                    per_person_intrinsics.append(pred_intrinsics_img.astype(np.float32))
                    continue

                gt3d = gt3d.reshape(-1, 3)
                pred3d = pred.reshape(-1, 3)
                gt3d_verts = gt_verts_list[i].reshape(-1, 3)
                pred3d_verts = pred_verts_list[i].reshape(-1, 3)

                gt_h = gt_heights_list[i]
                pred_h = pred_heights_list[i]

                height_errors.append(abs(pred_h - gt_h))

                gt_pelvis = gt3d[[1, 4], :].mean(axis=0, keepdims=True)
                pred_pelvis = pred3d[[1, 4], :].mean(axis=0, keepdims=True)

                gt3d = (gt3d - gt_pelvis)[H36M_EVAL_JOINTS, :].copy()
                gt3d_verts = (gt3d_verts - gt_pelvis).copy()
                
                pred3d = (pred3d - pred_pelvis)[H36M_EVAL_JOINTS, :].copy()
                pred3d_verts = (pred3d_verts - pred_pelvis).copy()

                #joints
                error_j, pa_error_j = cal_3d_position_error(pred3d, gt3d)
                mpjpe.append(error_j)
                pa_mpjpe.append(pa_error_j)
                per_person_img_paths.append(targets[idx]['img_path'])
                per_person_img_idxs.append(img_idx)
                per_person_mpjpes.append(error_j)
                per_person_pa_mpjpes.append(pa_error_j)
                per_person_uncertainties.append(float(pred_uncertainty[matchDict[str(i)]]))
                per_person_confs.append(float(pred_confs_filtered[matchDict[str(i)]]))
                per_person_betas.append(pred_betas[matchDict[str(i)]].cpu().numpy().astype(np.float32))
                per_person_poses.append(pred_poses_filtered[matchDict[str(i)]].cpu().numpy().astype(np.float32))
                per_person_trans.append(pred_transl[matchDict[str(i)]].cpu().numpy().astype(np.float32))
                per_person_intrinsics.append(pred_intrinsics_img.astype(np.float32))
                #vertices
                error_v, pa_error_v = cal_3d_position_error(pred3d_verts, gt3d_verts)
                mve.append(error_v)
                pa_mve.append(pa_error_v)
            
            _should_vis = vis and len(matched_verts_idx) > 0 and (
                (select_idxs is not None and img_idx in select_idxs) or
                (select_idxs is None and img_idx % vis_step == 0)
            )
            if _should_vis:
                img_name = targets[idx]['img_path'].split('/')[-1].split('.')[0]
                ori_img = tensor_to_BGR(unNormalize(samples[idx]).cpu())
                ori_img = pad_img(ori_img, model.input_size)

                # prepare matched pairs (gt_idx -> pred_idx)
                matched_pairs = []
                for gtIdx in gtIdxs:
                    if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                        continue
                    matched_pairs.append((int(gtIdx), int(matchDict[str(gtIdx)])))

                # build lists of verts and translations for gt and pred
                gt_list = [gt_verts[gt_i] for gt_i, _ in matched_pairs]
                pred_list = [pred_verts[p_i] for _, p_i in matched_pairs]

                gt_trans_list = [targets[idx]['transl'][gt_i].cpu().numpy() for gt_i, _ in matched_pairs]
                pred_trans_list = [pred_transl[p_i].detach().cpu().numpy() for _, p_i in matched_pairs]

                # colors = get_colors_rgb(len(matched_verts_idx))
                colors = [BASE_COLORS[i] for i in range(len(matched_verts_idx))]
                colors_gt = colors
                colors_pred = colors

                # render projected meshes on image
                # gt_mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                              verts = gt_list,
                #                              smpl_faces = smpl_layer.faces,
                #                              colors = colors_gt,
                #                              cam_intrinsics = targets[idx]['cam_intrinsics'].detach().cpu())

                # pred_mesh_img = vis_meshes_img(img = ori_img.copy(),
                #                                verts = pred_list,
                #                                smpl_faces = smpl_layer.faces,
                #                                colors = colors_pred,
                #                                cam_intrinsics = outputs['pred_intrinsics'][idx].detach().cpu())
                
                faces_list = [smpl_layer.faces] * max(1, len(pred_list))
                K_gt = targets[idx]['cam_intrinsics'][0].detach().cpu().numpy()
                K_vis = outputs['pred_intrinsics'][idx][0].detach().cpu().numpy()

                gt_mesh_img = render_meshes(img=ori_img.copy(), 
                                            l_mesh=gt_list, 
                                            l_face=faces_list, 
                                            cam_param={'focal': np.asarray([K_gt[0,0],K_gt[1,1]]), 'princpt': np.asarray([K_gt[0,-1],K_gt[1,-1]])}, 
                                            color=colors_gt,
                                            # heights=gt_heights_list
                                            )
                pred_mesh_img = render_meshes(img=ori_img.copy(),
                                            l_mesh=pred_list, 
                                            l_face=faces_list, 
                                            cam_param={'focal': np.asarray([K_vis[0,0],K_vis[1,1]]), 'princpt': np.asarray([K_vis[0,-1],K_vis[1,-1]])}, 
                                            color=colors_pred,
                                            # heights=pred_heights_list
                                            )

                # faces list repeated

                _, pred_sideview, pred_bev = render_side_views(ori_img.copy(), colors_pred, pred_list, faces_list, pred_trans_list, K_vis)
                _, gt_sideview, gt_bev = render_side_views(ori_img.copy(), colors_gt, gt_list, faces_list, gt_trans_list, K_gt)

                # compose final visualization: top row: ori | gt_proj | pred_proj ; bottom row: pred_side | gt_side | bev comparison
                top_row = np.hstack([ori_img, gt_mesh_img, pred_mesh_img])
                bottom_row = np.hstack([pred_sideview, gt_bev, pred_bev])
                # ensure same widths: if mismatched, resize bottom_row to match top_row width
                if bottom_row.shape[1] != top_row.shape[1]:
                    # simple horizontal tiling adjustment: resize bottom_row width via padding
                    pad_w = top_row.shape[1] - bottom_row.shape[1]
                    if pad_w > 0:
                        bottom_row = np.pad(bottom_row, ((0,0),(0,pad_w),(0,0)), mode='constant', constant_values=255)

                full_img = np.vstack([top_row, bottom_row])
                # downsample=2
                # full_img = cv2.resize(full_img, (full_img.shape[1]//2, full_img.shape[0]//2))
                cv2.imwrite(os.path.join(imgs_save_dir, f'{img_idx}_{img_name}.png'), full_img)
                
        progress_bar.update(1)

    if distributed:
        mve = accelerator.gather_for_metrics(mve)
        mpjpe = accelerator.gather_for_metrics(mpjpe)
        pa_mpjpe = accelerator.gather_for_metrics(pa_mpjpe)
        pa_mve = accelerator.gather_for_metrics(pa_mve)

        total_miss_count = sum(accelerator.gather_for_metrics([total_miss_count]))
        miss_ids = accelerator.gather_for_metrics(miss_ids)
        total_count = sum(accelerator.gather_for_metrics([total_count]))
        total_fp = sum(accelerator.gather_for_metrics([total_fp]))
        # gather focal stats lists
        pred_focals = accelerator.gather_for_metrics(pred_focals)
        gt_focals = accelerator.gather_for_metrics(gt_focals)
        focal_errors = accelerator.gather_for_metrics(focal_errors)
        # gather height stats lists
        height_errors = accelerator.gather_for_metrics(height_errors)
        # gather inference times
        inference_times = accelerator.gather_for_metrics(inference_times)
        # gather per-person results
        per_person_img_paths = accelerator.gather_for_metrics(per_person_img_paths)
        per_person_img_idxs = accelerator.gather_for_metrics(per_person_img_idxs)
        per_person_mpjpes = accelerator.gather_for_metrics(per_person_mpjpes)
        per_person_pa_mpjpes = accelerator.gather_for_metrics(per_person_pa_mpjpes)
        per_person_uncertainties = accelerator.gather_for_metrics(per_person_uncertainties)
        per_person_confs = accelerator.gather_for_metrics(per_person_confs)
        per_person_betas = accelerator.gather_for_metrics(per_person_betas)
        per_person_poses = accelerator.gather_for_metrics(per_person_poses)
        per_person_trans = accelerator.gather_for_metrics(per_person_trans)
        per_person_intrinsics = accelerator.gather_for_metrics(per_person_intrinsics)
        interpen_before = accelerator.gather_for_metrics(interpen_before)
        if do_tto:
            interpen_after = accelerator.gather_for_metrics(interpen_after)

    if len(mpjpe) <= num_processes:
        return "Failed to evaluate. Keep training!"
    
    precision, recall, f1 = compute_prf1(total_count,total_miss_count,total_fp)
    error_dict = {}
    error_dict['total_miss_count'] = total_miss_count
    error_dict['recall'] = recall
    error_dict['miss_ids'] = [i for i in miss_ids if i != 'dummy']
    error_dict['MPJPE'] = round(float(sum(mpjpe)/(len(mpjpe)-num_processes)), 1)
    error_dict['PA-MPJPE'] = round(float(sum(pa_mpjpe)/(len(pa_mpjpe)-num_processes)), 1)
    error_dict['MVE'] = round(float(sum(mve)/(len(mve)-num_processes)), 1)
    error_dict['PA-MVE'] = round(float(sum(pa_mve)/(len(pa_mve)-num_processes)), 1)

    # focal statistics: compute mean absolute error, and mean/std for pred and gt focals
    # remove placeholder zeros added at initialization
    pf = np.array(pred_focals, dtype=float)
    gf = np.array(gt_focals, dtype=float)
    fe = np.array(focal_errors, dtype=float)

    focal_mae = float(np.mean(fe))
    pred_mean = float(np.mean(pf))
    pred_std = float(np.std(pf, ddof=0))
    gt_mean = float(np.mean(gf))
    gt_std = float(np.std(gf, ddof=0))

    error_dict['focal'] = round(focal_mae, 4)
    error_dict['pred_focal'] = {'mean': round(pred_mean, 4), 'std': round(pred_std, 4)}
    error_dict['gt_focal'] = {'mean': round(gt_mean, 4), 'std': round(gt_std, 4)}

    # height statistics
    error_dict['height_error'] = np.round(float(np.mean(np.array(height_errors, dtype=float))), 4)

    # inference time: compute mean across all recorded forward calls (ms)
    try:
        it = np.array(inference_times, dtype=float)
        # if list contains nested lists because of gather, flatten
        it = it.flatten()
        # remove zero or near-zero entries if any accidental placeholders
        if it.size == 0:
            mean_it = 0.0
        else:
            mean_it = float(np.mean(it))
    except Exception:
        mean_it = 0.0

    error_dict['inference_time_ms'] = round(mean_it, 4)

    # interpenetration metric (mm)
    # NOTE: gather_for_metrics may return list-/tensor-like objects depending on backend,
    # so normalize before numeric ops/comparisons.
    interpen_before_arr = np.array(interpen_before, dtype=float).reshape(-1)
    error_dict['interpen_mm'] = round(float(np.mean(interpen_before_arr)) * 1000, 4)
    error_dict['num_interpen_cases'] = int(np.sum(interpen_before_arr > 0))
    if do_tto:
        interpen_after_arr = np.array(interpen_after, dtype=float).reshape(-1)
        if interpen_after_arr.size > 0:
            error_dict['interpen_after_tto_mm'] = round(float(np.mean(interpen_after_arr)) * 1000, 4)

    if accelerator.is_main_process:
        with open(os.path.join(results_save_path,'results.txt'),'w') as f:
            for k,v in error_dict.items():
                f.write(f'{k}: {v}\n')
        betas_arr = np.array(per_person_betas, dtype=np.float32)  # (N, num_betas)
        if select_idxs is not None:
            accelerator.print('[select mode] skipping per_sample_results.npz write to avoid overwriting full-run data')
        else:
            np.savez(
                os.path.join(results_save_path, 'per_sample_results.npz'),
                img_paths=np.array(per_person_img_paths),
                img_idxs=np.array(per_person_img_idxs, dtype=np.int64),
                mpjpe=np.array(per_person_mpjpes, dtype=np.float32),
                pa_mpjpe=np.array(per_person_pa_mpjpes, dtype=np.float32),
                pose_layer_std=np.array(per_person_uncertainties, dtype=np.float32),
                pred_conf=np.array(per_person_confs, dtype=np.float32),
                pred_betas=betas_arr,
                pred_poses=np.array(per_person_poses, dtype=np.float32),   # (N, 72) axis-angle, root orient first
                pred_trans=np.array(per_person_trans, dtype=np.float32),   # (N, 3) in camera space
                pred_intrinsics=np.array(per_person_intrinsics, dtype=np.float32),  # (N, 3, 3)
            )

        # beta11 analysis
        if betas_arr.shape[-1] >= 11:
            beta11 = betas_arr[:, 10]
            valid = ~np.isnan(beta11)
            beta11_valid = beta11[valid]
            near_zero = np.abs(beta11_valid) < 0.05
            accelerator.print(
                f'[beta11] n={valid.sum()}  near-zero (<0.05): {near_zero.sum()} ({100*near_zero.mean():.1f}%)  '
                f'mean={beta11_valid.mean():.4f}  std={beta11_valid.std():.4f}  '
                f'min={beta11_valid.min():.4f}  max={beta11_valid.max():.4f}'
            )
            error_dict['beta11_near_zero_pct'] = round(float(100 * near_zero.mean()), 2)

    return error_dict