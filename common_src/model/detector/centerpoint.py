import os
import tempfile
import pickle
from datetime import datetime

import numpy as np 
import torch
import torch.nn.functional as F
from collections import OrderedDict

from vod.evaluation import Evaluation
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_trransformation


import lightning as L
import torch.distributed as dist

from common_src.ops import Voxelization
from common_src.model.voxel_encoders import PillarFeatureNet
from common_src.model.middle_encoders import PointPillarsScatter
from common_src.model.backbones import SECOND
from common_src.model.necks import SECONDFPN
from common_src.model.heads import CenterHead

# Import the necessary segmentation helper for PointPainting
from common_src.model.segmentation_helpers import get_modified_deeplabv3 # Added for PointPainting

class CenterPoint(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Use config directly for save_hyperparameters to access it later via self.hparams.config
        self.save_hyperparameters(config) 
        
        self.img_shape = torch.tensor([1936 , 1216])
        self.data_root = config.get('data_root', None)
        self.class_names = config.get('class_names', None)
        self.output_dir = config.get('output_dir', None)
        self.pc_range = torch.tensor(config.get('point_cloud_range', None))
        
        # Ensure these configs are correctly retrieved from the main config
        voxel_layer_config = config.get('pts_voxel_layer', None)
        voxel_encoder_config = config.get('voxel_encoder', None)
        middle_encoder_config = config.get('middle_encoder', None)
        backbone_config = config.get('backbone', None)
        neck_config = config.get('neck', None)
        head_config = config.get('head', None)
        
        self.voxel_layer = Voxelization(**voxel_layer_config)
        self.voxel_encoder = PillarFeatureNet(**voxel_encoder_config)
        self.middle_encoder = PointPillarsScatter(**middle_encoder_config)
        self.backbone = SECOND(**backbone_config)
        self.neck = SECONDFPN(**neck_config)
        self.head = CenterHead(**head_config)
        
        self.optimizer_config = config.get('optimizer', None) # Used directly in configure_optimizers
        
        self.vod_kitti_locations = KittiLocations(root_dir = self.data_root, 
                                     output_dir= self.output_dir,
                                     frame_set_path='',
                                     pred_dir='',)
        self.inference_mode = config.get('inference_mode', 'val')
        self.save_results = config.get('save_preds_results', False)
        self.val_results_list =[]

        # --- PointPainting: Semantic Segmentation Model Setup ---
        self.segmentation_net = None
        if self.hparams.get('image_segmentation_model'): # Check if config exists
            seg_config = self.hparams.image_segmentation_model
            num_seg_classes = seg_config.get('num_classes', 4) # Default or from config
            
            self.segmentation_net = get_modified_deeplabv3(
                num_target_classes=num_seg_classes,
                freeze_backbone=True, # For inference, backbone state doesn't matter as much as weights being loaded
                use_coco_weights=False # Set to False if loading fully fine-tuned model, True if starting from COCO + custom head
            )
            
            finetuned_weights_path = seg_config.get('finetuned_weights_path', None)
            if finetuned_weights_path and os.path.exists(finetuned_weights_path):
                print(f"CenterPoint: Loading fine-tuned segmentation weights from {finetuned_weights_path}")
                try:
                    checkpoint = torch.load(finetuned_weights_path, map_location=self.device,weights_only=False)
                    if 'state_dict' in checkpoint: # Common for Lightning checkpoints
                        seg_state_dict = {k.replace('model.', '', 1): v 
                                          for k, v in checkpoint['state_dict'].items() 
                                          if k.startswith('model.')}
                        if not seg_state_dict: # If no 'model.' prefix, try loading all
                             seg_state_dict = checkpoint['state_dict']
                    else: # Assume it's a raw model state_dict
                        seg_state_dict = checkpoint
                    
                    self.segmentation_net.load_state_dict(seg_state_dict, strict=True) 
                except Exception as e:
                    print(f"Error loading segmentation weights: {e}. Using model with potentially un-fine-tuned head.")

            else:
                print(f"CenterPoint: Fine-tuned segmentation weights not found or path not specified ({finetuned_weights_path}). Model will use its initial weights.")

            # Freeze segmentation network and set to eval mode
            for param in self.segmentation_net.parameters():
                param.requires_grad = False
            self.segmentation_net.eval()
            self.segmentation_num_features = num_seg_classes
        else:
            print("CenterPoint: No image_segmentation_model configured for PointPainting.")
            self.segmentation_num_features = 0

    # --- PointPainting: Helper to get semantic scores ---
    def _get_semantic_scores(self, images_batch: torch.Tensor) -> torch.Tensor:
        if self.segmentation_net is not None and images_batch is not None:

             # If images_batch is a list of tensors, stack them into a single batch tensor
            if isinstance(images_batch, list):
                if not images_batch: # Handle case of an empty list
                    return None
                # Assuming all tensors in the list have the same shape
                images_batch = torch.stack(images_batch, dim=0)

            self.segmentation_net = self.segmentation_net.to(images_batch.device)
            seg_output = self.segmentation_net(images_batch)
            seg_logits = seg_output['out'] if isinstance(seg_output, dict) else seg_output
            return F.softmax(seg_logits, dim=1) # (B, NumSegClasses, H_img, W_img)
        return None

    # --- PointPainting: Helper to decorate points with semantics ---
    def _decorate_points_with_semantics(self, 
                                        pts_batch_list: list, # List of [N_pts, features] tensors
                                        images_batch: torch.Tensor, # (B, C, H_img, W_img)
                                        seg_scores_batch: torch.Tensor, # (B, NumSegClasses, H_img, W_img)
                                        calib_batch_list: list # List of calib_data dicts
                                        ) -> list:
        if seg_scores_batch is None:
            return pts_batch_list # Return original points if no scores

        decorated_pts_list = []
        B, NumSegClasses, H_img, W_img = seg_scores_batch.shape

        for i in range(B): # Iterate over batch samples
            points = pts_batch_list[i]           # (N_pts, num_features_orig)
            calib_data = calib_batch_list[i]     # Dict with P2, Tr_velo_to_cam etc.
            seg_scores_sample = seg_scores_batch[i] # (NumSegClasses, H_img, W_img)
            
            if points.numel() == 0: # Handle empty point clouds
                decorated_pts_list.append(points)
                continue

            P2 = calib_data['P2'].to(points.device)                 # (3,4) or (4,4)
            Tr_velo_to_cam_full = torch.eye(4, device=points.device, dtype=points.dtype)
            Tr_v2c_raw = calib_data['Tr_velo_to_cam'].to(points.device)
            if Tr_v2c_raw.shape == (3,4):
                Tr_velo_to_cam_full[:3, :] = Tr_v2c_raw
            elif Tr_v2c_raw.shape == (4,4):
                Tr_velo_to_cam_full = Tr_v2c_raw
            else: 
                decorated_pts_list.append(points) 
                continue

            points_xyz = points[:, :3]
            points_xyz1_lidar = F.pad(points_xyz, (0, 1), 'constant', 1.0)
            points_xyz1_cam = points_xyz1_lidar @ Tr_velo_to_cam_full.T
            
            P2_4x4 = torch.eye(4, device=points.device, dtype=points.dtype)
            P2_4x4[:P2.shape[0], :P2.shape[1]] = P2

            points_img_projected_homo = points_xyz1_cam @ P2_4x4.T 
            
            depth_in_cam = points_xyz1_cam[:, 2]

            points_u_unnorm = points_img_projected_homo[:, 0]
            points_v_unnorm = points_img_projected_homo[:, 1]
            perspective_w = points_img_projected_homo[:, 2] 

            u_coords_cont = points_u_unnorm / (perspective_w + 1e-8)
            v_coords_cont = points_v_unnorm / (perspective_w + 1e-8)

            u_coords = torch.round(u_coords_cont).long()
            v_coords = torch.round(v_coords_cont).long()

            point_semantic_features = torch.zeros((points.shape[0], NumSegClasses), device=points.device)

            valid_depth_mask = depth_in_cam > 0.1 
            valid_u_mask = (u_coords >= 0) & (u_coords < W_img)
            valid_v_mask = (v_coords >= 0) & (v_coords < H_img)
            valid_projection_mask = valid_depth_mask & valid_u_mask & valid_v_mask
            
            if valid_projection_mask.any():
                valid_indices_in_cloud = valid_projection_mask.nonzero(as_tuple=True)[0]
                valid_u = u_coords[valid_indices_in_cloud]
                valid_v = v_coords[valid_indices_in_cloud]
                
                sampled_scores = seg_scores_sample[:, valid_v, valid_u] 
                point_semantic_features[valid_indices_in_cloud] = sampled_scores.T 

            decorated_points = torch.cat([points, point_semantic_features], dim=1)
            decorated_pts_list.append(decorated_points)
            
        return decorated_pts_list
    
    ## Voxelization (Copied from original)
    def voxelize(self, points): # points here is a list of tensors for the batch
        voxel_dict = dict()
        voxels, coors, num_points = [], [], []
        for i, res in enumerate(points):
            # Ensure res is moved to CUDA if not already there
            res_voxels, res_coors, res_num_points = self.voxel_layer(res.cuda()) 
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors = torch.cat(coors, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['num_points'] = num_points
        voxel_dict['coors'] = coors

        return voxel_dict
    
    def _model_forward(self, pts_data_list, images_batch=None, calib_batch_list=None):
        decorated_pts_data_list = pts_data_list # Default to original if no painting
        if self.segmentation_net and images_batch is not None and calib_batch_list is not None:
            semantic_scores = self._get_semantic_scores(images_batch)
            if semantic_scores is not None:
                decorated_pts_data_list = self._decorate_points_with_semantics(
                    pts_data_list, images_batch, semantic_scores, calib_batch_list
                )
        
        # Voxelize decorated points (the list of tensors)
        voxel_dict = self.voxelize(decorated_pts_data_list)
    
        voxels = voxel_dict['voxels']
        num_points = voxel_dict['num_points']
        coors = voxel_dict['coors']
    
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        # Handle case where coors might be empty (e.g., empty batch)
        bs = coors[-1,0].item() + 1 if coors.numel() > 0 else 0
        
        bev_feats = self.middle_encoder(voxel_features, coors, bs)        
        backbone_feats = self.backbone(bev_feats)
        neck_feats = self.neck(backbone_feats)
        ret_dict = self.head(neck_feats)
        return ret_dict
    
    def training_step(self, batch, batch_idx):
        # <--- START DEBUGGING BLOCK: CHECK GT BOXES VS PC_RANGE --->
        # We print this periodically (e.g., every 50 steps) to avoid spamming the log.
        if batch_idx % 50 == 0:
            # The batch dictionary contains lists where each element corresponds to a sample.
            # We'll inspect the first sample in the batch as a representative example.
            gt_labels_3d_sample = batch['gt_labels_3d'][0]
            gt_bboxes_3d_sample = batch['gt_bboxes_3d'][0]

            # Check if there are any ground truth boxes in this sample
            if len(gt_bboxes_3d_sample.tensor) > 0:
                print(f"\n--- [Step {self.global_step}] Debugging GT Boxes vs PC_RANGE ---")
                print(f"Model's PC Range: {self.pc_range.cpu().numpy().tolist()}")

                # Get the center coordinates of each bounding box
                box_centers = gt_bboxes_3d_sample.gravity_center.cpu().numpy()
                labels = gt_labels_3d_sample.cpu().numpy()

                for i, center in enumerate(box_centers):
                    class_idx = labels[i]
                    # self.class_names should be ['Car', 'Pedestrian', 'Cyclist']
                    class_name = self.class_names[int(class_idx)]
                    x, y, z = center[0], center[1], center[2]
                    
                    # Check if the box center is outside the pc_range
                    is_outside = (x < self.pc_range[0].item() or x > self.pc_range[3].item() or
                                  y < self.pc_range[1].item() or y > self.pc_range[4].item() or
                                  z < self.pc_range[2].item() or z > self.pc_range[5].item())
                    
                    if is_outside:
                        print(f"  - GT Box '{class_name}' at [{x:.2f}, {y:.2f}, {z:.2f}] is OUTSIDE the PC_RANGE!")
                    else:
                        print(f"  - GT Box '{class_name}' at [{x:.2f}, {y:.2f}, {z:.2f}] is INSIDE the PC_RANGE.")
                print("--- End Debugging Block ---\n")
        # <--- END DEBUGGING BLOCK --->


        # --- Original Training Logic ---
        pts_data_list = batch['pts'] # This is now a list of tensors from collate_fn
        images_batch = batch.get('images') # (B, C, H, W)
        calib_batch_list = batch.get('calib_data') # List of dicts

        gt_labels_3d = batch['gt_labels_3d'] 
        gt_bboxes_3d = batch['gt_bboxes_3d'] 
            
        ret_dict = self._model_forward(pts_data_list, images_batch, calib_batch_list)
        
        loss_input = [gt_bboxes_3d, gt_labels_3d, ret_dict]
        
        losses = self.head.loss(*loss_input)
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.mean() # Ensure reduction if loss_value is not scalar
        
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        
        # Determine batch size dynamically from one of the inputs
        batch_size = len(pts_data_list)

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training (Copied from original)
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
            self.log(f'train/{loss_name}', loss_value, batch_size=batch_size, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        # Use self.optimizer_config directly, which was loaded from config in __init__
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_config)
        return optimizer
    
    
    def validation_step(self, batch, batch_idx):
        # assert len(batch['pts']) == 1, 'Batch size should be 1 for validation' # Keep if your validation dataloader guarantees batch size 1
        pts_data_list = batch['pts']
        images_batch = batch.get('images')
        calib_batch_list = batch.get('calib_data')
        metas = batch['metas']
        gt_labels_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']
        
        ret_dict = self._model_forward(pts_data_list, images_batch, calib_batch_list)
        
        loss_input = [gt_bboxes_3d, gt_labels_3d, ret_dict]
        
        # Ensure metas are correctly passed as a list, get_bboxes expects a list of dicts
        # If your validation batch size is 1, metas will be a list containing one dict.
        # If validation batch size can be > 1, then bbox_list should be a list of lists of bboxes, scores, labels
        bbox_list = self.head.get_bboxes(ret_dict, img_metas=metas)
        
        bbox_results = [
            dict(bboxes_3d = bboxes, 
                 scores_3d = scores, 
                 labels_3d = labels)
            for bboxes, scores, labels in bbox_list
        ]

        losses = self.head.loss(*loss_input)
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.mean()
        
        val_loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = val_loss
        
        # Determine batch size for validation logging, typically 1 if assert is kept
        val_batch_size = 1 # Or self.hparams.config.get('val_batch_size', 1)

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training (Copied from original)
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
            self.log(f'validation/{loss_name}', loss_value, batch_size=val_batch_size, sync_dist=True) # sync_dist=True for validation metrics is typical
        
        # Append validation results (Copied and adapted from original)
# --- START OF MODIFICATION ---
        # Loop through each sample in the batch to store results correctly
        # This is crucial for validation where results are processed per sample.
        for i, meta_for_sample in enumerate(metas): # Iterate through each sample's metadata
            # Ensure bbox_results_for_sample contains the results for the current sample
            # bbox_list is a list of (bboxes, scores, labels) tuples for the batch
            # We need to extract the i-th sample's results and wrap it in a list for consistency.
            if i < len(bbox_list):
                bboxes, scores, labels = bbox_list[i]
                bbox_results_for_sample = [dict(bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels)]
            else:
                # Handle cases where bbox_list might not have results for this sample
                # (e.g., if get_bboxes filtered out all predictions for this sample)
                bbox_results_for_sample = []

            # Construct the input_batch for this specific sample for `convert_valid_bboxes`
            # `convert_valid_bboxes` expects `input_batch['metas']` to be a list containing one dict,
            # and `input_batch['calib_data']` to be a list containing one dict.
            single_sample_input_batch = {
                'metas': [meta_for_sample], # Wrap the single meta dict in a list
                'calib_data': [calib_batch_list[i]], # Wrap the single calib dict in a list
                # Add other necessary single-sample components from the original 'batch' if convert_valid_bboxes uses them
                # For example, if it needs 'pts' for the sample:
                'pts': [pts_data_list[i]] # Wrap the single pts tensor in a list
            }

            self.val_results_list.append(dict(
                sample_idx = meta_for_sample['num_frame'],
                input_batch = single_sample_input_batch, # Store the correctly structured dict
                bbox_results = bbox_results_for_sample, # Store the single sample's bbox results
                losses = log_vars # Note: These are batch-level losses, might need adjustment if per-sample losses are desired
            ))
        # --- END OF MODIFICATION ---
    def on_validation_epoch_end(self): # Copied from original
        if (not self.save_results) or self.training: 
            tmp_dir = tempfile.TemporaryDirectory()
            working_dir = tmp_dir.name
        else:
            tmp_dir = None
            working_dir = self.output_dir

        preds_dst = os.path.join(working_dir, f'{self.inference_mode}_preds')
        os.makedirs(preds_dst, exist_ok=True)
        
        outputs = self.val_results_list
        self.val_results_list = [] # Clear for next epoch
        results = self.format_results(outputs, results_save_path=preds_dst)
        
        if self.inference_mode =='val': 
            gt_dst = os.path.join(self.data_root, 'lidar', 'training', 'label_2')
            
            evaluation = Evaluation(test_annotation_file=gt_dst)
            results = evaluation.evaluate(result_path=preds_dst, current_class=[0, 1, 2])
            
            self.log('validation/entire_area/Car_3d', results['entire_area']['Car_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/Pedestrian_3d', results['entire_area']['Pedestrian_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/Cyclist_3d', results['entire_area']['Cyclist_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/entire_area/mAP', (results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3, batch_size=1, sync_dist=True)
            self.log('validation/ROI/Car_3d', results['roi']['Car_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/Pedestrian_3d', results['roi']['Pedestrian_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/Cyclist_3d', results['roi']['Cyclist_3d_all'], batch_size=1, sync_dist=True)
            self.log('validation/ROI/mAP', (results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3, batch_size=1, sync_dist=True)
        
            print("Results: \n"
                f"Entire annotated area: \n"
                f"Car: {results['entire_area']['Car_3d_all']} \n"
                f"Pedestrian: {results['entire_area']['Pedestrian_3d_all']} \n"
                f"Cyclist: {results['entire_area']['Cyclist_3d_all']} \n"
                f"mAP: {(results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3} \n"
                f"Driving corridor area: \n"
                f"Car: {results['roi']['Car_3d_all']} \n"
                f"Pedestrian: {results['roi']['Pedestrian_3d_all']} \n"
                f"Cyclist: {results['roi']['Cyclist_3d_all']} \n"
                f"mAP: {(results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3} \n"
                )
            
        if isinstance(tmp_dir, tempfile.TemporaryDirectory):
            tmp_dir.cleanup() 
        return results
            
    def format_results(self, # Copied from original
                       outputs, 
                       results_save_path=None,
                       pklfile_prefix=None):
        
        det_annos = []
        print('\nConverting prediction to KITTI format')
        print(f'Writing results to {results_save_path}')
        for result in outputs:
            sample_idx = result['sample_idx']
            res_dict = result['bbox_results']
            input_batch = result['input_batch']
            
            annos = []
            box_dict = self.convert_valid_bboxes(res_dict[0], input_batch) # res_dict[0] because bbox_results is a list of lists here
            
            anno = {                 
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': [],
            }
            
            if len(box_dict['box2d']) > 0:
                box2d_preds = box_dict['box2d']
                box3d_preds_lidar = box_dict['box3d_lidar']
                box3d_location_cam = box_dict['location_cam']
                scores = box_dict['scores']
                label_preds = box_dict['label_preds']
                
                for box3d_lidar, location_cam, box2d, score, label in zip(box3d_preds_lidar, box3d_location_cam, box2d_preds, scores, label_preds):                                      
                    box2d[2:] = np.minimum(box2d[2:], self.img_shape.cpu().numpy()[:2])
                    box2d[:2] = np.maximum(box2d[:2], [0, 0])
                    anno['name'].append(self.class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(np.arctan2(location_cam[2], location_cam[0]) + box3d_lidar[6] - np.pi/2)
                    anno['bbox'].append(box2d)
                    anno['dimensions'].append(box3d_lidar[3:6])
                    anno['location'].append(location_cam[:3])
                    anno['rotation_y'].append(box3d_lidar[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)
            
            if results_save_path is not None:
                curr_file = f'{results_save_path}/{sample_idx}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lwh -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][2], dims[idx][1],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)
                
            annos[-1]['sample_idx'] = np.array([sample_idx] * len(annos[-1]['score']), dtype=np.int64)
            det_annos += annos
        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            with open(out, "wb") as f:
                pickle.dump(det_annos, f)
            print(f'Result is saved to {out}.')
        return det_annos
        
    def convert_valid_bboxes(self, box_dict, input_batch): # Copied from original
        # Convert the predicted bounding boxes to the format required by the evaluation metric
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        # Assuming input_batch['metas'] is a list where the first element is the meta for the current sample
        sample_idx = input_batch['metas'][0]['num_frame'] 
        
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=sample_idx)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
        device = box_preds.tensor.device
                
        box_preds_corners_lidar = box_preds.corners
        box_preds_bottom_center_lidar = box_preds.bottom_center 
        
        box_preds_corners_img_list = [] 
        box_preds_bottom_center_cam_list =[]
        
        for box_pred_corners, box_pred_bottom_center in zip(box_preds_corners_lidar, box_preds_bottom_center_lidar):
            
            box_pred_corners_lidar_homo= torch.ones((8,4))
            box_pred_corners_lidar_homo[:, :3] = box_pred_corners
            box_pred_corners_cam_homo = homogeneous_transformation(box_pred_corners_lidar_homo, local_transforms.t_camera_lidar)
            box_pred_corners_img = np.dot(box_pred_corners_cam_homo, local_transforms.camera_projection_matrix.T)
            box_pred_corners_img = torch.tensor((box_pred_corners_img[:, :2].T / box_pred_corners_img[:, 2]).T, device=device)
            box_preds_corners_img_list.append(box_pred_corners_img)

            box_pred_bottom_center_lidar_homo = torch.ones((1,4))
            box_pred_bottom_center_lidar_homo[:, :3] = box_pred_bottom_center
            box_pred_bottom_center_cam_homo = homogeneous_transformation(box_pred_bottom_center_lidar_homo, local_transforms.t_camera_lidar)
            box_pred_bottom_center_cam = torch.tensor(box_pred_bottom_center_cam_homo[:,:3])
            box_preds_bottom_center_cam_list.append(box_pred_bottom_center_cam)

        if box_preds_corners_img_list != []:
            box_preds_corners_img = torch.stack(box_preds_corners_img_list, dim=0)
            assert box_preds_bottom_center_cam_list != []
            box_preds_bottom_center_cam = torch.cat(box_preds_bottom_center_cam_list, dim=0).to(device)
        
            minxy = torch.min(box_preds_corners_img, dim=1)[0]
            maxxy = torch.max(box_preds_corners_img, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)

            self.img_shape = self.img_shape.to(device)
            self.pc_range = self.pc_range.to(device)
            
            valid_cam_inds = ((box_2d_preds[:, 0] < self.img_shape[0]) & (box_2d_preds[:, 1] < self.img_shape[1]) & (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
            valid_pcd_inds = ((box_preds.center > self.pc_range[:3]) & (box_preds.center < self.pc_range[3:]))
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
            
            if valid_inds.sum() > 0:
                return dict(
                    box2d=box_2d_preds[valid_inds, :].cpu().numpy(),
                    location_cam=box_preds_bottom_center_cam[valid_inds].cpu().numpy(),
                    box3d_lidar=box_preds[valid_inds].tensor.cpu().numpy(),
                    scores=scores[valid_inds].cpu().numpy(),
                    label_preds=labels[valid_inds].cpu().numpy(),
                    sample_idx=sample_idx)
            else:
                return dict(
                    box2d=np.zeros([0, 4]),
                    location_cam=np.zeros([0, 3]),
                    box3d_lidar=np.zeros([0, 7]),
                    scores=np.zeros([0]),
                    label_preds=np.zeros([0, 4]),
                    sample_idx=sample_idx)
        else:
            return dict(
                box2d=np.zeros([0, 4]),
                location_cam=np.zeros([0, 3]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)