import os
import tempfile
import pickle
from datetime import datetime

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models # Added for ResNet
from collections import OrderedDict

from vod.evaluation import Evaluation
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

import hydra # Added for Hydra instantiation if needed
import lightning as L
import torch.distributed as dist

from common_src.ops import Voxelization
from common_src.model.voxel_encoders import PillarFeatureNet
from common_src.model.middle_encoders import PointPillarsScatter
from common_src.model.backbones import SECOND
from common_src.model.necks import SECONDFPN
from common_src.model.heads import CenterHead

# --- Note: You can remove the import for segmentation_helpers as it's no longer used ---
# from common_src.model.segmentation_helpers import get_modified_deeplabv3 

class CenterPoint(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Use config directly for save_hyperparameters to access it later via self.hparams
        self.save_hyperparameters(config) 
        
        self.img_shape = torch.tensor([1936 , 1216])
        self.data_root = config.get('data_root', None)
        self.class_names = config.get('class_names', None)
        self.output_dir = config.get('output_dir', None)
        self.pc_range = torch.tensor(config.get('point_cloud_range', None))
        
        # --- LiDAR Path Components (Unchanged) ---
        voxel_layer_config = config.get('pts_voxel_layer', None)
        voxel_encoder_config = config.get('voxel_encoder', None)
        middle_encoder_config = config.get('middle_encoder', None)
        neck_config = config.get('neck', None)
        head_config = config.get('head', None)
        
        self.voxel_layer = Voxelization(**voxel_layer_config)
        self.voxel_encoder = PillarFeatureNet(**voxel_encoder_config)
        self.middle_encoder = PointPillarsScatter(**middle_encoder_config)
        
        # --- NEW: BEV Fusion - Image Backbone Setup ---
        self.image_backbone = None
        image_feature_channels = 0
        if self.hparams.get('image_backbone'):
            print("Initializing Image Backbone for BEV Fusion...")
            # Load a pre-trained ResNet-18
            self.image_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # We use features from layer3 (before the last downsampling stage).
            # This provides a good balance of semantic info and spatial resolution.
            self.image_backbone = nn.Sequential(*list(self.image_backbone.children())[:-3]) 
            # Freeze the image backbone - we use it as a fixed feature extractor
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            self.image_backbone.eval()
            
            # ResNet-18's layer3 outputs 256 channels
            image_feature_channels = self.hparams.image_backbone.get('num_features', 256)
        else:
            print("No image_backbone configured. Running in LiDAR-only mode.")

        # --- MODIFIED: Update BEV Backbone Input Channels for Fusion ---
        backbone_config = config.get('backbone', None)
        # Input to BEV backbone = features from LiDAR BEV + features from Image BEV
        lidar_bev_channels = middle_encoder_config.get('in_channels', 64)
        backbone_config['in_channels'] = lidar_bev_channels + image_feature_channels
        print(f"BEV Backbone input channels set to: {backbone_config['in_channels']} ({lidar_bev_channels} from LiDAR + {image_feature_channels} from Image)")
        
        # --- BEV Path Components (Instantiated with potentially updated config) ---
        self.backbone = SECOND(**backbone_config)
        self.neck = SECONDFPN(**neck_config)
        self.head = CenterHead(**head_config)
        
        self.optimizer_config = config.get('optimizer', None)
        
        self.vod_kitti_locations = KittiLocations(root_dir = self.data_root, 
                                     output_dir= self.output_dir,
                                     frame_set_path='',
                                     pred_dir='',)
        self.inference_mode = config.get('inference_mode', 'val')
        self.save_results = config.get('save_preds_results', False)
        self.val_results_list =[]

    # --- NEW: Helper for BEV Fusion ---
    def _get_projected_image_features(self, images_batch, calib_batch_list, lidar_bev_features):
        """
        Projects BEV grid coordinates into the image plane and samples features.
        """
        # Ensure we have everything needed for fusion
        if self.image_backbone is None or images_batch is None or calib_batch_list is None:
            return None
        
        # Ensure images_batch is a tensor and not an empty list or other type
        if not isinstance(images_batch, torch.Tensor) or images_batch.numel() == 0:
            return None

        # 1. Get image features from the pre-trained backbone
        self.image_backbone.to(images_batch.device)
        with torch.no_grad(): # Ensure no gradients are computed for the frozen backbone
            image_features = self.image_backbone(images_batch)
        
        batch_size, _, bev_h, bev_w = lidar_bev_features.shape
        
        # 2. Create a grid of real-world (X, Y, Z) coordinates corresponding to the BEV feature map
        pc_range = self.pc_range.to(images_batch.device)
        voxel_size = torch.tensor(self.hparams.voxel_size, device=images_batch.device)

        ys, xs = torch.meshgrid(
            torch.arange(bev_h, device=images_batch.device, dtype=torch.float32),
            torch.arange(bev_w, device=images_batch.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Convert BEV grid indices to real-world LiDAR coordinates
        world_x = xs * voxel_size[0] + pc_range[0] + (voxel_size[0] * 0.5)
        world_y = ys * voxel_size[1] + pc_range[1] + (voxel_size[1] * 0.5)
        world_z = torch.zeros_like(world_x) # Assume BEV plane is at Z=0 for projection
        
        world_coords = torch.stack([world_x, world_y, world_z], dim=-1).view(-1, 3)
        world_coords_homo = F.pad(world_coords, (0, 1), 'constant', 1.0)
        
        projected_image_features_list = []
        for i in range(batch_size):
            calib_data = calib_batch_list[i]
            # Check if this specific sample's calibration data is valid
            if not calib_data or 'P2' not in calib_data or 'T_cam_from_lidar' not in calib_data:
                # If calib data is missing for this sample, append zeros for its image features
                num_img_features = image_features.shape[1]
                zeros = torch.zeros(num_img_features, bev_h, bev_w, device=images_batch.device)
                projected_image_features_list.append(zeros)
                continue

            # Assumes calib_data provides these keys from your updated ViewOfDelft dataset
            P2 = calib_data['P2'].to(images_batch.device)
            T_cam_from_lidar = calib_data['T_cam_from_lidar'].to(images_batch.device)

            # 3. Project world coordinates to camera, then to image plane
            cam_coords_homo = world_coords_homo @ T_cam_from_lidar.T
            img_coords_homo = cam_coords_homo @ P2.T
            
            depth_in_cam = img_coords_homo[:, 2] # Depth for filtering
            img_coords = img_coords_homo[:, :2] / (depth_in_cam.unsqueeze(-1) + 1e-8)
            
            # 4. Normalize coordinates for grid_sample (-1 to 1 range)
            _, _, H_img_feat, W_img_feat = image_features.shape
            img_coords[:, 0] = (img_coords[:, 0] / (W_img_feat - 1)) * 2.0 - 1.0
            img_coords[:, 1] = (img_coords[:, 1] / (H_img_feat - 1)) * 2.0 - 1.0
            
            # Replace out-of-bounds coordinates to avoid artifacts from 'border' padding mode
            # Any coord outside [-1, 1] will be sampled as zeros due to padding_mode='zeros'
            grid = img_coords.view(1, bev_h, bev_w, 2)
            
            # 5. Sample features using F.grid_sample for efficiency
            with torch.no_grad(): # Ensure grid_sample also does not compute gradients
                sampled_features = F.grid_sample(
                    image_features[i].unsqueeze(0), # (1, C_img, H_img, W_img)
                    grid,
                    mode='bilinear',
                    padding_mode='zeros', # Sampled values for out-of-bounds coords will be 0
                    align_corners=False
                ) # Output shape: (1, C_img, bev_h, bev_w)
            projected_image_features_list.append(sampled_features.squeeze(0))

        return torch.stack(projected_image_features_list, dim=0)

    ## Voxelization (Unchanged from your original)
    def voxelize(self, points): # points here is a list of tensors for the batch
        voxel_dict = dict()
        voxels, coors, num_points = [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.voxel_layer(res.to(self.device)) 
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        # Handle case where a batch might be empty after all filtering
        if not voxels:
            return {'voxels': torch.empty(0, self.voxel_layer.max_num_points, self.voxel_layer.in_channels, device=self.device),
                    'coors': torch.empty(0, 4, device=self.device, dtype=torch.long),
                    'num_points': torch.empty(0, device=self.device, dtype=torch.long)}

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors = torch.cat(coors, dim=0)

        voxel_dict['voxels'] = voxels
        voxel_dict['num_points'] = num_points
        voxel_dict['coors'] = coors
        return voxel_dict
    
    def _model_forward(self, batch):
        """Main forward pass combining LiDAR and Image paths."""
        pts_data_list = batch['pts']
        images_batch = batch.get('images')
        calib_batch_list = batch.get('calib_data')

        # --- LiDAR Path ---
        voxel_dict = self.voxelize(pts_data_list)
        voxels, num_points, coors = voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors']

        # If after voxelization no points remain in the batch, return empty predictions
        if coors.numel() == 0:
            return self.head.get_empty_predictions(self.device)
        
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1,0].item() + 1 if coors.numel() > 0 else 0
        if batch_size == 0:
            return self.head.get_empty_predictions(self.device) # Return empty predictions if batch is empty

        lidar_bev_features = self.middle_encoder(voxel_features, coors, batch_size)
        
        # --- BEV Fusion Path ---
        image_bev_features = self._get_projected_image_features(images_batch, calib_batch_list, lidar_bev_features)

        if image_bev_features is not None:
            # Concatenate along the channel dimension (dim=1)
            fused_bev_features = torch.cat([lidar_bev_features, image_bev_features], dim=1)
        else:
            # If no image data, proceed with LiDAR-only
            fused_bev_features = lidar_bev_features
        
        # --- Detection Path (uses fused features) ---
        backbone_feats = self.backbone(fused_bev_features)
        neck_feats = self.neck(backbone_feats)
        ret_dict = self.head(neck_feats)
        return ret_dict
    
    def training_step(self, batch, batch_idx):
        # The model_forward now handles all fusion logic internally
        ret_dict = self._model_forward(batch)
        if not ret_dict: return None # Skip batch if model forward returns nothing
        gt_labels_3d = batch['gt_labels_3d'] 
        gt_bboxes_3d = batch['gt_bboxes_3d'] 
            
        loss_input = [gt_bboxes_3d, gt_labels_3d, ret_dict]
        losses = self.head.loss(*loss_input)
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.mean() 
        
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        
        batch_size = len(batch['pts'])
        for loss_name, loss_value in log_vars.items():
            self.log(f'train/{loss_name}', loss_value.item(), batch_size=batch_size, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_config)
        return optimizer
        
    def validation_step(self, batch, batch_idx):
        ret_dict = self._model_forward(batch)
        
        gt_labels_3d = batch['gt_labels_3d']
        gt_bboxes_3d = batch['gt_bboxes_3d']
        metas = batch['metas']
        
        loss_input = [gt_bboxes_3d, gt_labels_3d, ret_dict]
        bbox_list = self.head.get_bboxes(ret_dict, img_metas=metas)
        losses = self.head.loss(*loss_input)
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            log_vars[loss_name] = loss_value.mean()
        
        val_loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = val_loss
        
        val_batch_size = len(batch['pts'])
        for loss_name, loss_value in log_vars.items():
            self.log(f'validation/{loss_name}', loss_value.item(), batch_size=val_batch_size, sync_dist=True)
        
        # This loop correctly handles results for any validation batch size
        for i in range(val_batch_size):
            # Extract results for the i-th sample in the batch
            bboxes, scores, labels = bbox_list[i]
            bbox_results_for_sample = [dict(bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels)]
            
            # Structure input_batch for format_results to process one sample at a time
            single_sample_input_batch = {
                'metas': [metas[i]], 
                'calib_data': [batch['calib_data'][i]] if 'calib_data' in batch else None,
            }

            self.val_results_list.append(dict(
                sample_idx = metas[i]['num_frame'],
                input_batch = single_sample_input_batch,
                bbox_results = bbox_results_for_sample,
                losses = {k: v.item() for k,v in log_vars.items()} # Batch-level loss for this step
            ))

    # --- on_validation_epoch_end, format_results, convert_valid_bboxes ---
    # These methods from your original file should be kept here as they are needed for evaluation.
    # They don't need changes for this BEV Fusion method.
    def on_validation_epoch_end(self):
        # (Your existing on_validation_epoch_end logic here)
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
        self.format_results(outputs, results_save_path=preds_dst)
        
        if self.inference_mode =='val' and os.path.exists(preds_dst): 
            gt_dst = os.path.join(self.data_root, 'lidar', 'training', 'label_2')
            if not os.path.exists(gt_dst):
                print(f"Ground truth directory not found at {gt_dst}, skipping evaluation.")
                return

            try:
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
            except Exception as e:
                print(f"Error during VoD evaluation: {e}")

        if isinstance(tmp_dir, tempfile.TemporaryDirectory):
            tmp_dir.cleanup()

    def format_results(self, outputs, results_save_path=None, **kwargs):
        # (Your existing format_results logic here)
        print(f'\nConverting {len(outputs)} prediction results to KITTI format...')
        if results_save_path:
            print(f'Writing results to {results_save_path}')
        
        for result in outputs:
            sample_idx = result['sample_idx']
            res_dict_list = result['bbox_results']
            
            # In validation step, we wrapped the single sample result in a list.
            if not res_dict_list: continue # Skip if no bboxes were predicted for this sample
            res_dict = res_dict_list[0]
            
            input_batch = result['input_batch']
            box_dict = self.convert_valid_bboxes(res_dict, input_batch)
            
            # ... (rest of file writing logic from your original format_results) ...
            anno = { 'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [],
                     'dimensions': [], 'location': [], 'rotation_y': [], 'score': [] }
            
            if box_dict and len(box_dict['box2d']) > 0:
                for i in range(len(box_dict['box2d'])):
                    anno['name'].append(self.class_names[int(box_dict['label_preds'][i])])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(np.arctan2(box_dict['location_cam'][i][0], box_dict['location_cam'][i][2]) + box_dict['box3d_lidar'][i][6])
                    anno['bbox'].append(box_dict['box2d'][i])
                    anno['dimensions'].append(box_dict['box3d_lidar'][i][3:6])
                    anno['location'].append(box_dict['location_cam'][i])
                    anno['rotation_y'].append(box_dict['box3d_lidar'][i][6])
                    anno['score'].append(box_dict['scores'][i])

                if results_save_path:
                    curr_file = os.path.join(results_save_path, f'{sample_idx}.txt')
                    with open(curr_file, 'w') as f:
                        for i in range(len(anno['bbox'])):
                            # KITTI format: type, truncated, occluded, alpha, bbox_left, top, right, bottom, dim_h,w,l, loc_x,y,z, rot_y, score
                            f.write(f"{anno['name'][i]} {anno['truncated'][i]:.2f} {anno['occluded'][i]} {anno['alpha'][i]:.2f} ")
                            f.write(f"{anno['bbox'][i][0]:.2f} {anno['bbox'][i][1]:.2f} {anno['bbox'][i][2]:.2f} {anno['bbox'][i][3]:.2f} ")
                            f.write(f"{anno['dimensions'][i][2]:.2f} {anno['dimensions'][i][1]:.2f} {anno['dimensions'][i][0]:.2f} ") # HWL -> LWH, so dim is l,w,h now. KITTI wants h,w,l
                            f.write(f"{anno['location'][i][0]:.2f} {anno['location'][i][1]:.2f} {anno['location'][i][2]:.2f} {anno['rotation_y'][i]:.2f}\n")

    def convert_valid_bboxes(self, box_dict, input_batch):
        # (Your existing convert_valid_bboxes logic here)
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = input_batch['metas'][0]['num_frame'] 
        
        vod_frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=sample_idx)
        local_transforms = FrameTransformMatrix(vod_frame_data)
        
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
        device = box_preds.tensor.device
        
        if len(box_preds.tensor) == 0:
            return {}

        box_preds_corners_lidar = box_preds.corners
        box_preds_bottom_center_lidar = box_preds.bottom_center

        # Step 1: Create the homogeneous points tensor (already done)
        box_preds_corners_homogeneous = torch.cat(
            [box_preds_corners_lidar.view(-1, 3), 
            torch.ones(box_preds_corners_lidar.view(-1, 3).shape[0], 1).to(box_preds_corners_lidar.device)], 
            dim=1
        )

        # Step 2: Move the homogeneous points tensor to CPU and convert to NumPy array
        box_preds_corners_homogeneous_np = box_preds_corners_homogeneous.cpu().numpy()

        # Step 3: Call homogeneous_transformation with the NumPy array
        # (Assuming local_transforms.t_camera_lidar is already a NumPy array, which is typical for VOD calibrations)
        box_preds_corners_cam_homo_np = homogeneous_transformation(box_preds_corners_homogeneous_np, local_transforms.t_camera_lidar)

        # Step 4: Convert the result back to a PyTorch tensor and move it back to the original device (GPU)
        box_preds_corners_cam_homo = torch.from_numpy(box_preds_corners_cam_homo_np).to(box_preds_corners_lidar.device).float()


        box_preds_corners_cam_homo = box_preds_corners_cam_homo.view(-1, 8, 4)

        box_pred_corners_img = torch.matmul(box_preds_corners_cam_homo, torch.from_numpy(local_transforms.camera_projection_matrix.T).to(device).float())
        
        # Normalize by depth
        box_pred_corners_img = box_pred_corners_img[..., :2] / box_pred_corners_img[..., 2:3]
        
        min_xy = torch.min(box_pred_corners_img, dim=1)[0]
        max_xy = torch.max(box_pred_corners_img, dim=1)[0]
        box_2d_preds = torch.cat([min_xy, max_xy], dim=1)

        # Step 1: Convert box_preds_bottom_center_lidar to homogeneous coordinates (Nx4)
        box_preds_bottom_center_lidar_homogeneous = torch.cat(
            [box_preds_bottom_center_lidar,
            torch.ones(box_preds_bottom_center_lidar.shape[0], 1).to(box_preds_bottom_center_lidar.device)],
            dim=1
        )

        # Step 2: Move the homogeneous points tensor to CPU and convert to NumPy array
        box_preds_bottom_center_lidar_homogeneous_np = box_preds_bottom_center_lidar_homogeneous.cpu().numpy()

        # Step 3: Call homogeneous_transformation with the NumPy array
        # (Assuming local_transforms.t_camera_lidar is already a NumPy array)
        box_preds_bottom_center_cam_np = homogeneous_transformation(
            box_preds_bottom_center_lidar_homogeneous_np,
            local_transforms.t_camera_lidar
        )

        # Step 4: Convert the result back to a PyTorch tensor, move it back to the original device,
        # cast to float32, and then take the first 3 columns (x, y, z)
        box_preds_bottom_center_cam = torch.from_numpy(box_preds_bottom_center_cam_np).to(
            box_preds_bottom_center_lidar.device
        ).float()[:, :3]

        # Filtering logic
        valid_inds = (box_preds.center > self.pc_range.to(device)[:3]) & (box_preds.center < self.pc_range.to(device)[3:])
        valid_inds = valid_inds.all(-1)
        
        if valid_inds.sum() > 0:
            return dict(
                box2d=box_2d_preds[valid_inds].cpu().numpy(),
                location_cam=box_preds_bottom_center_cam[valid_inds].cpu().numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.cpu().numpy(),
                scores=scores[valid_inds].cpu().numpy(),
                label_preds=labels[valid_inds].cpu().numpy(),
                sample_idx=sample_idx)
        else:
            return {}
