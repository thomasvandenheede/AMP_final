# common_src/dataset/view_of_delft.py

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation
from common_src.model.utils import LiDARInstance3DBoxes # Ensure this path is correct for your project

class ViewOfDelft(Dataset):
    CLASSES_3D_DETECTION = ['Car', 'Pedestrian', 'Cyclist'] 
    
    # For parsing label_2.txt from VoD dataset
    LABEL_MAPPING_VOD = {
        'class': 0, 'truncated': 1, 'occluded': 2, 'alpha': 3,
        'bbox2d': slice(4,8), # left, top, right, bottom
        'bbox3d_dimensions': slice(8,11), # H, W, L (in camera coords)
        'bbox3d_location': slice(11,14), # X, Y, Z (in camera coords)
        'bbox3d_rotation': 14, # rotation_y (in camera coords)
    }

    # Define your segmentation classes and their mapping to integer IDs
    CLASSES_SEGMENTATION_MAP = {
        'Background': 0,
        'Car': 1,
        'Pedestrian': 2,
        'Cyclist': 3
    }
    
    def __init__(self, 
                 data_root: str, 
                 split: str,
                 mode: str = 'detection_pointpainting', # "detection_pointpainting" or "segmentation_finetune"
                 load_image_for_pointpainting: bool = True, 
                 segmentation_target_size: tuple = (256, 448), # H, W for rezize for the backbone
                 image_normalize_mean: list = [0.485, 0.456, 0.406],
                 image_normalize_std: list = [0.229, 0.224, 0.225],
                 cfg: dict = None # To pass general Hydra config if needed
                ):
        super().__init__()
        
        self.data_root = data_root
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        self.split = split
        assert mode in ['detection_pointpainting', 'segmentation_finetune'], f"Invalid mode: {mode}"
        self.mode = mode

        self.load_image_for_pointpainting = load_image_for_pointpainting if self.mode == 'detection_pointpainting' else False
        
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines() if line.strip()]
            if not self.sample_list:
                raise ValueError(f"No samples found in split file: {split_file}")

        self.vod_kitti_locations = KittiLocations(root_dir=data_root)

        self.seg_target_h, self.seg_target_w = segmentation_target_size
        self.img_mean = image_normalize_mean
        self.img_std = image_normalize_std

        common_image_transforms = [
            transforms.Resize(segmentation_target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.img_mean, std=self.img_std)
        ]
        
        if self.mode == 'segmentation_finetune':
            self.image_transform = transforms.Compose(common_image_transforms)
            self.mask_transform = transforms.Compose([
                transforms.Resize(segmentation_target_size, interpolation=transforms.InterpolationMode.NEAREST)
            ])
        elif self.load_image_for_pointpainting:
             self.image_transform_for_painting = transforms.Compose(common_image_transforms)

    def __len__(self):
        return len(self.sample_list)

    def _generate_segmentation_mask(self, pil_image: Image.Image, raw_labels: list) -> torch.Tensor:
        original_h, original_w = pil_image.height, pil_image.width
        background_id = self.CLASSES_SEGMENTATION_MAP.get('Background', 0)
        target_mask_np = np.full((original_h, original_w), fill_value=background_id, dtype=np.uint8)

        for label_str in raw_labels:
            label_parts = label_str.split(' ')
            if len(label_parts) < 8: continue
            class_name_from_label_file = label_parts[self.LABEL_MAPPING_VOD['class']]
            seg_class_id = self.CLASSES_SEGMENTATION_MAP.get(class_name_from_label_file)

            if seg_class_id is not None and seg_class_id != background_id:
                try:
                    bbox2d_str_list = label_parts[self.LABEL_MAPPING_VOD['bbox2d'].start : self.LABEL_MAPPING_VOD['bbox2d'].stop]
                    left, top, right, bottom = map(float, bbox2d_str_list)
                except (ValueError, TypeError): continue
                
                left_int, top_int = int(max(0, np.floor(left))), int(max(0, np.floor(top)))
                right_int, bottom_int = int(min(original_w, np.ceil(right))), int(min(original_h, np.ceil(bottom)))

                if right_int > left_int and bottom_int > top_int:
                    target_mask_np[top_int:bottom_int, left_int:right_int] = seg_class_id
        
        mask_pil = Image.fromarray(target_mask_np)
        if hasattr(self, 'mask_transform') and self.mask_transform:
            mask_pil = self.mask_transform(mask_pil)
        return torch.from_numpy(np.array(mask_pil)).long()

    

    def __getitem__(self, idx: int) -> dict:
        frame_id_str = self.sample_list[idx]
        frame_data = FrameDataLoader(kitti_locations=self.vod_kitti_locations, frame_number=frame_id_str)
        
        output_dict = {"meta": {"num_frame": frame_id_str, "frame_id": frame_id_str}}

        if self.mode == 'segmentation_finetune':
            try:
                pil_image = frame_data.image
                raw_labels = frame_data.raw_labels if self.split != 'test' else []
                if isinstance(pil_image, np.ndarray):
                    pil_image_for_transform = Image.fromarray(pil_image.astype(np.uint8)).convert('RGB')
                elif isinstance(pil_image, Image.Image):
                    pil_image_for_transform = pil_image.convert('RGB')
                else:
                    raise TypeError(f"Frame {frame_id_str}: frame_data.image is of unexpected type: {type(pil_image)}")

                output_dict['image'] = self.image_transform(pil_image_for_transform)
                output_dict['mask'] = self._generate_segmentation_mask(pil_image_for_transform, raw_labels)
            except FileNotFoundError as e:
                print(f"Error loading data for frame {frame_id_str} (segmentation): {e}")
                output_dict['image'] = torch.zeros(3, self.seg_target_h, self.seg_target_w)
                output_dict['mask'] = torch.zeros(self.seg_target_h, self.seg_target_w, dtype=torch.long)
            return output_dict

        elif self.mode == 'detection_pointpainting':
            try:
                output_dict['lidar_data'] = torch.from_numpy(frame_data.lidar_data.astype(np.float32))
            except FileNotFoundError:
                output_dict['lidar_data'] = torch.empty((0, 4), dtype=torch.float32)
            
            gt_labels_3d_list = []
            gt_bboxes_3d_list = []
            
            local_transforms = None # Initialize
            if self.split != 'test':
                try:
                    local_transforms = FrameTransformMatrix(frame_data)
                except Exception as e:
                    print(f"Warning: Failed to init FrameTransformMatrix for {frame_id_str}. Error: {e}")
                    
                if local_transforms:
                    raw_labels = frame_data.raw_labels
                    
                    # # <--- START DEBUGGING BLOCK 1: INITIAL COUNTS --->
                    # print(f"--- [Frame: {frame_id_str}] Processing Labels ---")
                    # initial_counts = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0, 'DontCare': 0, 'Other': 0}
                    # for label_str_debug in raw_labels:
                    #     cls_name = label_str_debug.split(' ')[0]
                    #     if cls_name in initial_counts:
                    #         initial_counts[cls_name] += 1
                    #     else:
                    #         initial_counts['Other'] += 1
                    # print(f"Initial objects loaded from file: {initial_counts}")
                    # # <--- END DEBUGGING BLOCK 1 --->

                    for label_str in raw_labels:
                        label_parts = label_str.split(' ')
                        if len(label_parts) < self.LABEL_MAPPING_VOD['bbox3d_rotation'] + 1: continue
                        class_name_3d = label_parts[self.LABEL_MAPPING_VOD['class']]
                        if class_name_3d in self.CLASSES_3D_DETECTION:
                            class_idx = self.CLASSES_3D_DETECTION.index(class_name_3d)
                            try:
                                loc_cam_str = label_parts[self.LABEL_MAPPING_VOD['bbox3d_location'].start : self.LABEL_MAPPING_VOD['bbox3d_location'].stop]
                                dim_cam_str = label_parts[self.LABEL_MAPPING_VOD['bbox3d_dimensions'].start : self.LABEL_MAPPING_VOD['bbox3d_dimensions'].stop]
                                rot_cam_str = label_parts[self.LABEL_MAPPING_VOD['bbox3d_rotation']]
                                loc_cam = np.array(list(map(float, loc_cam_str)))
                                dims_hwl_cam = np.array(list(map(float, dim_cam_str)))
                                rot_y_cam = float(rot_cam_str)
                            except (ValueError, IndexError): continue

                            loc_cam_homo = np.ones((1, 4)); loc_cam_homo[0, :3] = loc_cam
                            loc_lidar_homo = homogeneous_transformation(loc_cam_homo, local_transforms.t_lidar_camera)
                            loc_lidar = loc_lidar_homo[0, :3]
                            dims_lwh_lidar = dims_hwl_cam[[2, 1, 0]]
                            yaw_lidar = -rot_y_cam
                            yaw_lidar = (yaw_lidar + np.pi) % (2 * np.pi) - np.pi
                            
                            gt_labels_3d_list.append(class_idx)
                            gt_bboxes_3d_list.append(np.concatenate([loc_lidar, dims_lwh_lidar, [yaw_lidar]]))

                    # <--- START DEBUGGING BLOCK 2: FINAL COUNTS --->
                    # final_counts = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}
                    # for label_idx in gt_labels_3d_list:
                    #     class_name = self.CLASSES_3D_DETECTION[label_idx]
                    #     final_counts[class_name] += 1
                    # print(f"Objects remaining after processing: {final_counts}\n")
                    # # <--- END DEBUGGING BLOCK 2 --->
            
            if not gt_bboxes_3d_list:
                output_dict['gt_labels_3d'] = torch.empty((0,), dtype=torch.long)
                output_dict['gt_bboxes_3d'] = LiDARInstance3DBoxes(torch.empty((0, 7), dtype=torch.float32))
            else:
                output_dict['gt_labels_3d'] = torch.tensor(gt_labels_3d_list, dtype=torch.long)
                output_dict['gt_bboxes_3d'] = LiDARInstance3DBoxes(torch.tensor(np.array(gt_bboxes_3d_list), dtype=torch.float32))

            if self.load_image_for_pointpainting:
                try:
                    pil_image = frame_data.image
                    if isinstance(pil_image, np.ndarray):
                        pil_image_for_transform = Image.fromarray(pil_image.astype(np.uint8)).convert('RGB')
                    elif isinstance(pil_image, Image.Image):
                        pil_image_for_transform = pil_image.convert('RGB')
                    else:
                        raise TypeError(f"Frame {frame_id_str}: frame_data.image is unexpected type: {type(pil_image)}")

                    output_dict['image'] = self.image_transform_for_painting(pil_image_for_transform)
                    
                    if local_transforms:
                        P2_np = local_transforms.camera_projection_matrix 
                        Tr_velo_to_cam_np = local_transforms.t_camera_lidar 
                        
                        output_dict['calib_data'] = {
                            'P2': torch.from_numpy(P2_np.astype(np.float32)), 
                            'Tr_velo_to_cam': torch.from_numpy(Tr_velo_to_cam_np.astype(np.float32)),
                        }
                    else:
                        print(f"Warning: Using dummy calibration for {frame_id_str} due to missing transforms.")
                        output_dict['calib_data'] = {
                            'P2': torch.eye(3, 4, dtype=torch.float32), 
                            'Tr_velo_to_cam': torch.eye(4, 4, dtype=torch.float32),
                        }

                except (FileNotFoundError, AttributeError, TypeError) as e:
                    print(f"Error loading image/calib for PointPainting (frame {frame_id_str}): {e}")
                    output_dict['image'] = torch.zeros(3, self.seg_target_h, self.seg_target_w)
                    output_dict['calib_data'] = {
                        'P2': torch.eye(3, 4, dtype=torch.float32), 
                        'Tr_velo_to_cam': torch.eye(4, 4, dtype=torch.float32),
                    }
            return output_dict
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")