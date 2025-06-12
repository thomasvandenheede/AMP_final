import os
import numpy as np
from common_src.model.utils import LiDARInstance3DBoxes

import torch
from torch.utils.data import Dataset

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

class ViewOfDelft(Dataset):
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']
    LABEL_MAPPING = {
        'class': 0,
        'truncated': 1,
        'occluded': 2,
        'alpha': 3,
        'bbox2d': slice(4,8),
        'bbox3d_dimensions': slice(8,11),
        'bbox3d_location': slice(11,14),
        'bbox3d_rotation': 14,
    }

    def __init__(self, 
                 data_root='data/view_of_delft', 
                 sequential_loading=False,
                 split='train'):
        super().__init__()
        self.data_root = data_root
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}."
        self.split = split
        split_file = os.path.join(data_root, 'lidar', 'ImageSets', f'{split}.txt')
        with open(split_file, 'r') as f:
            lines = f.readlines()
        self.sample_list = [line.strip() for line in lines]
        self.vod_kitti_locations = KittiLocations(root_dir=data_root)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Determine current and previous frame IDs
        curr_frame = self.sample_list[idx]
        prev_frame = self.sample_list[idx-1] if idx > 0 else curr_frame

        # Load current frame data
        vod_curr = FrameDataLoader(kitti_locations=self.vod_kitti_locations,
                                   frame_number=curr_frame)
        tf_curr = FrameTransformMatrix(vod_curr)
        pts_curr = vod_curr.lidar_data  # Nx4

        # Load previous frame data
        vod_prev = FrameDataLoader(kitti_locations=self.vod_kitti_locations,
                                   frame_number=prev_frame)
        tf_prev = FrameTransformMatrix(vod_prev)
        pts_prev = vod_prev.lidar_data  # Mx4

        # Transform prev points from prev-LiDAR frame -> prev-camera -> curr-LiDAR frame
        device = torch.device('cpu')
        pts_prev_h = torch.cat([
            torch.tensor(pts_prev, dtype=torch.float32, device=device),
            torch.ones((pts_prev.shape[0], 1), dtype=torch.float32, device=device)
        ], dim=1)  # (M,4)
       # prev LiDAR -> prev camera
        cam_prev = homogeneous_transformation(pts_prev_h.numpy(), tf_prev.t_camera_lidar)
        cam_prev = torch.tensor(cam_prev, dtype=torch.float32, device=device)
        # prev camera -> curr LiDAR
        lidar_prev_in_curr = homogeneous_transformation(cam_prev.numpy(), tf_curr.t_lidar_camera)
        lidar_prev_in_curr = torch.tensor(lidar_prev_in_curr, dtype=torch.float32, device=device)
        pts_prev_trans = lidar_prev_in_curr[:, :3]  # (M,3)

        # Prepare combined points with intensity and frame flag
        # original pts: [x,y,z,intensity]
        intens_curr = torch.tensor(pts_curr[:,3], dtype=torch.float32, device=device).unsqueeze(1)
        pts_curr_xyz = torch.tensor(pts_curr[:, :3], dtype=torch.float32, device=device)
        flag_curr = torch.ones((pts_curr_xyz.shape[0],1), device=device)
        data_curr = torch.cat([pts_curr_xyz, intens_curr, flag_curr], dim=1)

        intens_prev = cam_prev.new_zeros((pts_prev_trans.shape[0],1))  # no intensity for prev, set 0
        flag_prev = torch.zeros((pts_prev_trans.shape[0],1), device=device)
        data_prev = torch.cat([pts_prev_trans, intens_prev, flag_prev], dim=1)

        lidar_data = torch.cat([data_prev, data_curr], dim=0)  # ((M+N),5)

        # Load GT for current frame only
        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != 'test':
            raw_labels = vod_curr.raw_labels
            for lbl in raw_labels:
                parts = lbl.split(' ')
                cls = parts[self.LABEL_MAPPING['class']]
                if cls in self.CLASSES:
                    gt_labels_3d_list.append(self.CLASSES.index(cls))
                    # parse and transform to LiDAR coords
                    loc_cam = np.array(parts[self.LABEL_MAPPING['bbox3d_location']], dtype=np.float32)
                    homo_cam = np.ones((1,4), dtype=np.float32)
                    homo_cam[0,:3] = loc_cam
                    loc_lid = homogeneous_transformation(homo_cam, tf_curr.t_lidar_camera)
                    dims = np.array(parts[self.LABEL_MAPPING['bbox3d_dimensions']], dtype=np.float32)[[2,1,0]]
                    rot = np.array([parts[self.LABEL_MAPPING['bbox3d_rotation']]], dtype=np.float32)
                    gt_bboxes_3d_list.append(np.concatenate([loc_lid[0,:3], dims, rot], axis=0))

        if len(gt_bboxes_3d_list)==0:
            gt_labels_3d = torch.zeros((1,), dtype=torch.int64)
            gt_bboxes_np = np.zeros((1,7), dtype=np.float32)
        else:
            gt_labels_3d = torch.tensor(gt_labels_3d_list, dtype=torch.int64)
            gt_bboxes_np = np.stack(gt_bboxes_3d_list, axis=0)

        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_np, box_dim=7, origin=(0.5,0.5,0))

        return {
            'pts': [lidar_data],
            'gt_labels_3d': [gt_labels_3d],
            'gt_bboxes_3d': [gt_bboxes_3d],
            'metas': [{'num_frame': curr_frame}]
        }

