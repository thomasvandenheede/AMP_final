# common_src/dataset/vod_multi_mod.py

import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix
from common_src.dataset import collate_vod_batch  # import the same collate function you already wrote
from common_src.model.utils import LiDARInstance3DBoxes

# Re‐use the same class list and label‐parsing logic as the original VoD loader:
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
LABEL_MAPPING = {
    'class': 0,
    'bbox3d_dimensions': slice(8, 11),  # (h, w, l)
    'bbox3d_location': slice(11, 14),   # (x,y,z) in camera coords
    'bbox3d_rotation': 14               # yaw in camera coords (LiDAR frame)
}


class VoDMultiModal(Dataset):
    """
    View‐of‐Delft dataset for LiDAR+Camera fusion.
    Returns a dict with keys:
      - pts        : (N,4) torch.FloatTensor of raw LiDAR points [x,y,z,int]
      - img        : (3,H_img,W_img) torch.FloatTensor, normalized [0,1]
      - depth_map  : (1,H_img,W_img) torch.FloatTensor of z (in camera frame) 
                     at pixels where a LiDAR point projected; NaN elsewhere
      - gt_labels_3d : torch.LongTensor  (#objs,)
      - gt_bboxes_3d : LiDARInstance3DBoxes  (#objs × 7)
      - metas: dict with 'sample_id', 'K' (3×3), 'T_cam_lidar' (4×4), 'image_shape'
    """

    def __init__(self,
                 data_root: str = 'data/view_of_delft',
                 split: str = 'train',
                 img_size: tuple = (1216, 1936),
                 img_transform: torch.nn.Module = None):
        super().__init__()

        self.data_root = data_root
        self.split = split  # 'train', 'val', or 'test'
        self.img_size = img_size  # (H_img, W_img)

        self.kitti_locations = KittiLocations(root_dir=self.data_root)

        # 1) Build sample_list from ImageSets/{split}.txt
        split_file = osp.join(self.data_root, 'lidar', 'ImageSets', f'{split}.txt')
        with open(split_file, 'r') as f:
            lines = [ln.strip() for ln in f.readlines()]
        self.sample_list = lines  # e.g., ['000001', '000002', ...]

        # 2) Image directory: images live flat under data_root/image_2/{sample_id}.png
        # choose the real folder names
        self.img_dir = osp.join(self.data_root, 'lidar', 'training', 'image_2')

        # 3) Default image transform: convert to tensor [0..1]
        if img_transform is None:
            self.img_transform = T.Compose([
                T.ToTensor(),  # PIL→(C,H,W) float [0,1]
            ])
        else:
            self.img_transform = img_transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """
        Returns a dictionary:
          {
            'pts'         : (N,4) FloatTensor,
            'img'         : (3,H_img,W_img) FloatTensor,
            'depth_map'   : (1,H_img,W_img) FloatTensor (NaNs for no-depth),
            'gt_labels_3d': LongTensor (#objs,),
            'gt_bboxes_3d': LiDARInstance3DBoxes (#objs,7),
            'metas'       : dict(sample_id, K, T_cam_lidar, image_shape)
          }
        """
        sample_id = self.sample_list[idx]
        num_frame = sample_id  # keep original naming

        # ---------------------------
        # 1) Load LiDAR + labels
        # ---------------------------
        frame_data = FrameDataLoader(
            kitti_locations=self.kitti_locations,
            frame_number=sample_id
        )

        # Raw LiDAR points: numpy (N,4)
        # Each row = [x, y, z, intensity]
        lidar_np = frame_data.lidar_data
        pts = torch.from_numpy(lidar_np).float()  # (N,4)

        # Build 3D labels exactly as in original loader:
        gt_labels_list = []
        gt_bboxes_list = []
        if self.split != 'test':
            # transform from camera to LiDAR
            transforms = FrameTransformMatrix(frame_data)
            T_cam_lidar = transforms.t_camera_lidar  # 4×4 array
            for obj_str in frame_data.raw_labels:
                parts = obj_str.strip().split()
                cls_str = parts[LABEL_MAPPING['class']]
                if cls_str not in CLASSES:
                    continue
                gt_labels_list.append(CLASSES.index(cls_str))

                # dims: [h, w, l] → reorder to [l, w, h]
                h, w, l = map(float, parts[LABEL_MAPPING['bbox3d_dimensions']])
                dims_lwh = [l, w, h]

                # camera → LiDAR for location
                loc_cam = np.array(list(map(float, parts[LABEL_MAPPING['bbox3d_location']])))
                loc_cam_h = np.append(loc_cam, 1.0)  # (4,)
                loc_lidar = (T_cam_lidar @ loc_cam_h)[:3]  # (3,)

                rot = float(parts[LABEL_MAPPING['bbox3d_rotation']])
                gt_bboxes_list.append(
                    np.array([*loc_lidar, *dims_lwh, rot], dtype=np.float32)
                )

            if len(gt_bboxes_list) == 0:
                # no objects: dummy zero
                gt_labels = torch.zeros((1,), dtype=torch.long)
                gt_bboxes_np = np.zeros((1, 7), dtype=np.float32)
            else:
                gt_labels = torch.tensor(gt_labels_list, dtype=torch.long)
                gt_bboxes_np = np.stack(gt_bboxes_list, axis=0)
            gt_bboxes_3d = LiDARInstance3DBoxes(
                gt_bboxes_np, box_dim=7, origin=(0.5, 0.5, 0.5)
            )
        else:
            # In 'test' split, we do not have labels
            gt_labels = torch.zeros((1,), dtype=torch.long)
            gt_bboxes_3d = LiDARInstance3DBoxes(
                np.zeros((1, 7), dtype=np.float32),
                box_dim=7,
                origin=(0.5, 0.5, 0.5)
            )

        # ---------------------------
        # 2) Load camera image
        # ---------------------------
        img_path = osp.join(self.img_dir, f'{sample_id}.jpg')
        img_pil  = Image.open(img_path).convert('RGB')
        W_img, H_img = img_pil.size
        # Optional: resize if (H_img,W_img) != self.img_size
        if (H_img, W_img) != self.img_size:
            img_pil = img_pil.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            H_img, W_img = self.img_size

        img_tensor = self.img_transform(img_pil)  # (3, H_img, W_img) float

        # ---------------------------
        # 3) Build sparse depth map
        # ---------------------------
        # 3.1  Get calibration matrices
        transforms = FrameTransformMatrix(frame_data)
        T_cam_lidar = transforms.t_camera_lidar  # 4×4 (camera ← lidar)
        P = transforms.camera_projection_matrix       # 3×4 (img ← camera)
        K = P[:, :3]                              # numpy (3×3)

        # 3.2  Convert LiDAR points to camera coords
        pts_xyz = lidar_np[:, :3]  # (N,3)
        N_pts = pts_xyz.shape[0]
        ones = np.ones((N_pts, 1), dtype=np.float32)
        pts_hom = np.concatenate((pts_xyz, ones), axis=1)  # (N,4)

        # (N,4) → (N,4) camera frame
        pts_cam_hom = (T_cam_lidar @ pts_hom.T).T       # (N,4)
        pts_cam = pts_cam_hom[:, :3]                    # (N,3)

        # (N,3) → (N,3) pixel homogeneous via P
        uvw = (P @ pts_cam_hom.T).T                     # (N,3)
        z_cam = pts_cam[:, 2]                           # (N,)
        # Avoid dividing by zero
        eps = 1e-6
        us = uvw[:, 0] / (uvw[:, 2] + eps)
        vs = uvw[:, 1] / (uvw[:, 2] + eps)

        # Round to nearest integer pixel
        u_int = np.round(us).astype(int)
        v_int = np.round(vs).astype(int)

        # Prepare depth_map full of NaNs
        depth_map = torch.full(
            (1, H_img, W_img), float('nan'), dtype=torch.float32
        )

        # Populate only valid projections
        valid_mask = (
            (u_int >= 0) & (u_int < W_img) &
            (v_int >= 0) & (v_int < H_img) &
            (z_cam > 0)
        )
        u_valid = u_int[valid_mask]
        v_valid = v_int[valid_mask]
        z_valid = z_cam[valid_mask]

        depth_map[0, v_valid, u_valid] = torch.from_numpy(z_valid).float()

        # ---------------------------
        # 4) Build metas dict
        # ---------------------------
        metas = {
            'sample_id': sample_id,
            # you can rename this to 'P' if you like, but it's the 3×4 proj. matrix
            'K': torch.from_numpy(K).float(),              # now CenterPointFusion will see metas[i]['K']
            'P': torch.from_numpy(transforms.camera_projection_matrix).float(),
            'T_cam_lidar': torch.from_numpy(T_cam_lidar).float(),
            'image_shape': torch.tensor([H_img, W_img], dtype=torch.int64),
        }

        return {
            'pts': pts,                          # (N,4) FloatTensor
            'img': img_tensor,                   # (3, H_img, W_img)
            'depth_map': depth_map,              # (1, H_img, W_img)
            'gt_labels_3d': gt_labels,           # LongTensor (#objs,)
            'gt_bboxes_3d': gt_bboxes_3d,        # LiDARInstance3DBoxes
            'metas': metas
        }


def build_vod_multi_mod_dataset(data_root: str,
                                split: str = 'train',
                                img_size=(1216, 1936),
                                img_transform=None,
                                **kwargs):
    """
    Helper function if you want to construct the dataset from a config.
    Returns a VoDMultiModal instance and a collate function.
    """
    ds = VoDMultiModal(data_root=data_root,
                       split=split,
                       img_size=img_size,
                       img_transform=img_transform)
    return ds, collate_vod_batch
