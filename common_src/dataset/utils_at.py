import torch
from torch.utils.data import DataLoader

def collate_vod_batch(batch):
    pts_list       = [s['pts']          for s in batch]
    gt_labels_list = [s['gt_labels_3d'] for s in batch]
    gt_boxes_list  = [s['gt_bboxes_3d'] for s in batch]
    metas_list     = [s['metas']        for s in batch]

    # stack into batched tensors
    img_batch       = torch.stack([s['img']       for s in batch], dim=0)
    depth_map_batch = torch.stack([s['depth_map'] for s in batch], dim=0)

    return dict(
        pts          = pts_list,
        img          = img_batch,
        depth_map    = depth_map_batch,
        gt_labels_3d = gt_labels_list,
        gt_bboxes_3d = gt_boxes_list,
        metas        = metas_list,
    )
