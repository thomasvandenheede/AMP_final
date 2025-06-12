def collate_vod_batch(batch):
    pts_list = []
    img_list = []
    gt_labels_3d_list = []
    gt_bboxes_3d_list = []
    k_list = []
    t_list = []
    meta_list = []
    for idx, sample in enumerate(batch):
        pts_list.append(sample['lidar_data'])
        img_list.append(sample['img'])
        gt_labels_3d_list.append(sample['gt_labels_3d'])
        gt_bboxes_3d_list.append(sample['gt_bboxes_3d'])
        k_list.append(sample['K'])
        t_list.append(sample['T_lidar_camera'])
        meta_list.append(sample['meta'])
    return dict(
        pts = pts_list,
        img = img_list,
        gt_labels_3d = gt_labels_3d_list,
        gt_bboxes_3d = gt_bboxes_3d_list,
        K = k_list,
        T_lidar_camera = t_list,
        metas = meta_list
    )