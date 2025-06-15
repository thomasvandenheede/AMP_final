import torch

def collate_vod_batch(batch):
    pts_list = []
    gt_labels_3d_list = []
    gt_bboxes_3d_list = []
    meta_list = []
    
    # Temporarily store image and calibration data from each sample
    temp_images = []
    temp_calib_data = []

    # Flag to check if all samples in the current batch have both image and calibration data
    all_samples_have_image_and_calib = True

    for sample in batch:
        pts_list.append(sample['lidar_data'])
        gt_labels_3d_list.append(sample['gt_labels_3d'])
        gt_bboxes_3d_list.append(sample['gt_bboxes_3d'])
        meta_list.append(sample['meta'])
        
        # Check if 'image' and 'calib_data' exist and are not None for the current sample
        has_image = 'image' in sample and sample['image'] is not None
        has_calib = 'calib_data' in sample and sample['calib_data'] is not None

        if has_image and has_calib:
            temp_images.append(sample['image'])
            temp_calib_data.append(sample['calib_data'])
        else:
            # If even one sample is missing image or calib_data,
            # the entire batch cannot be processed with image fusion for this round.
            all_samples_have_image_and_calib = False
            # We don't append anything to temp_images/calib_data from this sample if it's incomplete,
            # as the final output for images/calib_data will be None.

    # Form the final 'images' and 'calib_data' for the batch
    final_images_batch = None
    final_calib_data_batch = None # For calibration data, keep it as a list if present, else None

    if all_samples_have_image_and_calib:
        # If ALL samples in the batch have images and calibration data, stack the images
        # Assuming your ViewOfDelft dataset ensures `sample['image']` is a torch.Tensor when it's present.
        final_images_batch = torch.stack(temp_images)
        
        # For calibration data, it's typically a list of dictionaries or transformation matrices,
        # not something that can be directly stacked into a single tensor.
        final_calib_data_batch = temp_calib_data
    else:
        # If any sample was missing image/calib_data, then set the batch's image and calib_data to None.
        # This signals CenterPoint to proceed with LiDAR-only processing for this entire batch.
        final_images_batch = None
        final_calib_data_batch = None

    return dict(
        pts = pts_list,
        gt_labels_3d = gt_labels_3d_list,
        gt_bboxes_3d = gt_bboxes_3d_list,
        metas = meta_list,
        images = final_images_batch,      # This will now be a batched Tensor or None
        calib_data = final_calib_data_batch # This will now be a list of calib_data (if all present) or None
    )