# In common_src/dataset_segmentation.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
# Crucially, import FrameDataLoader and KittiLocations from your VoD tools
from vod.frame import FrameDataLoader 
from vod.configuration import KittiLocations # Assuming you pass kitti_locations to the dataset

class SegmentationDataset(Dataset):
    def __init__(self, kitti_locations_obj: KittiLocations, frame_ids_list: list, 
                 class_mapping: dict, desired_output_size: tuple, 
                 image_transform=None, # For image normalization, ToTensor etc.
                 # Spatial transforms like resize, flip should be handled carefully for both image and mask
                 ):
        self.kitti_locations = kitti_locations_obj
        self.frame_ids = frame_ids_list
        self.class_mapping = class_mapping # e.g., {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Background': 0}
        self.num_segmentation_classes = len(class_mapping)
        self.output_size_h, self.output_size_w = desired_output_size
        self.image_transform = image_transform

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id_str = self.frame_ids[idx]

        # Use FrameDataLoader to get image and annotations
        frame_data_loader = FrameDataLoader(kitti_locations=self.kitti_locations, frame_number=frame_id_str)

        try:
            pil_image = frame_data_loader.image # This is a PIL Image [cite: 1]
            annotations = frame_data_loader.raw_labels # List of annotation objects [cite: 1]
        except FileNotFoundError:
            print(f"Data not found for frame {frame_id_str}. Skipping or returning dummy.")
            # Handle appropriately, e.g., return a dummy image and mask
            dummy_img = torch.zeros(3, self.output_size_h, self.output_size_w)
            dummy_mask = torch.zeros(self.output_size_h, self.output_size_w, dtype=torch.long)
            return dummy_img, dummy_mask

        original_h, original_w = pil_image.height, pil_image.width

        # 1. Create an empty mask (initialized to background class ID)
        # The class_mapping should define which ID is background, e.g., 0
        background_class_id = self.class_mapping.get('Background', 0)
        target_mask_np = np.full((original_h, original_w), fill_value=background_class_id, dtype=np.uint8)

        # 2. Fill mask based on 2D bounding box annotations
        for ann_obj in annotations:
            class_name = ann_obj.cls_type # e.g., 'Car', 'Pedestrian' [cite: 1]
            if class_name in self.class_mapping:
                class_id = self.class_mapping[class_name]

                # ann_obj.bbox_2d contains [left, top, right, bottom] pixel coordinates [cite: 1]
                left, top, right, bottom = ann_obj.bbox_2d 

                # Ensure coordinates are integers and within image bounds
                left_int = int(max(0, np.floor(left)))
                top_int = int(max(0, np.floor(top)))
                right_int = int(min(original_w, np.ceil(right)))
                bottom_int = int(min(original_h, np.ceil(bottom)))

                if right_int > left_int and bottom_int > top_int: # Check for valid box
                    target_mask_np[top_int:bottom_int, left_int:right_int] = class_id

        # 3. Apply transformations
        # Image transforms (ToTensor, Normalize, potentially Resize)
        if self.image_transform:
            img_tensor = self.image_transform(pil_image)
        else: # Basic ToTensor if no transform provided
            img_tensor = transforms.ToTensor()(pil_image)


        # Mask transforms (Resize if needed, then ToTensor)
        # Resize mask to the same output size as the image if SimpleUNet expects fixed size
        # Use NEAREST interpolation for masks to preserve class IDs
        mask_pil = Image.fromarray(target_mask_np)
        resized_mask_pil = mask_pil.resize((self.output_size_w, self.output_size_h), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(resized_mask_pil)).long() # (H, W)

        return img_tensor, mask_tensor

# Example usage of this dataset (conceptual, you'd get frame_ids from a train/val split file)
# from torchvision import transforms
# image_transform = transforms.Compose([
#     transforms.Resize((256, 256)), 
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
# mask_transform_resize = transforms.Resize((256,256), interpolation=transforms.InterpolationMode.NEAREST)

# train_seg_dataset = SegmentationDataset(
#     image_dir="view_of_delft/lidar/training/image_2",
#     mask_dir="/path/to/your/generated_segmentation_masks/training",
#     frame_ids=["000000", "000001", ...], # Populate with actual frame IDs
#     transform=image_transform,
#     # mask_transform=lambda m: mask_transform_resize(Image.fromarray(m)) # if mask needs resize
# )