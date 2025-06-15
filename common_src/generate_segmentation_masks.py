# generate_segmentation_masks.py
import os
import numpy as np
from PIL import Image

# --- Configuration ---
# Root directory of your View of Delft dataset
DATA_ROOT = "/home/mramidi/final_assignment/data/view_of_delft/lidar"


# Input directories relative to DATA_ROOT
IMAGE_DIR_RELATIVE = "training/image_2" # Adjust if your images are in a different split folder
LABEL_DIR_RELATIVE = "training/label_2" # Adjust if your labels are in a different split folder
# Note: For validation, you'll need to adapt these paths or run the script again
# with appropriate validation paths.

# Output root directory for generated masks
OUTPUT_MASK_ROOT = "/home/mramidi/final_assignment/generated_masks_for_finetuning"

# --- Finalized Target Class Mapping ---
# This mapping defines the integer IDs for your *generated* masks.
# 0 is always background.
# Ensure 'Cyclist' matches the label in your KITTI files if it exists, otherwise 'bicycle'.
# Based on the sample .txt, 'bicycle' is present, so let's stick with that for generation.
# If your KITTI labels explicitly use 'Cyclist', change 'bicycle' to 'Cyclist' here.
CLASS_MAPPING_FINETUNE = {
    'Background': 0, # Default for unmapped classes and empty space
    'Car': 1,
    'Pedestrian': 2,
    'Cyclist': 3, # Assuming KITTI labels use 'bicycle'
    # Map 'truck', 'motor', 'ride_other' to 'Background' (0) for simplicity as per feedback.
    # If you want a general 'Vehicle' class, you'd add it here.
}
# Reverse mapping for easy lookup (optional, for debugging)
ID_TO_CLASS_NAME = {v: k for k, v in CLASS_MAPPING_FINETUNE.items()}


# --- Function to Create Semantic Mask from KITTI Labels ---
def create_mask_from_kitti_labels(image_path, label_txt_path, output_mask_path, class_mapping):
    """
    Generates a semantic segmentation mask (PNG) from KITTI-format .txt labels.
    Fills bounding boxes with corresponding class IDs from class_mapping.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Skipping mask generation for this frame.")
        return False # Indicate failure

    mask_np = np.zeros((height, width), dtype=np.uint8) # Initialize with background (0)

    if not os.path.exists(label_txt_path):
        # If label file doesn't exist, save an empty mask (all background)
        mask_img = Image.fromarray(mask_np, mode='L')
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        mask_img.save(output_mask_path)
        print(f"Warning: Label TXT file not found for {os.path.basename(image_path)}. Saved empty mask.")
        return True

    with open(label_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if not parts: continue

            class_name = parts[0]
            # Get class ID from the finetune mapping, default to Background (0)
            class_id = class_mapping.get(class_name, class_mapping['Background'])

            # Extract bbox coordinates: left, top, right, bottom
            # KITTI bbox is (left, top, right, bottom)
            bbox_left = int(float(parts[4]))
            bbox_top = int(float(parts[5]))
            bbox_right = int(float(parts[6]))
            bbox_bottom = int(float(parts[7]))

            # Clamp bbox coordinates to image boundaries
            bbox_left = max(0, bbox_left)
            bbox_top = max(0, bbox_top)
            bbox_right = min(width, bbox_right)
            bbox_bottom = min(height, bbox_bottom)

            # Fill the rectangle in the mask with the class ID
            if bbox_right > bbox_left and bbox_bottom > bbox_top: # Ensure valid box
                mask_np[bbox_top:bbox_bottom, bbox_left:bbox_right] = class_id

    # Save the generated mask
    mask_img = Image.fromarray(mask_np, mode='L') # 'L' mode for single-channel grayscale
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    mask_img.save(output_mask_path)
    print(f"Generated mask for {os.path.basename(image_path)} to {output_mask_path}")
    return True

# --- Main Batch Processing Logic ---
def batch_generate_masks(split_file, output_split_dir, class_mapping):
    """
    Processes all frames listed in a split file (e.g., train.txt)
    to generate segmentation masks.
    """
    split_file_path = os.path.join(DATA_ROOT, "ImageSets", split_file)
    print(f"\n--- Processing split: {split_file_path} ---")

    if not os.path.exists(split_file_path):
        print(f"Error: Split file '{split_file_path}' not found. Skipping this split.")
        return

    with open(split_file_path, 'r') as f:
        frame_ids = [line.strip() for line in f if line.strip()]

    num_generated = 0
    num_skipped = 0

    for frame_id in frame_ids:
        image_path = os.path.join(DATA_ROOT, IMAGE_DIR_RELATIVE, f"{frame_id}.jpg")
        label_txt_path = os.path.join(DATA_ROOT, LABEL_DIR_RELATIVE, f"{frame_id}.txt")
        output_mask_path = os.path.join(OUTPUT_MASK_ROOT, output_split_dir, f"{frame_id}.png")

        success = create_mask_from_kitti_labels(image_path, label_txt_path, output_mask_path, class_mapping)
        if success:
            num_generated += 1
        else:
            num_skipped += 1

    print(f"\n--- Summary for {split_file} ---")
    print(f"Total frames: {len(frame_ids)}")
    print(f"Masks generated: {num_generated}")
    print(f"Masks skipped (image/label not found): {num_skipped}")


if __name__ == "__main__":
    # Ensure the output base directory exists
    os.makedirs(OUTPUT_MASK_ROOT, exist_ok=True)

    # Process training set
    batch_generate_masks("train.txt", "training", CLASS_MAPPING_FINETUNE)

    # Process validation set (assuming a val.txt exists and similar directory structure)
    # You might need to adjust IMAGE_DIR_RELATIVE and LABEL_DIR_RELATIVE for validation data
    # if it's in a separate folder like 'validation/image_2'
    # batch_generate_masks("val.txt", "validation", CLASS_MAPPING_FINETUNE)
    # If your val.txt frames use images/labels from 'training' folder, no need to change IMAGE_DIR_RELATIVE/LABEL_DIR_RELATIVE
    # If you only have a train.txt and need to split, you'll have to manually create val.txt first.

    print("\nBatch mask generation script finished.")
    print(f"Generated masks saved under: {OUTPUT_MASK_ROOT}")