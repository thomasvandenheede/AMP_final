import os
import sys
import numpy as np
import argparse
from PIL import Image, ImageDraw

# --- Add Project Root to Python Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # Adjust if your script is nested differently
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your dataset class and VOD utilities
from common_src.dataset.view_of_delft import ViewOfDelft
from vod.frame import FrameDataLoader, FrameTransformMatrix

def draw_projected_3d_box(draw, corners_2d, color='red', width=2):
    """
    Draws a 3D bounding box on a PIL ImageDraw object.
    The corners are expected to be in a specific order.
    """
    # Define connections between the 8 corners
    # (Front face, Back face, Connecting lines)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting lines
    ]
    
    for i, j in edges:
        # Get the coordinates for the start and end points of the line
        start_point = (corners_2d[i, 0], corners_2d[i, 1])
        end_point = (corners_2d[j, 0], corners_2d[j, 1])
        draw.line([start_point, end_point], fill=color, width=width)

def main(args):
    """
    Loads a frame, processes its 3D ground truth, projects the 3D boxes
    onto the 2D image, and saves the result.
    """
    print(f"--- Projecting Ground Truth for Frame: {args.frame_id} ---")
    
    # 1. Instantiate the dataset to get access to the data loading and processing logic
    try:
        # We need the calib data, so we run in detection_pointpainting mode
        # but only care about the GT and calib parts
        dataset = ViewOfDelft(
            data_root=args.data_root,
            split=args.split,
            mode='detection_pointpainting',
            load_image_for_pointpainting=True 
        )
    except FileNotFoundError as e:
        print(f"Error: Could not initialize dataset. Check data_root path. Details: {e}")
        return

    try:
        sample_idx = dataset.sample_list.index(args.frame_id)
    except ValueError:
        print(f"Error: Frame ID '{args.frame_id}' not found in the '{args.split}' split.")
        return

    # 2. Get the processed data dictionary for the specific frame
    data_dict = dataset[sample_idx]
    
    gt_boxes_3d = data_dict['gt_bboxes_3d'] # LiDARInstance3DBoxes object
    gt_labels = data_dict['gt_labels_3d'].numpy()
    
    # Load the original image using FrameDataLoader for drawing
    frame_data_loader = FrameDataLoader(kitti_locations=dataset.vod_kitti_locations, frame_number=args.frame_id)
    image_data = frame_data_loader.image # This might be a NumPy array

    # ---> FIX: Ensure the image is a PIL Image object before drawing <---
    if isinstance(image_data, np.ndarray):
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image_data.astype(np.uint8)).convert('RGB')
    elif isinstance(image_data, Image.Image):
        image = image_data.convert('RGB')
    else:
        raise TypeError(f"Loaded image data is of an unexpected type: {type(image_data)}")
        
    draw = ImageDraw.Draw(image)

    # 3. Get calibration matrices
    # Use FrameTransformMatrix to get the correct, processed matrices
    transforms = FrameTransformMatrix(frame_data_loader)
    P2 = transforms.camera_projection_matrix
    T_cam_from_lidar = transforms.t_camera_lidar

    print(f"Found {gt_boxes_3d.tensor.shape[0]} ground truth objects to project.")

    # 4. Project each box and draw it
    class_colors = { 0: "red", 1: "lime", 2: "blue" } # Car, Pedestrian, Cyclist

    # Get the 8 corners of all boxes at once (in LiDAR coordinates)
    box_corners_lidar = gt_boxes_3d.corners.numpy() 

    for i in range(box_corners_lidar.shape[0]):
        corners3d_lidar = box_corners_lidar[i] # Shape (8, 3)
        
        # Make points homogeneous for transformation
        corners3d_lidar_homo = np.hstack([corners3d_lidar, np.ones((8, 1))])
        
        # Project points from LiDAR to Camera frame
        corners3d_cam_homo = corners3d_lidar_homo @ T_cam_from_lidar.T
        
        # Filter out points that are behind the camera
        points_in_front_of_camera = corners3d_cam_homo[:, 2] > 0.1
        if not np.all(points_in_front_of_camera):
            continue
            
        # Project to image plane
        corners2d_homo = corners3d_cam_homo @ P2.T
        
        # Perspective divide to get pixel coordinates
        corners2d = corners2d_homo[:, :2] / corners2d_homo[:, 2, np.newaxis]
        
        # Draw the box on the image
        color = class_colors.get(gt_labels[i], "yellow")
        draw_projected_3d_box(draw, corners2d, color=color)

    # 5. Save the output image
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.frame_id}_gt_projection.jpg")
    image.save(output_path, "JPEG")
    print(f"Successfully saved visualization to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Project VoD Ground Truth 3D Boxes onto 2D Images")
    parser.add_argument('frame_id', type=str, help="The frame ID to visualize (e.g., '000123').")
    parser.add_argument('--data_root', type=str, default='/home/mramidi/final_assignment/view_of_delft', 
                        help='Root directory of the View-of-Delft dataset.')
    parser.add_argument('--output_dir', type=str, default='./visualization_output',
                        help='Directory where the output images will be saved.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help="The dataset split to use ('train' or 'val').")
    
    args = parser.parse_args()
    
    # ---> CORRECTED LOGIC <---
    # This block now just prints a warning if a default path is used, but ALWAYS continues to run main().
    default_paths = ['/path/to/your/View-of-Delft-Dataset', '/home/mramidi/final_assignment/view_of_delft']
    if args.data_root in default_paths:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Using a default data_root path. Please ensure this is correct. !!!")
        print(f"!!! Path being used: {args.data_root} !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    # Always call the main function after parsing arguments
    main(args)
