import os
import sys
import numpy as np
import open3d as o3d
import torch

# --- Add common_src to path to import your custom modules ---
# Adjust this path if your script is in a different location relative to common_src
root = os.path.abspath(os.path.join(os.getcwd()))
if root not in sys.path:
    sys.path.insert(0, root)

try:
    # Assuming your ViewOfDelft class is in common_src/dataset/view_of_delft.py
    from common_src.dataset import ViewOfDelft 
except ImportError:
    print("Error: Could not import ViewOfDelft. Make sure the path to common_src is correct.")
    print("And your dataset file is at 'common_src/dataset/view_of_delft.py'")
    sys.exit(1)

# --- Configuration ---
# !!! IMPORTANT: SET YOUR DATA_ROOT PATH !!!
DATA_ROOT = "/home/mramidi/final_assignment/view_of_delft" 
# You can change this to 'val' or 'test' to see data from other splits
SPLIT = 'train' 
# Change this to the frame number you want to visualize
FRAME_ID_TO_VISUALIZE = "01340" 

def visualize_sample(data_dict: dict):
    """
    Visualizes LiDAR point cloud and 3D ground truth boxes using Open3D.
    """
    lidar_points = data_dict['lidar_data'][:, :3].numpy()
    gt_boxes = data_dict['gt_bboxes_3d']
    gt_labels = data_dict['gt_labels_3d'].numpy()

    print(f"Visualizing Frame: {data_dict['meta']['frame_id']}")
    print(f"Number of LiDAR points: {len(lidar_points)}")
    print(f"Number of GT boxes: {len(gt_boxes)}")

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)
    pcd.paint_uniform_color([0.8, 0.8, 0.8]) # Paint points gray

    # Define colors for different classes
    # Car: Red, Pedestrian: Green, Cyclist: Blue
    class_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    box_geometries = []
    if len(gt_boxes) > 0:
        box_corners = gt_boxes.corners.numpy()
        for i, corners in enumerate(box_corners):
            class_idx = gt_labels[i]
            # Lines connecting the 8 corners of a box
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            color = class_colors[class_idx]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
            box_geometries.append(line_set)
            
            print(f"- Box {i+1}: Class '{ViewOfDelft.CLASSES_3D_DETECTION[class_idx]}', Color: {'Red' if class_idx==0 else 'Green' if class_idx==1 else 'Blue'}")


    # Add point cloud and all box geometries to the list of things to draw
    geometries_to_draw = [pcd] + box_geometries
    
    # Create a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])
    geometries_to_draw.append(coord_frame)

    print("\nStarting Open3D visualizer...")
    print("Close the window to exit the script.")
    o3d.visualization.draw_geometries(geometries_to_draw)

if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT) or "path/to" in DATA_ROOT:
        print("Error: Please set the 'DATA_ROOT' variable in this script to your dataset's location.")
    else:
        # We need a list of frame IDs for the dataset constructor
        # We can just use the single frame we want to visualize
        sample_list = [FRAME_ID_TO_VISUALIZE]
        
        # Instantiate the dataset
        vod_dataset = ViewOfDelft(
            data_root=DATA_ROOT,
            split=SPLIT,
            mode='detection_pointpainting',
            load_image_for_pointpainting=False # No need for images for this visualization
        )
        
        if len(vod_dataset) == 0:
            print(f"Error: Could not find frame '{FRAME_ID_TO_VISUALIZE}' in the dataset for split '{SPLIT}'.")
        else:
            # Get the data dictionary for our specific frame
            data_sample = vod_dataset[0] 
            
            # Visualize it!
            visualize_sample(data_sample)