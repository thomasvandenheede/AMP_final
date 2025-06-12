import os
import sys
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

# ====== User-configurable settings ======
INDEX = 0               # change this line to display a different sample index
DATA_ROOT = 'data/view_of_delft'
SPLIT = 'test'
# =======================================

# Ensure project root is on Python path to locate common_src and vod modules
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, '..', '..'))
sys.path.insert(0, project_root)

from vod_multi_mod import VoDMultiModal

# Initialize dataset
dataset = VoDMultiModal(data_root=DATA_ROOT, split=SPLIT)
print(f"Total samples in dataset: {len(dataset)}")

# Fetch and print the specified sample
sample = dataset[INDEX]
print(f"\n--- MultiModal Sample {INDEX} ---")
pprint(sample)

# Extract image and depth map
img_tensor = sample['img']                # (3, H, W)
depth_tensor = sample['depth_map']        # (1, H, W)

# Convert to numpy for plotting
img = img_tensor.permute(1, 2, 0).cpu().numpy()
depth = depth_tensor[0].cpu().numpy()    # drop channel dim

# Prepare depth for visualization: replace NaN with 0
# and scale to valid range
depth_vis = np.nan_to_num(depth, nan=0.0)
max_val = np.nanmax(depth)
if max_val <= 0:
    print("Warning: depth map contains no positive values.")
    vmax = 1.0
else:
    vmax = max_val

# Plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img)
ax1.set_title('RGB Image')
ax1.axis('off')

im = ax2.imshow(depth_vis, cmap='plasma', vmin=0, vmax=vmax, interpolation='nearest')
ax2.set_title('Sparse Depth Map')
ax2.axis('off')
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
