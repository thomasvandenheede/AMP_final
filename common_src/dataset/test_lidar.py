import os
import sys
from pprint import pprint

# ====== User-configurable settings ======
INDEX = 1000               # change this line to print a different sample index
DATA_ROOT = 'data/view_of_delft'
SPLIT = 'train'
# =======================================

# Ensure project root is on Python path to locate common_src
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, '..', '..'))
sys.path.insert(0, project_root)

from view_of_delft import ViewOfDelft

# Load dataset
dataset = ViewOfDelft(data_root=DATA_ROOT, split=SPLIT)
print(f"Total samples in dataset: {len(dataset)}")

# Fetch and print the specified sample
sample = dataset[INDEX]
print(f"\n--- Sample {INDEX} ---")
pprint(sample)
