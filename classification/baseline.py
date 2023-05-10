# Imports

# For modules
import os
import sys

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

# Import module from the package
from data_preprocessing_packages.preprocessing import import_split_scale

import_split_scale()