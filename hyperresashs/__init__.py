import sys
import os
from importlib.metadata import version, PackageNotFoundError

# Check for any already-cached nnunetv2 modules
to_remove = [key for key in sys.modules if key == "nnunetv2" or key.startswith("nnunetv2.")]
for key in to_remove:
    print(f'hyperresashs found existing nnunetv2 module {key} in sys.modules. Removing it to avoid conflicts with hyperresashs submodule nnunetv2.')
    del sys.modules[key]

# Insert the submodule path at the front so it takes priority
_submodule_path = os.path.join(os.path.dirname(__file__), "submodules", "nnUNet")
sys.path.insert(0, _submodule_path)

# Insert the submodule path at the front so it takes priority
_submodule_path_inr = os.path.join(os.path.dirname(__file__), "submodules", "multi_contrast_inr")
sys.path.insert(0, _submodule_path_inr)

# Get the version from python importlib
try:
    # Replace 'your_package_name' with the actual name of your package
    __version__ = version("hyperresashs") 
except PackageNotFoundError:
    # Handle cases where the package isn't installed in the environment (e.g., running from source without proper installation)
    __version__ = "unknown"
