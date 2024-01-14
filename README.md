# Intercomparison-of-precipitation-extremes-from-FROGS-and-regional-datasets
Intercomparison of precipitation extremes from FROGS and regional datasets


# Spatial Patterns Plotting Function

This Python function, `plot_spatial_patterns`, is designed to generate spatial plots for multiple datasets. It uses libraries such as Pandas, Seaborn, NumPy, Xarray, Matplotlib, Cartopy, and others.

## Usage

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### How to Use the `plot_spatial_patterns` Function

To use the `plot_spatial_patterns` function, follow these steps:

1. Download the `plotting_functions.py` file containing the function definition.

2. Place the file in the directory where you want to use the function.

3. In your Python script or Jupyter Notebook, import the function:

```python
import sys
sys.path.append('/path/to/directory/containing/plotting_functions.py')

from plotting_functions import plot_spatial_patterns
```

### Running the Function

To use the `plot_spatial_patterns` function, you need to provide the necessary input parameters. Modify the function call in your script accordingly.

```python
# Import the function
from your_module import plot_spatial_patterns

# Example input parameters
spatial_patterns = ...  # Your spatial patterns data
datasets_names = ...    # List of dataset names
boundary_kwargs = ...   # Dictionary with boundary settings
cluster_kwargs = ...    # Dictionary with cluster settings
txt_kwargs = ...        # Dictionary with text settings

# Call the function
plot_spatial_patterns(spatial_patterns, datasets_names, boundary_kwargs, cluster_kwargs, txt_kwargs)
