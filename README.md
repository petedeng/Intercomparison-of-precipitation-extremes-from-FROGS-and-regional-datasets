# Intercomparison-of-precipitation-extremes-from-FROGS-and-regional-datasets
Plotting functions for "Intercomparison of precipitation extremes from FROGS and regional datasets"


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
    pandas
    seaborn
    numpy
    xarray
    math
    matplotlib
    cartopy
    ```

### How to Use the `plot_spatial_patterns` Function

To use the `plot_spatial_patterns` function, follow these steps:

1. Place the file in the directory where you want to use the function.

2. In your Python script or Jupyter Notebook, import the function:

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
spatial_patterns = ...  # Your spatial patterns data, xarray.DataArray object

datasets_names = ...    # List of dataset names, e.g., datasets_names = ["REGEN_LONG_V1-2019","GPCC_FDD_v2020","CPC_v1.0"]

boundary_kwargs = ...   # Dictionary with boundary settings, e.g., boundary_kwargs = {'lon_reg_min': 109.5, 'lon_reg_max': 155.5, 'lat_reg_min': -45.5, 'lat_reg_max': -9.5}

cluster_kwargs = ...    # Dictionary with cluster settings, e.g., cluster_kwargs = {'num_insitu': 3, 'num_sat': 6, 'num_reanal': 4, 'num_reg': 1}

txt_kwargs = ...
# Dictionary with text settings, e,g,
#ylabel_ts = 'R10mm (days)'
#ylabel_ts_ano = 'Anomaly of R10mm (days)'
#cbar_lb_sp = 'R10mm (days)'
#cbar_lb_sp_diff = 'Difference in R10mm (days) (Mean)'
#ylabel_box_abs = 'Trend in R10mm'
#
#ylabel_box_rel = 'Trend in precentage change of R10mm' 
#pr_indx_n = "R10mm"
#pr_indx_n_unit = "(unit: days)"
#pr_indx_n_trabs_unit = "(unit: days/decade)"
#pr_indx_n_trrel_unit = "(unit: %/\N{DEGREE SIGN}C)"
#
#region_n = "C.Australia"
##region_n2 = "W.North-America in the updated SREX regions in the sixth IPCC assessment report (AR6)"
#base_p_n = "1998–2017"
#trend_p_n = "2000-2019"
#
#reg_data_details = "regional in situ data, i.e., the Australian Gridded Climate Data (AGCD, previously \
#termed Australian Water Availability Project [AWAP]; Jones et al., 2009)"
#
#txt_kwargs = {'ylabel_ts': ylabel_ts, 'ylabel_ts_ano': ylabel_ts_ano, \
#    'region_n': region_n, 'pr_indx_n': pr_indx_n,\
#    'reg_data_details': reg_data_details,\
#    'base_p_n': base_p_n,\
#    'trend_p_n': trend_p_n,\
#    'ar6_reg_name': ar6_reg_name,\
#    'cbar_lb_sp': cbar_lb_sp,\
#    'cbar_lb_sp_diff': cbar_lb_sp_diff,\
#    'ylabel_box_abs': ylabel_box_abs,\
#    'ylabel_box_rel': ylabel_box_rel,\
#    'pr_indx_n_unit': pr_indx_n_unit,\
#    'pr_indx_n_trabs_unit': pr_indx_n_trabs_unit,\
#    'pr_indx_n_trrel_unit': pr_indx_n_trrel_unit}

# Call the function
plot_spatial_patterns(spatial_patterns, datasets_names, boundary_kwargs, cluster_kwargs, txt_kwargs)

# Other Functions: plot_spatial_patterns_diff and plot_time_series_and_trend_boxplots

The function, `plot_spatial_patterns_diff` is designed to generate spatial plots for the differences between each dataset and the mean of multiple datasets. 
The function, `plot_time_series_and_trend_boxplots` is designed to generate the time series and the trends (relative to time and GMST respectively) for the datasets.

The usage of the functions is similar to plot_spatial_patterns.
