import pandas as pd
import seaborn as sns
import numpy as np
import xarray as xr
import xesmf as xe
import glob
import statsmodels.api as sm
import pymannkendall as mk
import csv

import math

from scipy import stats

import regionmask
import cf_xarray

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import matplotlib.gridspec as gridspec

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_time_series_index_and_anomalies(time_series_index, time_series_anomalies, base_period, datasets_names, cluster_kwargs, txt_kwargs):
    # Default keyword arguments
##    cluster_kwargs = {'num_insitu': 1, 'num_sat': 1, 'num_reanal': 1, 'num_reg': 1}
##    txt_kwargs = {'ylabel_ts': 'Default Title Indx', 'ylabel_ts_ano': 'Default Title Indx Ano', \
##    'region_n': 'Default Region N', 'pr_indx_n': 'Default PR Index N',\
##    'reg_data_details': 'Default Reg Data Details',\
##    'base_p_n': 'Default Base Period N'}
    
    default_value = 1  # Define a default number for cluster_kwargs
    default_title = 'Default Title'  # Define a default title for txt_kwargs

    num_insitu = cluster_kwargs.get('num_insitu', default_value)
    num_sat = cluster_kwargs.get('num_sat', default_value)
    num_reanal = cluster_kwargs.get('num_reanal', default_value)
    num_reg = cluster_kwargs.get('num_reg', default_value)

    ylabel_ts = txt_kwargs.get('ylabel_ts', default_title)
    ylabel_ts_ano = txt_kwargs.get('ylabel_ts_ano', default_title)
    region_n = txt_kwargs.get('region_n', default_title)
    pr_indx_n = txt_kwargs.get('pr_indx_n', default_title)
    reg_data_details = txt_kwargs.get('reg_data_details', default_title)
    base_p_n = txt_kwargs.get('base_p_n', default_title)
    ar6_reg_name = txt_kwargs.get('ar6_reg_name', default_title)
    pr_indx_n_unit = txt_kwargs.get('pr_indx_n_unit', default_title)

###    # Extract the keyword arguments
###    ylabel_ts = kwargs.get('ylabel_ts', 'Default Title Indx')
###    ylabel_ts_ano = kwargs.get('ylabel_ts_ano', 'Default Title Indx Ano')
###    region_n = kwargs.get('region_n', 'Default Region N')
###    pr_indx_n = kwargs.get('pr_indx_n', 'Default PR Index N')
###    reg_data_details = kwargs.get('reg_data_details', 'Default Reg Data Details')
###    base_p_n = kwargs.get('base_p_n', 'Default Base Period N')

    # Update with any user-supplied keyword arguments
    txt_kwargs.update(txt_kwargs.get('txt_kwargs', {}))
    cluster_kwargs.update(cluster_kwargs.get('cluster_kwargs', {}))

    # Rest of your function code using 'data' instead of 'time_series_index'
    fig = plt.figure(figsize=(12, 12))
    
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1])
    
    # Create the subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])  # Spanning the entire width of the second row
    
    gs.update(wspace=0)  # Remove horizontal space between ax1 and ax2
    
    colors = ['purple'] * num_insitu + ['orange'] * num_sat + ['green'] * num_reanal + ['blue'] * num_reg
    
    
    ##############time series and boxplot
    
    for i in range(time_series_index.sizes['new_dim']):
        if i<=8:
            time_series_index.isel(new_dim=i).plot(ax=ax1, alpha=0.6, marker='${}$'.format(i+1), markersize=7, \
                                          markevery=5, label=f'{datasets_names[i]}', color=colors[i])
        else:
            time_series_index.isel(new_dim=i).plot(ax=ax1, alpha=0.6, marker='${}$'.format(i+1), markersize=10, \
                                          markevery=5, label=f'{datasets_names[i]}', color=colors[i])
    #ax1.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
    #ax1.plot(np.zeros(1), np.zeros([1,3]), color='w', alpha=0, label=' ')
    
    #ax1.set_ylim([0,100])
    
    #####
    data_clim = time_series_index.sel(time=slice(str(base_period['base_s']), str(base_period['base_e']))).mean(dim='time').values
    
    PROPS = {
            'boxprops':{'facecolor':'none', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
            'capprops':{'color':'black'}
            }
    
    sns.boxplot(data=data_clim, ax=ax2, color = 'white', width=0.2, whis=1.5, linewidth=1.2,\
                            flierprops = dict(marker='d', markerfacecolor = 'gray', markersize = 8),\
                            showmeans=True,\
                            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black","markersize":"0"},
                           **PROPS)
    
    ax2.set(xticklabels=[])  
    ax2.set(xlabel=None)
    
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    ax2.tick_params(bottom=False)  # remove the ticks
    
    
    #========================================
    # Add a vertical line to the right-hand side plot
    # Assuming that the first four time series in time_series_index are the ones for which you want to draw vertical lines
    # add the vertical lines beside the boxplot
    
    data_clim1 = time_series_index[0:num_insitu].sel(time=slice(str(base_period['base_s']), str(base_period['base_e']))).mean(dim='time').values
    
    data_clim_arr1 = np.array(data_clim1) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr1):
        x_pos = 1 # set the x position for the vertical line
        #ax2.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='purple', linewidth=5)
        
    ax2.set_ylim(ax1.get_ylim())
    
    y_min = np.nanmin(data_clim_arr1)
    y_max = np.nanmax(data_clim_arr1)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='purple',linewidth=2)
    
    
    data_clim2 = time_series_index[num_insitu:(num_insitu+num_sat)].sel(time=slice(str(base_period['base_s']), str(base_period['base_e']))).mean(dim='time').values
    
    data_clim_arr2 = np.array(data_clim2) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr2):
        x_pos = 2 # set the x position for the vertical line
        #ax2.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr2)
    y_max = np.nanmax(data_clim_arr2)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='orange',linewidth=2)
    
    
    data_clim4 = time_series_index[(num_insitu+num_sat):(num_insitu+num_sat+num_reanal)].sel(time=slice(str(base_period['base_s']), str(base_period['base_e']))).mean(dim='time').values
    
    data_clim_arr4 = np.array(data_clim4) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr4):
        x_pos = 3 # set the x position for the vertical line
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='green', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr4)
    y_max = np.nanmax(data_clim_arr4)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='green',linewidth=2)
    
    
    data_clim5 = time_series_index[(num_insitu+num_sat+num_reanal):(num_insitu+num_sat+num_reanal+num_reg)].sel(time=slice(str(base_period['base_s']), str(base_period['base_e']))).mean(dim='time').values
    
    data_clim_arr5 = np.array(data_clim5) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr5):
        x_pos = 4 # set the x position for the vertical line
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='blue', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr5)
    y_max = np.nanmax(data_clim_arr5)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='blue',linewidth=2)
    #=================================
    
    plt.autoscale()
        

    ax1.set_ylabel(ylabel_ts,fontsize=11.)
    ax1.set_xlabel(' ',fontsize=11.)
    
    import matplotlib.dates as mdates
    yrs = mdates.YearLocator(2)
    
    ax1.xaxis.set_minor_locator(yrs)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    
    ##############difference
    for i in range(time_series_anomalies.sizes['new_dim']):
        if i<=8:
            time_series_anomalies.isel(new_dim=i).plot(ax=ax3, alpha=0.6, marker='${}$'.format(i+1), markersize=7, \
                                          markevery=5, label=f'{datasets_names[i]}', color=colors[i])
        else:
            time_series_anomalies.isel(new_dim=i).plot(ax=ax3, alpha=0.6, marker='${}$'.format(i+1), markersize=10, \
                                          markevery=5, label=f'{datasets_names[i]}', color=colors[i])
    
    
    #ax3.set_ylabel('Difference in Rx1day (mm) - '+ar6_reg_name)
    ax3.set_ylabel(ylabel_ts_ano,fontsize=11.)
    ax3.set_xlabel('Time (year)',fontsize=11.)
    
    ax3.axhline(0, color='black', linestyle='-',linewidth=1)
    
    plt.autoscale()
    
    ################
    #ax3.yaxis.set_minor_locator(MultipleLocator(50))
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    
    import matplotlib.dates as mdates
    yrs = mdates.YearLocator(2)
    ax3.xaxis.set_minor_locator(yrs)
    #ax.xaxis.set_minor_formatter(ticker.FixedFormatter(ww.Week))
    
    
    # Add legend with multiple columns
    num_series = time_series_index.sizes['new_dim']
    print(num_series)
    num_cols = 5
    num_rows = (num_series + num_cols - 1) // num_cols
    print(num_rows)
    legend_handles, legend_labels = ax3.get_legend_handles_labels()
    plt.legend(legend_handles, legend_labels, ncol=num_cols, loc='upper center',
               bbox_to_anchor=(0.5, -0.2), frameon=False)
    
    
    plt.text(0.01, 1.03, 'a) AR6_'+ar6_reg_name,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax1.transAxes,fontsize=11.)
    
    
    plt.text(0.01, 1.03, 'b) AR6_'+ar6_reg_name,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax3.transAxes,fontsize=11.)
    
    ####################
    #hfont = {'fontname':'Helvetica'}
    
    #plt.rcParams['linespacing'] = 1.5  # Increase the value for more space
    #fig.suptitle(f"Plots of {pr_indx_n} over {region_n}",fontsize=21, wrap=True, y=1.5, \
    #             linespacing=2.,fontdict={'fontname': 'serif'}) #, 'fontstyle': 'italic'
    
    fig.suptitle(f"Plots of {pr_indx_n} over {region_n}",fontsize=21, wrap=True, y=1.2, \
                 linespacing=2.,fontdict={'fontname': 'serif'}) #, 'fontstyle': 'italic'
    
    fig.text(0.5, 0.93, f"Time series of {pr_indx_n} and its anomaly over {region_n} for global and regional observations", ha='center', wrap=True, fontsize=16)
    
    txt = ("Fig. 1. Time series of spatially-weighted average " + pr_indx_n + " " +pr_indx_n_unit+ " and its anomaly for datasets from "
       "the global Frequent Rainfall Observations on GridS (FROGS; Roca et al 2019) and "
       + reg_data_details + ". "
       "The spatially-weighted averages are calculated across " + region_n + ". "
       "Moreover, for (a), box and whiskers provide information on the distribution of " + pr_indx_n + " over the " + base_p_n + " climatological "
       "period for all data products. Each colour represents each product ‘cluster’, i.e. all data products (black), "
       "in situ (purple), satellite corrected (orange), "
       "reanalyses (green) and regional data (blue).")
    fig.text(0.01, -0.05, txt, wrap=True, ha='left', va='top', fontsize=13) #
    



def plot_spatial_patterns(spatial_patterns, datasets_names, boundary_kwargs, cluster_kwargs, txt_kwargs):
    
    default_value = 1  # Define a default number for cluster_kwargs
    default_deg = 1.  # Define a default number for cluster_kwargs
    default_title = 'Default Title'  # Define a default title for txt_kwargs

    num_insitu = cluster_kwargs.get('num_insitu', default_value)
    num_sat = cluster_kwargs.get('num_sat', default_value)
    num_reanal = cluster_kwargs.get('num_reanal', default_value)
    num_reg = cluster_kwargs.get('num_reg', default_value)

    ylabel_ts = txt_kwargs.get('ylabel_ts', default_title)
    ylabel_ts_ano = txt_kwargs.get('ylabel_ts_ano', default_title)
    region_n = txt_kwargs.get('region_n', default_title)
    pr_indx_n = txt_kwargs.get('pr_indx_n', default_title)
    reg_data_details = txt_kwargs.get('reg_data_details', default_title)
    base_p_n = txt_kwargs.get('base_p_n', default_title)
    ar6_reg_name = txt_kwargs.get('ar6_reg_name', default_title)
    cbar_lb_sp = txt_kwargs.get('cbar_lb_sp', default_title)
    cbar_lb_sp_diff = txt_kwargs.get('cbar_lb_sp_diff', default_title)
    pr_indx_n_unit = txt_kwargs.get('pr_indx_n_unit', default_title)

    lon_reg_min = boundary_kwargs.get('lon_reg_min', default_deg)
    lon_reg_max = boundary_kwargs.get('lon_reg_max', default_deg)
    lat_reg_min = boundary_kwargs.get('lat_reg_min', default_deg)
    lat_reg_max = boundary_kwargs.get('lat_reg_max', default_deg)
#    lon_map_ticks = boundary_kwargs.get('lon_map_ticks', default_deg)
#    lat_map_ticks = boundary_kwargs.get('lat_map_ticks', default_deg)

    # Update with any user-supplied keyword arguments
    txt_kwargs.update(txt_kwargs.get('txt_kwargs', {}))
    cluster_kwargs.update(cluster_kwargs.get('cluster_kwargs', {}))
    boundary_kwargs.update(boundary_kwargs.get('boundary_kwargs', {}))

    ####################
    indx_n = ['a)','b)','c)','d)','e)',\
          'f)','g)','h)','i)','j)',\
            'k)','l)','m)','n)']

    #levels_pr = np.arange(0, 301, 30)b
    #levels_pr = np.arange(0, 20.1, 0.5)
    ##############
    # Set custom percentiles for the color mapping
    raw_start_value = np.nanpercentile(spatial_patterns, 0.1)
    raw_end_value = np.nanpercentile(spatial_patterns, 99.9)

    data_range = raw_end_value - raw_start_value

    # Adjust interval size based on the range
    if data_range <= 2 and data_range > 0:
        interval_size_spatialp = 0.1
    elif data_range <= 5 and data_range >2 :
        interval_size_spatialp = 0.2
    elif data_range <= 10 and data_range >5 :
        interval_size_spatialp = 0.5
    elif data_range <= 20 and data_range >10 :
        interval_size_spatialp = 1
    elif data_range <= 50 and data_range >20 :
        interval_size_spatialp = 2
    elif data_range <= 100 and data_range >50 :
        interval_size_spatialp = 5
    elif data_range <= 200 and data_range >100 :
        interval_size_spatialp = 10
    elif data_range <= 500 and data_range >200 :
        interval_size_spatialp = 20
    elif data_range <= 1000 and data_range >500 :
        interval_size_spatialp = 50
    elif data_range <= 2000 and data_range >1000 :
        interval_size_spatialp = 100
    elif data_range <= 5000 and data_range >2000 :
        interval_size_spatialp = 200
    elif data_range <= 10000 and data_range >5000 :
        interval_size_spatialp = 500
    elif data_range <= 20000 and data_range >10000 :
        interval_size_spatialp = 1000
    elif data_range <= 50000 and data_range >20000 :
        interval_size_spatialp = 2000
    elif data_range <= 100000 and data_range >50000 :
        interval_size_spatialp = 5000
    
    # Adjust start and end values to be multiples of interval_size_spatialp
    start_value = math.floor(raw_start_value / interval_size_spatialp) * interval_size_spatialp
    end_value = math.ceil(raw_end_value / interval_size_spatialp) * interval_size_spatialp

    # Calculate number of colors
    num_colors = int((end_value - start_value) / interval_size_spatialp)
    print(num_colors)
    
    # Ensure there is at least one interval
    if num_colors < 1:
        num_colors = 1
    
    levels_pr = np.linspace(start_value, end_value, num_colors + 1)
    
    # Choose a colormap
    cmap = mpl.colormaps['viridis_r'](np.linspace(0, 1, num_colors))
    
    cbar_kwargs = {'orientation':'horizontal', 'shrink':0.3, 'aspect':40}
    
    def custom_round(n1, n2):
        # Ensure n1 is the smaller and n2 is the larger number
        if n1 > n2:
            n1, n2 = n2, n1
    
        # Round n1 up to the next multiple of 5 or 10
        remainder_n1 = n1 % 10
        if remainder_n1 == 0:
            rounded_n1 = n1
        elif remainder_n1 <= 5:
            rounded_n1 = n1 + (5 - remainder_n1)
        else:
            rounded_n1 = n1 + (10 - remainder_n1)
    
        # Round n2 down to the previous multiple of 5 or 10
        remainder_n2 = n2 % 10
        if remainder_n2 == 0:
            rounded_n2 = n2
        elif remainder_n2 < 5:
            rounded_n2 = n2 - remainder_n2
        else:
            rounded_n2 = n2 - (remainder_n2 - 5)
    
        return rounded_n1, rounded_n2
    
    lat_reg_min_n, lat_reg_max_n = custom_round(lat_reg_min, lat_reg_max)
    lon_reg_min_n, lon_reg_max_n = custom_round(lon_reg_min, lon_reg_max)
    
    ##lat_reg_min_n = math.floor(lat_reg_min)
    ##lat_reg_max_n = math.ceil(lat_reg_max)
    ##lon_reg_min_n = math.floor(lon_reg_min)
    ##lon_reg_max_n = math.ceil(lon_reg_max)
    
    ############
    lat_map_ticks = np.arange(lat_reg_min_n,lat_reg_max_n+0.1, 10)
    # using (-180,-120,-60,0,60,120,180) may be wrong，need to change it to (0, 60, 120, 180, 240, 300, 360)
    lon_map_ticks = np.arange(lon_reg_min_n,lon_reg_max_n+0.1, 20)
    ##
    lon_reg_central = (lon_reg_min+lon_reg_max)/2.
##
##    lat_reg_min = custom_round(lat_reg_min)
##    lat_reg_max = custom_round(lat_reg_max)
    
    subplot_kws = {'projection': ccrs.Mercator(central_longitude=lon_reg_central, min_latitude=lat_reg_min, max_latitude=lat_reg_max)}
    
    axs = spatial_patterns.plot(x="lon", y="lat", col="datasets", col_wrap=5, \
                                              cmap=mpl.colors.ListedColormap(cmap), \
                                                norm=mpl.colors.BoundaryNorm(levels_pr, len(cmap)),\
                                                extend='max',transform=ccrs.PlateCarree(),\
                                              subplot_kws=subplot_kws,\
                                               cbar_kwargs=cbar_kwargs) #, levels=levels_pr
                                                #norm=mpl.colors.BoundaryNorm(levels, len(cmap)),\
    
    print(spatial_patterns.shape)

    axs.cbar.set_label(label=cbar_lb_sp, size=13) #, weight='bold'

    # Assuming 'total_panels' is the total number of panels you have
    total_panels = len(datasets_names)  # or another appropriate value
    num_columns = 5  # You can set this based on your layout preference
    num_rows = math.ceil(total_panels / num_columns)
    
    
    for i, ax in enumerate(axs.axs.flat):
        #print(i)
        #0-27, really weird. Should be 0-24
        
        #############################
        if i>=total_panels: break
        
        ax.set_title(" ")
    
        if i ==0:
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white',linewidth=1.2))
        if i > 0 and i <= num_reg:
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white',linewidth=1.2))
        if i >num_reg and i <= (num_insitu+num_reg):
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='purple', facecolor='white',linewidth=1.2))
        if i >(num_insitu+num_reg) and i <=(num_sat+num_insitu+num_reg):
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='orange', facecolor='white',linewidth=1.2))

        if i >(num_sat+num_insitu+num_reg) and i <=(num_reanal+num_sat+num_insitu+num_reg):
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white',linewidth=1.2))
        
        ax.coastlines()
        
        ax.set_extent([lon_reg_min, lon_reg_max, lat_reg_min, lat_reg_max])
        
        
        ax.set_xticks(lon_map_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_map_ticks, crs=ccrs.PlateCarree())
        
        #ax.tick_params(labelsize=12)  # Set fontsize for ticks

        # Determine the row and column index
        row_idx = i // num_columns
        col_idx = i % num_columns
            
        if row_idx == num_rows -1 :
            ax.set_xlabel('Longitude', fontsize=11)
        else:
            ax.tick_params(labelbottom=False)  # Hides xticks labels
        
        if col_idx != 0:
            ax.tick_params(labelleft=False)  # Hides xticks labels
        else:
            ax.set_ylabel('Latitude', fontsize=11)
        
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        
        if i==0: print(f'=========={lon_formatter}=========')
        
        ax.tick_params(axis='both', labelsize=9, length=3.5) # set fontsize for ticks
        ax.tick_params(axis='both', which='minor', length=2)
    
        ax.set_aspect('equal')
        
        #ax.yaxis.set_minor_locator(MultipleLocator(5))
        #ax.xaxis.set_minor_locator(MultipleLocator(5))
        
    #plt.subplots_adjust(bottom=0.05, top=0.95)
    plt.subplots_adjust(top=0.93, bottom=0.22, wspace=0.05, hspace=0.25)
    
    #plt.tight_layout()
    
    #plt.draw()
    
    ####################
    #hfont = {'fontname':'Helvetica'}
    
    #plt.rcParams['linespacing'] = 1.5  # Increase the value for more space
    #fig.suptitle(f"Plots of {pr_indx_n} over {region_n}",fontsize=21, wrap=True, y=1.5, \
    #             linespacing=2.,fontdict={'fontname': 'serif'}) #, 'fontstyle': 'italic'
    #axs.set_size_inches(15, 10)  # Adjust the size as needed
    
    plt.figtext(0.5, 0.995, f"Map plot of {pr_indx_n} over {region_n}", ha='center', wrap=True, fontsize=26)
    
    txt = (f"Fig. 2. Map plots of {pr_indx_n} {pr_indx_n_unit} for the {base_p_n} climatological period for datasets from "
       f"the global Frequent Rainfall Observations on GridS (FROGS; Roca et al 2019) and "
       f"{reg_data_details}. Each plot's title is framed with a color representing each product ‘cluster’, "
       f"i.e. all data products (black), in situ (purple), satellite corrected (orange), "
       f"reanalyses (green) and regional data (blue).")
    plt.figtext(0.01, 0.08, txt, wrap=True, ha='left', va='top', fontsize=20) #


def plot_spatial_patterns_diff(spatial_patterns_diff, datasets_names, boundary_kwargs, cluster_kwargs, txt_kwargs):
    default_value = 1  # Define a default number for cluster_kwargs
    default_deg = 1.  # Define a default number for cluster_kwargs
    default_title = 'Default Title'  # Define a default title for txt_kwargs

    num_insitu = cluster_kwargs.get('num_insitu', default_value)
    num_sat = cluster_kwargs.get('num_sat', default_value)
    num_reanal = cluster_kwargs.get('num_reanal', default_value)
    num_reg = cluster_kwargs.get('num_reg', default_value)

    ylabel_ts = txt_kwargs.get('ylabel_ts', default_title)
    ylabel_ts_ano = txt_kwargs.get('ylabel_ts_ano', default_title)
    region_n = txt_kwargs.get('region_n', default_title)
    pr_indx_n = txt_kwargs.get('pr_indx_n', default_title)
    reg_data_details = txt_kwargs.get('reg_data_details', default_title)
    base_p_n = txt_kwargs.get('base_p_n', default_title)
    ar6_reg_name = txt_kwargs.get('ar6_reg_name', default_title)
    cbar_lb_sp = txt_kwargs.get('cbar_lb_sp', default_title)
    cbar_lb_sp_diff = txt_kwargs.get('cbar_lb_sp_diff', default_title)
    pr_indx_n_unit = txt_kwargs.get('pr_indx_n_unit', default_title)

    lon_reg_min = boundary_kwargs.get('lon_reg_min', default_deg)
    lon_reg_max = boundary_kwargs.get('lon_reg_max', default_deg)
    lat_reg_min = boundary_kwargs.get('lat_reg_min', default_deg)
    lat_reg_max = boundary_kwargs.get('lat_reg_max', default_deg)
#    lon_map_ticks = boundary_kwargs.get('lon_map_ticks', default_deg)
#    lat_map_ticks = boundary_kwargs.get('lat_map_ticks', default_deg)

    # Update with any user-supplied keyword arguments
    txt_kwargs.update(txt_kwargs.get('txt_kwargs', {}))
    cluster_kwargs.update(cluster_kwargs.get('cluster_kwargs', {}))
    boundary_kwargs.update(boundary_kwargs.get('boundary_kwargs', {}))

    ####################
    indx_n = ['a)','b)','c)','d)','e)',\
              'f)','g)','h)','i)','j)',\
                'k)','l)','m)','n)']
    
    #levels_pr = np.arange(-120, 121, 10)
    #levels_pr = np.arange(-10, 11, 1)
    
    #,'ticks': levels_bar
    
    
    # Set custom percentiles for the color mapping
    raw_start_value = np.nanpercentile(spatial_patterns_diff, 1.)
    raw_end_value = np.nanpercentile(spatial_patterns_diff, 99.)
    
    print(raw_start_value, raw_end_value)
    
    if abs(raw_start_value) != abs(raw_end_value):
        v_limit = max(abs(raw_start_value), abs(raw_end_value))
    
#    v_limit = math.ceil(v_limit / 10.0) * 10.0
#    print(v_limit)
    
    
#    start_value = v_limit * (-1)
#    end_value = v_limit
    
    data_range = v_limit*2

    # Adjust interval size based on the range
    if data_range <= 2 and data_range > 0:
        interval_size_spatialp = 0.1
    elif data_range <= 5 and data_range >2 :
        interval_size_spatialp = 0.2
    elif data_range <= 10 and data_range >5 :
        interval_size_spatialp = 0.5
    elif data_range <= 20 and data_range >10 :
        interval_size_spatialp = 1
    elif data_range <= 50 and data_range >20 :
        interval_size_spatialp = 2
    elif data_range <= 100 and data_range >50 :
        interval_size_spatialp = 5
    elif data_range <= 200 and data_range >100 :
        interval_size_spatialp = 10
    elif data_range <= 500 and data_range >200 :
        interval_size_spatialp = 20
    elif data_range <= 1000 and data_range >500 :
        interval_size_spatialp = 50
    elif data_range <= 2000 and data_range >1000 :
        interval_size_spatialp = 100
    elif data_range <= 5000 and data_range >2000 :
        interval_size_spatialp = 200
    elif data_range <= 10000 and data_range >5000 :
        interval_size_spatialp = 500
    elif data_range <= 20000 and data_range >10000 :
        interval_size_spatialp = 1000
    elif data_range <= 50000 and data_range >20000 :
        interval_size_spatialp = 2000
    elif data_range <= 100000 and data_range >50000 :
        interval_size_spatialp = 5000
    
    # Adjust start and end values to be multiples of interval_size_spatialp
    start_value = math.floor(raw_start_value / interval_size_spatialp) * interval_size_spatialp
    end_value = math.ceil(raw_end_value / interval_size_spatialp) * interval_size_spatialp

    # Calculate number of colors
    num_colors = int((end_value - start_value) / interval_size_spatialp)
    print(num_colors)
    
    # Ensure there is at least one interval
    if num_colors < 1:
        num_colors = 1
    
    levels_pr = np.linspace(start_value, end_value, num_colors + 1)
    
    # Choose a colormap
    cmap = mpl.colormaps['BrBG'](np.linspace(0, 1, num_colors))
        
    cbar_kwargs = {'orientation':'horizontal', 'shrink':0.3, 'aspect':40}
    ############
    lon_reg_central = (lon_reg_min+lon_reg_max)/2.

    def custom_round(n1, n2):
        # Ensure n1 is the smaller and n2 is the larger number
        if n1 > n2:
            n1, n2 = n2, n1
    
        # Round n1 up to the next multiple of 5 or 10
        remainder_n1 = n1 % 10
        if remainder_n1 == 0:
            rounded_n1 = n1
        elif remainder_n1 <= 5:
            rounded_n1 = n1 + (5 - remainder_n1)
        else:
            rounded_n1 = n1 + (10 - remainder_n1)
    
        # Round n2 down to the previous multiple of 5 or 10
        remainder_n2 = n2 % 10
        if remainder_n2 == 0:
            rounded_n2 = n2
        elif remainder_n2 < 5:
            rounded_n2 = n2 - remainder_n2
        else:
            rounded_n2 = n2 - (remainder_n2 - 5)
    
        return rounded_n1, rounded_n2
    
    lat_reg_min_n, lat_reg_max_n = custom_round(lat_reg_min, lat_reg_max)
    lon_reg_min_n, lon_reg_max_n = custom_round(lon_reg_min, lon_reg_max)
    
    ##lat_reg_min_n = math.floor(lat_reg_min)
    ##lat_reg_max_n = math.ceil(lat_reg_max)
    ##lon_reg_min_n = math.floor(lon_reg_min)
    ##lon_reg_max_n = math.ceil(lon_reg_max)
    
    ############
    lat_map_ticks = np.arange(lat_reg_min_n,lat_reg_max_n+0.1, 10)
    # using (-180,-120,-60,0,60,120,180) may be wrong，need to change it to (0, 60, 120, 180, 240, 300, 360)
    lon_map_ticks = np.arange(lon_reg_min_n,lon_reg_max_n+0.1, 20)

    lon_reg_central = (lon_reg_min+lon_reg_max)/2.

#    lat_reg_min = custom_round(lat_reg_min)
#    lat_reg_max = custom_round(lat_reg_max)

    subplot_kws = {'projection': ccrs.Mercator(central_longitude=lon_reg_central, min_latitude=lat_reg_min, max_latitude=lat_reg_max)}
    #cmap = 'BrBG'
    
    axs = spatial_patterns_diff.plot(x="lon", y="lat", col="datasets", col_wrap=5, \
                                            extend='both',\
                                            cmap=mpl.colors.ListedColormap(cmap), \
                                            norm=mpl.colors.BoundaryNorm(levels_pr, len(cmap)),\
                                            transform=ccrs.PlateCarree(),\
                                            subplot_kws=subplot_kws,\
                                            cbar_kwargs=cbar_kwargs) #, levels=levels_pr, cmap=cmap, \vmin=v_limit*(-1), vmax=v_limit
    #extend = "max", "min"
    ##for ax in axs.axs.flat:
    ##    ax.add_feature(cfeature.LAND)
    ##    ax.add_feature(cfeature.OCEAN)
    
    print(spatial_patterns_diff.shape)

    axs.cbar.set_label(label=cbar_lb_sp_diff, size=13) #, weight='bold'
    
    ##############
    ##y_extent = np.arange(-45, -10.4, 10)
    ### using (-180,-120,-60,0,60,120,180) may be wrong，need to change it to (0, 60, 120, 180, 240, 300, 360)
    ##x_extent = np.arange(110, 155.6, 10)

    # Assuming 'total_panels' is the total number of panels you have
    total_panels = len(datasets_names)  # or another appropriate value
    num_columns = 5  # You can set this based on your layout preference
    num_rows = math.ceil(total_panels / num_columns)
    
    for i, ax in enumerate(axs.axs.flat):
        #print(i)
        #0-23
        
        if i>=total_panels: break
    
        ax.set_title(" ")
        #ax.set_title(datasets_names[i])
        
        #ax.add_feature(cfeature.LAND)
        #ax.add_feature(cfeature.OCEAN)
    
        if i >= 0 and i < num_reg:
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white',linewidth=1.2))
        if i >= num_reg and i < (num_insitu+num_reg):
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='purple', facecolor='white',linewidth=1.2))
        if i >= (num_insitu+num_reg) and i < (num_sat+num_insitu+num_reg):
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='orange', facecolor='white',linewidth=1.2))
        if i >= (num_sat+num_insitu+num_reg) and i < (num_reanal+num_sat+num_insitu+num_reg):
            print(i, datasets_names[i])
            ax.annotate(datasets_names[i], xy=(0.5, 1.1), xycoords='axes fraction',
                    fontsize=13, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white',linewidth=1.2))
        
        ax.coastlines()
        
        #min_lat = diff_pr_indx_spatial_global["lat"].min()
        #max_lat = diff_pr_indx_spatial_global["lat"].max()
        #print(min_lat, max_lat)
        #ax.set_extent([90.5, 144.5, -14.5, 24.5])
        ax.set_extent([lon_reg_min, lon_reg_max, lat_reg_min, lat_reg_max])
        ###############
        
        #ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        
        
        ax.set_xticks(lon_map_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_map_ticks, crs=ccrs.PlateCarree())
        
        #ax.tick_params(labelsize=12)  # Set fontsize for ticks

        # Determine the row and column index
        row_idx = i // num_columns
        col_idx = i % num_columns
            
        if row_idx == num_rows -1 :
            ax.set_xlabel('Longitude', fontsize=11)
        else:
            ax.tick_params(labelbottom=False)  # Hides xticks labels
        
        if col_idx != 0:
            ax.tick_params(labelleft=False)  # Hides xticks labels
        else:
            ax.set_ylabel('Latitude', fontsize=11)
        
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        
        if i==0: print(f'=========={lon_formatter}=========')
        
        ax.tick_params(axis='both', labelsize=9, length=3.5) # set fontsize for ticks
        ax.tick_params(axis='both', which='minor', length=2)
        
        #ax.yaxis.set_minor_locator(MultipleLocator(5))
        #ax.xaxis.set_minor_locator(MultipleLocator(5))
        
    #plt.subplots_adjust(bottom=0.05, top=0.95)
    plt.subplots_adjust(top=0.93, bottom=0.22, wspace=0.05, hspace=0.25)
    
    plt.figtext(0.5, 0.995, f"Map plot of difference for {pr_indx_n} between each dataset and the multi-data mean over {region_n}", ha='center', wrap=True, fontsize=26)
    
    txt = f"Fig. 3. Same as Fig. 2, but for difference of {pr_indx_n} {pr_indx_n_unit} between each dataset and the multi-data mean."
    plt.figtext(0.01, 0.08, txt, wrap=True, ha='left', va='top', fontsize=20) #



def plot_trend_abs_rel_boxplots(trend_abs, trend_rel, cluster_kwargs, txt_kwargs):

    default_value = 1  # Define a default number for cluster_kwargs
    default_title = 'Default Title'  # Define a default title for txt_kwargs

    num_insitu = cluster_kwargs.get('num_insitu', default_value)
    num_sat = cluster_kwargs.get('num_sat', default_value)
    num_reanal = cluster_kwargs.get('num_reanal', default_value)
    num_reg = cluster_kwargs.get('num_reg', default_value)

    ylabel_ts = txt_kwargs.get('ylabel_ts', default_title)
    ylabel_ts_ano = txt_kwargs.get('ylabel_ts_ano', default_title)
    region_n = txt_kwargs.get('region_n', default_title)
    pr_indx_n = txt_kwargs.get('pr_indx_n', default_title)
    reg_data_details = txt_kwargs.get('reg_data_details', default_title)
    base_p_n = txt_kwargs.get('base_p_n', default_title)
    ar6_reg_name = txt_kwargs.get('ar6_reg_name', default_title)
    ylabel_box_abs = txt_kwargs.get('ylabel_box_abs', default_title)
    ylabel_box_rel = txt_kwargs.get('ylabel_box_rel', default_title)
    pr_indx_n_trabs_unit = txt_kwargs.get('pr_indx_n_trabs_unit', default_title)
    pr_indx_n_trrel_unit = txt_kwargs.get('pr_indx_n_trrel_unit', default_title)
    trend_p_n = txt_kwargs.get('trend_p_n', default_title)
    

    # Update with any user-supplied keyword arguments
    txt_kwargs.update(txt_kwargs.get('txt_kwargs', {}))
    cluster_kwargs.update(cluster_kwargs.get('cluster_kwargs', {}))


    # %%
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 12))
    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4), sharey=True, gridspec_kw={'width_ratios': [2.5, 1]})
    
    plt.subplots_adjust(wspace=0, hspace=0.3)
    
    colors = ['purple'] * num_insitu + ['orange'] * num_sat + ['green'] * num_reanal + ['blue'] * num_reg
    
    ###########################
    #data_clim = combined_all.sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim = trend_abs
    
    PROPS = {
            'boxprops':{'facecolor':'none', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
            'capprops':{'color':'black'}
            }
    
    sns.boxplot(data=data_clim, ax=ax1, color = 'white', width=0.2, whis=1.5, linewidth=1.2,\
                            flierprops = dict(marker='d', markerfacecolor = 'gray', markersize = 8),\
                            showmeans=True,\
                            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black","markersize":"0"},
                           **PROPS)
    
    ax1.set(xticklabels=[])  
    ax1.set(xlabel=None)
    ax1.tick_params(bottom=False)  # remove the ticks
    
    
    #========================================
    # Add a vertical line to the right-hand side plot
    # Assuming that the first four time series in combined_all are the ones for which you want to draw vertical lines
    # add the vertical lines beside the boxplot
    
    #data_clim1 = combined_all[0:5].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim1 = trend_abs[num_reg:(num_reg+num_insitu)]
    
    #print(combined_all[0:1])
    
    data_clim_arr1 = np.array(data_clim1) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr1):
        x_pos = 1 # set the x position for the vertical line
        #ax1.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax1.plot([x_pos], [d], marker='_', markersize=8, color='purple', linewidth=5)
        
    ax1.set_ylim(ax1.get_ylim())
    
    y_min = np.nanmin(data_clim_arr1)
    y_max = np.nanmax(data_clim_arr1)
    
    ax1.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='purple',linewidth=2)
    
    ##################
    #data_clim2 = combined_all[5:13].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim2 = trend_abs[(num_reg+num_insitu):(num_reg+num_insitu+num_sat)]
    
    data_clim_arr2 = np.array(data_clim2) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr2):
        x_pos = 2 # set the x position for the vertical line
        #ax1.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax1.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr2)
    y_max = np.nanmax(data_clim_arr2)
    
    ax1.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='orange',linewidth=2)
    
    ##################
    #data_clim4 = combined_all[17:23].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim4 = trend_abs[(num_reg+num_insitu+num_sat):(num_reg+num_insitu+num_sat+num_reanal)]
    
    data_clim_arr4 = np.array(data_clim4) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr4):
        x_pos = 3 # set the x position for the vertical line
        
        ax1.plot([x_pos], [d], marker='_', markersize=8, color='green', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr4)
    y_max = np.nanmax(data_clim_arr4)
    
    ax1.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='green',linewidth=2)
    
    ##################
    #data_clim5 = combined_all[23:25].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim5 = trend_abs[0:num_reg]
    
    data_clim_arr5 = np.array(data_clim5) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr5):
        x_pos = 4 # set the x position for the vertical line
        
        ax1.plot([x_pos], [d], marker='_', markersize=8, color='blue', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr5)
    y_max = np.nanmax(data_clim_arr5)
    
    ax1.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='blue',linewidth=2)
    #=================================
    
    # Set the right-hand side y-axis label
    
    ax1.set_ylabel(ylabel_box_abs, fontsize=11)
    #ax1.set_xlabel('Time (year)')
    
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.', 'regional'], fontsize=11)
    
    plt.text(0.01, 1.03, 'a) AR6_'+ar6_reg_name,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax1.transAxes, fontsize=11)
    
    
    plt.autoscale()
    
    ####################
    
    
    ###########################
    #data_clim = combined_all.sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim = trend_rel
    
    sns.boxplot(data=data_clim, ax=ax2, color = 'white', width=0.2, whis=1.5, linewidth=1.2,\
                            flierprops = dict(marker='d', markerfacecolor = 'gray', markersize = 8),\
                            showmeans=True,\
                            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black","markersize":"0"},
                           **PROPS)
    
    ax2.set(xticklabels=[])  
    ax2.set(xlabel=None)
    ax2.tick_params(bottom=False)  # remove the ticks
    
    
    #========================================
    # Add a vertical line to the right-hand side plot
    # Assuming that the first four time series in combined_all are the ones for which you want to draw vertical lines
    # add the vertical lines beside the boxplot
    
    #data_clim1 = combined_all[0:5].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim1 = trend_rel[num_reg:(num_reg+num_insitu)]
    
    #print(combined_all[0:1])
    
    data_clim_arr1 = np.array(data_clim1) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr1):
        x_pos = 1 # set the x position for the vertical line
        #ax2.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='purple', linewidth=5)
        
    ax2.set_ylim(ax1.get_ylim())
    
    y_min = np.nanmin(data_clim_arr1)
    y_max = np.nanmax(data_clim_arr1)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='purple',linewidth=2)
    
    ##################
    #data_clim2 = combined_all[5:13].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim2 = trend_rel[(num_reg+num_insitu):(num_reg+num_insitu+num_sat)]
    
    data_clim_arr2 = np.array(data_clim2) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr2):
        x_pos = 2 # set the x position for the vertical line
        #ax2.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr2)
    y_max = np.nanmax(data_clim_arr2)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='orange',linewidth=2)
    
    ##################
    #data_clim4 = combined_all[17:23].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim4 = trend_rel[(num_reg+num_insitu+num_sat):(num_reg+num_insitu+num_sat+num_reanal)]
    
    data_clim_arr4 = np.array(data_clim4) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr4):
        x_pos = 3 # set the x position for the vertical line
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='green', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr4)
    y_max = np.nanmax(data_clim_arr4)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='green',linewidth=2)
    
    ##################
    #data_clim5 = combined_all[23:25].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim5 = trend_rel[0:num_reg]
    
    data_clim_arr5 = np.array(data_clim5) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr5):
        x_pos = 4 # set the x position for the vertical line
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='blue', linewidth=5)
    
    y_min = np.nanmin(data_clim_arr5)
    y_max = np.nanmax(data_clim_arr5)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='blue',linewidth=2)
    #=================================
    
    # Set the right-hand side y-axis label
    
    ax2.set_ylabel(ylabel_box_rel, fontsize=11)
    #ax1.set_xlabel('Time (year)')
    
    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.', 'regional'], fontsize=11)
    
    plt.text(0.01, 1.03, 'b) AR6_'+ar6_reg_name,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax2.transAxes, fontsize=11)
    
    plt.autoscale()
    
    ####################
    
    fig.text(0.5, 0.93, f"Box and whiskers for the absolute and relative trends of {pr_indx_n} over {region_n}", ha='center', wrap=True, fontsize=16)
    
    txt = (f"Fig. 4. Box and whiskers for the absolute {pr_indx_n_trabs_unit} and relative {pr_indx_n_trrel_unit} "
    f"trends of {pr_indx_n} during {trend_p_n} over {region_n}. Each colour represents each product ‘cluster’, "
    f"i.e. all data products (black), in situ (purple), satellite corrected (orange), "
    f"reanalyses (green) and regional data (blue).")
    fig.text(0.01, 0.05, txt, wrap=True, ha='left', va='top', fontsize=13) #
    
#    fig.savefig(out_dir+file_name+"_"+pr_indx_name+'_boxplot_trend_trper_rob.png',dpi=dpi_set,format=format_n1,bbox_inches='tight')
#    fig.savefig(out_dir+file_name+"_"+pr_indx_name+'_boxplot_trend_trper_rob.eps',dpi=dpi_set,format=format_n2,bbox_inches='tight')
    
    plt.show()

