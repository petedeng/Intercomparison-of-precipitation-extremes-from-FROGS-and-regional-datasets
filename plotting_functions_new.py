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


def plot_time_series_and_trend_boxplots(time_series_index, time_series_anomalies, base_period, datasets_names, \
                                        trend_abs, trend_rel, cluster_kwargs, txt_kwargs, reg_data_type='default'):
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


    ylabel_box_abs = txt_kwargs.get('ylabel_box_abs', default_title)
    ylabel_box_rel = txt_kwargs.get('ylabel_box_rel', default_title)
    pr_indx_n_trabs_unit = txt_kwargs.get('pr_indx_n_trabs_unit', default_title)
    pr_indx_n_trrel_unit = txt_kwargs.get('pr_indx_n_trrel_unit', default_title)
    trend_p_n = txt_kwargs.get('trend_p_n', default_title)
    

    # Update with any user-supplied keyword arguments
    txt_kwargs.update(txt_kwargs.get('txt_kwargs', {}))
    cluster_kwargs.update(cluster_kwargs.get('cluster_kwargs', {}))

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

    fig = plt.figure(figsize=(12, 9))
    
    gs1 = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1], height_ratios=[1, 1])
    
    # Create the subplots
    ax1 = fig.add_subplot(gs1[0, 0])
    ax2 = fig.add_subplot(gs1[0, 1])
    ax3 = fig.add_subplot(gs1[1, :])  # Spanning the entire width of the second row
    
    gs1.update(left = 0, right = 0.63, wspace=0)  # Remove horizontal space between ax1 and ax2
    
    gs2 = gridspec.GridSpec(2, 1,  height_ratios=[1, 1])
    ax4 = fig.add_subplot(gs2[0])
    ax5 = fig.add_subplot(gs2[1])
    gs2.update(left = 0.7, right = 1)
    
    if reg_data_type=='default':
        colors = ['purple'] * num_insitu + ['orange'] * num_sat + ['green'] * num_reanal + ['blue'] * num_reg

    elif reg_data_type=='satellite':
        colors = ['purple'] * num_insitu + ['orange'] * num_sat + ['green'] * num_reanal + ['orange'] * num_reg
    
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
            'whiskerprops':{'color':'black', 'linestyle': 'dashed'},
            'capprops':{'color':'black'}
            }
    
    sns.boxplot(data=data_clim, ax=ax2, color = 'white', width=0.2, whis=1.5, linewidth=1.2,\
                            flierprops = dict(marker='d', markerfacecolor = 'gray', markersize = 8),\
                            showmeans=True,\
                            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black","markersize":"0"},
                           **PROPS)
    
    #ax2.set(xticklabels=[])  
    #ax2.set(xlabel=None)

    if reg_data_type=='default':
        ax2.set_xlim(-0.5, 4.5)
    
        ax2.set_xticks([0., 1., 2., 3., 4.])
        ax2.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.', 'regional'], fontsize=11., rotation=30)

    elif reg_data_type=='satellite':

        ax2.set_xlim(-0.5, 3.5)
    
        ax2.set_xticks([0., 1., 2., 3.])
        ax2.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.'], fontsize=11., rotation=30)

    # Add tick marks inside the frame
    ax2.tick_params(axis='x', direction='in')
    
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    #ax2.tick_params(bottom=False)  # remove the ticks
    
    
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
        #ax2.plot(x_pos, d, marker='_', markersize=8, color='purple', linewidth=5)

        # Add text label near each marker
        #label_offset = 1.  # Adjust this value as needed
        ax2.text(x_pos + 0.1, d, f'{i+1}', fontsize=9, verticalalignment='center')  # Adjust the position and font size as needed
        
    ax2.set_ylim(ax1.get_ylim())
    
    y_min = np.nanmin(data_clim_arr1)
    y_max = np.nanmax(data_clim_arr1)
    
    #ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle=(0, (5, 1)), color='purple',linewidth=2)
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='--', color='purple',linewidth=1.2)
    
    
    data_clim2 = time_series_index[num_insitu:(num_insitu+num_sat)].sel(time=slice(str(base_period['base_s']), str(base_period['base_e']))).mean(dim='time').values
    
    data_clim_arr2 = np.array(data_clim2) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr2):
        x_pos = 2 # set the x position for the vertical line
        #ax2.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)

        # Add text label near each marker
        #label_offset = 1.  # Adjust this value as needed
        ax2.text(x_pos + 0.1, d, f'{i+1+num_insitu}', fontsize=9, verticalalignment='center')
    
    y_min = np.nanmin(data_clim_arr2)
    y_max = np.nanmax(data_clim_arr2)
    
    if reg_data_type=='default':
        ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='--', color='orange',linewidth=1.2)
    elif reg_data_type=='satellite':
        y_min_glb_satellite = y_min
        y_max_glb_satellite = y_max
    
    data_clim4 = time_series_index[(num_insitu+num_sat):(num_insitu+num_sat+num_reanal)].sel(time=slice(str(base_period['base_s']), str(base_period['base_e']))).mean(dim='time').values
    
    data_clim_arr4 = np.array(data_clim4) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr4):
        x_pos = 3 # set the x position for the vertical line
        
        ax2.plot([x_pos], [d], marker='_', markersize=8, color='green', linewidth=5)

        # Add text label near each marker
        #label_offset = 1.  # Adjust this value as needed
        ax2.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat}', fontsize=9, verticalalignment='center')
    
    y_min = np.nanmin(data_clim_arr4)
    y_max = np.nanmax(data_clim_arr4)
    
    ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='--', color='green',linewidth=1.2)
    
    
    data_clim5 = time_series_index[(num_insitu+num_sat+num_reanal):(num_insitu+num_sat+num_reanal+num_reg)].sel(time=slice(str(base_period['base_s']), str(base_period['base_e']))).mean(dim='time').values
    
    data_clim_arr5 = np.array(data_clim5) # convert data_clim to numpy array

    if reg_data_type=='default':
        for i, d in enumerate(data_clim_arr5):
            x_pos = 4 # set the x position for the vertical line
            
            ax2.plot([x_pos], [d], marker='_', markersize=8, color='blue', linewidth=5)
    
            # Add text label near each marker
            #label_offset = 1.  # Adjust this value as needed
            ax2.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat+num_reanal}', fontsize=9, verticalalignment='center')

            y_min = np.nanmin(data_clim_arr5)
            y_max = np.nanmax(data_clim_arr5)

            ax2.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='--', color='blue',linewidth=1.2)

    elif reg_data_type=='satellite':
        for i, d in enumerate(data_clim_arr5):
            x_pos = 2 # set the x position for the vertical line
            
            ax2.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)
    
            # Add text label near each marker
            #label_offset = 1.  # Adjust this value as needed
            ax2.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat+num_reanal}', fontsize=9, verticalalignment='center')
    
            y_min = np.nanmin(data_clim_arr5)
            y_max = np.nanmax(data_clim_arr5)

            y_min_n = np.nanmin([y_min_glb_satellite, y_min])
            y_max_n = np.nanmax([y_max_glb_satellite, y_max])
    
            ax2.vlines(x=x_pos, ymin=y_min_n, ymax=y_max_n, linestyle='--', color='orange',linewidth=1.2)
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
    
    

    ##############
    ##############
    ##############
    # %%
#    fig, (ax4, ax5) = plt.subplots(nrows=2, figsize=(12, 12))
    #fig, (ax4, ax5) = plt.subplots(ncols=2, figsize=(10, 4), sharey=True, gridspec_kw={'width_ratios': [2.5, 1]})
    
#    plt.subplots_adjust(wspace=0, hspace=0.3)
    
#    colors = ['purple'] * num_insitu + ['orange'] * num_sat + ['green'] * num_reanal + ['blue'] * num_reg
    
    ###########################
    #data_clim = combined_all.sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim = trend_abs
    
    PROPS = {
            'boxprops':{'facecolor':'none', 'edgecolor':'black'},
            'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},
            'capprops':{'color':'black'}
            }
    
    sns.boxplot(data=data_clim, ax=ax4, color = 'white', width=0.2, whis=1.5, linewidth=1.2,\
                            flierprops = dict(marker='d', markerfacecolor = 'gray', markersize = 8),\
                            showmeans=True,\
                            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black","markersize":"0"},
                           **PROPS)
    
#    ax4.set(xticklabels=[])  
#    ax4.set(xlabel=None)
#    ax4.tick_params(bottom=False)  # remove the ticks
    
    print('==========')
    #=========================================
    # Add a vertical line to the right-hand side plot
    # Assuming that the first four time series in combined_all are the ones for which you want to draw vertical lines
    # add the vertical lines beside the boxplot
    
    #data_clim1 = combined_all[0:5].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim1 = trend_abs[num_reg:(num_reg+num_insitu)]
    
    #print(combined_all[0:1])
    
    data_clim_arr1 = np.array(data_clim1) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr1):
        x_pos = 1 # set the x position for the vertical line
        #ax4.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax4.plot([x_pos], [d], marker='_', markersize=8, color='purple', linewidth=5)

        # Add text label near each marker
        #label_offset = 1.  # Adjust this value as needed
        ax4.text(x_pos + 0.1, d, f'{i+1}', fontsize=9, verticalalignment='center')  # Adjust the position and font size as needed
        
    ax4.set_ylim(ax4.get_ylim())
    
    y_min = np.nanmin(data_clim_arr1)
    y_max = np.nanmax(data_clim_arr1)

    print('111')
    
    ax4.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='purple',linewidth=2)
    
    ##################
    #data_clim2 = combined_all[5:13].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim2 = trend_abs[(num_reg+num_insitu):(num_reg+num_insitu+num_sat)]
    
    data_clim_arr2 = np.array(data_clim2) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr2):
        x_pos = 2 # set the x position for the vertical line
        #ax4.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax4.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)

        ax4.text(x_pos + 0.1, d, f'{i+1+num_insitu}', fontsize=9, verticalalignment='center')  # Adjust the position and font size as needed

    y_min = np.nanmin(data_clim_arr2)
    y_max = np.nanmax(data_clim_arr2)

    print('222')

    if reg_data_type=='default':
        ax4.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='orange',linewidth=2)
    elif reg_data_type=='satellite':
        y_min_glb_satellite = y_min
        y_max_glb_satellite = y_max
    
    ##################
    #data_clim4 = combined_all[17:23].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim4 = trend_abs[(num_reg+num_insitu+num_sat):(num_reg+num_insitu+num_sat+num_reanal)]
    
    data_clim_arr4 = np.array(data_clim4) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr4):
        x_pos = 3 # set the x position for the vertical line
        
        ax4.plot([x_pos], [d], marker='_', markersize=8, color='green', linewidth=5)

        ax4.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat}', fontsize=9, verticalalignment='center') 
    
    y_min = np.nanmin(data_clim_arr4)
    y_max = np.nanmax(data_clim_arr4)

    print('333')
    
    ax4.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='green',linewidth=2)
    
    ##################
    #data_clim5 = combined_all[23:25].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim5 = trend_abs[0:num_reg]
    
    data_clim_arr5 = np.array(data_clim5) # convert data_clim to numpy array

    if reg_data_type=='default':
        for i, d in enumerate(data_clim_arr5):
            x_pos = 4 # set the x position for the vertical line
            
            ax4.plot([x_pos], [d], marker='_', markersize=8, color='blue', linewidth=5)
    
            # Add text label near each marker
            #label_offset = 1.  # Adjust this value as needed
            ax4.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat+num_reanal}', fontsize=9, verticalalignment='center')

            y_min = np.nanmin(data_clim_arr5)
            y_max = np.nanmax(data_clim_arr5)

            ax4.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='blue',linewidth=2)

    elif reg_data_type=='satellite':
        for i, d in enumerate(data_clim_arr5):
            x_pos = 2 # set the x position for the vertical line
            
            ax4.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)
    
            # Add text label near each marker
            #label_offset = 1.  # Adjust this value as needed
            ax4.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat+num_reanal}', fontsize=9, verticalalignment='center')
    
            y_min = np.nanmin(data_clim_arr5)
            y_max = np.nanmax(data_clim_arr5)

            y_min_n = np.nanmin([y_min_glb_satellite, y_min])
            y_max_n = np.nanmax([y_max_glb_satellite, y_max])
    
            ax4.vlines(x=x_pos, ymin=y_min_n, ymax=y_max_n, linestyle='-', color='orange',linewidth=2)

    print('444')
    
    #=================================
    
    # Set the right-hand side y-axis label
    
    ax4.set_ylabel(ylabel_box_abs, fontsize=11)
    #ax4.set_xlabel('Time (year)')
    
#    ax4.set_xticks([0, 1, 2, 3, 4])
#    ax4.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.', 'regional'], fontsize=11)
    
    if reg_data_type=='default':
        ax4.set_xlim(-0.5, 4.5)
    
        ax4.set_xticks([0., 1., 2., 3., 4.])
        ax4.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.', 'regional'], fontsize=11.)

    elif reg_data_type=='satellite':
        ax4.set_xlim(-0.5, 3.5)
    
        ax4.set_xticks([0., 1., 2., 3.])
        ax4.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.'], fontsize=11.)

    # Add tick marks inside the frame
    ax4.tick_params(axis='x', direction='in')
    
    plt.text(0.01, 1.03, 'c) AR6_'+ar6_reg_name,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax4.transAxes, fontsize=11)
    
    
#    plt.autoscale()
    
    ####################
    
    
    ###########################
    #data_clim = combined_all.sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim = trend_rel
    
    sns.boxplot(data=data_clim, ax=ax5, color = 'white', width=0.2, whis=1.5, linewidth=1.2,\
                            flierprops = dict(marker='d', markerfacecolor = 'gray', markersize = 8),\
                            showmeans=True,\
                            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black","markersize":"0"},
                           **PROPS)
    
#    ax5.set(xticklabels=[])  
#    ax5.set(xlabel=None)
#    ax5.tick_params(bottom=False)  # remove the ticks
    
    print('+++++++++')
    #========================================
    # Add a vertical line to the right-hand side plot
    # Assuming that the first four time series in combined_all are the ones for which you want to draw vertical lines
    # add the vertical lines beside the boxplot
    
    #data_clim1 = combined_all[0:5].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim1 = trend_rel[num_reg:(num_reg+num_insitu)]
    
    #print(combined_all[0:1])
    
    data_clim_arr1 = np.array(data_clim1) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr1):
        x_pos = 1.0 # set the x position for the vertical line
        #ax5.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax5.plot([x_pos], [d], marker='_', markersize=8, color='purple', linewidth=5)

        ax5.text(x_pos + 0.1, d, f'{i+1}', fontsize=9, verticalalignment='center')
        
    ax5.set_ylim(ax5.get_ylim())
    
    y_min = np.nanmin(data_clim_arr1)
    y_max = np.nanmax(data_clim_arr1)

    print('666')
    
    ax5.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='purple',linewidth=2)
    
    ##################
    #data_clim2 = combined_all[5:13].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim2 = trend_rel[(num_reg+num_insitu):(num_reg+num_insitu+num_sat)]
    
    data_clim_arr2 = np.array(data_clim2) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr2):
        x_pos = 2.0 # set the x position for the vertical line
        #ax5.axhline(y=d, linestyle='--', color='gray', linewidth=1)
        
        ax5.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)

        ax5.text(x_pos + 0.1, d, f'{i+1+num_insitu}', fontsize=9, verticalalignment='center')
    
    y_min = np.nanmin(data_clim_arr2)
    y_max = np.nanmax(data_clim_arr2)

    print('777')
    
    if reg_data_type=='default':
        ax5.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='orange',linewidth=2)
    elif reg_data_type=='satellite':
        y_min_glb_satellite = y_min
        y_max_glb_satellite = y_max
    
    
    ##################
    #data_clim4 = combined_all[17:23].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim4 = trend_rel[(num_reg+num_insitu+num_sat):(num_reg+num_insitu+num_sat+num_reanal)]
    
    data_clim_arr4 = np.array(data_clim4) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr4):
        x_pos = 3.0 # set the x position for the vertical line
        
        ax5.plot([x_pos], [d], marker='_', markersize=8, color='green', linewidth=5)

        ax5.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat}', fontsize=9, verticalalignment='center')
    
    y_min = np.nanmin(data_clim_arr4)
    y_max = np.nanmax(data_clim_arr4)

    print('888')
    
    ax5.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='green',linewidth=2)
    
    ##################
    #data_clim5 = combined_all[23:25].sel(time=slice('2001', '2013')).mean(dim='time').values
    data_clim5 = trend_rel[0:num_reg]
    
    data_clim_arr5 = np.array(data_clim5) # convert data_clim to numpy array
    for i, d in enumerate(data_clim_arr5):
        x_pos = 4.0 # set the x position for the vertical line
        
        ax5.plot([x_pos], [d], marker='_', markersize=8, color='blue', linewidth=5)

        ax5.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat+num_reanal}', fontsize=9, verticalalignment='center')
    
    if reg_data_type=='default':
        for i, d in enumerate(data_clim_arr5):
            x_pos = 4 # set the x position for the vertical line
            
            ax5.plot([x_pos], [d], marker='_', markersize=8, color='blue', linewidth=5)
    
            # Add text label near each marker
            #label_offset = 1.  # Adjust this value as needed
            ax5.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat+num_reanal}', fontsize=9, verticalalignment='center')

            y_min = np.nanmin(data_clim_arr5)
            y_max = np.nanmax(data_clim_arr5)

            ax5.vlines(x=x_pos, ymin=y_min, ymax=y_max, linestyle='-', color='blue',linewidth=2)

    elif reg_data_type=='satellite':
        for i, d in enumerate(data_clim_arr5):
            x_pos = 2 # set the x position for the vertical line
            
            ax5.plot([x_pos], [d], marker='_', markersize=8, color='orange', linewidth=5)
    
            # Add text label near each marker
            #label_offset = 1.  # Adjust this value as needed
            ax5.text(x_pos + 0.1, d, f'{i+1+num_insitu+num_sat+num_reanal}', fontsize=9, verticalalignment='center')
    
            y_min = np.nanmin(data_clim_arr5)
            y_max = np.nanmax(data_clim_arr5)

            y_min_n = np.nanmin([y_min_glb_satellite, y_min])
            y_max_n = np.nanmax([y_max_glb_satellite, y_max])
    
            ax5.vlines(x=x_pos, ymin=y_min_n, ymax=y_max_n, linestyle='-', color='orange',linewidth=2)

    print('999')
    
    #=================================
    
    print('000')

    # Set the right-hand side y-axis label
    
    ax5.set_ylabel(ylabel_box_rel, fontsize=11)
    #ax4.set_xlabel('Time (year)')
    
#    ax5.set_xticks([0, 1, 2, 3, 4])
#    ax5.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.', 'regional'], fontsize=11)
    
    print('000===')

    if reg_data_type=='default':
        ax5.set_xlim(-0.5, 4.5)
    
        print('111===')
    
        ax5.set_xticks([0., 1., 2., 3., 4.])
        print('222===')
        ax5.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.', 'regional'], fontsize=11.)

    elif reg_data_type=='satellite':
        ax5.set_xlim(-0.5, 3.5)
    
        print('111===')
    
        ax5.set_xticks([0., 1., 2., 3.])
        print('222===')
        ax5.set_xticklabels(['All', 'in situ', 'sat. corr.', 'reanal.'], fontsize=11.)

    print('333===')
    # Add tick marks inside the frame
    ax5.tick_params(axis='x', direction='in')
    print('444===')
    
    plt.text(0.01, 1.03, 'd) AR6_'+ar6_reg_name,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax5.transAxes, fontsize=11)
    
#    plt.autoscale()
    
    ####################

    # Add legend with multiple columns
    num_series = time_series_index.sizes['new_dim']
    print(num_series)
    num_cols = 5
    num_rows = (num_series + num_cols - 1) // num_cols
    print(num_rows)
    legend_handles, legend_labels = ax3.get_legend_handles_labels()
    plt.legend(legend_handles, legend_labels, ncol=num_cols, loc='upper left',
               bbox_to_anchor=(-2.0, -0.25), frameon=False)
    

    fig.suptitle(f"Plots of {pr_indx_n} over {region_n} in the updated SREX regions in the sixth IPCC assessment report (AR6)",fontsize=21, wrap=True, y=1.2, \
                 linespacing=2.,fontdict={'fontname': 'serif'}) #, 'fontstyle': 'italic'
    
    fig.text(0.5, 0.93, f"Time series and trends of {pr_indx_n} and over {region_n} in the updated SREX regions in the sixth IPCC assessment report (AR6)", ha='center', wrap=True, fontsize=16)
    
    if reg_data_type=='default':
        txt = (f"Fig. 1. Time series and trends of spatially-weighted average {pr_indx_n} {pr_indx_n_unit} over {region_n} ({ar6_reg_name}), which are based on datasets from "
               f"the global Frequent Rainfall Observations on GridS (FROGS; Roca et al 2019) and {reg_data_details}. "
               f"(a) and (b): time series of {pr_indx_n} and its anomaly. (c) and (d): box and whiskers "
               f"for the absolute {pr_indx_n_trabs_unit} and relative {pr_indx_n_trrel_unit} "
               f"trends of {pr_indx_n} during {trend_p_n}. "
               f"Moreover, for (a), the box and whiskers provide information on the distribution of {pr_indx_n} over the {base_p_n} climatological "
               f"period for all data products. Each colour represents each product ‘cluster’, i.e. all data products (black), "
               f"in situ (purple), satellite corrected (orange), "
               f"reanalyses (green) and regional in situ data (blue).")
    elif reg_data_type=='satellite':
        txt = (f"Fig. 1. Time series and trends of spatially-weighted average {pr_indx_n} {pr_indx_n_unit} over {region_n} ({ar6_reg_name}), which are based on datasets from "
               f"the global Frequent Rainfall Observations on GridS (FROGS; Roca et al 2019) and {reg_data_details}. "
               f"(a) and (b): time series of {pr_indx_n} and its anomaly. (c) and (d): box and whiskers "
               f"for the absolute {pr_indx_n_trabs_unit} and relative {pr_indx_n_trrel_unit} "
               f"trends of {pr_indx_n} during {trend_p_n}. "
               f"Moreover, for (a), the box and whiskers provide information on the distribution of {pr_indx_n} over the {base_p_n} climatological "
               f"period for all data products. Each colour represents each product ‘cluster’, i.e. all data products (black), "
               f"in situ (purple), satellite corrected (orange), "
               f"reanalyses (green).")
    
    fig.text(-0.05, -0.1, txt, wrap=True, ha='left', va='top', fontsize=13) #
    
##    fig.text(0.5, 0.93, f"Box and whiskers for the absolute and relative trends of {pr_indx_n} over {region_n}", ha='center', wrap=True, fontsize=16)
##    
##    txt = (f"Fig. 4. Box and whiskers for the absolute {pr_indx_n_trabs_unit} and relative {pr_indx_n_trrel_unit} "
##    f"trends of {pr_indx_n} during {trend_p_n} over {region_n}. Each colour represents each product ‘cluster’, "
##    f"i.e. all data products (black), in situ (purple), satellite corrected (orange), "
##    f"reanalyses (green) and regional data (blue).")
##    fig.text(0.01, 0.05, txt, wrap=True, ha='left', va='top', fontsize=13) #
    
#    fig.savefig(out_dir+file_name+"_"+pr_indx_name+'_boxplot_trend_trper_rob.png',dpi=dpi_set,format=format_n1,bbox_inches='tight')
#    fig.savefig(out_dir+file_name+"_"+pr_indx_name+'_boxplot_trend_trper_rob.eps',dpi=dpi_set,format=format_n2,bbox_inches='tight')
    
    plt.show()