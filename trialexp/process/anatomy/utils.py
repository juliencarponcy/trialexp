import pandas as pd
import matplotlib.patches as patches
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def get_region_boundary(df_cell):
    #Find the region boundaries
    def get_region_boundary(df):
        return pd.Series({'min_mm': df.dv_mm.min(),
                'max_mm': df.dv_mm.max()})
    
    region_boundary = df_cell.groupby('acronym').apply(get_region_boundary)
    region_boundary = region_boundary.sort_values('min_mm').reset_index()
    return region_boundary

def assign_region_layer(df):
    # assign non-overlapping regions to its own layer for easier plotting later

    region_boundary = df.copy()

    region_boundary['layer'] = -1
    region_boundary.loc[0,'layer'] = 0
    
    def check_overlap(region1, region2):
      return not ((region1.min_mm > region2.max_mm) or (region1.max_mm < region2.min_mm))
    
    for idx, region in region_boundary.iloc[1:].iterrows():
    
        # loop through all the existing layer
        for i in range(region_boundary.layer.max()+1):
            region_in_layer = region_boundary[region_boundary.layer == i]
    
            #check if the current region can be added
            if not any([check_overlap(region, r) for _,r in region_in_layer.iterrows()]):
                region_boundary.loc[idx, 'layer'] = i
                break
        else: #execute when for loop is finished, i.e. no break is encountered
            region_boundary.loc[idx, 'layer'] = i+1        

    return region_boundary

def add_regions(ax, region_boundary, dv_bins):
    colorIdx = 0
    dv_bin_size = np.mean(np.diff(dv_bins))
    init_x_lim = ax.get_xlim()[1]
    colors = plt.cm.tab20.colors
    
    for idx, row in region_boundary.iterrows():
        # convert the dv coordinates to the bin coordinates so that the plot looks right
        y = (row.min_mm - dv_bins[0])/dv_bin_size
        height = (row.max_mm - row.min_mm)/dv_bin_size
        x = init_x_lim+10+row.layer*5
        region = row.acronym
    
        rect = patches.Rectangle((x, y), 4.8, height, color=colors[colorIdx%len(colors)])
        ax.add_patch(rect)
        colorIdx += 1
    
        # also add the region text
        t = ax.text(x+1, y+height/2, region, fontsize=10)

    # expand the xlim
    ax.set_xlim([0, init_x_lim+10+(region_boundary.layer.max()+1)*5])

    return ax


def format_cell4merge(df_cell):
    # Convert the dataframe into proper format for merging

    df_cell = df_cell.copy()

    def get_session_date(session_id):
        if type(session_id) is str:
        # only return the date of the session
            return '-'.join(session_id.split('-')[:-1])
    
    # extract the session ID and probe name from the cluID so that we can merge to Sharptrack results
    df_cell[['session_id','probe','id']] = df_cell.cluID.str.split('_',expand=True)
    df_cell['session_date'] = df_cell['session_id'].apply(get_session_date)
    df_cell['probe']  = df_cell['probe'].str.replace('Probe','')
    return df_cell

def plot_firing_rate_regions(df_cell, depth_col='dv_mm'):
    # plot firing rate of brain regions with different depth
    df_cell = df_cell.copy()
    df_cell['depth_group'], dv_bins = pd.cut(df_cell[depth_col],30, retbins=True)
    region_boundary = get_region_boundary(df_cell)
    region_boundary = assign_region_layer(region_boundary)
    
    plt.figure(figsize=(8,len(region_boundary)*1))
    ax = sns.barplot(df_cell, y='depth_group', x='firingRate')
    ax = add_regions(ax, region_boundary, dv_bins)
    ax.set(ylabel='dv_mm', xlabel='Firing Rate (Hz)')