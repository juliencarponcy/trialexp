import pandas as pd
import matplotlib.patches as patches
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

def get_region_boundary(df_cell, dep_col,group_method='consecutive'):
    df_cell = df_cell.sort_values(dep_col)
    df_cell['group'] = (df_cell['name']!=df_cell['name'].shift()).cumsum() #detect region boundaries
    #Find the region boundaries
    def get_boundary(df,method):
        if method =='consecutive':
            return pd.Series({'min_mm': df[dep_col].min(),
                    'max_mm': df[dep_col].max(),
                    'name':df.iloc[0]['name'],
                'acronym': df.iloc[0].acronym})
        else:
           return pd.Series({'min_mm': df[dep_col].min(),
                    'max_mm': df[dep_col].max()}) 
    
    if group_method == 'consecutive':
        region_boundary = df_cell.groupby(['group']).apply(get_boundary, method=group_method)
    else:
        region_boundary = df_cell.groupby(['name','acronym']).apply(get_boundary, method=group_method)

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
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    init_x_lim = xlim[1]
    colors = plt.cm.tab20.colors
    rect_width = (xlim[1]-xlim[0])*0.2
    
    unique_region = region_boundary.acronym.unique()
    color_table = {r:colors[i%len(colors)] for i,r in enumerate(unique_region)}
    
    for idx, row in region_boundary.iterrows():
        # convert the dv coordinates to the bin coordinates so that the plot looks right
        y = (row.min_mm - dv_bins[0])/dv_bin_size
        height = (row.max_mm - row.min_mm)/dv_bin_size
        x = init_x_lim+rect_width+row.layer*rect_width
        region = row.acronym
    
        rect = patches.Rectangle((x, y), rect_width, height, color=color_table[region])
        ax.add_patch(rect)
        colorIdx += 1
    
        # also add the region text
        t = ax.text(x+rect_width/4, y+height/2, region, fontsize=10)

    # expand the xlim
    ax.set_xlim([0, init_x_lim+rect_width*1.2+(region_boundary.layer.max()+1)*rect_width])
    ax.set_ylim([ylim[0]+1, ylim[1]])
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

def draw_region_legend(ax, region_boundary):
    y = ax.get_ylim()[1] +2
    x = ax.get_xlim()[1]+2
    
    for idx, region in region_boundary.iterrows():
        ax.text(x,y, f'{region.acronym}: {region["name"]}')
        y += 1
    

def plot_firing_rate_regions(df_cell, depth_col='depth_mm', group_method='consecutive'):
    # plot firing rate of brain regions with different depth
    df_cell = df_cell.copy()
    df_cell['depth_group'], dv_bins = pd.cut(df_cell[depth_col],30, retbins=True)
    region_boundary = get_region_boundary(df_cell, depth_col,group_method)
    region_boundary = assign_region_layer(region_boundary)
    # display(region_boundary)
    
    plt.figure(figsize=(8,max(len(region_boundary)*0.6,12)),dpi=200)
    ax = sns.barplot(df_cell, y='depth_group', x='firingRate')
    
    #set a consistent max rate so that figures from different sessions are comparable
    max_rate = ax.get_xlim()[1]
    ax.set_xlim([0,max(40,max_rate)])
    
    ax = add_regions(ax, region_boundary, dv_bins)
    ax.set(ylabel=depth_col, xlabel='Firing Rate (Hz)')
    
    draw_region_legend(ax, region_boundary)
    return ax
    
def get_session_date(session_id):
    if type(session_id) is str:
    # only return the date of the session
        return '-'.join(session_id.split('-')[:-1])