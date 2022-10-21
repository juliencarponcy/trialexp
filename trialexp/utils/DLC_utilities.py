# Utility functions for pycontrol and pyphotometry files processing

import numpy as np

#----------------------------------------------------------------------------------
# Plotting
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# Helpers
#----------------------------------------------------------------------------------

def get_regions_to_store(bodyparts_to_ave, names_of_ave_regions, bodyparts_to_store):
    '''
    determine which regions to store in a coordinates dict, based on the parameters
    used for get_deeplabcut_trials() method
    '''
    
    
    if names_of_ave_regions or bodyparts_to_store:
        if names_of_ave_regions and bodyparts_to_store:
            regions_to_store = names_of_ave_regions + bodyparts_to_store
        elif names_of_ave_regions and not bodyparts_to_store:
            regions_to_store = bodyparts_to_ave
        elif not names_of_ave_regions and bodyparts_to_store:
            regions_to_store = bodyparts_to_store
    return regions_to_store

#----------------------------------------------------------------------------------
# Processing helper
#----------------------------------------------------------------------------------

def normalize_coords(coord_dict, normalize_betwen=['Left_paw','spout'], bins_nb=200):
    '''
    Get the coordinates of maximum density of two regions in order to normalize trajectories.
    Only for 2D for now.
    coord_dict is a dictionary which keys are regions computed, and values are X-Y ndarrays
    return the coordinates normalized between the coords of max density of two regions.
    normalize_betwen is a 2 items list which state the start and stop region
    to normalize between.
    bins_nb is the number of bins used to compute the np.histogram2d functions.
    The trade-off for bins_nb: too high value will only have a few timestamps
    in a bin, leading to poor aggregation and then random-ish maximum coord.
    Values too low will lead to a good aggregation but much less pixel-wise
    precision.
    Used by session.get_deeplabcut_trials()
    '''
    if len(normalize_betwen) != 2:
        raise Exception('normalize_betwen must be a list of two regions (str)')
    
    min_max_coord = np.ndarray((2,2))
    for idx_r, region in enumerate(normalize_betwen):
        nan_free_coords = np.delete(coord_dict[region], np.isnan(coord_dict[region][:,0]),0)
        xmin = nan_free_coords[:,0].min()
        xmax = nan_free_coords[:,0].max()
        ymin = nan_free_coords[:,1].min()
        ymax = nan_free_coords[:,1].max()

        H, xedges, yedges = np.histogram2d(coord_dict[region][:,0],coord_dict[region][:,1], 
            bins=bins_nb , range=[[xmin, xmax], [ymin, ymax]])

        ind = np.unravel_index(np.argmax(H, axis=None), H.shape)
        min_max_coord[idx_r,:] = [xedges[ind[0]],yedges[ind[1]]]

    rangeXY = [min_max_coord[1,0] - min_max_coord[0,0], min_max_coord[1,1] - min_max_coord[0,1]]

    norm_coord_dict = dict()
    for region in coord_dict.keys():
        norm_coord_dict[region] = np.ndarray(shape=coord_dict[region].shape)
        norm_coord_dict[region][:,0] = (coord_dict[region][:,0]-min_max_coord[0,0]) / rangeXY[0]
        norm_coord_dict[region][:,1] = (coord_dict[region][:,1]-min_max_coord[0,1]) / rangeXY[1]

    return norm_coord_dict


#----------------------------------------------------------------------------------
# Load analog data
#----------------------------------------------------------------------------------

