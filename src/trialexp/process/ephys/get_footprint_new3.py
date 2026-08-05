import numpy as np
import scipy
from scipy import interpolate
import math

def get_sign(unit):
    troughs = []
    peaks = []

    for chan in unit:
        troughs.append(np.min(chan))
        peaks.append(np.max(chan))

    troughs_min = np.min(troughs)
    peaks_max = np.max(peaks)
    if abs(troughs_min) > abs(peaks_max):
        sign = 1
    else:
        sign = 0

    return sign

def get_amps(unit):
    '''Given a 2d array, find the largest amplitude signal from trough to peak for each array element in axis = 0.'''
    
    amps = []
    for chan, wv in enumerate(unit):
        trough_idx = np.where(wv==np.min(wv))[0][0]
        peak_idx = np.where(wv[trough_idx:]==wv[trough_idx:].max())[0][0]+trough_idx

        trough_h = wv[trough_idx]
        peak_h = wv[peak_idx]
        amp = (peak_h+abs(trough_h))
        amps.append(amp) 

    amps = np.array(amps)
    amps = amps-np.mean(np.sort(amps)[:10])
        
    return np.array(amps)


def get_amp_map(amps,xcoords,ycoords):

    # only use 500 um around peak site in ycoords, to speed up 2d interpolation
    I_max2 = np.where(amps==np.max(amps))[0]
    maxAmpY = ycoords[I_max2]
    index1 = np.where((ycoords>= maxAmpY-500) & (ycoords <=maxAmpY+500))
    amps = amps[index1]
    xcoords = xcoords[index1]
    ycoords = ycoords[index1]

    '''create a high resolution amplitude matrix, using 2d interpolate
    matrix boundry is defined by max and min of site position.
    '''
    xmin = np.min(xcoords)
    xmax = np.max(xcoords)
    ymin = np.min(ycoords)
    ymax = np.max(ycoords)
    width = np.ceil(xmax-xmin+1).astype(int)
    height = np.ceil(ymax-ymin+1).astype(int)
    xcoords2 = np.round(xcoords-xmin).astype(int)
    ycoords2 = np.round(ycoords-ymin).astype(int)
    width1 = np.arange(0,width)
    height1 = np.arange(0,height)
    xx, yy = np.meshgrid(width1, height1)
    if len(np.unique(xcoords)) > 1:
        GD1 = interpolate.griddata((ycoords2,xcoords2),amps,(yy,xx), method = 'cubic')
    else:
        spl = interpolate.interp1d(ycoords2, amps, kind='cubic', fill_value="extrapolate")
        GD1 = spl(yy)   

    return GD1

def polar_to_cartesian(theta,r):
    xall = []
    yall = []
    for i,r1 in enumerate(r):
        x = r1 * math.cos(theta[i])
        y = r1 * math.sin(theta[i])        
        xall.append(x)
        yall.append(y)

    return xall, yall

def create_sampling_matrix(pointN):
    # sample amp at different radius from the peak site
    # first design the sampling position matrix
    rho = np.arange(0,pointN)
    # Initialize theta array and set up parameters
    theta = np.zeros_like(rho)
    count1 = 0
    # Define beta range from -pi to pi with steps of pi/6
    beta_values = np.arange(-np.pi, np.pi, np.pi/6)
    # Initialize xq and yq as empty lists
    xq = np.empty((0,pointN))
    yq = np.empty((0,pointN))
    for beta in beta_values:
        x, y = polar_to_cartesian(theta + beta, rho)  # Use numpy's pol2cart equivalent
        xq = np.append(xq,np.array(x).reshape(1,-1),axis=0)
        yq = np.append(yq,np.array(y).reshape(1,-1),axis=0)

    return xq, yq

def get_row_spacing(xcoords,ycoords):
    xc1 = np.unique(xcoords)
    ysort = np.sort(ycoords[xcoords==xc1[0]]);
    ysort_diff = np.diff(ysort)  # Compute the differences between consecutive elements
    ysort_diff = ysort_diff[ysort_diff>0]
    # Find unique differences and their counts
    C, ia, ic = np.unique(ysort_diff, return_inverse=True, return_index=True)
    a_counts = np.bincount(ic)
    # Ensure C is a column vector 
    if C.ndim == 1:
        C = C[:, np.newaxis]
    # Combine values and their counts
    value_counts = np.column_stack((C, a_counts))
    # Sort by the counts (second column) in descending order
    value_counts_sort = value_counts[value_counts[:, 1].argsort()[::-1]]
    # Extract the row spacing (first value in the sorted array)
    row_spacing = value_counts_sort[0, 0]

    return row_spacing




def get_footprint_radius(unit,xcoords, ycoords, threshold=30):

    ''' Given a unit waveform of shape 384 channels x 82 time samples, return distance metric calculated as the vector length from the maximum amplitude channel about which
    the average amplitude is less than or equal to threshold.
    
    threshold = arbitrary voltage threshold used to find the footprint.'''

    # get the maximum amplitude per channel. 
    sign = get_sign(unit)
    if sign == 0:
        unit = -unit
        
    amps = get_amps(unit)

    # if 4 shank probe, only use shank where peak site is at, otherwise all sites
    shank_spacing = 250
    I_max = np.argmax(amps)
    maxAmpSite_shank = np.round(xcoords[I_max]/shank_spacing)
    alSites_shank = np.round(xcoords/shank_spacing)
    siteOnShank_index = (alSites_shank == maxAmpSite_shank)
    xcoords = xcoords[siteOnShank_index]
    ycoords = ycoords[siteOnShank_index]
    amps = amps[siteOnShank_index]

    # if staggering is too much, then bad for 2d interp, like NP20 double length
    row_spacing = get_row_spacing(xcoords,ycoords)
    length_expected = np.round(len(xcoords) / len(np.unique(xcoords))) * row_spacing
    length_true = np.max(ycoords) - np.min(ycoords)
    # collapse all x into one column
    if length_true > length_expected * 1.2:
        xcoords[:] = np.min(xcoords)

    # create high resolution amplitude map, at 1um spacing
    GD1 = get_amp_map(amps,xcoords,ycoords)
    # find max amp index
    flat_index = np.nanargmax(GD1)
    row, col = np.unravel_index(flat_index, GD1.shape)


    if len(np.unique(xcoords)) > 1:
        # create sampling points for 401um sweep radius
        pointN  = 401
        xq, yq = create_sampling_matrix(pointN)
        # add sampling points around the peak site position
        xq2 = np.round(xq+col).astype(int)
        yq2 = np.round(yq+row).astype(int)
        # sample amplitude using sampling points
        vq = np.full(xq2.shape, np.nan)
        for i in range(xq2.shape[0]):
            for j in range(xq2.shape[1]):
                if (
                    1 <= yq2[i, j] < GD1.shape[0] and 
                    1 <= xq2[i, j] < GD1.shape[1]
                ):
                    vq[i, j] = GD1[yq2[i, j], xq2[i, j]]
    else:
        rho = np.arange(0,401)
        yq2 = np.empty((0,401))
        a1 = np.round(rho+row).astype(int);
        yq2 = np.append(yq2,np.array(a1).reshape(1,-1),axis=0)
        a2 = np.round(-rho+row).astype(int);
        yq2 = np.append(yq2,np.array(a2).reshape(1,-1),axis=0)
        yq2 = yq2.astype(int)
        # Initialize vq with NaN values
        vq = np.full(yq2.shape, np.nan)
        # Iterate through each element in xq2 and yq2
        for i in range(0,2):
            for j in range(yq2.shape[1]):
                if (1 <= yq2[i,j] < GD1.shape[0]):
                    vq[i,j] = GD1[yq2[i,j]]
                
    vq_mean = np.nanmean(vq,axis=0)
    # Find the first index where vq_mean <= 30
    indices = np.where(vq_mean <= 30)[0]  # Get indices where the condition is met
    # Select the first index or assign NaN if none found
    footprint = indices[0] if indices.size > 0 else 100

    return footprint