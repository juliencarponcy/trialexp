'''
These utilities contained originally much more methods, which have been moved
to more appropriate modules during refactoring.

It still contains mostly date matching patterns methods and helper methods
to find events timestamps for differential trial alignment.

It is possibly best to store the remaining methods in appropriate modules
'''

from datetime import datetime
from re import search

import numpy as np


'''
following is depracted until possible re-use elsewhere
#----------------------------------------------------------------------------------
# Plotting
#----------------------------------------------------------------------------------
# should move in a plotting module
def plot_longitudinal(results, plot_individuals=True):
    fontsize = 12
    condition_IDs = results['condition_ID'].unique()
    metric_names = results['metric_name'].unique()
    group_IDs = results['group_ID'].unique()

    fig, axs = plt.subplots(len(metric_names), len(condition_IDs), sharex= 'all', sharey = 'row', squeeze=False, figsize=(10,10))
    
    mean_df = results.groupby(['metric_name','condition_ID', 'group_ID', 'session_ID']).mean()
    sem_df = results.groupby(['metric_name','condition_ID', 'group_ID', 'session_ID']).sem()

    for col, cond_ID in enumerate(condition_IDs):
        for row, metric_name in enumerate(metric_names):
            
            mean_df = results.groupby(['metric_name','condition_ID', 'group_ID', 'session_ID']).mean()
            sem_df = results.groupby(['metric_name','condition_ID', 'group_ID', 'session_ID']).sem()
            
            for group_ID in group_IDs:

                subset_mean = mean_df.loc[(metric_name,cond_ID,group_ID)]
                subset_sem = sem_df.loc[(metric_name,cond_ID,group_ID)]

                axs[row, col].plot(subset_mean.index.values, subset_mean.metric.values)
                axs[row, col].fill_between(subset_mean.index.values, subset_mean.metric.values - subset_sem.metric.values,
                    subset_mean.metric.values + subset_sem.metric.values, color='gray', alpha=0.5)

                axs[row, col].set_xlabel('session nb', fontsize=fontsize)
                axs[row, col].set_ylabel(metric_name, fontsize=fontsize)
                axs[row, col].set_title('Title', fontsize=fontsize)

'''
    
def get_datetime_from_datestr(datestr: str):
    '''
    here is the order and format of Year(2 or 4 digits)-months-day and time which matters.
    All possibilities present in a folder should be listed
    for an exhaustive list of possibilities see:
    https://www.programiz.com/python-programming/datetime/strptime 
    '''

    date_patterns = ["%Y-%m-%d_%H-%M-%S", "%y-%m-%d_%H-%M-%S", "%m-%d-%y_%H-%M-%S", "%Y-%m-%d-%H%M%S"]

    for pattern in date_patterns:
        try:
            datetime_match = datetime.strptime(datestr, pattern)
            #print(s_date,datetime_match)
            return datetime_match
        except:
            #print(s_date,'exception')
            continue

def get_datestr_from_filename(filename: str):
    ''' 
        Should probably be refactored with strftime
        
        PyControl and other files like videos and DLC data may
        have different date formats in their filenames.
        The purpose of this function is to reconcile these differences
        
        list all the possible decimal format for date strings
        here the order of year month and date doesn't matter
        datestring will be later processed as a datetime
        
        Add more patterns as needed

    '''

    re_patterns = [
        '\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', 
        '\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}',
        '\d{4}-\d{2}-\d{2}-\d{6}' # ppd files date format
        ]
    
    # loop over all the date patterns to find a match
    for re_date in re_patterns:
        # print(idx)
        match_str = search(re_date, filename)
        if match_str:
            break
    
    # return datestr if match
    if match_str is not None:
        datestr = match_str.group(0)
    # or empty str if no match
    else:
        datestr = ''
    
    return datestr


def blank_spurious_detection(df_item, blank_timelim):
    '''
    Delete events within blank_timelim, which are suspected to be caused by
    unwanted contacts (e.g. drop to whisker), or artifacts in the detections.
    '''
    if isinstance(df_item, list):
        tlist = [t for t in df_item if (t < blank_timelim[0] or t > blank_timelim[1])]
        return tlist

def find_last_time_before_list(list_ev, list_lim):
    '''
    Utility function to use as apply to DataFrame in order
    to find the last event occuring before another event,
    like the last bar_off occrurign before a spout touch
    list_ev is the list of events to detect (contained in a cell of dataframe)
    list_lim is the list of events to use as limit
    '''
    if len(list_ev) >= 1 and len(list_lim) >= 1:
        last_time = max([i for i in list_ev if i < find_min_time_list(list_lim)], default=np.NaN)
    
    # TODO check implementation for limit cases (when no lim events found)
    elif isinstance(list_ev, int) and find_min_time_list(list_lim) is not np.NaN:
        
        if find_min_time_list(list_lim) > list_ev:
            last_time = list_ev
        else:
            last_time = np.NaN

    elif len(list_ev) == 0 or len(list_lim) == 0:
        last_time = np.NaN
    else:
        print(list_ev,type(list_ev))
    return last_time


def find_min_time_list(x):
    if len(x) >= 1:
        min_time = min([i for i in x if i>0], default=np.NaN)
    elif isinstance(x, int) and x > 0:
        min_time = x
    elif len(x) == 0:
        min_time = np.NaN
    else:
        print(x,type(x))

    return min_time

def find_max_time_list(x):
    if len(x) >= 1:
        max_time = max([i for i in x if i>0], default=np.NaN)
    elif isinstance(x, int) and x > 0:
        max_time = x
    elif len(x) == 0:
        max_time = np.NaN
    else:
        print(x,type(x))

    return max_time

# Make use of timelim former param for success definition (if event found within specified limits), 
# expected to possibly be legacy/deprecated method. But could be useful for some cases?
def find_if_event_within_timelim(df_item, timelim):
    if isinstance(df_item, list):
        within_lim = any(ele >= timelim[0] and ele <= timelim[1] for ele in df_item)
    else:
        within_lim = False

    return within_lim

def time_delta_by_row(df_events_row, col_idx_start, col_idx_end):
    #print(df_events_row)
    start_time = min([i for i in df_events_row[col_idx_start] if i > 0], default=np.NaN)
    if isinstance(start_time, float):
        return np.NaN
    else:
        end_time = min([i for i in df_events_row[col_idx_end] if i > start_time], default=np.NaN)
        if isinstance(end_time, int):
            return end_time - start_time
        else:
            return np.NaN

'''
# Kouichi's helper, appears not to be used
def cmap10():
    """
    Default plot colors of matplotlib.pyplot.plot, turned into colormap
    """
    cmap = (mpl.colors.ListedColormap([u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
        u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])
        ) # default 10 colors

    return cmap
'''
def get_methods(obj):
    """
    obj.__dict__
    vars(obj)
    These return attribute names and values.

    dir(obj)
    returns both attributes and methods

    See also:
    get_methods(get_attributes_and_properties)

    https://stackoverflow.com/questions/34439/finding-what-methods-a-python-object-has
    """

    spacing=20
    methodList = []
    for method_name in dir(object):
        try:
            if callable(getattr(object, method_name)):
                methodList.append(str(method_name))
        except Exception:
            methodList.append(str(method_name))
    processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
    for method in methodList:
        try:
            print(str(method.ljust(spacing)) + ' ' +
                processFunc(str(getattr(object, method).__doc__)[0:90]))
        except Exception:
            print(method.ljust(spacing) + ' ' + ' getattr() failed')

        object_methods = [method_name for method_name in dir(obj)
                if callable(getattr(obj, method_name))]
        print(object_methods)
        return object_methods

def get_attributes_and_properties(obj):
    """
    obj.__dict__
    vars(obj)
    These return attribute names and values.

    dir(obj)
    returns both attributes and methods

    See also:
    get_methods(obj)
    """
    attrnames = list(obj.__dict__.keys())

    propnames = [p for p in dir(obj) if isinstance(getattr(obj, p), property)]

    print('Attributes:')
    print(attrnames)
    print('Properties:')
    print(propnames)

    return [attrnames, propnames]


#----------------------------------------------------------------------------------
# Load analog data
#----------------------------------------------------------------------------------

def load_analog_data(file_path):
    '''Load a pyControl analog data file and return the contents as a numpy array
    whose first column is timestamps (ms) and second data values.'''
    with open(file_path, 'rb') as f:
        return np.fromfile(f, dtype='<i').reshape(-1,2)
