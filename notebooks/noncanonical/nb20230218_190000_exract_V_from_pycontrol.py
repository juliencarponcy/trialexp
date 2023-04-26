#!/usr/bin/env python
# coding: utf-8

# This script will provide summary for the parameters used in a session and their changes, as well as session duration and sumary data from the last trial (the last line starting with #) for experiment notes

# In[ ]:


import os

nb_name = "nb20230218_190000_exract_V_from_pycontrol.ipynb" #TODO change this

basename, ext = os.path.splitext(nb_name)
input_path = os.path.join(os.getcwd(), nb_name)

get_ipython().system('jupyter nbconvert "{input_path}" --to="python" --output="{basename}"')


# In[ ]:


import re
from datetime import datetime 
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilenames(initialdir=r"\\ettin\Magill_Lab\Julien\Data\head-fixed\pycontrol",
    filetypes = [('Text files','*.txt')],
    multiple =True)


# In[ ]:


#TODO sort them in the order of animals and then for datetime
# Sort them in the order of they happened

m = [re.search('\-(\d{4}\-\d{2}\-\d{2}\-\d{6}).txt', fp) for fp in file_path]
    
dt_obj = []    
ind = []    
for i, dt in enumerate([datetime.strptime(m_.group(1), '%Y-%m-%d-%H%M%S') for m_ in m]):
    dt_obj.append(dt)
    ind.append(i)   

sorted_ind = sorted(ind, key=lambda i: dt_obj[i])

file_path = list(file_path) 
file_path = [ file_path[i] for i in sorted_ind]


# In[ ]:


for fp in file_path: # list of file paths

    if len(fp) == 0:
        raise Exception("Cancelled")

    with open(fp, 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    v_lines = [line for line in all_lines if bool(re.match('^V\s\d+\s', line))]

    print('```python')
    print(f'"{fp}"')

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sExperiment\sname\s+:\s(.+)', all_lines[i])
        i += 1
    exp_name = m.group(0)
    print(f"{exp_name}")


    m = None
    i = 0
    while m is None:
        m = re.match('^I\sTask\sname\s\:\s(.+)', all_lines[i])
        i += 1
    task_name = m.group(0)
    print(f"{task_name}")

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sSetup\sID\s\:\s(.+)', all_lines[i])
        i += 1
    setup_id = m.group(0)
    print(f"{setup_id}")

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sSubject\sID\s\:\s(.+)', all_lines[i])
        i += 1
    subject_id = m.group(0)
    print(f"{subject_id}")

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sStart\sdate\s\:\s(.+)', all_lines[i])
        i += 1
    start_date = m.group(0)
    print(f"{start_date}")

    i = -1
    m = None
    while m is None:
        m = re.match('^\w\s(\d+)',all_lines[i])
        i -= 1
    print(f"{float(m.group(1))/60000:.1f} min\n")


    flag_notyet = True
    for string in v_lines:
        if flag_notyet:
            if not bool(re.match('^V\s0\s', string)):
                flag_notyet = False
                print('')

        print(string)


    print()

    i = -1
    m = None
    while m is None and i >= -1 * len(all_lines):
        m = re.match('^#.+',all_lines[i])
        i -= 1
    if m is not None:    
        print(f"{m.group(0):s}")

    print('```')
    print('\n\n')



