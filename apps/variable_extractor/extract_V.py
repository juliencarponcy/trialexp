'''
# This script will provide summary for the parameters used in a session and their changes, as well as session duration and sumary data from the last trial (the last line starting with #) for experiment notes

You can compile this into standalone application
1. install auto-py-to-exe
2. Execute auto-py-to-exe by directly typing `auto-py-to-exe` into terminal
3. In the launched window, click 'Browse' and choose this script file
4. Choose 'One File'
5. Choose 'Console Based'
6. Click 'Convert .py to .exe', 
7. The exe file should be in the `output` folder

Usage

1. Execute this script or the compiled exe
2. Click the button 'choose a list of pycontrol file` to select the pycontrol file you want to extract.
The extracted variables should be inserted into the textbox
3. Click `copy to clipboard` to copy content to clipboard

'''

# %%
import re
from datetime import datetime 
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import csv
import json

#Build the GUI
root = tk.Tk(screenName='Extract V')

root.rowconfigure(0, minsize=800, weight=1)
root.columnconfigure(1, minsize=1000, weight=1)

control_frame = tk.Frame(root, relief=tk.RAISED, bd=2)

button = tk.Button(control_frame, text='Choose a list of pycontrol files')
copy_button = tk.Button(control_frame, text='Copy to clipboard')


button.grid(row=0,column=0, sticky='ew', padx=5, pady=5)
copy_button.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

control_frame.grid(row=0, column=0, sticky='ns')
scrollbar = tk.Scrollbar(root)
text_box = tk.Text(root, yscrollcommand=scrollbar.set)

scrollbar.config(command=text_box.yview)

scrollbar.grid(row=0, column=2,sticky='ns')
text_box.grid(row=0, column=1, sticky='nsew')


def extract_info_v1(fp):
    with open(fp, 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    v_lines = [line for line in all_lines if bool(re.match('^V\s\d+\s', line))]

    output_text = '```python\n'
    output_text += f'"{fp}"\n'

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sExperiment\sname\s+:\s(.+)', all_lines[i])
        i += 1
    exp_name = m.group(0)
    output_text+=f"{exp_name}\n"


    m = None
    i = 0
    while m is None:
        m = re.match('^I\sTask\sname\s\:\s(.+)', all_lines[i])
        i += 1
    task_name = m.group(0)
    output_text+=f"{task_name}\n"

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sSetup\sID\s\:\s(.+)', all_lines[i])
        i += 1
    setup_id = m.group(0)
    output_text+=f"{setup_id}\n"

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sSubject\sID\s\:\s(.+)', all_lines[i])
        i += 1
    subject_id = m.group(0)
    output_text+=f"{subject_id}\n"

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sStart\sdate\s\:\s(.+)', all_lines[i])
        i += 1
    start_date = m.group(0)
    output_text+=f"{start_date}\n"

    i = -1
    m = None
    while m is None:
        m = re.match('^\w\s(\d+)',all_lines[i])
        i -= 1
    output_text+=f"{float(m.group(1))/60000:.1f} min\n\n"


    flag_notyet = True
    for string in v_lines:
        if flag_notyet:
            if not bool(re.match('^V\s0\s', string)):
                flag_notyet = False
                print('')

        output_text+=string+'\n'


    output_text+='\n'

    i = -1
    m = None
    while m is None and i >= -1 * len(all_lines):
        m = re.match('^#.+',all_lines[i])
        i -= 1
    if m is not None:    
        output_text+=f"{m.group(0):s}"

    output_text+='\n```'
    output_text+='\n\n\n'
    
    return output_text

def concat_variable_line(lines):
    s = ''
    d = [(x[0], json.loads(x[3])) for x in lines]
    for t, d in d:
        for k, v in d.items():
            if not k.endswith('___'):
                s += f'{t} \t {k} {v}\n'
    return s

def extract_info_v2(fp):
    with open(fp, 'r') as f:
        s = '```python\n'
        s += f'"{fp}"\n'
        rd = list(csv.reader(f, delimiter='\t', quotechar='"'))
        
        info2print = ['task_name', 'subject_id', 'start_time', 'end_time']
        info_lines = list(filter(lambda x: x[2] in info2print, rd))
        info_lines = [f'{x[2]}: {x[3]}' for x in info_lines]
        
        run_start = list(filter(lambda x: x[1] in ['variable'] and x[2] == 'run_start', rd))
        user_set = list(filter(lambda x: x[1] in ['variable'] and x[2] == 'user_set', rd))
        
        last_status = list(filter(lambda x: x[1] == 'print', rd))[-1][3]
        
        s += '\n'.join(info_lines)
        s += '\n'
        s += f'{float(rd[-1][0]) / 60:.1f} min'
        s += '\n\n'
        s += concat_variable_line(run_start)
        s += '\n\n'
        s += concat_variable_line(user_set)
        s += '\n\n'
        s += last_status
        s += '\n```'
        s += '\n\n\n'
        return s
    

def get_variable_info(event):
    file_path = filedialog.askopenfilenames(initialdir=r"\\ettin\Magill_Lab\Julien\ASAP\DATA",
        filetypes = [('Pycontrol files','*.txt;*.tsv')],
        multiple =True)

    
    #TODO sort them in the order of animals and then for datetime
    # Sort them in the order of they happened

    m = [re.search('\-(\d{4}\-\d{2}\-\d{2}\-\d{6}).', fp) for fp in file_path]
        
    dt_obj = [datetime.strptime(m_.group(1), '%Y-%m-%d-%H%M%S') for m_ in m]
    sorted_ind = sorted(range(len(dt_obj)), key=lambda i: dt_obj[i])
    file_path = [Path(file_path[i]) for i in sorted_ind]

    output_text = ''
    print(file_path)
    for fp in file_path: # list of file paths
        if fp.suffix == '.txt':
            output_text += extract_info_v1(fp)
        else:
            output_text += extract_info_v2(fp)
            
    
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, output_text)

def copy2clipboard(event):
    root.clipboard_clear()
    root.clipboard_append(text_box.get(1.0, tk.END))
    print('Content copied to clipboard')
    


button.bind('<Button-1>', get_variable_info)
copy_button.bind('<Button-1>', copy2clipboard)
root.mainloop()


