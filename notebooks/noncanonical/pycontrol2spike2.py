'''
# Provides a quick way to export a Spike2 file from a pycontrol data file at Desktop for debugging/test purposes.

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

#TODO to export P and V values

See also

trialexp/process/pycontrol/spike2_export.py
workflow/scripts/03_export_spike2.py
trialexp/process/pycontrol/utils.py > export_session()
notebooks/noncanonical/extract_V.py

'''

# %%
import re
from datetime import datetime 
import tkinter as tk
from tkinter import filedialog
from trialexp.process.pycontrol.spike2_export import Spike2Exporter
import os

desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')

#Build the GUI
root = tk.Tk(screenName='pyControl to Spike2')

root.rowconfigure(0, minsize=800, weight=1)
root.columnconfigure(1, minsize=1000, weight=1)

control_frame = tk.Frame(root, relief=tk.RAISED, bd=2)

button = tk.Button(control_frame, text='Choose a pycontrol file') #TODO support list
copy_button = tk.Button(control_frame, text='Copy to clipboard')


button.grid(row=0,column=0, sticky='ew', padx=5, pady=5)
copy_button.grid(row=1, column=0, sticky='ew', padx=5, pady=5)

control_frame.grid(row=0, column=0, sticky='ns')
scrollbar = tk.Scrollbar(root)
text_box = tk.Text(root, yscrollcommand=scrollbar.set)

scrollbar.config(command=text_box.yview)

scrollbar.grid(row=0, column=2,sticky='ns')
text_box.grid(row=0, column=1, sticky='nsew')

#TODO get Event and State information

#TODO collect data

def export_pycontrol(smrx_filename, maxtime_ms, event_dict, state_dict ):

    spike2exporter = Spike2Exporter(smrx_filename, maxtime_ms)

    y_index = 0
    for k in event_dict.keys():

        spike2exporter.write_event(event_dict[k], k, y_index)
        y_index += 1
    
    for k in state_dict.keys():
        spike2exporter.write_marker_for_state(state_dict[k], k, y_index)
        y_index += 1

    del spike2exporter

def get_variable_info(event):
    file_path = filedialog.askopenfilenames(initialdir=r"\\ettin\Magill_Lab\Julien\Data\head-fixed\pycontrol",
        filetypes = [('Text files','*.txt')],
        multiple =False)
    file_path = file_path[0]
    if not file_path:
        return 

    
    #TODO sort them in the order of animals and then for datetime
    # Sort them in the order of they happened

    m = re.search('\-(\d{4}\-\d{2}\-\d{2}\-\d{6}).txt', file_path)
        
    # dt_obj = []    
    #ind = []

    # dt_obj = datetime.strptime(m.group(1), '%Y-%m-%d-%H%M%S')#TODO
    #ind.append(i)   

    # sorted_ind = sorted(ind, key=lambda i: dt_obj[i])

    # file_path = list(file_path) 
    # file_path = [ file_path[i] for i in sorted_ind]

    output_text = ''
    print(file_path)
    #for fp in file_path: # list of file paths
    fp = file_path

    if len(fp) == 0:
        raise Exception("Cancelled")

    with open(fp, 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    v_lines = [line for line in all_lines if bool(re.match('^V\s\d+\s', line))]
    d_lines = [line for line in all_lines if bool(re.match('^D\s\d+\s', line))]

    output_text += '```python\n'
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

    m = None
    i = 0
    while m is None:
        m = re.match('^I\sStart\sdate\s\:\s(.+)', all_lines[i])
        i += 1
    start_date = m.group(0)
    output_text+=f"{start_date}\n"
    
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

    # states
    m = None
    i = 0
    while m is None:
        m = re.match('^S\s(\{.+\})',all_lines[i])
        i += 1

    states_ = eval(m.group(1))
    states = {value: key for key, value in states_.items()}
    states_ms = {key: [] for key in states_.keys()}

    # events
    m = None
    i = 0
    while m is None:
        m = re.match('^E\s(\{.+\})',all_lines[i])
        i += 1
    
    events_ = eval(m.group(1))
    events = {value: key for key, value in events_.items()}
    events_ms = {key: [] for key in events_.keys()}

    last_state = None
    # extract D
    for string in d_lines:
        m = re.match('^D\s(\d+)\s(\d+)',string) 

        # if states
        if int(m.group(2)) in [int(k) for k in states.keys()]:#TODO
            if last_state != None:
                states_ms[last_state].append(int(m.group(1))) # end of previous state

            if states[int(m.group(2))] == last_state:
                raise Exception("States are expected to change")
            
            states_ms[states[int(m.group(2))]].append(int(m.group(1)))

            last_state = states[int(m.group(2))]
        else: # events
            events_ms[events[int(m.group(2))]].append(int(m.group(1)))
    
    
    m = re.match('^D\s(\d+)\s\d+', d_lines[-1])
    maxtime_ms = int(m.group(1))
    
    smrx_filename = re.sub('.txt$', '.smrx', os.path.join(desktop, os.path.basename(fp)))

    export_pycontrol(smrx_filename, maxtime_ms, events_ms, states_ms)
    #TODO support print and V
    
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, output_text)
    

button.bind('<Button-1>', get_variable_info)
root.mainloop()


