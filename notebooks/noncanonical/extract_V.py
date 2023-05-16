# %% [markdown]
# This script will provide summary for the parameters used in a session and their changes, as well as session duration and sumary data from the last trial (the last line starting with #) for experiment notes

# %%
import re
from datetime import datetime 
import tkinter as tk
from tkinter import filedialog

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


# root.withdraw()

def get_variable_info(event):
    file_path = filedialog.askopenfilenames(initialdir=r"\\ettin\Magill_Lab\Julien\Data\head-fixed\pycontrol",
        filetypes = [('Text files','*.txt')],
        multiple =True)
    
    if not file_path:
        return 

    
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

    output_text = ''
    print(file_path)
    for fp in file_path: # list of file paths

        if len(fp) == 0:
            raise Exception("Cancelled")

        with open(fp, 'r') as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]

        v_lines = [line for line in all_lines if bool(re.match('^V\s\d+\s', line))]

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
        
    text_box.insert(tk.END, output_text)

def copy2clipboard(event):
    print('test')
    root.clipboard_clear()
    root.clipboard_append('test')
    


button.bind('<Button-1>', get_variable_info)
copy_button.bind('<Button-2>', copy2clipboard)
root.mainloop()


