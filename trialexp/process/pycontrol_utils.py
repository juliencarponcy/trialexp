from trialexp.process.data_import import Event, State
import pandas as pd
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from plotly.subplots import make_subplots
import numpy as np 

def parse_events(session):
    #parse the event and state information and return it as a dataframe

    #parse the events list to distinguish between state and event
    state_names = session.state_IDs.keys()
    events = session.events

    for i, evt in enumerate(events):
        if session.events[i].name in state_names:
            events[i] = State(evt.time, evt.name)

    #parse the print line and turn them into events
    print_evts = []
    for ln in session.print_lines:
        s = ln.split()
        print_evts.append(
            Event(int(s[0]), ' '.join(s[1:])))

    # merge the print list and event list and sort them by timestamp
    all_events = events+print_evts
    all_events = sorted(all_events, key=lambda x:x.time)

    #convert events into a data frame
    # state change is regarded as a special event type
    evt_list = []
    last_state = None
    for evt in all_events:
        if type(evt) is State:
            last_state = evt.name
            event = {
               'state':last_state,
                'event_name':'state_change',
                'time':evt.time,
            }
        else:
            event = {
                'state':last_state,
                  'event_name':evt.name,
                    'time':evt.time,
            }

        evt_list.append(event)


    df_events = pd.DataFrame(evt_list)

    # remove rsync
    df_events = df_events[df_events.event_name!='rsync'].copy()
        
    return df_events


def add_trial_number(df_events, trigger):
    # trigger is a tuple containing the state and event_name e.g. ('waiting_for_spout','state_change')
    # I really liked that
    df = df_events.copy()


    df['trial_number'] = 0

    df.loc[(df.state=='waiting_for_spout') & (df.event_name=='state_change'), 'trial_number'] = 1
    df.trial_number = df.trial_number.cumsum()
    
    return df

def plot_session(df:pd.DataFrame, keys: list = None, state_def: list = None, print_expr: list = None, 
                    event_ms: list = None, export_smrx: bool = False, smrx_filename: str = None, verbose :bool = False,
                    print_to_text: bool = True):
        """
        Visualise a session using Plotly as a scrollable figure

        keys: list
            subset of self.times.keys() to be plotted as events
            Use [] to plot nothing

        state_def: dict, list, or None = None
            must be None (default)
            or dictionary of 
                'name' : str
                    Channel name
                'onset' : str 
                    key for onset 
                'offset' : str
                    key for offset
            or list of such dictionaries

            eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
            eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}

            For each onset, find the first offset event before the next onset 

        event_ms: list of dict
                'name':'name of something'
                'time_ms': X
            allow plotting timestamps as an event

        state_ms: list of dict #TODO

        verbose :bool = False


        """

        # see  \Users\phar0528\Anaconda3\envs\trialexp\Lib\site-packages\sonpy\MakeFile.py
        #NOTE cannot put file path in the pydoc block

        raw_symbols  = SymbolValidator().values
        symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]
        # 40 symbols

        fig = go.Figure()
        if keys is None:
            keys = df.event_name.unique()
        else:
            for k in keys: 
               assert k in df.event_name.unique(), f"{k} is not found in self.time.keys()"

        def find_states(state_def_dict: dict):
            """
            state_def: dict, list, or None = None
            must be None (default)
            or dictionary of 
                'name' : str
                    Channel name
                'onset' : str 
                    key for onset 
                'offset' : str
                    key for offset
            or list of such dictionaries

            eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
            eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}

            For each onset, find the first offset event before the next onset 
            """
            if state_def_dict is None:
                return None

            all_on_ms = df[(df.state == state_def_dict['onset']) & (df.event_name == 'state_change')].time.values
            all_off_ms = df[(df.state == state_def_dict['offset']) & (df.event_name == 'state_change')].time.values

            onsets_ms = [np.NaN] * len(all_on_ms)
            offsets_ms = [np.NaN] * len(all_on_ms)

            for i, this_onset in enumerate(all_on_ms):  # slow
                good_offset_list_ms = []
                for j, _ in enumerate(all_off_ms):
                    if i < len(all_on_ms)-1:
                        if all_on_ms[i] < all_off_ms[j] and all_off_ms[j] < all_on_ms[i+1]:
                            good_offset_list_ms.append(all_off_ms[j])
                    else:
                        if all_on_ms[i] < all_off_ms[j]:
                            good_offset_list_ms.append(all_off_ms[j])

                if len(good_offset_list_ms) > 0:
                    onsets_ms[i] = this_onset
                    offsets_ms[i] = good_offset_list_ms[0]
                else:
                    ...  # keep them as nan

            onsets_ms = [x for x in onsets_ms if not np.isnan(x)]  # remove nan
            offsets_ms = [x for x in offsets_ms if not np.isnan(x)]

            state_ms = map(list, zip(onsets_ms, offsets_ms,
                           [np.NaN] * len(onsets_ms)))
            # [onset1, offset1, NaN, onset2, offset2, NaN, ....]
            state_ms = [item for sublist in state_ms for item in sublist]
            
            return state_ms

        y_index = 0
        
        for kind, k in enumerate(keys):
            y_index += 1
            df_evt2plot = df[df.event_name==k]
            line1 = go.Scatter(x=df_evt2plot.time, y=[k]
                        * len(df_evt2plot), name=k, mode='markers', marker_symbol=symbols[y_index % 40])
            fig.add_trace(line1)

            if export_smrx:
                write_event(MyFile, df.time[k], k, y_index, EventRate, time_vec_ms)


        if event_ms is not None:
            if isinstance(event_ms, dict):
                event_ms = [event_ms]
            
            for dct in event_ms:
                y_index += 1
                line3 = go.Scatter(
                    x=[t/1000 for t in dct['time_ms']],
                    y=[dct['name']] * len(dct['time_ms']),
                    name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
                fig.add_trace(line3)


        if state_def is not None:
            # Draw states as gapped lines
            # Assuming a list of lists of two names

            if isinstance(state_def, list):# multiple entry
                state_ms = None
                for state in state_def:
                    assert isinstance(state, dict)
                    
                    y_index +=1
                    state_ms = find_states(state)
                    print(state['name'], state_ms)
                    print([state['name']] * len(state_ms))

                    line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[state['name']] * len(state_ms), 
                        name=state['name'], mode='lines', line=dict(width=5))
                    fig.add_trace(line1)

            else:
                state_ms = None
        else:
            state_ms = None
             

        fig.update_xaxes(title='Time (s)')
        # fig.update_yaxes(fixedrange=True) # Fix the Y axis

        # fig.update_layout(
            
        #     title =dict(
        #         text = f"{self.task_name}, {self.subject_ID} #{self.number}, on {self.datetime_string} via {self.setup_ID}"
        #     )
        # )

        fig.show()

        if export_smrx:
            del MyFile
            #NOTE when failed to close the file, restart the kernel to delete the corrupted file(s)
            print(f'saved {smrx_filename}')