from trialexp.process.data_import import Event, State
import pandas as pd
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from plotly.subplots import make_subplots
import numpy as np 

def parse_events(session):
    #parse the event and state information and return it as a dataframe
    # df_events = session.df_events
    # df_conditions = session.conditions

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
    # trigger is a tuple containing the state and event_name e.g. ('waiting_for_spout','state')
    
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
        
        print_expr: list of dict #TODO need more testing
            'name':'name of channel'
            'expr': The expression '^\d+(?= ' + expr + ')' will be used for re.match()
            list of regular expressions to be searched for self.print_lines and shown as an event channel

            eg. {
                'name':'water success',
                'expr':'.?water success' # .? is needed if it is unknown whether there is any character ahead        
            }

        event_ms: list of dict
                'name':'name of something'
                'time_ms': X
            allow plotting timestamps as an event

        state_ms: list of dict #TODO

        export_smrx: Bool = False
            Save the plotted channels to a Spike 2 .smrx file.
            An event channel and a state channel will be represetnted as an event and marker channels.
            For the latter, onset and offset of a state is coded by 1 and 0 for code0.
            Use
                pip install sonpy
            to install the sonpy module.

            This metthod seems unstable. Tha same session data may fail ot succedd to export Spike2 file. Try restating kernel a few times. 
            Apparently addition of time.sleep(0.05) helped to make this more stable.
            Use verbose option to see what's going on.
            When failed, the file size tends to be 11KB. Verbose will show [-1].
            Restart the kernel to delete the corrupeted .smrx file.

            Stylise the Spike2 display using notebooks|noncanonical|display_style.s2s

        smrx_filename: str = None

        verbose :bool = False

        print_to_text: bool = True
            print_lines will be converted to text (and TextMark chaanel in Spike2)

        """

        # see  \Users\phar0528\Anaconda3\envs\trialexp\Lib\site-packages\sonpy\MakeFile.py
        #NOTE cannot put file path in the pydoc block

        raw_symbols  = SymbolValidator().values
        symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]
        # 40 symbols

        fig = go.Figure()
        # if keys is None:
        #     keys = self.times.keys()
        # else:
        #     for k in keys: 
        #        assert k in self.times.keys(), f"{k} is not found in self.time.keys()"

        if export_smrx:
            from sonpy import lib as sp
            import time
            if smrx_filename is None:
                raise Exception('smrx_filename is required')
            #TODO assert .smlx

            mtc = re.search('\.smrx$', smrx_filename)
            if mtc is None:
                raise Exception('smrx_filename has to end with .smrx')

            MyFile = sp.SonFile(smrx_filename)
            CurChan = 0
            UsedChans = 0
            Scale = 65535/20
            Offset = 0
            ChanLow = 0
            ChanHigh = 5
            tFrom = 0
            tUpto = sp.MaxTime64()         # The maximum allowed time in a 64-bit SON file
            dTimeBase = 1e-6               # s = microseconds
            x86BufSec = 2.
            EventRate = 1/(dTimeBase*1e3)  # Hz, period is 1000 greater than the timebase
            SubDvd = 1                     # How many ticks between attached items in WaveMarks

            max_time_ms1 = np.max([np.max(self.times[k]) for k in keys if any(self.times[k])]) #TODO when no data 

            list_of_match = [re.match('^\d+', L) for L in self.print_lines if re.match('^\d+', L) is not None]
            max_time_ms2 = np.max([int(m.group(0)) for m in list_of_match])

            max_time_ms = np.max([max_time_ms1, max_time_ms2])
            time_vec_ms = np.arange(0, max_time_ms, 1000/EventRate)
            # time_vec_micros = np.arange(0, max_time_ms*1000, 10**6 * 1/EventRate)

            samples_per_s = EventRate
            interval = 1/samples_per_s

            samples_per_ms = 1/1000 * EventRate
            interval = 1/samples_per_s

            MyFile.SetTimeBase(dTimeBase)  # Set timebase


        def write_event(MyFile, X_ms, title, y_index, EventRate, time_vec_ms):
            (hist, ___) = np.histogram(X_ms, bins=time_vec_ms) # time is 1000 too small

            eventfalldata = np.where(hist)

            MyFile.SetEventChannel(y_index, EventRate)
            MyFile.SetChannelTitle(y_index, title)
            if eventfalldata[0] is not []:
                MyFile.WriteEvents(int(y_index), eventfalldata[0]*1000) #dirty fix but works
                time.sleep(0.05)# might help?

            if verbose:
                print(f'{y_index}, {title}:')
                nMax = 10
                # nMax = int(MyFile.ChannelMaxTime(int(y_index))/MyFile.ChannelDivide(int(y_index))) 
                print(MyFile.ReadEvents(int(y_index), nMax, tFrom, tUpto)) #TODO incompatible function arguments.
                # [-1] when failed

                # ReadEvents(self: sonpy.amd64.sonpy.SonFile, 
                #     chan: int, 
                #     nMax: int, # probably the end of the range to read in the unit of number of channel divide
                #     tFrom: int, 
                #     tUpto: int = 8070450532247928832, 
                #     Filter: sonpy.amd64.sonpy.MarkerFilter = <sonpy.MarkerFilter> in mode 'First', with trace column -1 and items
                #     Layer 1 [
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        def write_marker_for_state(MyFile,X_ms, title, y_index, EventRate, time_vec_ms):
            #TODO nearly there, but file cannot be open

            # remove NaN
            X_notnan_ms = [x for x in X_ms if not np.isnan(x)]

            (hist, ___) = np.histogram(X_notnan_ms, bins=time_vec_ms) # time is 1000 too small

            eventfalldata = np.where(hist)

            nEvents = len(eventfalldata[0])

            MarkData = np.empty(nEvents, dtype=sp.DigMark)
            for i in range(nEvents):
                if (i+1) % 2 == 0:
                    MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 0) #offset
                elif (i+1) % 2 == 1:
                    MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 1) #onset
                else:
                    raise Exception('oh no')
            MyFile.SetMarkerChannel(y_index, EventRate)
            MyFile.SetChannelTitle(y_index, title)
            if eventfalldata[0] is not []:
                MyFile.WriteMarkers(int(y_index), MarkData)
                time.sleep(0.05)# might help?

            if verbose:             
                print(f'{y_index}, {title}:')
                print(MyFile.ReadMarkers(int(y_index), nEvents, tFrom, tUpto)) #TODO failed Tick = -1

        def write_textmark(MyFile, X_ms, title, y_index, txt, EventRate, time_vec_ms):

            (hist, ___) = np.histogram(X_ms, bins=time_vec_ms) # time is 1000 too small

            eventfalldata = np.where(hist)

            nEvents = len(eventfalldata[0])

            MarkData = np.empty(nEvents, dtype=sp.DigMark)

            TMrkData = np.empty(nEvents, dtype=sp.TextMarker)

            for i in range(nEvents):
                if (i+1) % 2 == 0:
                    MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 0) #offset
                elif (i+1) % 2 == 1:
                    MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 1) #onset
                else:
                    raise Exception('oh no')
                TMrkData[i] = sp.TextMarker(txt[i], MarkData[i]) #TODO
                
            MyFile.SetTextMarkChannel(y_index, EventRate, max(len(s) for s in txt)+1)
            MyFile.SetChannelTitle(y_index, title)
            if eventfalldata[0] is not []:
                MyFile.WriteTextMarks(y_index, TMrkData)
                time.sleep(0.05)# might help?

            if verbose:
                print(f'{y_index}, {title}:')
                try:
                    print(MyFile.ReadTextMarks(int(y_index), nEvents, tFrom, tUpto))
                except:
                    print('error in print')

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

            all_on_ms = self.times[state_def_dict['onset']]
            all_off_ms = self.times[state_def_dict['offset']]

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



        # if print_expr is not None: #TODO
        #     if isinstance(print_expr, dict):
        #         print_expr = [print_expr]

        #     for dct in print_expr:
        #         y_index += 1
        #         expr = '^\d+(?= ' + dct['expr'] + ')'
        #         list_of_match = [re.match(expr, L) for L in self.print_lines if re.match(expr, L) is not None]
        #         ts_ms = [int(m.group(0)) for m in list_of_match]
        #         line2 = go.Scatter(
        #             x=[TS_ms/1000 for TS_ms in ts_ms], y=[dct['name']] * len(ts_ms), 
        #             name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
        #         fig.add_trace(line2)

        #         if export_smrx:
        #             write_event(
        #                 MyFile, ts_ms, dct['name'], y_index, EventRate, time_vec_ms)

        # if event_ms is not None:
        #     if isinstance(event_ms, dict):
        #         event_ms = [event_ms]
            
        #     for dct in event_ms:
        #         y_index += 1
        #         line3 = go.Scatter(
        #             x=[t/1000 for t in dct['time_ms']],
        #             y=[dct['name']] * len(dct['time_ms']),
        #             name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
        #         fig.add_trace(line3)

        #         if export_smrx:
        #             write_event(
        #                 MyFile, dct['time_ms'], dct['name'], y_index, EventRate, time_vec_ms)

        # if print_to_text:

        #     EXPR = '^(\d+)\s(.+)'
        #     list_of_match = [re.match(EXPR, L) for L in self.print_lines if re.match(EXPR, L) is not None]
        #     ts_ms = [int(m.group(1)) for m in list_of_match]
        #     txt = [m.group(2) for m in list_of_match]
  
        #     df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

        #     y_index += 1
        #     txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['print_lines']*len(ts_ms), 
        #         text=txt, textposition="top center", 
        #         mode="markers", marker_symbol=symbols[y_index % 40])
        #     fig.add_trace(txtsc)

        #     if export_smrx:
        #         write_textmark( MyFile, ts_ms, 'print lines', y_index, txt, EventRate, time_vec_ms)

        # if state_def is not None:
        #     # Draw states as gapped lines
        #     # Assuming a list of lists of two names

        #     if isinstance(state_def, dict):# single entry
        #         state_def = [state_def]
        #         # state_ms = find_states(state_def)

        #         # line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[state_def['name']] * len(state_ms), 
        #         #     name=state_def['name'], mode='lines', line=dict(width=5))
        #         # fig.add_trace(line1)

        #     if isinstance(state_def, list):# multiple entry
        #         state_ms = None
        #         for i in state_def:
        #             assert isinstance(i, dict)
                    
        #             y_index +=1
        #             state_ms = find_states(i)

        #             line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[i['name']] * len(state_ms), 
        #                 name=i['name'], mode='lines', line=dict(width=5))
        #             fig.add_trace(line1)

        #             if export_smrx:
        #                 write_marker_for_state(MyFile, state_ms, i['name'], y_index, EventRate, time_vec_ms)
        #     else:
        #         state_ms = None
        # else:
        #     state_ms = None
             

        fig.update_xaxes(title='Time (s)')
        fig.update_yaxes(fixedrange=True) # Fix the Y axis

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