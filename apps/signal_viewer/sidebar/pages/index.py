"""The home page of the app."""

from sidebar import styles
from sidebar.templates import template

import reflex as rx

from trialexp.process.folder_org.utils import build_session_info_cohort
from ..components.sidebar import SessionSelectState
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import xarray as xr 
from pathlib import Path
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

def plot_session(signal2plot):
    
    
    root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
    df_session_info = build_session_info_cohort(root_path)
    df_session_info = df_session_info.query("animal_id=='TT013'")
    
    xr_photometry = xr.open_dataset(Path(df_session_info.iloc[0].path)/'processed'/'xr_photometry.nc', engine='h5netcdf')
    df_pycontrol = pd.read_pickle(Path(df_session_info.iloc[0].path)/'processed'/'df_pycontrol.pkl')
    

    def find_states(state_def_dict: dict):
        """
        state_def: dict, list, or None = None
        must be None (default)
        or dictionary of 
            'name' : str 
                Channel name
            'onset' : str | list of str 
                key for onset 
            'offset' : str | list of str 
                key for offset
        or list of such dictionaries

        eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
        eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}
        eg. {'name':'trial', 'onset':'CS_Go', 'offset': ['refrac_period', 'break_after_abortion']}

        For each onset, find the first offset event before the next onset 
        You can use multiple definitions with OR operation, eg. 'offset' determined by 'abort' or 'success', whichever comes first            
        """
        if state_def_dict is None:
            return None

        if isinstance(state_def_dict['onset'], str):
            all_on_ms = self.times[state_def_dict['onset']]
        elif isinstance(state_def_dict['onset'], list):
            # OR operation
            all_on_ms = []
            for li in state_def_dict['onset']:
                assert isinstance(li, str), 'onset must be str or list of str'
                all_on_ms.extend(self.times[li])
            all_on_ms = sorted(all_on_ms)
            
        else:
            raise Exception("onset is in a wrong type") 

        if isinstance(state_def_dict['offset'], str):
            all_off_ms = self.times[state_def_dict['offset']]
        elif isinstance(state_def_dict['offset'], list):
            # OR operation
            all_off_ms = []
            for li in state_def_dict['offset']:
                assert isinstance(li, str), 'offset must be str or list of str'                    
                all_off_ms.extend(self.times[li])
            all_off_ms = sorted(all_off_ms)
        else:
            raise Exception("offset is in a wrong type") 

        onsets_ms = [np.nan] * len(all_on_ms)
        offsets_ms = [np.nan] * len(all_on_ms)

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
                        [np.nan] * len(onsets_ms)))
        # [onset1, offset1, NaN, onset2, offset2, NaN, ....]
        state_ms = [item for sublist in state_ms for item in sublist]
        return state_ms
    
    signal2plot = [k for k in xr_photometry.data_vars.keys() if k.startswith('analog')]

    # Doesn't work in Reflex, need to figure out how to make it work
    fig = FigureResampler(make_subplots(rows=len(signal2plot), cols=1, shared_xaxes=True,
                        vertical_spacing=0.02))  # Reduced vertical spacing

    for idx, k in enumerate(signal2plot, start=1):
        line1 = go.Scattergl(x=xr_photometry['time']/1000, 
                          y=xr_photometry[k], 
                          name=f"{k} signal", 
                          mode='lines')
        fig.add_trace(line1, row=idx, col=1)

    fig.update_layout(
        height=200*len(signal2plot),
        plot_bgcolor='white'  # White background
    )
    
    # Remove grid from all x and y axes
    fig.update_xaxes(showgrid=False)
    
    return fig

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

    # if print_to_text:

    #     EXPR = '^(\d+)\s(.+)' #NOTE . doesn't capture \n and re.DOTALL is required below
    #     list_of_match = [re.match(EXPR, L, re.DOTALL) for L in self.print_lines if re.match(EXPR, L) is not None]
    #     ts_ms = [int(m.group(1)) for m in list_of_match]
    #     txt = [m.group(2) for m in list_of_match]

    #     # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

    #     y_index += 1
    #     txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['print_lines']*len(ts_ms), 
    #         text=txt, textposition="top center", 
    #         mode="markers", marker_symbol=symbols[y_index % 40])
    #     fig.add_trace(txtsc)

    # if vchange_to_text:
    #     EXPR = '^([1-9]\d*)\s(.+)' #NOTE Need to ignore the defaults (V 0 ****)
    #     list_of_match = [re.match(EXPR, L) for L in self.v_lines if re.match(EXPR, L) is not None]
    #     ts_ms = [int(m.group(1)) for m in list_of_match]
    #     txt = [m.group(2) for m in list_of_match]

    #     # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

    #     y_index += 1
    #     txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['V changes']*len(ts_ms), 
    #         text=txt, textposition="top center", 
    #         mode="markers", marker_symbol=symbols[y_index % 40])
    #     fig.add_trace(txtsc)



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

    #     else:
    #         state_ms = None
    # else:
    #     state_ms = None
        


    # fig.update_xaxes(title='Time (s)')
    # fig.update_yaxes(fixedrange=True) # Fix the Y axis

    # fig.update_layout(
        
    #     title =dict(
    #         text = f"{self.task_name}, {self.subject_ID} #{self.number}, on {self.datetime_string} via {self.setup_ID}"
    #     )
    # )

    # fig.show()


class GraphState(rx.State):
    """The state for the graph component."""

    session_id: str = ""


    fig = plot_session(['analog_1','analog_2'])
    
    @rx.event
    def refresh_graph(self):
        cohort = SessionSelectState.cohort
        session_id = SessionSelectState.session_id
        animal_id = SessionSelectState.animal_id
        
        session2plot = SessionSelectState.df_session_info.query(f"cohort=='{cohort}' & animal_id=='{animal_id}' & session_id=='{session_id}'")
        if len(session2plot)>0:
            session2plot = session2plot.iloc[0]
            
        
            

def graph() -> rx.Component:
    return rx.plotly(data=GraphState.fig, width="80%", height="0vh")


@template(route="/", title="Home")
def index() -> rx.Component:
    """The home page.

    Returns:
        The UI for the home page.
    """
    return graph()
