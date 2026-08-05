from shiny import module, ui, render, reactive
from pathlib import Path
import base64
import uuid
import pandas as pd
from trialexp.process.folder_org.utils import get_session_config_info, get_photom_config_info
import xarray as xr

@module.ui
def gallery_ui(animal_id):
    return ui.card(
        ui.card_header(animal_id),
        ui.output_ui('image_cards'),
    )
    
    
    
@module.server
def gallery_server(input, output, session, animal_id, df_img2plot):
        
    @output
    @render.ui
    def image_cards():
        card_list = []
        df_animal = df_img2plot()
        df_animal = df_animal[df_animal.animal_id==animal_id]
        for _, row in df_animal.iterrows():
            p = Path(row.src)
            if p.exists():
                id = str(uuid.uuid4()).replace('-','')
                card_list.append(figure_ui(id))
                figure_server(id, figure_info=row)
        
        return ui.layout_column_wrap(*card_list, width=1/4)

        
    
@module.ui
def figure_ui():
    return ui.output_ui('image_card')

@module.server
def figure_server(input, output, session, figure_info):
    try:
        df_pycontrol = pd.read_pickle(Path(figure_info.session_root)/'processed'/'df_pycontrol.pkl')
        df_conditions = pd.read_pickle(Path(figure_info.session_root)/'processed'/'df_conditions.pkl')
        df_info, df_start_params, df_user_param, df_trial_outcome_counts = get_session_config_info(df_pycontrol, df_conditions)

    except:
        df_pycontrol = pd.DataFrame()
        df_conditions = pd.DataFrame()
        df_info = pd.DataFrame()
        df_start_params = pd.DataFrame()
        df_user_param = pd.DataFrame()
        df_trial_outcome_counts = pd.DataFrame()
        
    #Each figure_server is a separate module, so function inside can access its own figure_info
    # basically it generate a different show_figure function and attached it to the card
    @output
    @render.ui
    def image_card():
        p = Path(figure_info.src)
        with open(p, 'rb') as f:
            mime_type = p.suffix
            b64_str = base64.b64encode(f.read()).decode("utf-8")
            id = figure_info.session_id.replace('-','_')
            return  ui.card(
                            ui.card_header(ui.input_action_link('show_figure',  label=figure_info.task_name + '/' + figure_info.session_id)),
                            ui.img(src=f"data:{mime_type};base64,{b64_str}"), # create html tag directly,
                            ui.input_action_button('show_session_info',  label='Show session info')
                        )
            
    
    @reactive.Effect
    @reactive.event(input.show_figure)
    def show_figure():
        p = Path(figure_info.src)
        with open(p, 'rb') as f:
            mime_type = p.suffix
            b64_str = base64.b64encode(f.read()).decode("utf-8")
            id = figure_info.session_id.replace('-','_')
            card = ui.card(
                            ui.card_header(figure_info.session_id),
                            ui.img(src=f"data:{mime_type};base64,{b64_str}") # create html tag directly
                        )
            
            m = ui.modal(card, size='l', easy_close=True, fade=False)
            ui.modal_show(m)
            
            
    @output
    @render.data_frame
    def df_info_show():
        return render.DataTable(df_info, height=None)
    
    @output
    @render.data_frame
    def df_start_params_show():
        return render.DataTable(df_start_params, height=None)       
    
    @output
    @render.data_frame
    def df_user_param_show():
        return render.DataTable(df_user_param, height=None)

    @output
    @render.data_frame
    def df_trial_outcome_counts_show():
        return render.DataTable(df_trial_outcome_counts, height=None)
    
    @output
    @render.data_frame
    def df_photom_info_show():
        try:
            xr_photometry = xr.open_dataset(Path(figure_info.session_root)/'processed'/'xr_photometry.nc')
            df_photom_info = get_photom_config_info(xr_photometry)
            xr_photometry.close()
        except:
            df_photom_info = pd.DataFrame()
        
        return render.DataTable(df_photom_info, height=None)
    
        
    @reactive.Effect
    @reactive.event(input.show_session_info)
    def show_session_info():
        
        info =  ui.layout_column_wrap(
                    ui.card(
                        ui.card_header('Session Info'),
                        ui.output_data_frame('df_info_show')
                        
                    ),
                    ui.card(
                        ui.card_header('Start Parameters'),
                        ui.output_data_frame('df_start_params_show')
                    ),
                    ui.card(
                        ui.card_header('User Parameters'),
                        ui.output_data_frame('df_user_param_show')
                    ),
                    ui.card(
                        ui.card_header('Trial Outcome Counts'),
                        ui.output_data_frame('df_trial_outcome_counts_show')
                    ),
                      ui.card(
                        ui.card_header('Photometry signals'),
                        ui.output_data_frame('df_photom_info_show')
                    ),
                    width=1/3,
                    height= 800,
                )
                
        m = ui.modal(info, size='xl', easy_close=True, fade=False)
        ui.modal_show(m)


        
#%%

