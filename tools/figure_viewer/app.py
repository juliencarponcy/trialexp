from shiny import App, reactive, ui, render
from loguru import logger
from trialexp.process.folder_org.utils import build_session_info_cohort, load_and_concat_dataset, filter_sessions
import os
from gallery import gallery_server, gallery_ui
import pandas as pd

# logger.debug('Reading sessions data from ettin...')
root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
print('Server starting')


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select("cohort", "Select cohort", choices=[]),
        ui.input_checkbox_group('animal_id', 'Animals', choices=[]),
        ui.input_checkbox_group('task_name', 'Task', choices=[]),
        ui.input_radio_buttons('modality', 'Modality', choices =['Photometry', 'Neuropixels', 'Behaviour']),
        ui.input_radio_buttons('figure_list', 'Figures', choices=[0,1]),
        ui.input_action_button('plot_btn', 'Plot figures'),
        width=600        
    ),
    ui.output_ui('gallery')
)


def server(input, output, session):

    # Variables defined inside here user session specific
    df_img2plot = reactive.value() # used to store the dataframe of images to plot

    
    df_session_info = build_session_info_cohort(root_path)
    df = df_session_info
    cohorts = df.cohort.unique().tolist()
    ui.update_select('cohort', choices=cohorts, selected=cohorts[-1])
    
    
    @reactive.calc
    def session_info():
        idx = df_session_info.cohort==input.cohort()
        if len(input.animal_id())>0:
            idx = idx & df_session_info.animal_id.isin(input.animal_id())
        if len(input.task_name())>0:
            idx = idx & df_session_info.task_name.isin(input.task_name())
        return df_session_info[idx].sort_values('expt_datetime')
    
    @reactive.effect
    @reactive.event(input.cohort)
    def animal_id():
        df = session_info()
        ui.update_checkbox_group('animal_id', choices=df.animal_id.unique().tolist())
        
        
    @reactive.effect
    @reactive.event(input.animal_id)
    def update_task():
        df = session_info()
        ui.update_checkbox_group('task_name', choices=df.task_name.unique().tolist())
        
    @reactive.effect
    @reactive.event(input.task_name, input.modality)
    def update_figure_list():
        df = session_info()
        figure_names=[]
        
        if input.modality() == 'Photometry':
            for p in df.path:
                figure_names += list((p/'processed'/'figures'/'photometry').glob('*.png'))
                figure_names += list((p/'processed'/'figures'/'timewarp').glob('*.png'))
        elif input.modality() == 'Neuropixels':
            for p in df.path:
                figure_names += list((p/'processed'/'figures'/'ephys').rglob('*.png'))
        else:
            for p in df.path:
                figure_names += list((p/'processed'/'figures').glob('*.png'))

        
        figure_names = [f.name for f in figure_names]
        figure_names = sorted(list(set(figure_names)))
        if len(figure_names) == 0: # hack for handling the case when no image can be found, radio button forces us to at least provide on option
            figure_names =['None']
        ui.update_radio_buttons('figure_list', choices=figure_names, selected=None)
        
        
    @reactive.effect
    @reactive.event(input.plot_btn)
    def update_figures_list():
        df = session_info()
        fig_name = input.figure_list()
        img_info = []
            
        # search for images in each of the session
        for _, row in df.iterrows():
            # search for the files
            file_list = list((row.path/'processed'/'figures').rglob(fig_name))
            if len(file_list) > 0:
                src = file_list[0]
                img_info.append({'animal_id': row.animal_id,
                                'task_name': row.task_name,
                                'expt_datetime': row.expt_datetime,
                                'session_id': row.session_id,
                                'session_root': row.path,
                                'src': src})
                
        df_img2plot.set(pd.DataFrame(img_info)) # need to assign a new object so that shiny know it has been changed


    @render.ui
    def gallery():
        # Determine how many galleries we need, one for each animals
        df = df_img2plot()
        animal_ids = df.animal_id.unique()
        return [gallery_ui(id, id) for id in animal_ids]
    
    #also need to provide the server code for the module here
    # these function will only be called when the corresponsding UI are shown
    # so we can just pre-define it for all possible animal ids
    [gallery_server(id, id, df_img2plot) for id in df_session_info.animal_id.unique()]

app = App(app_ui, server)