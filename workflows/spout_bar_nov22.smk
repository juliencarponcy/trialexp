rule process_pycontrol:
    input:
        pycontrol_file = '{base_path}/{session}.txt'
    output:
        event_dataframe = '{base_path}/processed/{session}.pkl'