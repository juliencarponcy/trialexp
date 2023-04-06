import os
from dotenv import load_dotenv
load_dotenv()

######################
## Analysis settings

trial_window = [-2000, 4000]
timelim = [1000, 4000] # in ms

#################
## Debug settings
root_folder = os.environ.get('SESSION_ROOT_FOLDER')
if root_folder=='':
    print(f'You should set the environmental variable SESSION_ROOT_FOLDER first')
debug_folder = root_folder+ r'/pavlovian_nobar_nodelay/kms053-2022-07-27-183042'