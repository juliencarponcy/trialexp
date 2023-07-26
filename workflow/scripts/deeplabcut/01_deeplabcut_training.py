#%%
import deeplabcut
#%%
path_config_file = '/home/MRC.OX.AC.UK/ndcn1330/ettin/Teris/ASAP/deeplabcut/side_2_hands_newobj-julien-2022-08-26/config.yaml'
deeplabcut.train_network(path_config_file, gputouse=0, saveiters=300, displayiters=100, maxiters=200000)