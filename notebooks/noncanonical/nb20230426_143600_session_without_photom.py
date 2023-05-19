#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os

nb_name = "nb20230426_143600_session_without_photom.ipynb" #TODO change this

basename, ext = os.path.splitext(nb_name)
input_path = os.path.join(os.getcwd(), nb_name)

get_ipython().system('jupyter nbconvert "{input_path}" --to="python" --output="{basename}"')


# In[7]:


import os
import glob


def find_subfolders_without_ppd_files(root_folder):
    subfolders_without_ppd = []

    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item, 'pyphotometry')

        # Check if the item is a directory
        if os.path.isdir(item_path):
            ppd_files = glob.glob(os.path.join(item_path, '*.ppd'))

            if not ppd_files:
                subfolders_without_ppd.append(item_path)

    return subfolders_without_ppd


if __name__ == "__main__":
    root_folder = r"Z:\Julien\Data\head-fixed\by_sessions\reaching_go_spout_bar_nov22"
    subfolders_without_ppd = find_subfolders_without_ppd_files(root_folder)

    print("Subfolders without .ppd files:")
    for folder in subfolders_without_ppd:
        print(folder)

