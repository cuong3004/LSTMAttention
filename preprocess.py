#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import librosa 
from tqdm import tqdm
import os
import soundfile as sf


# In[2]:


df = pd.read_csv("urbansound8k/UrbanSound8K.csv")
df['fold'] = df['fold'].astype(str)
df["path"] = "urbansound8k/fold"+df['fold']+"/"+df['slice_file_name']


# In[3]:


data_dir = 'preprocessing_data'
dir_list = [f"fold{i+1}" for i in range(10)]
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    for dir_name in dir_list:
        os.makedirs(f"{data_dir}/{dir_name}")


# In[35]:


def get_new_name(name_old):
    name_old = name_old.split("/")
    name_old[0] = data_dir
    name_new = "/".join(name_old)
    return name_new

def open_and_save_file(name_input, name_output):
    wave, sr = librosa.load(name_input)
    sf.write(name_output, wave, sr, 'PCM_24')


# In[36]:


for i in tqdm(range(len(df))):
    path_file = df.iloc[i,-1]
    new_name = get_new_name(path_file)
    open_and_save_file(path_file, new_name)


# In[ ]:


# wave, sr = librosa.load("stereo_file.wav")


# In[20]:


# import matplotlib.pyplot as plt
# plt.plot(wave)


# In[ ]:


