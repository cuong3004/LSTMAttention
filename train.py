import pandas as pd
import os
from sklearn.model_selection import train_test_split
from custom_data import *
import torch
from model import AudioLSTM
import pytorch_lightning as pl
from litmodule import LitClassification
from callbacks import input_monitor_train, input_monitor_valid, checkpoint_callback, early_stop_callback

FILE_PATH = "preprocessing_data/"
df = pd.read_csv(os.path.join(FILE_PATH, "UrbanSound8K.csv"))

files = df["slice_file_name"].values.tolist()
folder_fold = df["fold"].values
label = df["classID"].values.tolist()
path = [
    os.path.join(FILE_PATH + "fold" + str(folder) + "/" + file) for folder, file in zip(folder_fold, files)
]

X_train, X_test, y_train, y_test = train_test_split(path, label, random_state=42, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=43, test_size=0.15)

train_dataset = AudioDataset(
    file_path=X_train,
    class_id=y_train
)

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn
)

valid_dataset = AudioDataset(
    file_path=X_valid,
    class_id=y_valid
)

test_dataset = AudioDataset(
    file_path=X_test,
    class_id=y_test
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn
)

x, y = next(iter(train_loader))

EPOCH = 50
OUT_FEATURE = 10
model_audio = AudioLSTM(n_feature=168, out_feature=OUT_FEATURE)

model = LitClassification(model_audio)

callbacks = [input_monitor_train, input_monitor_valid, checkpoint_callback, 
# early_stop_callback
]

trainer = pl.Trainer(
                gpus=1 if torch.cuda.is_available() else 0 , 
                callbacks = callbacks, 
                max_epochs=EPOCH,
                )
trainer.fit(model, train_loader, valid_loader)
















# from custom_data import CustomData, CustomDataMel
# import torch 
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from utils import *
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import librosa.display
# from model import CnnLstm, CNNModel, LstmModel
# from torch.utils.data import DataLoader, random_split
# import pytorch_lightning as pl
# from litmodule import LitClassification
# from torchvision.models import mobilenet_v2
# import torch.nn as nn
# import os
# import pandas as pd
# from glob import glob
# from callbacks import input_monitor_train, input_monitor_valid, checkpoint_callback, early_stop_callback


# # In[2]:


# batch_size = 64
# num_classes = 10
# num_workers = 2

# npobj = np.load("normalize.npz")
# mean, std = npobj['mean'], npobj['std']
# print("mean, std : ", mean, std)

# pl.seed_everything(1234)


# # In[3]:


# melspectrogram_parameters = {
#         "n_mels": 128,
#         "fmin": 40,
#         # "fmax": 32000
#     }


# # In[4]:


# transform_audio = A.Compose([

#          NoiseInjection(p=0.5),
#          ShiftingTime(p=0.5),
#          PitchShift(p=0.5),
#          MelSpectrogram(parameters=melspectrogram_parameters, always_apply=True),
#     #      SpectToImage(always_apply=True)
#     ])

# transform_image = A.Compose([
#     A.Normalize(mean=mean, std=std),
#     ToTensorV2()
# ])


# # In[5]:


# dataset = CustomDataMel(
#                 csv_file="UrbanSound8K.csv",
#                 data_dir="preprocessing_data",
#                 transform_audio=transform_audio,
#                 transform_image = transform_image
#                 )


# # In[6]:


# train_len = int(len(dataset)*0.8)
# valid_len = len(dataset)-train_len
# data_train, data_valid = random_split(dataset,[train_len, valid_len])


# train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
# valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# # In[7]:


# cnn_model = mobilenet_v2()
# cnn_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
# cnn_model.classifier[1] = torch.nn.Linear(1280,num_classes)
# # model_lstm = LstmModel(n_feature=172, num_classes=10, n_hidden=256, n_layers=2)


# # In[8]:


# model = LitClassification(cnn_model)

# callbacks = [input_monitor_train, input_monitor_valid, checkpoint_callback, early_stop_callback]


# # In[9]:

# gpus = 1 if torch.cuda.is_available() else 0
# trainer = pl.Trainer(
#                 gpus=gpus, 
#                 callbacks = callbacks, 
#                 # max_epochs=1,
#                 )


# # In[ ]:


# trainer.fit(model, train_loader, valid_loader)