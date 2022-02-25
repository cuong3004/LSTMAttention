import pandas as pd
import os
from sklearn.model_selection import train_test_split
from custom_data import *
import torch
from model import AudioLSTM, AudioLSTMAttention
import pytorch_lightning as pl
from litmodule import LitClassification
from callbacks import input_monitor_train, input_monitor_valid, checkpoint_callback, early_stop_callback
from utils import *

pl.seed_everything(1234)

FILE_PATH = "../ConvLstmMultipleFeature/preprocessing_data/"
df = pd.read_csv(os.path.join(FILE_PATH, "UrbanSound8K.csv"))

files = df["slice_file_name"].values.tolist()
folder_fold = df["fold"].values
label = df["classID"].values.tolist()
path = [
    os.path.join(FILE_PATH + "fold" + str(folder) + "/" + file) for folder, file in zip(folder_fold, files)
]

X_train, X_test, y_train, y_test = train_test_split(path, label, random_state=42, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=43, test_size=0.15)

transform_audio = A.Compose([

         NoiseInjection(p=0.4),
         ShiftingTime(p=0.4),
        #  PitchShift(p=0.2),
        #  MelSpectrogram(parameters=melspectrogram_parameters, always_apply=True),
    #      SpectToImage(always_apply=True)
    ])
# transform_audio = None

train_dataset = AudioDataset(
    file_path=X_train,
    class_id=y_train,
    transform=transform_audio
)

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn
)

valid_dataset = AudioDataset(
    file_path=X_valid,
    class_id=y_valid,
    # transform=transform_audio
)

test_dataset = AudioDataset(
    file_path=X_test,
    class_id=y_test,
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
model_audio = AudioLSTMAttention(n_feature=128, out_feature=OUT_FEATURE)
# model_audio = AudioLSTM(n_feature=40, out_feature=OUT_FEATURE)

model = LitClassification(model_audio)
# model = LitClassification.load_from_checkpoint("/content/drive/MyDrive/LSTMAtention/checkpoint/model-v18.ckpt")

callbacks = [input_monitor_train, input_monitor_valid, checkpoint_callback, 
# early_stop_callback
]

trainer = pl.Trainer(
                gpus=1 if torch.cuda.is_available() else 0 , 
                callbacks = callbacks, 
                max_epochs=EPOCH,
                )
trainer.fit(model, train_loader, valid_loader)


# trainer.test(model, test_loader)