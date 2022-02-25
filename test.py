import torch 
from litmodule import LitClassification
from custom_data import *
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import pytorch_lightning as pl
import pandas as pd
import os
from sklearn.model_selection import train_test_split
pl.seed_everything(1234)
batch_size = 32
model_file_checkpoint = "checkpoint/model-v13.ckpt"
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

model = LitClassification.load_from_checkpoint(model_file_checkpoint)
model.eval()

test_dataset = AudioDataset(
    file_path=X_test,
    class_id=y_test,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

label_list = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


all_preds = []
for batch in tqdm(test_loader):
    x, y= batch

    with torch.no_grad():
        pred = model(x)
    # print(pred)
    # assert False
    pred = pred.argmax(dim=1)
    all_preds.append(pred)

all_preds = torch.cat(all_preds,dim=0)
print(all_preds.shape)
print(len(test_dataset.class_id))
cm = confusion_matrix(y_test, all_preds)

# plt.figure(figsize=(12.5,10))
plot_confusion_matrix(cm, label_list, normalize=True)
plt.savefig("test_pt.png")
plt.clf()
