import torch 
from litmodule import LitClassification
# import tensorflow as tf 
# from pytorch2keras.converter import pytorch_to_keras
import os
import numpy as np
from torch.autograd import Variable


model_file_checkpoint = "checkpoint/model-v13.ckpt"


model = LitClassification.load_from_checkpoint(model_file_checkpoint)
model.eval()

#  Export to onnx
# input_sample = torch.randn((1, 442, 128))
# model.to_onnx("model.onnx", input_sample, export_params=True)
# assert os.path.isfile("model.onnx")

# Export to torchscript
torch.jit.save(model.to_torchscript(), "model.pt")
assert os.path.isfile("model.pt")