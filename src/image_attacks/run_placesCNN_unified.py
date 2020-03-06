import torch
from torchvision import transforms as trn
import os
import numpy as np

def hook_feature(module, input, output):

    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


def returnTF():
    """
    load the image transformer
    :return:
    """

    tf = trn.Compose([
        trn.Scale((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():

    # this model has a last conv feature map as 14x14

    model_file = 'whole_wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
    useGPU = 0
    if useGPU == 1:
        model = torch.load(model_file)
    else:
        model = torch.load(model_file, map_location=lambda storage, loc: storage) # allow cpu

    model.eval()

    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet

    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)

    return model

features_blobs = []
