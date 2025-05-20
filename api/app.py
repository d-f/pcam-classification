import numpy as np
import torchvision
from PIL import Image
from pathlib import Path
import torch
from flask import Flask, request, jsonify
from torchvision.transforms import Compose
import json
from typing import Dict
from torchvision.transforms import Lambda

app = Flask(__name__)


def read_json(json_path: Path) -> Dict:
    """
    reads JSON file and returns dictionary of file contents
    """
    with open(json_path, mode="r") as opened_json:
        return json.load(opened_json)


DEVICE = torch.device('cuda')
MODEL_PATH = '../models/model_1.pth.tar'
UPLOAD_PATH = '../data/uploads/'
DS_STATS = read_json('../data/ds_mean_std.csv')
CLASS_LIST = ["Contains no tumor", "Contains tumor"]
    
try:
    model = torch.load(MODEL_PATH, weights_only=False)
    model.to(DEVICE)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(e)


@app.route('/')
def home():
    if MODEL_LOADED:
        return jsonify({"status": "ready", "message": "Model is loaded"})
    else:
        return jsonify({"status": "not ready", "message": "Model did not load, check server logs for details"})


@app.route('/predict', methods=['POST'])
def predict():
    img_filename = request.args['filename']
    tensor = prepare_img(
        filepath=Path(UPLOAD_PATH).joinpath(img_filename),
        mean=DS_STATS['mean'],
        std=DS_STATS['std']
        )
    pred = model(tensor)
    return jsonify({'prediction': CLASS_LIST[pred.argmax(dim=1).item()]})


def prepare_img(filepath, mean, std):
    transforms = Compose([
        torchvision.transforms.ToTensor(),
        Lambda(lambda x: x.unsqueeze(0)),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    loaded_img = Image.open(filepath)
    norm_tensor = transforms(loaded_img).to(DEVICE)
    return norm_tensor
    
    

