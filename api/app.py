import matplotlib.pyplot as plt
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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

    generate_gradcam(
        model=model, 
        pred_class=pred.argmax(dim=1).item(),
        save_dir=Path('../data/output_imgs/'),
        model_name=MODEL_PATH.rpartition("/")[2],
        input_img=Image.open(Path(UPLOAD_PATH).joinpath(img_filename)),
        input_tensor=tensor,
        input_filename=img_filename
        )

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


def generate_gradcam(
        model, 
        pred_class, 
        save_dir, 
        model_name, 
        input_img, 
        input_tensor,
        input_filename
        ):
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    target_layers = [model.features[-4][0]]
    targets = [ClassifierOutputTarget(pred_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor, targets=targets)

    plt.imshow(input_img)
    plt.imshow(grayscale_cam[0], alpha=0.5, cmap="turbo")
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(
        save_dir.joinpath(f"{model_name}-{input_filename}-{pred_class}-gradcam.jpg"),
        bbox_inches='tight',
        pad_inches=0
        )
    plt.close()
