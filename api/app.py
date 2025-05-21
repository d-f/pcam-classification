from torch import nn
from tqdm import tqdm
from hyppo.ksample import MMD
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from pathlib import Path
import torch
from flask import Flask, request, jsonify
from torchvision.transforms import Compose
import json
from typing import Dict, Type
from torchvision.transforms import Lambda
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader

def create_app():
    app = Flask(__name__)
    
    app.config.from_mapping(
        DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        MODEL_PATH='../models/model_1.pth.tar',
        UPLOAD_PATH='../data/uploads/',
        CLASS_LIST=["Contains no tumor", "Contains tumor"],
    )

    # Load dataset statistics
    with open('../data/ds_mean_std.csv', mode="r") as opened_json:
        app.config['DS_STATS'] = json.load(opened_json)

    # Initialize model at app startup
    with app.app_context():
        init_model(app)
    

    @app.route('/')
    def home():
        if app.config.get('MODEL_LOADED', False):
            return jsonify({"status": "ready", "message": "Model is loaded"})
        else:
            return jsonify({"status": "not ready", "message": "Model did not load, check server logs for details"})
    

    @app.route('/predict', methods=['POST'])
    def predict():
        img_filename = request.args['filename']
        tensor = prepare_img(
            filepath=Path(app.config['UPLOAD_PATH']).joinpath(img_filename),
            mean=app.config['DS_STATS']['mean'],
            std=app.config['DS_STATS']['std'],
            device=app.config['DEVICE']
        )
        model = app.config['MODEL']
        pred = model(tensor)
        
        generate_gradcam(
            model=model, 
            pred_class=pred.argmax(dim=1).item(),
            save_dir=Path('../data/output_imgs/'),
            model_name=app.config['MODEL_PATH'].rpartition("/")[2],
            input_img=Image.open(Path(app.config['UPLOAD_PATH']).joinpath(img_filename)),
            input_tensor=tensor,
            input_filename=img_filename
        )
        
        return jsonify({'prediction': app.config['CLASS_LIST'][pred.argmax(dim=1).item()]})
    

    @app.route('/monitor_data', methods=['POST'])
    def monitor_data():
        with torch.no_grad():
            model = app.config['MODEL']
            old_classifier = model.classifier

            model.classifier = nn.Identity()

            # list for embedded train data
            train_prod = []
            # list for embedded new input data
            new_prod = []

            train_dl, new_dl = create_dataloaders(
                pcam_root="C:\\personal_ML\\DINOVIT_PCAM\\pcam\\", 
                mean=app.config['DS_STATS']['mean'], 
                std=app.config['DS_STATS']['std'],
                batch_size=1,
                upload_path=app.config['UPLOAD_PATH']
            )

            for ds_tuple in tqdm(train_dl):
                out = model(ds_tuple[0].to(app.config['DEVICE']))
                train_prod.append(out[0].cpu().numpy())

            for ds_tuple in tqdm(new_dl):
                out = model(ds_tuple[0].unsqueeze(0).to(app.config['DEVICE']))
                new_prod.append(out[0].cpu().numpy())

            # calculate max mean discrepency test stat and p value
            stat, p_value = MMD().test(np.array(train_prod), np.array(new_prod))

            model.classifier = old_classifier

            return jsonify({"stat": stat, "p value": p_value, "drift detected": p_value<=0.05})

    return app


def prepare_img(filepath, mean, std, device):
    transforms = Compose([
        torchvision.transforms.ToTensor(),
        Lambda(lambda x: x.unsqueeze(0)),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    loaded_img = Image.open(filepath)
    norm_tensor = transforms(loaded_img).to(device)
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


class NewDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, upload_dir):
        self.transforms = transforms
        self.uploads = [x for x in Path(upload_dir).iterdir()]

    def __len__(self):
        return len(self.uploads)
    
    def __getitem__(self, index):
        img_path = self.uploads[index]
        loaded_img = Image.open(img_path)
        norm_tensor = self.transforms(loaded_img)
        return norm_tensor


def create_dataloaders(pcam_root: Path, mean: tuple, std: tuple, batch_size: int, upload_path: Path) -> tuple[Type[DataLoader]]:
    """
    creates PyTorch DataLoaders

    ds_root:    Path to the PCAM dataset files
    mean:       mean for each channel across the train dataset
    std:        standard deviation for each channel across the train dataset 
    batch_size: number of images in each batch 

    returns tuple of DataLoaders for each dataset partition
    """
    eval_transforms = Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(size=32),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    train_ds = torchvision.datasets.PCAM(split="train", root=pcam_root, download=True, transform=eval_transforms)
    new_ds = NewDataset(transforms=eval_transforms, upload_dir=upload_path)

    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=False)
    new_dl = DataLoader(dataset=new_ds, batch_size=batch_size, shuffle=False)

    return train_dl, new_dl


def init_model(app):
    """Initialize the model and store it in app config"""
    try:
        model = torch.load(app.config['MODEL_PATH'], weights_only=False)
        model.to(app.config['DEVICE'])
        app.config['MODEL'] = model
        app.config['MODEL_LOADED'] = True
    except Exception as e:
        app.config['MODEL_LOADED'] = False
        app.logger.error(f"Model failed to load: {e}")


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
