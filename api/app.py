import csv
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
from typing import Type
from torchvision.transforms import Lambda
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score
from train_model import model_summary, define_optim, define_loss, evaluate, train


def create_app():
    app = Flask(__name__)
    
    # set config
    app.config.from_mapping(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        model_path='../models/model_1.pth.tar',
        upload_path='../data/uploads/',
        class_list=["Contains no tumor", "Contains tumor"],
    )

    # load dataset statistics
    with open('../data/ds_mean_std.csv', mode="r") as opened_json:
        app.config['ds_stats'] = json.load(opened_json)

    # initialize model at app startup
    with app.app_context():
        init_model(app)

    @app.route('/')
    def home():
        if app.config.get('model_loaded', False):
            return jsonify({"status": "ready", "message": "model is loaded"})
        else:
            return jsonify({"status": "not ready", "message": "model did not load, check server logs for details"})
    

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        predicts classification of an image
        request requires filename and gradcam_save_dir arguments
        """
        img_filename = request.args['filename']
        # convert image to tensor
        tensor = prepare_img(
            filepath=Path(app.config['upload_path']).joinpath(img_filename),
            mean=app.config['ds_stats']['mean'],
            std=app.config['ds_stats']['std'],
            device=app.config['device']
        )

        model = app.config['model']
        pred = model(tensor)
        
        generate_gradcam(
            model=model, 
            pred_class=pred.argmax(dim=1).item(),
            save_dir=Path(request.args['gradcam_save_dir']),
            model_name=app.config['model_path'].rpartition("/")[2],
            input_img=Image.open(Path(app.config['upload_path']).joinpath(img_filename)),
            input_tensor=tensor,
            input_filename=img_filename
        )
        
        return jsonify({'prediction': app.config['class_list'][pred.argmax(dim=1).item()]})
    

    @app.route('/monitor_data', methods=['POST'])
    def monitor_data():
        """
        monitors data drift with maximum mean discrepancy
        request requires save_fp, pcam_dir, batch_size arguments
        """
        with torch.no_grad():
            model = app.config['model']

            # keep track of old classifier
            old_classifier = model.classifier

            # replace classifier with Identity to 
            # embed the input rather than classify
            model.classifier = nn.Identity()

            # list for embedded train data
            train_prod = []
            # list for embedded new input data
            new_prod = []

            # drift is detected between original training set
            # and the new inputs
            train_dl, new_dl = create_dataloaders(
                pcam_root=request.args['pcam_dir'], 
                mean=app.config['ds_stats']['mean'], 
                std=app.config['ds_stats']['std'],
                batch_size=int(request.args['batch_size']),
                upload_path=app.config['upload_path'],
                pcam_split="train"
            )

            # embed pcam dataset
            for ds_tuple in tqdm(train_dl):
                out = model(ds_tuple[0].to(app.config['device']))
                for pred in out:
                    train_prod.append(pred.cpu().numpy())

            # embed new inputs
            for ds_tuple in tqdm(new_dl):
                out = model(ds_tuple[0].unsqueeze(0).to(app.config['device']))
                for pred in out:
                    new_prod.append(pred.cpu().numpy())

            # calculate max mean discrepancy test stat and p value
            stat, p_value = MMD().test(np.array(train_prod), np.array(new_prod))

            # return classifier to original state
            model.classifier = old_classifier

            result_dict = {"stat": stat, "p value": p_value}

            # save results
            save_json(dict=result_dict, fp=request.args['save_fp'])

            return jsonify(result_dict)
        
    @app.route('/eval', methods=['POST'])
    def eval():
        """
        evaluates model on a combination of the original test set,
        and new inputs that have been labeled
        request requires pcam_dir, batch_size, label_fp, save_fp
        """
        model = app.config['model']
        model.eval()
        loss = torch.nn.CrossEntropyLoss()
        # create original pcam test set and dataloader from new inputs
        pcam_loader, new_loader = create_dataloaders(
            pcam_root=request.args['pcam_dir'],
            mean=app.config['ds_stats']['mean'], 
            std=app.config['ds_stats']['std'],
            batch_size=int(request.args['batch_size']),
            upload_path=app.config['upload_path'],
            pcam_split="test",
            label_file=request.args['label_fp']           
        )
        with torch.no_grad():
            pcam_accuracy = 0
            new_accuracy = 0
            pcam_loss_value = 0
            new_loss_value = 0
            pcam_roc_targets = []
            pcam_roc_scores = []
            new_roc_targets = []
            new_roc_scores = []
            pcam_correct = 0
            pcam_samples = 0
            new_correct = 0
            new_samples = 0
            for data, targets in tqdm(pcam_loader):
                data = data.to(app.config['device'])
                targets = targets.to(app.config['device'])
                scores = model(data)
                batch_train_loss = loss(scores, targets)
                pcam_loss_value += batch_train_loss.item()
                
                for target in targets.cpu().numpy():
                    pcam_roc_targets.append(target)
                soft_scores = torch.nn.functional.softmax(scores, dim=1)
                for score in soft_scores:
                    pcam_roc_scores.append(score[1].detach().cpu().numpy())

                num_correct = (scores.argmax(dim=1)==targets).sum().item()
                pcam_correct += num_correct
                pcam_samples += len(targets)

            for data, targets in tqdm(new_loader):
                data = data.to(app.config['device'])
                targets = targets.to(app.config['device']).squeeze(1)

                scores = model(data)

                batch_train_loss = loss(scores, targets)
                new_loss_value += batch_train_loss.item()
                
                for target in targets.cpu().numpy():
                    new_roc_targets.append(target)
                soft_scores = torch.nn.functional.softmax(scores, dim=1)
                for score in soft_scores:
                    new_roc_scores.append(score[1].detach().cpu().numpy())

                num_correct = (scores.argmax(dim=1)==targets).sum().item()
                new_correct += num_correct
                new_samples += len(targets)

            pcam_accuracy = pcam_correct / pcam_samples
            new_accuracy = new_correct / new_samples
            pcam_loss_value /= len(pcam_loader)
            new_loss_value /= len(new_loader)
            pcam_roc = roc_auc_score(y_true=pcam_roc_targets, y_score=pcam_roc_scores)
            new_roc = roc_auc_score(y_true=new_roc_targets, y_score=new_roc_scores)

            combined_acc = (pcam_correct + new_correct) / (pcam_samples + new_samples)
            combined_roc = roc_auc_score(
                y_true=new_roc_targets+pcam_roc_targets, 
                y_score=new_roc_scores+pcam_roc_scores
                )

            result_dict = {
                'pcam accuracy': pcam_accuracy, 
                'new accuracy': new_accuracy,
                'combined accuracy': combined_acc,
                'pcam loss': pcam_loss_value, 
                'new loss': new_loss_value,
                'pcam roc': pcam_roc,
                'new roc': new_roc,
                'combined roc': combined_roc
                }
            
            save_json(dict=result_dict, fp=request.args['save_fp'])
            return jsonify(result_dict)
        
    @app.route('/retrain', methods=['POST'])
    def retrain():
        """
        request requires weight_path, trainable_weights, lr, num_epochs, 
        patience, model_save_path, model_save_name, batch_size
        """ 
        model = app.config['model']
        # all or classifier only
        trainable_weights = request.args['trainable_weights']

        if trainable_weights == 'all':
            for param in model.parameters():
                param.requires_grad = True

        elif trainable_weights == 'classifier_only':
            for param, name in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        model_summary(model)

        loss = define_loss()
        optimizer = define_optim(model=model, learning_rate=float(request.args['lr']))
        train_dl, val_dl, test_dl = create_retrain_dataloaders(
            upload_dir=app.config['upload_path'], 
            label_file=request.args['label_fp'],
            mean=app.config['ds_stats']['mean'],
            std=app.config['ds_stats']['std'],
            batch_size=int(request.args['batch_size'])
            )
        train_loss_list, train_acc_list, train_roc_list, val_loss_list, val_acc_list, val_roc_list = train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        train_dl=train_dl,
        val_dl=val_dl,
        num_epochs=int(request.args['num_epochs']),
        patience=int(request.args['patience']),
        device=app.config['device'],
        model_save_path=Path(request.args['model_save_path']),
        model_save_name=request.args['model_save_name']
        )
        model = torch.load(Path(request.args['model_save_path']).joinpath(request.args['model_save_name']), weights_only=False)

        test_acc, test_loss, test_roc = evaluate(
            model=model,
            data_loader=test_dl,
            device=app.config['device'],
            loss=loss,
            mode="Test"
        )
        result_dict = {
            'train losses': train_loss_list,
            'train accuracy': train_acc_list,
            'train roc': train_roc_list,
            'val loss': val_loss_list,
            'val accuracy': val_acc_list,
            'val roc': val_roc_list,
            'test accuracy': test_acc,
            'test loss': test_loss,
            'test roc': test_roc
        }
        save_json(dict=result_dict, fp=request.args['save_fp'])
        return jsonify(result_dict)
    
    return app


def save_json(dict, fp):
    with open(fp, mode="w") as opened_json:
        json.dump(dict, opened_json)


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


class InferenceDataset(torch.utils.data.Dataset):
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
    

def read_csv(csv_path):
    with open(csv_path) as opened_csv:
        reader = csv.reader(opened_csv)
        return [x for x in reader]


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, upload_dir, label_file):
        self.transforms = transforms
        self.upload_dir = Path(upload_dir)
        self.labels = read_csv(label_file)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        ds_tuple = self.labels[index]
        img_path = self.upload_dir.joinpath(ds_tuple[0])
        loaded_img = Image.open(img_path)
        norm_tensor = self.transforms(loaded_img)
        return norm_tensor, torch.tensor([int(x) for x in ds_tuple[1]])


def create_dataloaders(
        pcam_root: Path, 
        mean: tuple, 
        std: tuple, 
        batch_size: int, 
        upload_path: Path, 
        pcam_split: str,
        label_file: Path=None
        ) -> tuple[Type[DataLoader]]:
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
    pcam_ds = torchvision.datasets.PCAM(split=pcam_split, root=pcam_root, download=True, transform=eval_transforms)
    if not label_file:
        new_ds = InferenceDataset(transforms=eval_transforms, upload_dir=upload_path)
    else:
        new_ds = LabeledDataset(transforms=eval_transforms, upload_dir=upload_path, label_file=label_file)

    pcam_dl = DataLoader(dataset=pcam_ds, batch_size=batch_size, shuffle=False)
    new_dl = DataLoader(dataset=new_ds, batch_size=batch_size, shuffle=False)

    return pcam_dl, new_dl


def create_retrain_dataloaders(upload_dir, label_file, mean, std, batch_size):
    transforms = Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(size=32),
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomApply([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomGrayscale()
        ], p=0.2)
    ])
    eval_transforms = Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(size=32),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    pcam_train = torchvision.datasets.PCAM(split="train", download=False, root="C:\\personal_ML\\DINOVIT_PCAM\\pcam\\", transform=transforms)
    new_ds = LabeledDataset(upload_dir=upload_dir, label_file=label_file, transforms=transforms)

    combined_ds = ConcatDataset([pcam_train, new_ds])

    val_ds = torchvision.datasets.PCAM(split="val", download=False, root="C:\\personal_ML\\DINOVIT_PCAM\\pcam\\", transform=eval_transforms)
    test_ds = torchvision.datasets.PCAM(split="test", download=False, root="C:\\personal_ML\\DINOVIT_PCAM\\pcam\\", transform=eval_transforms)

    train_dl = DataLoader(combined_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


def init_model(app):
    """Initialize the model and store it in app config"""
    try:
        model = torch.load(app.config['model_PATH'], weights_only=False)
        model.to(app.config['device'])
        app.config['model'] = model
        app.config['model_loaded'] = True
    except Exception as e:
        app.config['model_loaded'] = False
        app.logger.error(f"model failed to load: {e}")


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
