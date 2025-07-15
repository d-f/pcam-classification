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
from typing import Type, Dict, List
from torchvision.transforms import Lambda
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score
from train_model import model_summary, define_optim, define_loss, evaluate, train
from statsmodels.stats.proportion import proportions_ztest


def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return "App is running"

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        predicts classification of an image

        request parameters
        filename: filename of the image
        gradcam_save_dir: folder to save the gradcam image result to
        upload_path: folder that contains the image
        ds_stats_fp: json file that contains the channel-wise mean and 
            standard deviation of the train dataset
        model_path: path to the model weight file
        """
        class_list = ["Contains no tumor", "Contains tumor"]
        img_filename = request.args['filename']
        upload_path = request.args['upload_path']
        ds_stats = load_ds_stats(request.args['ds_stats_fp'])
        device = torch.device('cuda')
        model_path = request.args['model_path']
        gradcam_save_dir = Path(request.args['gradcam_save_dir'])

        # convert image to tensor
        tensor = prepare_img(
            filepath=Path(upload_path).joinpath(img_filename),
            mean=ds_stats['mean'],
            std=ds_stats['std'],
            device=device
        )

        model = init_model(model_path=model_path, device=device)
        pred = model(tensor)
        
        generate_gradcam(
            model=model, 
            pred_class=pred.argmax(dim=1).item(),
            save_dir=gradcam_save_dir,
            model_name=model_path.rpartition("/")[2],
            input_img=Image.open(Path(upload_path).joinpath(img_filename)),
            input_tensor=tensor,
            input_filename=img_filename
        )
        
        return jsonify({'prediction': class_list[pred.argmax(dim=1).item()]})
    

    @app.route('/monitor_data', methods=['POST'])
    def monitor_data():
        """
        monitors data drift with maximum mean discrepancy

        request parameters
        model_path: path to the model weight file
        ds_stats_fp: json file that contains the channel-wise mean and 
            standard deviation of the train dataset
        upload_path: folder that contains new images
        pcam_dir: folder that contains PCAM dataset
        batch_size: number of images in the dataloader batch
        save_fp: filepath to save results to 
        """
        with torch.no_grad():
            device = torch.device('cuda')
            model_path = request.args['model_path']
            model = init_model(model_path=model_path, device=device)
            ds_stats = load_ds_stats(request.args['ds_stats_fp'])
            upload_path = request.args['upload_path']
            pcam_dir = request.args['pcam_dir']
            batch_size = int(request.args['batch_size'])
            save_fp = request.args['save_fp']

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
                pcam_root=pcam_dir, 
                mean=ds_stats['mean'], 
                std=ds_stats['std'],
                batch_size=batch_size,
                upload_path=upload_path,
                pcam_split="train"
            )

            # embed pcam dataset
            for ds_tuple in tqdm(train_dl):
                out = model(ds_tuple[0].to(device))
                for pred in out:
                    train_prod.append(pred.cpu().numpy())

            # embed new inputs
            for ds_tuple in tqdm(new_dl):
                out = model(ds_tuple[0].unsqueeze(0).to(device))
                for pred in out:
                    new_prod.append(pred.cpu().numpy())

            # calculate max mean discrepancy test stat and p value
            stat, p_value = MMD().test(np.array(train_prod), np.array(new_prod))

            # return classifier to original state
            model.classifier = old_classifier

            result_dict = {"stat": stat, "p value": p_value}

            # save results
            save_json(dict=result_dict, fp=save_fp)

            return jsonify(result_dict)
        
    @app.route('/eval', methods=['POST'])
    def eval():
        """
        evaluates model on a combination of the original test set,
        and new inputs that have been labeled

        request parameters 
        model_path: path to the model weight file
        ds_stats_fp: json file that contains the channel-wise mean and 
            standard deviation of the train dataset
        upload_path: folder that contains new images
        label_fp: csv file with labels for new images
        pcam_dir: folder that contains PCAM dataset
        batch_size: number of images in the dataloader batch
        save_fp: filepath to save results to
        """
        device = torch.device('cuda')
        model_path = request.args['model_path']
        model = init_model(model_path=model_path, device=device)
        model.eval()
        loss = torch.nn.CrossEntropyLoss()
        ds_stats = load_ds_stats(request.args['ds_stats_fp'])
        model = init_model(model_path=model_path, device=device)
        upload_path = request.args['upload_path']
        label_fp = request.args['label_fp']
        pcam_dir = request.args['pcam_dir']
        batch_size = int(request.args['batch_size'])
        save_fp = request.args['save_fp']

        # create original pcam test set dataloader and dataloader from new inputs
        pcam_loader, new_loader = create_dataloaders(
            pcam_root=pcam_dir,
            mean=ds_stats['mean'], 
            std=ds_stats['std'],
            batch_size=batch_size,
            upload_path=upload_path,
            pcam_split="test",
            label_file=label_fp          
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
            # get predictions for pcam dataset
            for data, targets in tqdm(pcam_loader):
                # add data and labels to GPU
                data = data.to(device)
                targets = targets.to(device)
                # predict classification
                scores = model(data)
                # calculate loss
                batch_train_loss = loss(scores, targets)

                pcam_loss_value += batch_train_loss.item()

                for target in targets.cpu().numpy():
                    pcam_roc_targets.append(target)
                
                # add softmax scores to list for ROC calculation
                soft_scores = torch.nn.functional.softmax(scores, dim=1)
                for score in soft_scores:
                    pcam_roc_scores.append(score[1].detach().cpu().numpy())

                num_correct = (scores.argmax(dim=1)==targets).sum().item()
                # add number of correct classifications
                pcam_correct += num_correct
                # add total number of samples
                pcam_samples += len(targets)

            # calculate meticcs for new inputs
            for data, targets in tqdm(new_loader):
                # put iamge and label on GPU
                data = data.to(device)
                targets = targets.to(device).squeeze(1)

                # predict classification
                scores = model(data)

                # calculate loss
                batch_train_loss = loss(scores, targets)
                new_loss_value += batch_train_loss.item()
                
                for target in targets.cpu().numpy():
                    new_roc_targets.append(target)
                # add softmax scores to list for ROC calculation
                soft_scores = torch.nn.functional.softmax(scores, dim=1)
                for score in soft_scores:
                    new_roc_scores.append(score[1].detach().cpu().numpy())

                num_correct = (scores.argmax(dim=1)==targets).sum().item()
                # add number of correctly precited images
                new_correct += num_correct
                # add total number of samples
                new_samples += len(targets)

            # calculate accuracy for pcam
            pcam_accuracy = pcam_correct / pcam_samples
            # calculate accuracy for new inputs
            new_accuracy = new_correct / new_samples
            pcam_loss_value /= len(pcam_loader)
            new_loss_value /= len(new_loader)
            # calculate roc for pcam
            pcam_roc = roc_auc_score(y_true=pcam_roc_targets, y_score=pcam_roc_scores)
            # calculate roc for new inputs
            new_roc = roc_auc_score(y_true=new_roc_targets, y_score=new_roc_scores)
            # accuracy of combined datasets
            combined_acc = (pcam_correct + new_correct) / (pcam_samples + new_samples)
            # roc of combined datasets
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
            # save result dict
            save_json(dict=result_dict, fp=save_fp)
            return jsonify(result_dict)
        
    @app.route('/retrain', methods=['POST'])
    def retrain():
        """
        retrains a model on some additional data

        request parameters
        weight_path: path to the model weights
        trainable_weights: string indicating what parameters are frozen, 
            choices={'all', 'classifier only'}
        lr: learning rate
        num_epochs: number of rounds of training
        patience: number of epochs to train past best validation peformance
        model_save_path: folder to save model to
        model_save_name: name of file to save model to
        batch_size: number of images in the dataloader batch
        pcam_dir: folder that contains PCAM dataset
        """ 
        model_path = request.args['model_path']
        upload_path = request.args['upload_path']
        pcam_dir = request.args['pcam_dir']
        ds_stats = load_ds_stats(request.args['ds_stats_fp'])
        device = torch.device('cuda')
        model = init_model(model_path=model_path, device=device)
        # all or classifier only
        trainable_weights = request.args['trainable_weights']
        lr = float(request.args['lr'])
        label_fp = request.args['label_fp']
        batch_size = int(request.args['batch_size'])
        num_epochs = int(request.args['num_epochs'])
        patience = int(request.args['patience'])
        ms_path = Path(request.args['model_save_path'])
        model_save_name = request.args['model_save_name']
        save_fp = request.args['save_fp']

        # selectively freeze certain parameters
        if trainable_weights == 'all':
            for param in model.parameters():
                param.requires_grad = True

        elif trainable_weights == 'classifier_only':
            for param, name in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # print model summary
        model_summary(model)

        loss = define_loss()
        optimizer = define_optim(model=model, learning_rate=lr)
        # create dataloaders which take pcam data and newly labeled inputs
        train_dl, val_dl, test_dl = create_retrain_dataloaders(
            upload_dir=upload_path, 
            label_file=label_fp,
            mean=ds_stats['mean'],
            std=ds_stats['std'],
            batch_size=batch_size,
            pcam_root=pcam_dir
            )
        # train model
        train_loss_list, train_acc_list, train_roc_list, val_loss_list, val_acc_list, val_roc_list = train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        train_dl=train_dl,
        val_dl=val_dl,
        num_epochs=num_epochs,
        patience=patience,
        device=device,
        model_save_path=ms_path,
        model_save_name=model_save_name
        )
        # load best performing model
        model = torch.load(ms_path.joinpath(model_save_name), weights_only=False)
        # test model
        test_acc, test_loss, test_roc = evaluate(
            model=model,
            data_loader=test_dl,
            device=device,
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
        # save results
        save_json(dict=result_dict, fp=save_fp)
        return jsonify(result_dict)
    
    @app.route('/uncertainty_range', methods=["POST"])
    def uncertainty_range():
        """
        determines descriptive stats of the uncertainty values on the pcam test set

        request parameters 
        ds_stats_fp: json file that contains the channel-wise mean and 
            standard deviation of the train dataset
        batch_size: number of images in the dataloader batch
        n_rep: number of repeditions in the Monte Carlo simulation
        pcam_dir: folder that contains the PCAM dataset
        model_path: path to the model weights
        """
        ds_stats = load_ds_stats(request.args['ds_stats_fp'])
        batch_size = int(request.args['batch_size'])
        device = torch.device('cuda')
        n_rep = int(request.args['n_rep'])
        model_path = request.args['model_path']
        pcam_dir = request.args['pcam_dir']

        highest_uncertainty = 0
        lowest_uncertainty = float('inf')
        correct = []
        incorrect = []
        eval_transforms = Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(size=32),
        torchvision.transforms.Normalize(mean=ds_stats['mean'], std=ds_stats['std'])
        ])
        test_ds = torchvision.datasets.PCAM(split='test', root=pcam_dir, download=True, transform=eval_transforms)
        test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
        
        model = init_model(model_path=model_path, device=device)

        with torch.no_grad():
            model.train() # turn dropout on
            for data, targets in tqdm(test_dl):
                data = data.to(device)
                targets = targets.to(device)
                # create empty array to add MC predictions to
                batch_scores = np.empty(shape=(n_rep, len(data), 2))

                # MC dropout
                for i in range(n_rep):
                    scores = model(data)
                    batch_scores[i] = scores.cpu().numpy()

                # measure mean and standard deviation of the predictions
                batch_mean = np.mean(batch_scores, axis=0).tolist()
                batch_std = np.mean(np.std(batch_scores, axis=0), axis=1).tolist()

                batch_argmax = [x.index(max(x)) for x in batch_mean]

                # keep track of correct preds
                correct_idx = [i for i, (p, t) in enumerate(zip(batch_argmax, targets)) if p == t]
                # keep track of incorrect preds
                incorrect_idx = [i for i, (p, t) in enumerate(zip(batch_argmax, targets)) if p != t]

                correct += [batch_std[i] for i in correct_idx]
                incorrect += [batch_std[i] for i in incorrect_idx]

                largest = max(batch_std)
                smallest = min(batch_std)

                # keep track of min and max
                highest_uncertainty = max(largest, highest_uncertainty)
                lowest_uncertainty = min(smallest, lowest_uncertainty)

        avg_correct = np.mean(correct)
        avg_incorrect = np.mean(incorrect)
        std_correct = np.std(correct)
        std_incorrect = np.std(incorrect)

        return jsonify({
            "avg correct": avg_correct,
            "avg incorrect": avg_incorrect,
            "standard deviation correct": std_correct,
            "standard deviation incorrect": std_incorrect,
            "highest": highest_uncertainty,
            "lowest": lowest_uncertainty
        })

    @app.route('/ab_test_uncertainty', methods=["POST"])
    def ab_test_uncertainty():
        """
        performs an A/B test on two different uncertainty thresholds
        setting one of them as 'inf' is equivalent to no uncertainty thresholding

        request parameters 
        model_path: path to the model weights
        uncertainty_a: one of the thresholds of uncertainty considered for rejecting highly uncertain predictions
        uncertainty_b: one of the thresholds of uncertainty considered for rejecting highly uncertain predictions
        ds_stats_fp: json file that contains the channel-wise mean and 
            standard deviation of the train dataset 
        batch_size: number of images in the dataloader batch
        n_rep: number of repeditions in the Monte Carlo simulation
        pcam_dir: folder that contains the PCAM dataset
        """
        a_skipped = 0
        a_correct = 0
        a_total = 0

        b_skipped = 0
        b_correct = 0
        b_total = 0

        model_path = request.args['model_path']
        uncertainty_a = float(request.args['uncertainty_a'])
        uncertainty_b = float(request.args['uncertainty_b'])
        ds_stats = load_ds_stats(request.args['ds_stats_fp'])
        device = torch.device('cuda')
        batch_size = int(request.args['batch_size'])
        n_rep = int(request.args['n_rep'])
        pcam_dir = request.args['pcam_dir']

        model = init_model(model_path=model_path, device=device)

        model_summary(model)

        eval_transforms = Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(size=32),
        torchvision.transforms.Normalize(mean=ds_stats['mean'], std=ds_stats['std'])
        ])
        test_ds = torchvision.datasets.PCAM(split='test', root=pcam_dir, download=True, transform=eval_transforms)
        test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            model.train() # turn dropout on
            for data, targets in tqdm(test_dl):
                data = data.to(device)
                targets = targets.to(device)

                # create empty array to add MC predictions to
                batch_scores = np.empty(shape=(n_rep, len(data), 2))

                # MC dropout
                for i in range(n_rep):
                    scores = model(data)
                    batch_scores[i] = scores.cpu().numpy()

                batch_mean = np.mean(batch_scores, axis=0)
                
                batch_argmax = batch_mean.argmax(axis=1)
                
                batch_uncertainty = np.mean(np.std(batch_scores, axis=0), axis=1).tolist()

                # keep track of predictions that have uncertainty levels below threshold a
                pass_a = [(i, x) for i, x in enumerate(batch_uncertainty) if x < uncertainty_a]
                # keep track of predictions that have uncertainty levels below threshold b
                pass_b = [(i, x) for i, x in enumerate(batch_uncertainty) if x < uncertainty_b]

                a_total += len(pass_a)
                b_total += len(pass_b)

                # track coverage
                a_skipped += len(batch_argmax) - len(pass_a)
                b_skipped += len(batch_argmax) - len(pass_b)

                a_correct += len([(targets[x[0]]==x[1]) for x in pass_a])
                b_correct += len([(targets[x[0]]==x[1]) for x in pass_b])

        a_accuracy = a_correct / a_total
        b_accuracy = b_correct / b_total

        # measure whether difference between A and B is significant
        z_stat, p_value = proportions_ztest(
            count=np.array([a_correct, b_correct]),
            nobs=np.array([a_total, b_total])
        )

        results = {
            "Accuracy A": a_accuracy,
            "Accuracy B": b_accuracy,
            "A Coverage": a_total,
            "B Coverage": b_total,
            "Z Statistic": z_stat,
            "P Value": p_value
        }
        return jsonify(results)

    return app


def load_ds_stats(label_fp: str) -> Dict:
    """
    loads file containing dataset mean and 
    standard deviation

    label_fp: file path of the .json file
    """
    with open(label_fp, mode="r") as opened_json:
        return json.load(opened_json)    


def save_json(dict: Dict, fp: Path) -> None:
    """
    saves json file

    dict: dicitonary object to json serialize
    fp: file path of the .json file to save
    """
    with open(fp, mode="w") as opened_json:
        json.dump(dict, opened_json)


def prepare_img(
        filepath: str | Path, 
        mean: float, 
        std: float, 
        device: Type[torch.device]
        ) -> torch.tensor:
    """
    reads an image file, transforms, reshapes and normalizes
    the values and returns the normalized tensor

    file_path: file path of the image to prepare
    mean: channel-wise mean of the images in the training ds
    std: channel-wise standard deviation of the images in the training ds
    device: pytorch device context manager
    """
    transforms = Compose([
        torchvision.transforms.ToTensor(),
        Lambda(lambda x: x.unsqueeze(0)),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    loaded_img = Image.open(filepath)
    norm_tensor = transforms(loaded_img).to(device)
    return norm_tensor


def generate_gradcam(
        model: Type[torchvision.models.efficientnet_b0], 
        pred_class: int, 
        save_dir: Path | str, 
        model_name: str, 
        input_img: Image, 
        input_tensor: torch.tensor,
        input_filename: str
        ) -> None:
    """
    generates and saved gradcam for an image

    model: torchvision model
    pred_class: predicted class
    save_dir: path to save the gradcam image to
    model_name: name of the model file to keep track of in the gradcam image name
    input_img: PIL Image of the input
    input_tensor: torch tensor of the input image
    input_filename: filename of the input image
    """
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
        """
        loads and prepares image tensor
        """
        img_path = self.uploads[index]
        loaded_img = Image.open(img_path)
        norm_tensor = self.transforms(loaded_img)
        return norm_tensor
    

def read_csv(csv_path: str) -> List:
    """
    reads csv file
    """
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
        """
        loads and prepares image tensor
        """
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

    ds_root: Path to the PCAM dataset files
    mean: mean for each channel across the train dataset
    std: standard deviation for each channel across the train dataset 
    batch_size: number of images in each batch 
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


def create_retrain_dataloaders(
    upload_dir: str, 
    label_file: str, 
    mean: List, 
    std: List, 
    batch_size: int,
    pcam_root: str
    ) -> Type[torch.utils.data.DataLoader]:
    """
    creates dataloaders with additional data

    upload_dir: dir with additional images
    label_file: csv file with classes of additional images
    mean: list of the mean of each channel of the images in the train dataset
    std: list of the standard deviation of each channel of the iamges in the train dataset
    batch_size: number of images per dataloader batch
    pcam_root: directory with the pcam dataset
    """
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
    pcam_train = torchvision.datasets.PCAM(split="train", download=False, root=pcam_root, transform=transforms)
    new_ds = LabeledDataset(upload_dir=upload_dir, label_file=label_file, transforms=transforms)

    combined_ds = ConcatDataset([pcam_train, new_ds])

    val_ds = torchvision.datasets.PCAM(split="val", download=False, root=pcam_root, transform=eval_transforms)
    test_ds = torchvision.datasets.PCAM(split="test", download=False, root=pcam_root, transform=eval_transforms)

    train_dl = DataLoader(combined_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl


def init_model(
    model_path: str, 
    device: Type[torch.device]
    ) -> Type[torchvision.models.EfficientNet]:
    """
    loads EfficientNet model
    """
    model = torch.load(model_path, weights_only=False)
    model.to(device)
    return model


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
