from typing import Type, Dict, Tuple, List
import numpy as np
import torch
import json
import argparse
from pathlib import Path
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def parse_cla():
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-pcam_dir", type=Path)
    parser.add_argument("-ds_stat_path", type=Path)
    parser.add_argument("-freeze_base", action="store_true")
    parser.add_argument("-lr", type=float)
    parser.add_argument("-num_epochs", type=int)
    parser.add_argument("-patience", type=int)
    parser.add_argument("-batch_size", type=int)
    parser.add_argument("-model_save_path", type=Path)
    parser.add_argument("-model_save_name", type=str)
    parser.add_argument("-result_file_path", type=Path)
    return parser.parse_args()


def create_dataloaders(ds_root: Path, mean: tuple, std: tuple, batch_size: int) -> tuple[Type[DataLoader]]:
    """
    creates PyTorch DataLoaders

    ds_root:    Path to the PCAM dataset files
    mean:       mean for each channel across the train dataset
    std:        standard deviation for each channel across the train dataset 
    batch_size: number of images in each batch 

    returns tuple of DataLoaders for each dataset partition
    """
    transforms = Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(size=32),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])
    eval_transforms = Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(size=32),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    train_ds = torchvision.datasets.PCAM(split="train", root=ds_root, download=False, transform=transforms)
    val_ds = torchvision.datasets.PCAM(split="val", root=ds_root, download=False, transform=eval_transforms)
    test_ds = torchvision.datasets.PCAM(split="test", root=ds_root, download=False, transform=eval_transforms)

    train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl, test_dl


def read_json(json_path: Path) -> Dict:
    """
    reads JSON file and returns dictionary of file contents
    """
    with open(json_path, mode="r") as opened_json:
        return json.load(opened_json)
    

def define_model(num_classes: int) -> Type[torchvision.models.efficientnet_b0]:
    """
    defines the model architecture and replaces the classifier with a 
    classifier with the correct number of neurons

    num_classes: number of class neurons

    returns efficientnet_b0 model
    """
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    output_layer = model.classifier[1]
    new_out = torch.nn.Linear(in_features=output_layer.in_features, out_features=num_classes)    
    model.classifier[1] = new_out
    
    return model


def freeze_base(model: Type[torchvision.models.efficientnet_b0]) -> None:
    """
    freezes all parameters in the model except the classifier layer

    model: efficientnet_b0
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def model_summary(model: Type[torchvision.models.efficientnet_b0]) -> None:
    """
    prints each layer of the model's name and parameter shape,
    as well as the number of trainable and frozen parameters

    model: efficientnet_b0
    """
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        print(name, param.shape)
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    print(f"Number of trainable parameters:{trainable}")
    print(f"Number of frozen parameters:{frozen}")


def define_optim(
        model: Type[torchvision.models.efficientnet_b0], 
        learning_rate: float
        ) -> Type[torch.optim.Adam]:
    """
    defines the optimizer

    model:         efficientnet_b0
    learning_rate: weight to scale the gradient values
    """
    relevant_params = [x for x in model.parameters() if x.requires_grad]
    return torch.optim.Adam(params=relevant_params, lr=learning_rate)


def define_loss() -> Type[torch.nn.CrossEntropyLoss]:
    """
    defines the loss function
    """
    return torch.nn.CrossEntropyLoss()


def evaluate(
        model: Type[torchvision.models.efficientnet_b0], 
        data_loader: DataLoader, 
        device: Type[torch.device], 
        loss: Type[torch.nn.CrossEntropyLoss], 
        mode: str
        ) -> Tuple:
    """
    evaluates model on the given DataLoader

    model:       efficientnet_b0
    data_loader: PyTorch DataLoader
    device:      torch.device
    loss:        cross entropy loss function
    mode:        'Validate', 'Test', etc. for tqdm description

    returns tuple of accuracy, loss and ROC-AUC
    """
    model.eval()
    with torch.no_grad():
        accuracy = 0
        loss_value = 0
        roc_targets = []
        roc_scores = []
        total_correct = 0
        total_samples = 0
        for data, targets in tqdm(data_loader, desc=mode):
            data = data.to(device)
            one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=2)
            targets = targets.to(device)

            scores = model(data)

            batch_train_loss = loss(scores, targets)
            loss_value += batch_train_loss.item()
             
            
            for one_hot in one_hot_targets.cpu().numpy():
                roc_targets.append(one_hot)
            for score in scores.detach().cpu().numpy():
                roc_scores.append(score)

            num_correct = (scores.argmax(dim=1)==targets).sum().item()
            total_correct += num_correct
            total_samples += len(targets)

        accuracy = total_correct / total_samples
        loss_value /= len(data_loader)
        roc = roc_auc_score(y_true=roc_targets, y_score=roc_scores)
        return accuracy, loss_value, roc
    

def save_checkpoint(model: Type[torchvision.models.efficientnet_b0], save_path: Path) -> None:
    """
    saves model checkpoint
    """
    torch.save(model, save_path)


def train(
        model: Type[torchvision.models.efficientnet_b0], 
        loss: Type[torch.nn.CrossEntropyLoss], 
        optimizer: Type[torch.optim.Adam], 
        train_dl: Type[DataLoader], 
        val_dl: Type[DataLoader], 
        num_epochs: int, 
        patience: int, 
        device: Type[torch.device], 
        model_save_path: Path, 
        model_save_name: Path
        ) -> Tuple:
    """
    trains model and keeps track of training and validation metrics

    model:           efficientnet_b0
    loss:            cross entropy loss function
    optimizer:       Adam optimizer algorithm
    train_dl:        training DataLoader
    val_dl:          validation DataLoader
    num_epochs:      number of rounds of training
    patience:        number of epochs the model is trained past validation loss beginning to increase
    device:          torch device
    model_save_path: directory to save model in
    model_save_name: name to save model file

    returns tuple of lists of metrics
    """
    patience_counter = 0
    best_val_loss = np.inf
    train_loss_list = []
    train_acc_list = []
    train_roc_list = []
    val_loss_list = []
    val_acc_list = []
    val_roc_list = []

    for epoch_idx in range(num_epochs):
        model.train()
        if patience_counter == patience:
            break
        epoch_train_loss = 0
        epoch_train_acc = 0
        epoch_train_roc = 0
        roc_targets = []
        roc_scores = []
        total_correct = 0
        total_samples = 0
        for data, targets in tqdm(train_dl, desc="Train"):
            data = data.to(device)
            targets = targets.to(device)
            
            scores = model(data)
            loss_value = loss(scores, targets)

            optimizer.zero_grad()
            loss_value.backward()

            optimizer.step()

            with torch.no_grad():
                one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=2)
                for one_hot in one_hot_targets.cpu().numpy():
                    roc_targets.append(one_hot)
                for score in scores.detach().cpu().numpy():
                    roc_scores.append(score)
                num_correct = (scores.argmax(dim=1)==targets).sum().item()
                total_correct += num_correct
                total_samples += len(targets)
                epoch_train_loss += loss_value.item()

        epoch_train_loss /= len(train_dl)
        epoch_train_acc = total_correct / total_samples

        epoch_train_roc = roc_auc_score(y_true=roc_targets, y_score=roc_scores)

        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)
        train_roc_list.append(epoch_train_roc)

        print(f"Epoch: {epoch_idx}, train loss: {epoch_train_loss}, train acc: {epoch_train_acc}, train roc: {epoch_train_roc}")

        val_acc, val_loss, val_roc = evaluate(model=model, data_loader=val_dl, loss=loss, device=device, mode="Validate")

        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        val_roc_list.append(val_roc)

        print(f"val loss: {val_loss}, val acc: {val_acc}, val roc: {val_roc}")

        if val_loss < best_val_loss:
            patience_counter = 0
            best_val_loss = val_loss
            save_checkpoint(model=model, save_path=model_save_path.joinpath(model_save_name))
        else:
            patience_counter += 1

    return train_loss_list, train_acc_list, train_roc_list, val_loss_list, val_acc_list, val_roc_list


def save_results(
        train_loss_list: List, 
        train_acc_list: List, 
        train_roc_list: List, 
        val_loss_list: List, 
        val_acc_list: List, 
        val_roc_list: List, 
        test_loss: List, 
        test_acc: Type[torch.Tensor], 
        test_roc: float, 
        save_path: Path
        ) -> None:
    """
    saves the results of the training, validation and testing

    train_loss_list: list of training loss values per epoch
    train_acc_list:  list of accuracy on the training set per epoch
    train_roc_list:  list of the roc on the training set per epoch
    val_loss_list:   list of the validation loss values per epoch
    val_acc_list:    list of the validation accuracy values per epoch
    val_roc_list:    list of the validation roc values per epoch
    test_loss:       test loss value
    test_acc:        test accuracy 
    test_roc:        test ROC-AUC
    save_path:       path to save JSON file

    """
    result_dict = {
        "train_loss_list": train_loss_list, 
        "train_acc_list":train_acc_list, 
        "train_roc_list": [x.astype(np.float64) for x in train_roc_list], 
        "val_loss_list": val_loss_list, 
        "val_acc_list": val_acc_list, 
        "val_roc_list": [x.astype(np.float64) for x in val_roc_list], 
        "test_loss": test_loss, 
        "test_acc": test_acc, 
        "test_roc": test_roc
    }
    with open(save_path, mode="w") as opened_json:
        json.dump(result_dict, opened_json)


def main():
    args = parse_cla()
    ds_stats = read_json(args.ds_stat_path)
    train_dl, val_dl, test_dl = create_dataloaders(ds_root=args.pcam_dir, mean=ds_stats["mean"], std=ds_stats["std"], batch_size=args.batch_size)
    model = define_model(num_classes=2)
    if args.freeze_base:
        freeze_base(model)
    model_summary(model)
    loss = define_loss()
    optimizer = define_optim(model=model, learning_rate=args.lr)
    device = torch.device("cuda")
    model.to(device)
    train_loss_list, train_acc_list, train_roc_list, val_loss_list, val_acc_list, val_roc_list = train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        train_dl=train_dl,
        val_dl=val_dl,
        num_epochs=args.num_epochs,
        patience=args.patience,
        device=device,
        model_save_path=args.model_save_path,
        model_save_name=args.model_save_name
    )

    model = torch.load(args.model_save_path.joinpath(args.model_save_name))
    test_acc, test_loss, test_roc = evaluate(
        model=model,
        data_loader=test_dl,
        device=device,
        loss=loss,
        mode="Test"
    )
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)
    print("Test ROC", test_roc)
    save_results(
        train_loss_list=train_loss_list, 
        train_acc_list=train_acc_list, 
        train_roc_list=train_roc_list, 
        val_loss_list=val_loss_list, 
        val_acc_list=val_acc_list, 
        val_roc_list=val_roc_list,
        test_loss=test_loss,
        test_acc=test_acc,
        test_roc=test_roc,
        save_path=args.result_file_path
    )


if __name__ == "__main__":
    main()
