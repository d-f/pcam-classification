from PIL import Image
import matplotlib.pyplot as plt
from typing import Type, Dict, Tuple, List
import numpy as np
import torch
import json
import argparse
from pathlib import Path
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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


def create_dataloader(ds_root: Path, mean: tuple, std: tuple) -> tuple[Type[DataLoader]]:
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
    test_ds = torchvision.datasets.PCAM(split="test", root=ds_root, download=False, transform=eval_transforms)
    test_dl = DataLoader(dataset=test_ds, batch_size=1, shuffle=True)
    
    return test_dl


def read_json(json_path: Path) -> Dict:
    """
    reads JSON file and returns dictionary of file contents
    """
    with open(json_path, mode="r") as opened_json:
        return json.load(opened_json)


def unnormalize_img(data: np.array, mean: np.array, std: np.array) -> np.array:
    """
    reverses image normalization for visualization
    """
    std = np.array(std)
    mean = np.array(mean)
    return (data * std.reshape(3,1,1)) + mean.reshape(3,1,1)


def grad_cam(
        model: Type[torchvision.models.efficientnet_b0], 
        data_loader: DataLoader, 
        device: Type[torch.device], 
        ds_stats: Dict
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
    for param in model.parameters():
        param.requires_grad = True
    for data, labels in data_loader:
        data = data.to(device)
        data.requires_grad = True
        labels = labels.to(device)
        print(labels.item())
        target_layers = [model.features[-4][0]]
        targets = [ClassifierOutputTarget(labels.item())]

        un_norm = unnormalize_img(data.detach().cpu().numpy(), ds_stats["mean"], ds_stats["std"])
        img = (un_norm.mean(axis=0) * 255).astype(np.uint8)
        img = Image.fromarray(img.mean(axis=0))

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(data, targets=targets)
            
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(np.moveaxis(un_norm[0], 0, -1))
        ax1.axis('off')

        ax2.imshow(img)
        ax2.imshow(grayscale_cam[0], alpha=0.5, cmap="turbo")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()


def define_model(num_classes: int) -> Type[torchvision.models.efficientnet_b0]:
    """
    defines the model architecture and replaces the classifier with a 
    classifier with the correct number of neurons

    num_classes: number of class neurons

    returns efficientnet_b0 model
    """
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1, dropout=0.3)
    output_layer = model.classifier[1]
    new_out = torch.nn.Linear(in_features=output_layer.in_features, out_features=num_classes)    
    model.classifier[1] = new_out
    
    return model


def main():
    args = parse_cla()
    ds_stats = read_json(args.ds_stat_path)
    test_dl = create_dataloader(ds_root=args.pcam_dir, mean=ds_stats["mean"], std=ds_stats["std"])
    device = torch.device("cuda")
    model = torch.load(args.model_save_path.joinpath(args.model_save_name))
    model = define_model(num_classes=2)
    model.to(device)

    grad_cam(
        model=model,
        data_loader=test_dl,
        device=device,
        save_dir=Path("C:\\personal_ML\\pcam_classification\\"),
        ds_stats=ds_stats
    )


if __name__ == "__main__":
    main()
