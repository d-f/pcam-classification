import json
import argparse
from pathlib import Path
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
from typing import Type, Tuple


def parse_cla():
    """
    parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-pcam_dir", type=Path)
    parser.add_argument("-save_dir", type=Path)
    return parser.parse_args()


def create_dataloaders(ds_root: Path) -> Type[DataLoader]:
    """
    loads the train dataset from the ds_root directory

    ds_root: folder containing the pcam dataset
    """
    transforms = Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(size=32),
])
    train = torchvision.datasets.PCAM(
        split="train", 
        root=ds_root, 
        download=False, 
        transform=transforms
        )

    train = DataLoader(dataset=train, batch_size=32, shuffle=False)

    return train


def determine_norm_values(train_dataloader: Type[DataLoader]) -> Tuple:
    """
    determines channel wise mean and standard deviation of the entire dataset

    train_dataloader: PyTorch DataLoader containing the train partition of the PCAM dataset
    """
    ds_mean = 0
    ds_std = 0

    for img_batch, _ in tqdm(train_dataloader):
        batch_mean = img_batch.mean(dim=(0, 2, 3))
        batch_std = img_batch.std(dim=(0, 2, 3))
        ds_mean += batch_mean
        ds_std += batch_std

    ds_mean /= len(train_dataloader)
    ds_std /= len(train_dataloader)
    return ds_mean, ds_std


def save_results(mean: Tuple, std: Tuple, save_dir: Path) -> None:
    """
    saves the statistics in a JSON file

    mean:     channel-wise mean of the entire training dataset
    std:      channel-wise standard deviation of the entire training dataset
    save_dir: Path in which to save the JSON file
    """
    json_dict = {
        "mean": mean.detach().tolist(),
        "std": std.detach().tolist(),
    }
    with open(save_dir.joinpath("ds_mean_std.json"), mode="w") as opened_json:
        json.dump(json_dict, opened_json)

def main():
    args = parse_cla()
    train = create_dataloaders(ds_root=args.pcam_dir)
    mean, std = determine_norm_values(train)
    save_results(mean=mean, std=std, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
