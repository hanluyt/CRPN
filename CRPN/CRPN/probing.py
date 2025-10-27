#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Time: 2025/10/22 16:30
# @Author: hanluyt

import torch
from torchvision import transforms
from .models.model_emotion import PerCept
import os 
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import pandas as pd

def make_dataset(directory: str) -> Union[List[str], List[Tuple[str, int]]]:
    """
    Automatically adapts to the dataset structure:
    - If the input directory directly contains files: return [path1, path2, ...]
    - If the directory contains subfolders: return [(path, class_index), ...]
     - Mixed structure (both files and folders) is not allowed.
     """
    class_to_idx = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    directory = os.path.expanduser(directory)
    instances = []

    # List all visible entries
    entries = [entry for entry in os.scandir(directory) if not entry.name.startswith('.')]
    subdirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    # Case 1: The directory contains only files
    if files and not subdirs:
        return [f.path for f in files]
    
    # Case 2: The directory contains only subfolders
    elif subdirs and not files:
        for subdir in sorted(subdirs, key=lambda x: x.name):
            class_name = subdir.name
            if class_name not in class_to_idx:
                raise ValueError(
                    f"Subfolder '{class_name}' is not in the allowed category list: {list(class_to_idx.keys())}"
                )
            class_index = class_to_idx[class_name]

            # Walk through all files in the subfolder
            for root, _, fnames in sorted(os.walk(subdir.path, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_index)
                    instances.append(item)
        return instances
    
    # Case 3: Mixed structure (files and subfolders)
    elif files and subdirs:
        raise ValueError(
            f"The directory '{directory}' contains both files and subfolders. "
            "Please keep only one type (all files or all subfolders)."
        )

    # Case 4: Empty directory
    else:
        raise ValueError(f"No valid files or subfolders found in {directory}.")


class FERDataSet(Dataset):
    def __init__(self,
                 dir: str,
                 loader: Callable[[str], Any],
                 transform: Optional[Callable] = None
                 ) -> None:
        self.root = dir
        self.samples = make_dataset(self.root)

        self.loader = loader
        self.transform = transform
   
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        while True:
            try:
                sample = self.samples[index]
                if isinstance(sample, tuple) and len(sample) == 2:
                    path, label = sample
                else:
                    path = sample
                    label = None

                img = self.loader(path)
                break
            except Exception as e:
                print(e)
                index = random.randint(0, len(self.samples) - 1)

        if self.transform is not None:
            img = self.transform(img)

        if label is not None:
            return path, img, label
        else:
            return path, img

    def __len__(self) -> int:
        return len(self.samples)

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = cv2.imread(path)
        return Image.fromarray(img)

class FERImageFolder(FERDataSet):
    def __init__(
            self,
            dir: str,
            transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader
    ):
        super(FERImageFolder, self).__init__(dir, loader, transform=transform)
        self.imgs = self.samples

def valid_transform(mean, std):
    t = []
    t.append(transforms.Resize(256))
    t.append(transforms.CenterCrop(224))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_dataset(eval_data_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = valid_transform(mean, std)
    dataset = FERImageFolder(eval_data_path, transform=transform)

    return dataset

class CRPNProbing:
    def __init__(self, eval_data_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PerCept(n_class=7, baseline=False)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        weight_path = os.path.join(current_dir, ".", "backbone_weight", "resnet18_percept_7100.pth")
        weight_path = os.path.abspath(weight_path)


        checkpoint = torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

        dataset_val = build_dataset(eval_data_path)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        self.dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                  sampler=sampler_val,
                                                  batch_size=min(len(dataset_val), 128),
                                                  num_workers=8,
                                                  pin_memory=True,
                                                  drop_last=False)

    def test_acc(self):
        # return accuracy on the test set
        total_im, total_correct = 0, 0
        self.model.eval()

        for step, data in enumerate(self.dataloader_val):
            assert len(data) == 3
            path, imgs, labels = data
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            total_im += imgs.size(0)
            with torch.no_grad():
                return_all = self.model(imgs)

                if isinstance(return_all, tuple):
                    logits = return_all[-1]
                else:
                    logits = return_all

                correct_num = (logits.max(-1)[-1] == labels).sum()
                total_correct += correct_num

        val_acc = (total_correct / total_im).item()

        print('Test ----- test acc: %2.5f' % (val_acc))
        return val_acc

    
    def evaluate(self, output_filename):
        """
        Perform model evaluation on the test dataset and save predictions.

        Parameters
        ----------
        output_filename : str
            The file name (without extension) for saving the output CSV file.
            The CSV will be saved as "<output_filename>.csv" in the current directory.
        """
        self.model.eval()
        records = []

        for step, data in enumerate(self.dataloader_val):
            if len(data) == 3:
                path, imgs, labels = data
            elif len(data) == 2:
                path, imgs = data
                labels = None
            else:
                raise ValueError(f"Expected data with 2 or 3 elements, got {len(data)}.")

            imgs = imgs.to(self.device, non_blocking=True)
            with torch.no_grad():
                return_all = self.model(imgs)
                if isinstance(return_all, tuple):
                    logits = return_all[-1]
                else:
                    logits = return_all

                preds = logits.argmax(dim=-1).cpu().numpy()

            if labels is not None:
                labels = labels.cpu().numpy()
                for p, pred, label in zip(path, preds, labels):
                    records.append([p, int(pred), int(label)])
            else:
                for p, pred in zip(path, preds):
                    records.append([p, int(pred)])

        # Set up columns
        if labels is not None:
            df = pd.DataFrame(records, columns=["path", "pred", "label"])
        else:
            df = pd.DataFrame(records, columns=["path", "pred"])

        df.to_csv(f"{output_filename}.csv", index=False)
        return df
    
    def get_features(self, output_filename):
        # return crpn features (256-dim)
        self.model.eval()
        records = []

        for step, data in enumerate(self.dataloader_val):
            if len(data) == 3:
                path, imgs, labels = data
            elif len(data) == 2:
                path, imgs = data
                labels = None
            else:
                raise ValueError(f"Expected data with 2 or 3 elements, got {len(data)}.")

            imgs = imgs.to(self.device, non_blocking=True)
            with torch.no_grad():
                features = self.model(imgs)[0]
                features = features.cpu().numpy()
            
            if labels is not None:
                labels = labels.cpu().numpy()
                for p, f, label in zip(path, features, labels):
                    # Flatten: [p, f0, f1, ..., f255, label]
                    record = [p] + list(f) + [int(label)]
                    records.append(record)
            else:
                for p, f in zip(path, features):
                    record = [p] + list(f) 
                    records.append(record)
        
        # Set up columns
        if labels is not None:
            feature_cols = [f"f{i}" for i in range(1, 257)]
            df = pd.DataFrame(records, columns=["path"] + feature_cols + ["label"])
        else:
            feature_cols = [f"f{i}" for i in range(1, 257)]
            df = pd.DataFrame(records, columns=["path"] + feature_cols)

        df.to_csv(f"{output_filename}.csv", index=False)
        return df



    

