# CRPN

CRPN is a Python package designed for exploring emotion perception.

Data Folder Organization Guide
--------
**1. Labeled Dataset**

allowed category list: 'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'

If your dataset has labels (emotion categories), organize the images into subfolders where each subfolder represents an emotion label. The folder structure should look like this:

```text
dataset/
├── happy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── sad/
│   ├── img3.jpg
│   ├── img4.jpg
│   └── ...
├── anger/
│   ├── img5.jpg
│   ├── img6.jpg
│   └── ...
└── neutral/
    ├── img7.jpg
    ├── img8.jpg
    └── ...
```

**2. Unlabeled Dataset**

If your dataset does not have labels (i.e., images are not categorized by emotion), place all images into a single folder. The folder structure should look like this:

```text
dataset/
├── img1.jpg
├── img2.jpg
├── img3.jpg
├── img4.jpg
├── ...
```

Requirements
--------
See requirements.txt for required python libraries.
```
pip install requirements.txt
```

We conduct all experiments with the PyTorch toolbox and four NVIDIA GeForce RTX 4090 GPUs.

Usage
-----
**1. Installation**

Clone the repository to your local machine
```
git clone https://github.com/hanluyt/CRPN.git
```

**2. Download Pre-trained models and put them in the CRPN/backbone_weight**

We use the ResNet-50 pre-trained on VGGface2 as the backbone for the emotion and non-emotion encoder.
```
wget https://drive.google.com/file/d/1i_aYzKsvnnPcwm3kV0_Y8F9ofoVxSm5x/view?usp=drive_link
```
For the percept encoder, we use ResNet-18 pre-trained on Ms-Celeb-1M as the backbone.
```
wget https://drive.google.com/file/d/10NVjrvhacFlHcdW88Kn_PspYUDjXV4xQ/view?usp=drive_link
```

For the final CRPN, the parameters were saved in resnet18_percept_7100.pth.
```
wget https://drive.google.com/file/d/1IgzE0K1nI3EXXjo_eyOciWd1XlJfCgX6/view?usp=drive_link
```

**3. Basic Usage**

(1) Import the CRPN package in your Python script:

```
from CRPN import CRPNProbing
```

(2) Initialize the tool:

```
crpn = CRPNProbing(your_data_path)
```

(3) Calculate Emotion Recognition Accuracy Using `crpn.test_acc()` (only for labeled dataset)

(4) Predict Emotion Categories Using `crpn.evaluate(output_filename)`

output_filename (str): The name of the file where the predictions will be saved.

Predictions: {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

(5) Extract 256-Dimensional Features with `crpn.get_features(output_filename)`

