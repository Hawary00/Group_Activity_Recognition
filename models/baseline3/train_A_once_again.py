import torch
from torch import nn, optim

import sys
import os
# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)

from data.person_dataset import PersonDataset


from model import Person_Activity_Classifier
from utils.data_utils import load_annotations
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.train_utils import trian
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from constants import actions


videos_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos"
annotations_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_tracking_annotation"
annotations = load_annotations(videos_dir, annotations_dir)


# Step1: Data Augmentation and Transformations
train_transformers = transforms.Compose(
    [
    transforms.Resize(224),            # Resize shorter side to 256     
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
    transforms.RandomRotation(degrees=5),                   # Randomly rotate images within ±5 degrees
    transforms.ToTensor(),                                   # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225]),         # (mean and std are the same used during ResNet pre-training)
])

val_test_transformers = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                                   # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225]), 
])

batch_size = 32

train_dataset = PersonDataset(videos_dir, annotations, "train", transform=train_transformers)
print("len tarin data", len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataset = PersonDataset(videos_dir, annotations, "val", transform=val_test_transformers)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
print("len val data", len(val_dataset))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter("runs/baseline3A_{}".format(timestamp))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Person_Activity_Classifier(len(actions)).to(device)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

Epochs = 5


val_accuracy, avg_vloss, avg_loss, train_accuracy = trian(

    model,
    Epochs,
    val_dataloader,
    device, 
    loss_fun, 
    optimizer, 
    train_dataloader,
    writer = writer,
    use_AMP=False
    )


