import torch
from torch import nn, optim

import sys
import os
# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)

from data.features_dataset import FeaturesDataset, NewfeatureData, custom_collate


from model import FeaturesClassifier, Person_Activity_Classifier
from utils.data_utils import load_annotations
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.train_utils import trian
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from constants import group_activities, num_features, actions
import pickle




transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

videos_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos"
annotations_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_tracking_annotation"

annotations = load_annotations(videos_dir, annotations_dir)

batch_size = 4

train_dataset = NewfeatureData(annotations, videos_dir, "train", transform)
print("len tarin data", len(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size, num_workers=4, collate_fn=custom_collate, shuffle=True)
val_dataset = NewfeatureData(annotations, videos_dir, "val", transform)
val_dataloader = DataLoader(val_dataset, batch_size, num_workers=4, collate_fn=custom_collate, shuffle=False)
print("len val data", len(val_dataset))

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter("runs/baseline3B_{}".format(timestamp))

# Define the directory where you want to store the model
save_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline3/outputs/trained_model/b/"
os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = FeaturesClassifier(num_features, len(group_activities)).to(device)

features_model = Person_Activity_Classifier(len(actions))
features_model.load_state_dict(torch.load("/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline2/model_20250309_005202_4", weights_only=True))


model = FeaturesClassifier(features_model, num_features, len(group_activities)).to(device)
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

Epochs = 10


val_accuracy, avg_vloss, avg_loss, train_accuracy = trian(

    model,
    Epochs,
    val_dataloader,
    device, 
    loss_fun, 
    optimizer, 
    train_dataloader,
    save_dir,
    writer = writer,
    use_AMP=False
    )















# features_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/features"

# with open(os.path.join(features_dir, "train_features.pkl"), "rb") as f:
#     train_features = pickle.load(f)
# with open(os.path.join(features_dir, "val_features.pkl"), "rb") as f:
#     val_features = pickle.load(f)

# batch_size = 64

# train_dataset = FeaturesDataset(train_features)
# print("len tarin data", len(train_dataset))
# train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
# val_dataset = FeaturesDataset(val_features)
# val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
# print("len val data", len(val_dataset))

# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter("runs/baseline3B_{}".format(timestamp))

# # Define the directory where you want to store the model
# save_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/trained_model/baseline3_B/"
# os.makedirs(save_dir, exist_ok=True)




# device = "cuda" if torch.cuda.is_available() else "cpu"
# # model = FeaturesClassifier(num_features, len(group_activities)).to(device)

# features_model = Person_Activity_Classifier(len(actions))
# features_model.load_state_dict(torch.load("/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline2/model_20250309_005202_4", weights_only=True))
# features_model = nn.Sequential(*list(features_model.children())[:-1]).to(device)

# model = FeaturesClassifier( num_features, len(group_activities)).to(device)
# loss_fun = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Epochs = 20


# val_accuracy, avg_vloss, avg_loss, train_accuracy = trian(

#     model,
#     Epochs,
#     val_dataloader,
#     device, 
#     loss_fun, 
#     optimizer, 
#     train_dataloader,
#     save_dir,
#     writer = writer,
#     use_AMP=False
#     )


