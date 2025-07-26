import sys
import os

# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)

import torch
from eval_utils.eval_metrics import f1_score
from eval_utils.eval_metrics import eval_model
from data.data_loader import Person_Activity_DataSet
from model import PersonActivityTemporalClassifier
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.data_utils import load_annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import person_collate_fn


person_activity_clases = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
person_activity_labels = {class_name.lower():i for i, class_name in enumerate(person_activity_clases)}


test_transformers = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

videos_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos"
annotations_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_tracking_annotation"
annotations = load_annotations(videos_dir, annotations_dir)

batch_size = 2

PROJECT_ROOT = r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets"
sys.path.append(PROJECT_ROOT)

test_dataset = Person_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=person_activity_labels,
    split=[4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
    transform=test_transformers,
    seq=True,
  

)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=person_collate_fn

)
print("len test data: ", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/trained_model/model_20250721_234311_epoch3"
saved_model = PersonActivityTemporalClassifier(num_classes=9).to(device)
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.to(device)

save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/confusion_matrix.png"  # Change to your desired path

evalution = eval_model(model=saved_model, test_loader=test_loader, device=device,
                        class_names=person_activity_clases, save_path=save_path)
print(evalution)

