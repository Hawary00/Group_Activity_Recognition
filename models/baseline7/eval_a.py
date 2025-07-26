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
from model import Person_Activity_Temporal_Classifer
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


saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline7/outputs/trained_model/a/model_20250723_141648_epoch5"
saved_model = Person_Activity_Temporal_Classifer(num_classes=9, hidden_size=512,
                                                    num_layers=1).to(device)
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.to(device)

save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline7/outputs/confusion_matrix_a.png"  # Change to your desired path

evalution = eval_model(model=saved_model, test_loader=test_loader, device=device,
                        class_names=person_activity_clases, save_path=save_path)
print(evalution)

'''
{'Test accuracy': 79.61854899027674, 
'f1_score': np.float64(0.7948606502497627), 'classificaton report': {'Waiting': {'precision': 0.46586910626319494, 'recall': 0.5756521739130435, 'f1-score': 0.514974718008557, 'support': 1150.0}, 'Setting': {'precision': 0.8056338028169014, 'recall': 0.7688172043010753, 'f1-score': 0.7867950481430537, 'support': 372.0}, 'Digging': {'precision': 0.5862068965517241, 'recall': 0.3095599393019727, 'f1-score': 0.4051638530287984, 'support': 659.0}, 'Falling': {'precision': 0.8586956521739131, 'recall': 0.8381962864721485, 'f1-score': 0.8483221476510067, 'support': 377.0}, 'Spiking': {'precision': 0.9230769230769231, 'recall': 0.6129032258064516, 'f1-score': 0.7366720516962844, 'support': 372.0}, 'Blocking': {'precision': 0.8167202572347267, 'recall': 0.8912280701754386, 'f1-score': 0.8523489932885906, 'support': 855.0}, 'Jumping': {'precision': 0.13333333333333333, 'recall': 0.20454545454545456, 'f1-score': 0.16143497757847533, 'support': 88.0}, 'Moving': {'precision': 0.6103500761035008, 'recall': 0.5931952662721893, 'f1-score': 0.6016504126031508, 'support': 1352.0}, 'Standing': {'precision': 0.8693582349171473, 'recall': 0.8777151307884278, 'f1-score': 0.8735166957961549, 'support': 10819.0}, 'accuracy': 0.7961854899027674, 'macro avg': {'precision': 0.6743604758301517, 'recall': 0.6302014168418002, 'f1-score': 0.6423198775326746, 'support': 16044.0}, 'weighted avg': {'precision': 0.7996558007767328, 'recall': 0.7961854899027674, 'f1-score': 0.7948606502497627, 'support': 16044.0}}}
'''