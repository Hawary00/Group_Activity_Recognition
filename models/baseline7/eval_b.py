import sys
import os

# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)

import torch
from eval_utils.eval_metrics import f1_score
from eval_utils.eval_metrics import eval_model
from data.data_loader import Group_Activity_DataSet
from model import Group_Activity_Temporal_Classifer, Person_Activity_Temporal_Classifer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.data_utils import load_annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import group_collate_fn


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

group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_clases)}


test_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
    transform=test_transformers,
    crops=True,
    seq=True,
  

)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=group_collate_fn

)
print("len test data: ", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline7/outputs/trained_model/b/model_20250723_194053_epoch6"

# Load person model
saved_person_model_path ='/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline7/outputs/trained_model/a/model_20250723_141648_epoch5' ############### make sure to select right path.
person_model = Person_Activity_Temporal_Classifer(num_classes=9, hidden_size=512,
                                                  num_layers=1)
person_model.load_state_dict(torch.load(saved_person_model_path)) 
person_model.eval()  # Make sure you're not training it anymore

# Use in group model
saved_model = Group_Activity_Temporal_Classifer(person_model, num_classes=8,
                                                 hidden_size=512, num_layers=1)
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.to(device)

# saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/trained_model/model_20250309_005202_4"
# saved_model = GroupActivityClassifier(len(person_activity_clases)).to(device)
# saved_model.load_state_dict(torch.load(saved_model_path))
# saved_model.to(device)

save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/confusion_matrix_group.png"  # Change to your desired path

evalution = eval_model(model=saved_model, test_loader=test_loader, device=device,
                        class_names=group_activity_clases, save_path=save_path)
print(evalution)


'''
Test Accuracy: 83.84%
Confusion matrix saved to /mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/confusion_matrix_group.png
{'Test accuracy': 83.84442782348542, 'f1_score': np.float64(0.835001276273001), 'classificaton report': {'r_set': {'precision': 0.9207317073170732, 'recall': 0.7864583333333334, 'f1-score': 0.848314606741573, 'support': 192.0}, 'r_spike': {'precision': 0.888268156424581, 'recall': 0.9190751445086706, 'f1-score': 0.9034090909090909, 'support': 173.0}, 'r-pass': {'precision': 0.7926829268292683, 'recall': 0.9285714285714286, 'f1-score': 0.8552631578947368, 'support': 210.0}, 'r_winpoint': {'precision': 0.5256410256410257, 'recall': 0.47126436781609193, 'f1-score': 0.49696969696969695, 'support': 87.0}, 'l_winpoint': {'precision': 0.6746987951807228, 'recall': 0.5490196078431373, 'f1-score': 0.6054054054054054, 'support': 102.0}, 'l-pass': {'precision': 0.8647540983606558, 'recall': 0.9336283185840708, 'f1-score': 0.8978723404255319, 'support': 226.0}, 'l-spike': {'precision': 0.9111111111111111, 'recall': 0.9162011173184358, 'f1-score': 0.9136490250696379, 'support': 179.0}, 'l_set': {'precision': 0.8834355828220859, 'recall': 0.8571428571428571, 'f1-score': 0.8700906344410876, 'support': 168.0}, 'accuracy': 0.8384442782348541, 'macro avg': {'precision': 0.8076654254608155, 'recall': 0.7951701468897532, 'f1-score': 0.798871744732095, 'support': 1337.0}, 'weighted avg': {'precision': 0.8365032407420998, 'recall': 0.8384442782348541, 'f1-score': 0.835001276273001, 'support': 1337.0}}}
'''