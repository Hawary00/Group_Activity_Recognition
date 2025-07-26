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
from model import Group_Activity_Classifer_Temporal
from models.baseline3.model import Person_Activity_Classifier
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.data_utils import load_annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import collate_fn


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
    collate_fn=collate_fn

)
print("len test data: ", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path for model i need to evaluate
saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline6/outputs/trained_model/model_20250723_013053_epoch14"
# Load person model
saved_person_model_baseline3_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline2/model_20250309_005202_4" #make sure to select right path
person_model = Person_Activity_Classifier(num_classes=9)
person_model.load_state_dict(torch.load(saved_person_model_baseline3_path)) 
person_model.eval()  # Make sure you're not training it anymore

# Use in group model
saved_model = Group_Activity_Classifer_Temporal(num_classes=8,
                                        person_feature_extraction=person_model,
                                        hidden_size=512)
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.to(device)

# saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/trained_model/model_20250309_005202_4"
# saved_model = GroupActivityClassifier(len(person_activity_clases)).to(device)
# saved_model.load_state_dict(torch.load(saved_model_path))
# saved_model.to(device)

save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline6/outputs/confusion_matrix.png"  # Change to your desired path

evalution = eval_model(model=saved_model, test_loader=test_loader, device=device,
                        class_names=group_activity_clases, save_path=save_path)
print(evalution)

'''
{'Test accuracy': 66.04338070306656, 
'f1_score': np.float64(0.6462214357274076), 
'classificaton report': 
{'r_set': {'precision': 0.7409326424870466, 'recall': 0.7447916666666666, 'f1-score': 0.7428571428571429, 'support': 192.0}, 
'r_spike': {'precision': 0.7152317880794702, 'recall': 0.6242774566473989, 'f1-score': 0.6666666666666666, 'support': 173.0}, 
'r-pass': {'precision': 0.5889328063241107, 'recall': 0.7095238095238096, 'f1-score': 0.6436285097192225, 'support': 210.0}, 
'r_winpoint': {'precision': 0.4166666666666667, 'recall': 0.05747126436781609, 'f1-score': 0.10101010101010101, 'support': 87.0}, 
'l_winpoint': {'precision': 0.4945652173913043, 'recall': 0.8921568627450981, 'f1-score': 0.6363636363636364, 'support': 102.0}, 
'l-pass': {'precision': 0.6875, 'recall': 0.6814159292035398, 'f1-score': 0.6844444444444444, 'support': 226.0}, 
'l-spike': {'precision': 0.6626506024096386, 'recall': 0.6145251396648045, 'f1-score': 0.6376811594202898, 'support': 179.0}, 
'l_set': {'precision': 0.7987012987012987, 'recall': 0.7321428571428571, 'f1-score': 0.7639751552795031, 'support': 168.0}, 
'accuracy': 0.6604338070306657, 
'macro avg': 
{'precision': 0.6381476277574419, 'recall': 0.6320381232452488, 'f1-score': 0.6095783519701259, 'support': 1337.0}, 
'weighted avg': 
{'precision': 0.6615833838521922, 'recall': 0.6604338070306657, 'f1-score': 0.6462214357274076, 'support': 1337.0}}}
'''