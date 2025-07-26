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
from model import GroupTemporalClassifier
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from train_A_once_again import val_test_transformers, annotations, videos_dir
from constants import actions

group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
from utils.data_utils import load_annotations


val_test_transformers = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                                   # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225]), 
])

videos_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos"
annotations_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_tracking_annotation"
annotations = load_annotations(videos_dir, annotations_dir)

batch_size = 32

group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_clases)}

PROJECT_ROOT = r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets"
sys.path.append(PROJECT_ROOT)

test_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
    transform=val_test_transformers,
    crops=False,
    seq=True,

)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
print("len test data: ", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline4/outputs/trained_model/model_20250721_125421_epoch26"
saved_model = GroupTemporalClassifier(len(group_activity_clases),
                                      input_size=2048, hidden_size=512, num_layers=1).to(device)
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.to(device)

save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline4/outputs/confusion_matrix.png"  # Change to your desired path

evalution = eval_model(model=saved_model, test_loader=test_loader, device=device,
                        class_names=group_activity_clases, save_path=save_path)
print(evalution)


'''
{'Test accuracy': 16.903515332834704, 'f1_score': 0.048882846643898185,
 'classificaton report': 
{'r_set': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 192.0},
'r_spike': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 173.0}, 
'r-pass': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 210.0}, 
'r_winpoint': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 87.0}, 
'l_winpoint': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 102.0}, 
'l-pass': {'precision': 0.16903515332834704, 'recall': 1.0, 'f1-score': 0.2891874600127959, 'support': 226.0}, 
'l-spike': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 179.0}, 
'l_set': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 168.0}, 
'accuracy': 0.16903515332834704, 
'macro avg': {'precision': 0.02112939416604338, 'recall': 0.125, 'f1-score': 0.036148432501599485, 'support': 1337.0}, 
'weighted avg': {'precision': 0.028572883060737794, 'recall': 0.16903515332834704, 'f1-score': 0.048882846643898185, 'support': 1337.0}}}
'''