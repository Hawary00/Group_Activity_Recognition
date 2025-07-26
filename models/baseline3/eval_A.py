import sys
import os

# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)

import torch
from eval_utils.eval_metrics import f1_score
from eval_utils.eval_metrics import eval_model
from data.person_dataset import PersonDataset
from model import Person_Activity_Classifier
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from train_A_once_again import val_test_transformers, annotations, videos_dir
from constants import actions
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
test_dataset = PersonDataset(videos_dir, annotations, "test", transform=val_test_transformers)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
print("len test data: ", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



saved_model_path = "D:\Deep Learning. DR Mostafa\Group_Activity_Recognition\models\baseline3\outputs\trained_model\a\model_20250309_005202_4"
saved_model = Person_Activity_Classifier(len(actions)).to(device)
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.to(device)

save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline3/outputs/confusion_matrix.png"  # Change to your desired path

evalution = eval_model(model=saved_model, test_loader=test_loader, device=device,
                        class_names=actions, save_path=save_path)
print(evalution)



"""
len test data:  127472
Model set to eval mode: True
Test Accuracy: 76.06%
Confusion matrix saved to /mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline2/outputs/confusion_matrix.png
{'Test accuracy': 76.06376302246768, 'f1_score': 0.7505115842454524, 
'classificaton report': 
{'waiting': {'precision': 0.5887615320100643, 'recall': 0.253125, 'f1-score': 0.3540388333193242, 'support': 8320.0}, 
'setting': {'precision': 0.7336417713152676, 'recall': 0.37298387096774194, 'f1-score': 0.4945422143016262, 'support': 2976.0}, 
'digging': {'precision': 0.3469352318110783, 'recall': 0.37898330804248864, 'f1-score': 0.36225183573565406, 'support': 5272.0}, 
'falling': {'precision': 0.7800729040097205, 'recall': 0.6385941644562334, 'f1-score': 0.7022789425706473, 'support': 3016.0}, 
'spiking': {'precision': 0.6381066506890354, 'recall': 0.7157258064516129, 'f1-score': 0.6746911624960406, 'support': 2976.0}, 
'blocking': {'precision': 0.8213474640423921, 'recall': 0.7931286549707602, 'f1-score': 0.8069914466344366, 'support': 6840.0}, 
'jumping': {'precision': 0.11442786069651742, 'recall': 0.032670454545454544, 'f1-score': 0.05082872928176796, 'support': 704.0}, 
'moving': {'precision': 0.4810823948796889, 'recall': 0.5490014792899408, 'f1-score': 0.512802798048275, 'support': 10816.0}, 
'standing': {'precision': 0.8324042458027425, 'recall': 0.8815971901284777, 'f1-score': 0.8562947833844876, 'support': 86552.0}, 
'accuracy': 0.7606376302246768, 'macro avg': {'precision': 0.5929755616951673, 'recall': 0.5128677698725234, 'f1-score': 0.5349689717524733, 'support': 127472.0}, 
'weighted avg': {'precision': 0.7539755789861918, 'recall': 0.7606376302246768, 'f1-score': 0.7505115842454524, 'support': 127472.0}}}
"""