import sys
import os

# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)

import torch
from eval_utils.eval_metrics import f1_score
from eval_utils.eval_metrics import eval_model
from data.features_dataset import NewfeatureData, custom_collate
from model import FeaturesClassifier, Person_Activity_Classifier
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from train_A_once_again import val_test_transformers, annotations, videos_dir
from constants import actions
from utils.data_utils import load_annotations
from constants import num_features, group_activities



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                                   # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225]), 
])

videos_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos"
annotations_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_tracking_annotation"
annotations = load_annotations(videos_dir, annotations_dir)

batch_size = 4
test_dataset = NewfeatureData(annotations, videos_dir, "test", transform)
test_loader = DataLoader(test_dataset, batch_size, num_workers=4, collate_fn=custom_collate, shuffle=False)
print("len test data: ", len(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



features_model = Person_Activity_Classifier(len(actions))
features_model.load_state_dict(torch.load("/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline2/model_20250309_005202_4", weights_only=True))

# model = FeaturesClassifier(features_model, num_features, len(group_activities)).to(device)

saved_model_path = "D:\Deep Learning. DR Mostafa\Group_Activity_Recognition\models\baseline3\outputs\trained_model\b\model_20250311_165849_9.pth"
saved_model = FeaturesClassifier(features_model, num_features, len(group_activities)).to(device)
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.to(device)

save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline3/outputs/confusion_matrix_basleine3_B.png"  # Change to your desired path

evalution = eval_model(model=saved_model, test_loader=test_loader, device=device,
                        class_names=group_activities, save_path=save_path)
print(evalution)


"""
Test Accuracy: 77.41%
Confusion matrix saved to /mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline2/outputs/confusion_matrix_basleine3_B.png
{'Test accuracy': 77.41211667913238, 
'f1_score': 0.7747911703679683, 
'classificaton report': 
{'r_spike': {'precision': 0.9308176100628931, 'recall': 0.8554913294797688, 'f1-score': 0.891566265060241, 'support': 173.0}, 
'r_set': {'precision': 0.7873563218390804, 'recall': 0.7135416666666666, 'f1-score': 0.7486338797814208, 'support': 192.0}, 
'r-pass': {'precision': 0.68359375, 'recall': 0.8333333333333334, 'f1-score': 0.7510729613733905, 'support': 210.0}, 
'r_winpoint': {'precision': 0.5384615384615384, 'recall': 0.40229885057471265, 'f1-score': 0.4605263157894737, 'support': 87.0}, 
'l-spike': {'precision': 0.9580838323353293, 'recall': 0.8938547486033519, 'f1-score': 0.9248554913294798, 'support': 179.0}, 
'l_set': {'precision': 0.8417721518987342, 'recall': 0.7916666666666666, 'f1-score': 0.8159509202453987, 'support': 168.0}, 
'l-pass': {'precision': 0.771551724137931, 'recall': 0.7920353982300885, 'f1-score': 0.7816593886462883, 'support': 226.0}, 
'l_winpoint': {'precision': 0.5396825396825397, 'recall': 0.6666666666666666, 'f1-score': 0.5964912280701754, 'support': 102.0}, 
'accuracy': 0.7741211667913238, 'macro avg': {'precision': 0.7564149335522558, 'recall': 0.743611082527657, 'f1-score': 0.7463445562869835, 'support': 1337.0}, 
'weighted avg': {'precision': 0.781554029835408, 'recall': 0.7741211667913238, 'f1-score': 0.7747911703679683, 'support': 1337.0}}}
"""