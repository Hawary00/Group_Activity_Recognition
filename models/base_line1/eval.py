import sys
import os

# Automatically find the project root (assuming train.py is inside models/base_line1/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to sys.path
sys.path.append(project_root)
import torch
from eval_utils.eval_metrics import f1_score
# from train import test_loader
from eval_utils.eval_metrics import eval_model
# from train import device, test_loader, group_activity_clases
from model import b1_classifier
from data.data_loader import Group_Activity_DataSet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


PROJECT_ROOT = r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets"
sys.path.append(PROJECT_ROOT)


group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_clases)}
# # Define the model architecture
# model = b1_classifier(8)  # Replace with your actual model class
# model.load_state_dict(torch.load("model.pth", map_location=device))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transformers = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225])
])


test_dataset = Group_Activity_DataSet(
    videos_path= f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path= f"{PROJECT_ROOT}/annot_all.pkl",
    labels=group_activity_labels,
    split=[4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
    transform=test_transformers,

)


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_activity_project/models/base_line1/model_20250216_184456_4"
saved_model = b1_classifier(8)
saved_model.load_state_dict(torch.load(saved_model_path))
saved_model.to(device)

save_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_activity_project/models/base_line1/outputs/confusion_matrix.png"  # Change to your desired path

evalution = eval_model(model=saved_model, test_loader=test_loader, device=device,
                        class_names=group_activity_clases, save_path=save_path)
print(evalution)


# Output
"""
Test Accuracy: 78.91%
'f1_score': 0.7896756996102968, 

classificaton report': 
{'r_set': {'precision': 0.7863304578633046, 'recall': 0.6857638888888888, 'f1-score': 0.732612055641422, 'support': 1728.0},
'r_spike': {'precision': 0.8532792427315754, 'recall': 0.8105330764290302, 'f1-score': 0.8313570487483531, 'support': 1557.0},
'r-pass': {'precision': 0.7039080459770115, 'recall': 0.81005291005291, 'f1-score': 0.753259532595326, 'support': 1890.0},
'r_winpoint': {'precision': 0.9654088050314465, 'recall': 0.7841634738186463, 'f1-score': 0.8653981677237491, 'support': 783.0},
 'l_winpoint': {'precision': 0.8785340314136125, 'recall': 0.9139433551198257, 'f1-score': 0.8958889482114255, 'support': 918.0}, 
 'l-pass': {'precision': 0.6888068880688807, 'recall': 0.8259587020648967, 'f1-score': 0.7511737089201878, 'support': 2034.0}, 
 'l-spike': {'precision': 0.8849385908209437, 'recall': 0.8497827436374923, 'f1-score': 0.8670044331855605, 'support': 1611.0}, 
 'l_set': {'precision': 0.7837837837837838, 'recall': 0.6712962962962963, 'f1-score': 0.7231920199501247, 'support': 1512.0},
'accuracy': 0.7890800299177263,
 'macro avg': {'precision': 0.8181237307113198, 'recall': 0.7939368057884983, 'f1-score': 0.8024857393720186, 'support': 12033.0}, 
'weighted avg': {'precision': 0.7971312819264064, 'recall': 0.7890800299177263, 'f1-score': 0.7896756996102968, 'support': 12033.0}}}
"""