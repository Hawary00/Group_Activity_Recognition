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
from model import Group_Activity_Classifer, PersonActivityTemporalClassifier
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

saved_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/trained_model/group/model_20250722_180734_epoch11"
# Load person model
saved_person_model_path = "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline5/outputs/trained_model/model_20250721_234311_epoch3"############### make sure to select right path
person_model = PersonActivityTemporalClassifier(num_classes=9, hidden_size=512)
person_model.load_state_dict(torch.load(saved_person_model_path)) 
person_model.eval()  # Make sure you're not training it anymore

# Use in group model
saved_model = Group_Activity_Classifer(person_model, num_classes=8)
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


''''
{'Test accuracy': 59.31189229618548, 
'f1_score': np.float64(0.586296718777446), 
'classificaton report': 
{'r_set': {'precision': 0.7439024390243902, 'recall': 0.6354166666666666, 'f1-score': 0.6853932584269663, 'support': 192.0}, 
'r_spike': {'precision': 0.5531914893617021, 'recall': 0.7514450867052023, 'f1-score': 0.6372549019607843, 'support': 173.0}, 
'r-pass': {'precision': 0.5714285714285714, 'recall': 0.5142857142857142, 'f1-score': 0.5413533834586466, 'support': 210.0}, 
'r_winpoint': {'precision': 0.5094339622641509, 'recall': 0.3103448275862069, 'f1-score': 0.38571428571428573, 'support': 87.0}, 
'l_winpoint': {'precision': 0.5227272727272727, 'recall': 0.45098039215686275, 'f1-score': 0.4842105263157895, 'support': 102.0}, 
'l-pass': {'precision': 0.5102040816326531, 'recall': 0.7743362831858407, 'f1-score': 0.6151142355008787, 'support': 226.0}, 
'l-spike': {'precision': 0.7346938775510204, 'recall': 0.4022346368715084, 'f1-score': 0.51985559566787, 'support': 179.0}, 
'l_set': {'precision': 0.6766467065868264, 'recall': 0.6726190476190477, 'f1-score': 0.6746268656716418, 'support': 168.0}, 
'accuracy': 0.5931189229618549, 
'macro avg': 
{'precision': 0.6027785500720734, 'recall': 0.5639578318846312, 'f1-score': 0.5679403815896078, 'support': 1337.0}, 
'weighted avg': 
{'precision': 0.6108177305344947, 'recall': 0.5931189229618549, 'f1-score': 0.586296718777446, 'support': 1337.0}}}
'''