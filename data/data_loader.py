from torch.utils.data import  Dataset
import torch
from PIL import Image
from pathlib import Path
import pickle
import cv2
import torchvision.transforms as transforms
from data  import boxinfo
# dataset_root = "/teamspace/studios/this_studio/Group-Activity-Recognition/data"
# annot_path = f"{dataset_root}/annot_all.pkl"
# videos_path = f"{dataset_root}/videos"

group_activity_clases = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]
group_activity_labels = {class_name:i for i, class_name in enumerate(group_activity_clases)}


class Group_Activity_DataSet(Dataset):
    def __init__(self, videos_path: str, annot_path: str, labels: dict = {},
                  split: list = [], transform=None):

        """
        Args:
            video_path: path to video frames
            annot_path: path to annotations fle
            split: List of clip IDs to use
            labels: Group activity labels dictionary
            transform: optional transform to apply
        """

        self.videos_path = Path(videos_path)
        self.labels =  labels
        self.transform = transform
        

        # Load annotation and store only metadata
        with open(annot_path, 'rb') as f:
            videos_annot = pickle.load(f) 

        
        # Create indexx mapping for efficient retrieval
        self.data = []
        for clip_id in split:
            clip_dirs = videos_annot[str(clip_id)]

            for clip_dir in clip_dirs.keys():
                frames_data = clip_dirs[str(clip_dir)]['frame_boxes_dct']
                category = clip_dirs[str(clip_dir)]['category']
                dir_frames = list(clip_dirs[str(clip_dir)]['frame_boxes_dct'].items())

                # return a full image of target frame with its group label (frame, tensor(8))
                for frame_id, boxes in dir_frames:

                    self.data.append({
                            'frame_path': f"{videos_path}/{str(clip_id)}/{str(clip_dir)}/{frame_id}.jpg",
                            'category': category,
                        })


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        sample = self.data[idx]
        num_class = len(self.labels)
        label = torch.zeros(num_class)
        label[self.labels[sample['category']]] = 1

        frame = cv2.imread(sample['frame_path'])

        if self.transform:
            frame = Image.fromarray(frame)
            frame = self.transform(frame)



        return frame, label


    #     # Class mapping: 9 activity classes, 
    #     # and converting activity class strings to numeric labels
    #     self.class_mapping = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3,
    #                           'l_winpoint': 4, 'l-pass': 5, 'l-spike': 6, 'l_set':7} 
        

    # def __len___(self):
    #     return len(self.labels)


    
    # def get_item(self, idx):
    #     image = Image.open(self.image_path[idx]).convert('RGB')
    #     label = self.labels[idx]

    #     if self.transform:
    #         image = self.transform(image)

    #     return image, label
    


PROJECT_ROOT = r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets"


# Step 3: Data Augmentation and Transformations
train_transforms = transforms.Compose([
    transforms.Resize(256),            # Resize shorter side to 256     
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
    transforms.RandomRotation(degrees=5),                   # Randomly rotate images within ±5 degrees
    transforms.ToTensor(),                                   # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225]),         # (mean and std are the same used during ResNet pre-training)
])
train_dataset = Group_Activity_DataSet(
    videos_path=f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path=f"{PROJECT_ROOT}/annot_all.pkl",
    split=[1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    labels=group_activity_labels, 
    transform=train_transforms
)


# print(len(train_dataset))  # Should print the number of data samples
# sample = train_dataset[0]  # Try loading the first sample
# print(sample)              # Should print the sample data (e.g., video frames, label, etc.)
