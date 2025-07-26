from torch.utils.data import Dataset
from utils.data_utils import ViedeoAnnotation, splits
from PIL import Image
import os
import torch
from constants import group_activities


class GroupDataset(Dataset):
    def __init__(self, videios_dir: str, annotations: dict[str, ViedeoAnnotation], split: str, transform=None):
        for video_id in splits[split]:
            # Extracting Clip Activities
            annotation = annotations[str(video_id)]
            for clip_id in annotation.clip_activities:
                # Retrives activity labels for each clip
                clip_activity = annotation.clip_activities[clip_id]
                # Sorting Frame path and labels
                for frame_id in annotation.clip_annotations[clip_id].frame_anntations:
                    image_path = os.path.join(videios_dir, str(video_id), clip_id, f"{frame_id}.jpg")
                    self.data.append((image_path, clip_activity))


        self.transform =transform
    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        example = self.data[idx]
        image = Image.open(example[0])
        label = torch.tensor(group_activities[example[1]])

        if self.transform:
            image = self.transform(image)

        return image, label
