import torch 
from torch.utils.data import Dataset
from constants import group_activities, num_features
from utils.data_utils import VideoAnnotation, splits, splits_for_firstTry
from collections import defaultdict
import os
from PIL import Image


# Dont use it, Use last one
class FeaturesDataset(Dataset):
    def __init__(self, features):
        self.data = []
        for video_id in features:
            for clip_id in features[video_id]:
                clip_activity = features[video_id][clip_id]["label"]
                clip_features = features[video_id][clip_id]["features"]
                frame_features = torch.split(clip_features, num_features)
                for frame_feature in frame_features:
                    self.data.append((frame_feature, clip_activity))
   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        image_features = self.data[idx][0]
        label = torch.tensor(group_activities[self.data[idx][1]])

        return image_features, label
    


# class NewfeatureData(Dataset):
#     def __init__(self, annotations, videos_dir, split: str, transform=None):
#         for split in splits_for_firstTry:
#             split_features = defaultdict(dict)

#             for video_id in splits_for_firstTry[split]:
#                 annotation = annotations[str(video_id)]
#                 for clip_id in annotation.clip_activities:
#                     clip_activity = annotation.clip_activities[clip_id]
#                     clip_features = []

#                     for frame_id, boxes in annotation.clip_annotations[clip_id].frame_annotations.items():
#                         image_path = os.path.join(videos_dir, str(video_id), clip_id, f"{frame_id}.jpg")
#                         image = Image.open(image_path)
#                         self.cropped_images = []

#                         for box in boxes:
#                             cropped_image = image.crop(
#                                 (
#                                     box.x,
#                                     box.y,
#                                     box.w,
#                                     box.h,
#                                 )
#                             )
#                             self.cropped_images.append(transform(cropped_image).unsqueeze(0))  # Add batch dimension
#                             # self.data.append(transform(cropped_image).unsqueeze(0))  # Add batch dimension
#                             print("start cropeed")

#             split_features[video_id][clip_id] = {"features": clip_features, "label": clip_activity}
#         print("end cropped")
#     def __len__(self):
#         return len(self.split_features)

#     def __getitem__(self, idx: int):
#         image_features = self.split_features[idx][0]
#         # label = torch.tensor(group_activities[self.split_features[idx][1]])
#         label = self.split_features[self.video_id][self.clip_id]["label"]
#         return image_features, label
    
# from torch.utils.data import Dataset
# from collections import defaultdict
# import os
# from PIL import Image
# import torch

# class NewfeatureData(Dataset):
#     def __init__(self, annotations, videos_dir, split: str, transform=None):
#         self.split_features = defaultdict(dict)
#         self.data_list = []  # Flattened index list
#         self.transform = transform

#         for video_id in splits_for_firstTry[split]:  # Only use the given split
#             annotation = annotations[str(video_id)]
            
#             for clip_id in annotation.clip_activities:
#                 clip_activity = annotation.clip_activities[clip_id]
#                 clip_features = []
                
#                 for frame_id, boxes in annotation.clip_annotations[clip_id].frame_annotations.items():
#                     image_path = os.path.join(videos_dir, str(video_id), clip_id, f"{frame_id}.jpg")
                    
#                     try:
#                         image = Image.open(image_path).convert("RGB")
#                     except Exception as e:
#                         print(f"Error loading image {image_path}: {e}")
#                         continue

#                     cropped_images = []  # Store all crops for this frame
#                     for box in boxes:
#                         cropped_image = image.crop((box.x, box.y, box.w, box.h))
#                         if self.transform:
#                             cropped_image = self.transform(cropped_image)
#                         cropped_images.append(cropped_image.unsqueeze(0))  # Add batch dimension
                    
#                     clip_features.append(torch.cat(cropped_images))  # Stack cropped images
            
#                 self.split_features[video_id][clip_id] = {"features": clip_features, "label": clip_activity}
#                 self.data_list.append((video_id, clip_id))  # Flattened index
     
#         print("Finished processing dataset.")

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx: int):
#         video_id, clip_id = self.data_list[idx]  # Retrieve correct index
#         image_features = self.split_features[video_id][clip_id]["features"]
#         label = torch.tensor(self.split_features[video_id][clip_id]["label"], dtype=torch.long)
#         return image_features, label






class NewfeatureData(Dataset):
    def __init__(self, annotations, videos_dir, split: str, transform=None):
        self.data_list = []  # Store only references, not actual data
        self.annotations = annotations  # Keep annotation reference, not features
        self.videos_dir = videos_dir
        self.transform = transform

        for video_id in splits[split]:  # Process only relevant split
            annotation = annotations[str(video_id)]
            for clip_id in annotation.clip_activities:
                self.data_list.append((video_id, clip_id))  # Store only IDs

        print("Finished indexing dataset.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        video_id, clip_id = self.data_list[idx]
        annotation = self.annotations[str(video_id)]
        clip_activity = annotation.clip_activities[clip_id]

        cropped_images = []
        for frame_id, boxes in annotation.clip_annotations[clip_id].frame_annotations.items():
            image_path = os.path.join(self.videos_dir, str(video_id), clip_id, f"{frame_id}.jpg")
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            for box in boxes:
                cropped_image = image.crop((box.x, box.y, box.w, box.h))
                if self.transform:
                    cropped_image = self.transform(cropped_image)
                cropped_images.append(cropped_image.unsqueeze(0))  # Add batch dimension

        # Stack only for the batch, not pre-store in memory
        image_features = torch.cat(cropped_images) if cropped_images else torch.empty(0)  
        label = torch.tensor(group_activities[clip_activity], dtype=torch.long)

        return image_features, label


from torch.nn.utils.rnn import pad_sequence

def custom_collate(batch):
    images, labels = zip(*batch)  # Unzip batch
    images = [torch.cat(list(img), dim=0) if isinstance(img, (list, tuple)) else img for img in images]
    images_padded = pad_sequence(images, batch_first=True, padding_value=0)  # Pad to the longest sequence
    labels = torch.stack(labels)  # Convert labels to tensor
    return images_padded, labels
