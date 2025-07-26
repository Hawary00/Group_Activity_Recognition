import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from constants import actions, num_features
from models.baseline3.model import Person_Activity_Classifier
from utils.data_utils import load_annotations, splits


device = "cuda" if torch.cuda.is_available() else "cpu"


videos_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_/videos"
annotations_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/volleyball_tracking_annotation"
features_dir = "/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets/features/"

annotations = load_annotations(videos_dir, annotations_dir)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

features_model = Person_Activity_Classifier(len(actions))
features_model.load_state_dict(torch.load("/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline2/model_20250309_005202_4", weights_only=True))
features_model = nn.Sequential(*list(features_model.children())[:-1]).to(device)
features_model.eval()

with torch.no_grad():
    for split in splits:
        split_features = defaultdict(dict)

        for video_id in splits[split]:
            annotation = annotations[str(video_id)]
            for clip_id in annotation.clip_activities:
                clip_activity = annotation.clip_activities[clip_id]
                clip_features = []

                for frame_id, boxes in annotation.clip_annotations[clip_id].frame_annotations.items():
                    image_path = os.path.join(videos_dir, str(video_id), clip_id, f"{frame_id}.jpg")
                    image = Image.open(image_path)
                    cropped_images = []

                    for box in boxes:
                        cropped_image = image.crop(
                            (
                                box.x,
                                box.y,
                                box.w,
                                box.h,
                            )
                        )
                        cropped_images.append(transform(cropped_image).unsqueeze(0))  # Add batch dimension

                    cropped_images = torch.cat(cropped_images).to(device)
                    dnn_repr = features_model(cropped_images)
                    dnn_repr = dnn_repr.view(1, cropped_images.size(0), -1)
                    max_pool = nn.AdaptiveMaxPool2d((1, num_features)) # [12, 2048] -> [1, 2048]
                    dnn_repr = max_pool(dnn_repr)
                    dnn_repr = torch.squeeze(dnn_repr)
                    clip_features.append(dnn_repr)

                clip_features = torch.cat(clip_features)
                split_features[video_id][clip_id] = {"features": clip_features, "label": clip_activity}

        pkl_file_path = os.path.join(features_dir, f"{split}_features.pkl")
        with open(pkl_file_path, "wb") as f:
            pickle.dump(split_features, f)

