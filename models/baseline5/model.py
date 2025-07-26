import torch
import torch.nn as nn
import torchvision.models as models


class PersonActivityTemporalClassifier(nn.Module):
    """
    Classifier for person-level activity using ResNet50 + LSTM.
    """
    def __init__(self, num_classes: int, hidden_size: int = 512, num_layers: int = 1):
        super(PersonActivityTemporalClassifier, self).__init__()

        # ResNet50 feature extractor (remove final FC)
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # Output: [B, 2048, 1, 1]

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, T, C, H, W] - Batch, BBoxes, Frames, Channels, Height, Width
        """
        B, N, T, C, H, W = x.shape
        x = x.view(B * N * T, C, H, W)  # Merge for ResNet
        x = self.feature_extractor(x)  # → [B*N*T, 2048, 1, 1]
        x = x.view(B * N, T, -1)       # → [B*N, T, 2048]

        lstm_out, _ = self.lstm(x)     # → [B*N, T, hidden]
        last_hidden = lstm_out[:, -1, :]  # → [B*N, hidden]

        return self.classifier(last_hidden)  # → [B*N, num_classes]


class Group_Activity_Classifer(nn.Module):
    def __init__(self, person_feature_extraction, num_classes):
        super(Group_Activity_Classifer, self).__init__()

        self.resnet50 = person_feature_extraction.feature_extractor
        self.lstm = person_feature_extraction.lstm

        for module in [self.resnet50,  self.lstm]:
            for param in module.parameters():
                param.requires_grad = False
                
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [Batch, 12, hidden_size] -> [Batch, 1, 2048]
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes), 
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape # seq => frames
        x = x.view(b*bb*seq, c, h, w) # (b * bb * seq, c, h, w)
        x1 = self.resnet50(x) # (batch * bbox * seq, 2048, 1 , 1)

        x1 = x1.view(b*bb, seq, -1) # (batch * bbox, seq, 2048)
        x2, (h , c) = self.lstm(x1) # (batch * bbox, seq, hidden_size)

        x = torch.cat([x1, x2], dim=2) # Concat the Resnet50 representation and LSTM layer for every  
        x = x.contiguous()             # person and pool over all people in a scene.
        x = x[:, -1, :]                # (batch * bbox, hidden_size)
        
        x = x.view(b, bb, -1) # (batch , bbox, hidden_size)
        x = self.pool(x) # (batch, 1, 2048)
        x = x.squeeze(dim=1) # (batch, 2048)

        x = self.fc(x) # (batch, num_class)
        return x

# class GroupActivityClassifier(nn.Module):
#     """
#     Group activity classifier using frozen person-level features from ResNet + LSTM.
#     """
#     def __init__(self, person_model: PersonActivityTemporalClassifier, num_classes: int):
#         super(GroupActivityClassifier, self).__init__()

#         # Share feature extractor and LSTM from person model (frozen)
#         self.feature_extractor = person_model.feature_extractor
#         self.lstm = person_model.lstm
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
#         for param in self.lstm.parameters():
#             param.requires_grad = False

#         # Pool across players (N = 12 usually)
#         self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # → [B, 1, 2048]

#         self.classifier = nn.Sequential(
#             nn.Linear(2048, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: [B, N, T, C, H, W] - Batch, Players, Time, Channels, Height, Width
#         """
#         B, N, T, C, H, W = x.shape
#         x = x.view(B * N * T, C, H, W)
#         features = self.feature_extractor(x)      # → [B*N*T, 2048, 1, 1]
#         features = features.view(B * N, T, -1)    # → [B*N, T, 2048]

#         _, (h_n, _) = self.lstm(features)         # → [num_layers, B*N, hidden]
#         lstm_out = h_n[-1]                        # Use last layer hidden: [B*N, hidden]

#         lstm_out = lstm_out.view(B, N, -1)        # → [B, N, hidden]
#         # Pool across players 
#         lstm_out = torch.max(lstm_out, dim=1).values  # or torch.mean(...)

#         return self.classifier(lstm_out)          # → [B, num_classes]



def person_collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame and selecting the last frame label. 
    """
    clips, labels = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []
    padded_labels = []

    for clip, label in zip(clips, labels) :
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)))
            
            clip = torch.cat((clip, clip_padding), dim=0)
            label = torch.cat((label, label_padding), dim=0)
            
        padded_clips.append(clip)
        padded_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_labels = torch.stack(padded_labels)
    
    padded_labels = padded_labels[:, :, -1, :]  # utils the label of last frame for each player
    b, bb, num_class = padded_labels.shape # batch, bbox, num_clases
    padded_labels = padded_labels.view(b*bb, num_class)

    return padded_clips, padded_labels



def group_collate_fn(batch):
    """
    collate function to pad bounding boxes to 12 per frame and selecting the last frame label. 
    """
    clips, labels = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []

    for clip in clips:
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            clip = torch.cat((clip, clip_padding), dim=0)
    
        padded_clips.append(clip)
       
    padded_clips = torch.stack(padded_clips)
    labels = torch.stack(labels)
    
    labels = labels[:,-1, :] # utils the label of last frame
    
    return padded_clips, labels
