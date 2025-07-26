import torch
import torch.nn as nn
import torchvision.models as models

class Person_Activity_Temporal_Classifer(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(Person_Activity_Temporal_Classifer, self).__init__()
        
        self.resnet50 = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )

        self.layer_norm = nn.LayerNorm(2048)
        
        self.lstm_1 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape # seq => frames
        x = x.view(b*bb*seq, c, h, w) # (batch * bbox * seq, c, h, w)
        x = self.resnet50(x) # (batch * bbox * seq, 2048, 1 , 1)

        x = x.view(b*bb, seq, -1) # (batch * bbox, seq, 2048)
        x = self.layer_norm(x)
        x, (h , c) = self.lstm_1(x) # (batch * bbox, seq, hidden_size)

        x = x[:, -1, :] # (batch * bbox, hidden_size)
        x = self.fc(x) # (batch * bbox, num_class)  
        
        return x

class Group_Activity_Temporal_Classifer(nn.Module):
    def __init__(self, person_feature_extraction, hidden_size, num_layers, num_classes):
        super(Group_Activity_Temporal_Classifer, self).__init__()

        self.resnet50 = person_feature_extraction.resnet50
        self.lstm_1 = person_feature_extraction.lstm_1

        for module in [self.resnet50, self.lstm_1]:
            for param in module.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        # Layer normalization for better stability (will be shared through the network)
        self.layer_norm = nn.LayerNorm(2048)
        
        self.lstm_2 = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x.shape => batch, bbox, frames, channals , hight, width
        b, bb, seq, c, h, w = x.shape # seq => frames
        x = x.view(b*bb*seq, c, h, w) # (b *bb *seq, c, h, w)
        x1 = self.resnet50(x) # (batch * bbox * seq, 2048, 1 , 1)

        x1 = x1.view(b*bb, seq, -1) # (batch * bbox, seq, 2048)
        x1 = self.layer_norm(x1) # (batch * bbox, seq, 2048)
        x2, (h_1 , c_1) = self.lstm_1(x1) # (batch * bbox, seq, hidden_size)

        x = torch.cat([x1, x2], dim=2) # Concat the Resnet50 representation and LSTM layer for every  
        x = x.contiguous()             # person and pool over all people in a scene (same as paper).
       
        x = x.view(b*seq, bb, -1) # (batch * seq, bbox, hidden_size)
        x = self.pool(x) # (batch * seq, 1, 2048)
       
        x = x.view(b, seq, -1) # (batch, seq, 2048)
        x = self.layer_norm(x)
        x, (h_2 , c_2) = self.lstm_2(x) # (batch, seq, hidden_size)

        x = x[:, -1, :] # (batch, hidden_size)
        x = self.fc(x)  # (batch, num_class)
        return x

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
