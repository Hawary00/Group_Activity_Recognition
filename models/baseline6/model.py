import torch
import torch.nn as nn


class Group_Activity_Classifer_Temporal(nn.Module):
    def __init__(self, person_feature_extraction, hidden_size, num_classes):
        super(Group_Activity_Classifer_Temporal, self).__init__()
   
        self.feature_extraction = nn.Sequential(*list(person_feature_extraction.resnet50.children())[:-1])

        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [12, 2048] -> [1, 2048]
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            batch_first=True
        ) 

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        b, bb, seq , c, h, w = x.shape # batch, bbox, seq, channals, hight, width
        x = x.view(b*bb*seq, c, h, w) # [b*bb*seq, c, h, w]
        x = self.feature_extraction(x) # [b*bb*seq, 2048, 1, 1] 

        x = x.view(b*seq, bb, -1) # (b*seq, bb, 2048)
        x = self.pool(x) # [b*seq, 1, 2048] 
        
        x = x.squeeze(dim=1) # [b*seq, 2048]
        x = x.view(b, seq, -1) # [b, seq, 2048]

        x, (h, c) = self.lstm(x) # [b, seq, hidden]
        x = x [:, -1, :] # [b, hidden] 

        x = self.fc(x) # [b, num_classes] 
        return x 

def collate_fn(batch):
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