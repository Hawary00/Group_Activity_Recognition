import torch
import torch.nn as nn
import torchvision.models as models

class GroupTemporalClassifier(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(GroupTemporalClassifier, self).__init__()

        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extraction = nn.Sequential(*list(resnet50.children())[:-1])
        
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers=num_layers, batch_first=True)

        self.fc = nn.Sequential(
        nn.Linear(input_size + hidden_size , 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )

    def forward(self, x):
        # Input shape: (batch, 9, 3, 244, 244)
        b, seq, c, h, w = x.shape
        x1 = x.view(b * seq, c, h, w)  # (batch * 9, 3, 244, 244)

        x1 = self.feature_extraction(x1)  # (batch * 9, 2048, 1, 1)
        x1 = x1.view(b, seq, -1)  # (batch, 9, 2048)
        x2, (h, c) = self.lstm(x1)  # x: (batch, 9 , hidden_size)

        x = torch.cat([x1, x2], dim=2) # Concat the Resnet50 representation of the frame and Lstm temporal representation 
                                        
        x = x[:, -1, :]  # (batch, hidden_size + 2048)
        x = self.fc(x)  # (64, num_classes)

        return x
