import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Person_Activity_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Person_Activity_Classifier, self).__init__()

        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        # for layer in [self.resnet50.layer1]:
        #     for params in self.resnet50.layer1.parameters():
        #         params.requires_grad = False


        # Modefiy the final fully conected layer to match the number of our class
        self.resnet50.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features=num_classes)

    def forward(self, x):
        return self.resnet50(x)


class FeaturesClassifier(nn.Module):
    def __init__(self, Person_Activity_Classifier, input_dim, num_classes):
        super().__init__()
        #  Extract feature layers of ResNet50 (excluding final FC layer)
        # self.fc_in_features = Person_Activity_Classifier.in_features
        self.feature_extraction = nn.Sequential(*list(Person_Activity_Classifier.resnet50.children())[:-1])

        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))  # [12, 2048] -> [1, 2048]
        
        self.fc = nn.Sequential(
             nn.Linear(input_dim, 1024),
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
        b, bb, c, h, w = x.shape # batch, bbox, channals, hight, width
        x = x.view(b*bb, c, h, w) # [b*bb, c, h, w]
        x = self.feature_extraction(x) # [b*bb, 2048, 1, 1] 

        x = x.view(b, bb, -1) # (b, bb, 2048)
        x = self.pool(x) # [b, 1, 2048] 
        
        x = x.squeeze(dim=1) # [b, 2048]
        x = self.fc(x) # [b, num_classes] 
        return x
    



















