import torch
import torch.nn as nn


from torchvision.models import resnet50, ResNet50_Weights

# model = resnet50(weights=ResNet50_Weights.DEFAULT)

class b1_classifier(nn.Module):
    def __init__(self, num_classes):
        super(b1_classifier, self).__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)

        # # Freeze the first two layers
        # layers_to_freeze = list(self.resnet50.children())[:2]  # Get the first 2 layers
        # for layer in layers_to_freeze:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        # Freeze Layers (layer1)

        for layer in [self.resnet50.layer1]:
            for params in self.resnet50.layer1.parameters():
                params.requires_grad = False


        # Modefiy the final fully conected layer to match the number of our class
        self.resnet50.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features=num_classes)

    

    def forward(self, x):
        return self.resnet50(x)


# model =  b1_classifier(8)

# for param in model.parameters():
#     assert param.requires_grad, "Gradient updates are disabled!"

# Check trainable parameters
# trainable = [p for p in model.parameters() if p.requires_grad]
# print(f"Trainable parameters: {len(trainable)}")
# for name, param in model.named_parameters():
#     print(f"{name}: requires_grad = {param.requires_grad}")