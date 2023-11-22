import torch.nn as nn
import torch
from torchvision import transforms

class e2e_net(nn.Module):
    
    def __init__(self) -> None:
        super(e2e_net, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.conv_layer = nn.Sequential(
                                        nn.Conv2d(3, 24, 5, stride=2),
                                        nn.ELU(),
                                        nn.Conv2d(24, 36, 5, stride=2),
                                        nn.ELU(),
                                        nn.Conv2d(36, 48, 5, stride=2),
                                        nn.ELU(), 
                                        nn.Conv2d(48, 64, 3),
                                        nn.ELU(),
                                        nn.Conv2d(64, 64, 3)
        )
        self.pred_net = MLP()
        self.criterion = nn.SmoothL1Loss()


    def forward(self, input, angle_label):
        
        input = input.view(input.size(0), 3, 120, 160)
        # Normalize input tensor
        input = self.norm(input)
        
        # Generate feature map
        feature_map = self.conv_layer(input)
        feature_map = feature_map.view(feature_map.size(0), -1)

        # predict the angle
        angle_pred = self.pred_net(feature_map)
        
        # Generate the loss 
        loss = self.criterion(angle_pred, angle_label)

        return loss



    def validate(self, input):

        input = input.view(input.size(0), 3, 120, 160)
        # Normalize input tensor
        input = self.norm(input)
        
        # Conv the input img and predict angle
        feature_map = self.conv_layer(input)        
        feature_map = feature_map.view(feature_map.size(0), -1)
        # Predict the angle
        angle_pred = self.pred_net(feature_map)

        return angle_pred




class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_features=64 * 8 * 13, out_features=100)
        self.elayer1 = nn.ELU()
        self.layer2 = nn.Linear(in_features=100, out_features=50)
        self.elayer2 = nn.ELU()
        self.layer3 = nn.Linear(in_features=50, out_features=10)
        self.layer4 = nn.Linear(in_features=10, out_features=1)
    

    def forward(self, input):
        output = self.layer1(input)
        output = self.elayer1(output)
        output = self.layer2(output)
        output = self.elayer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        return output
