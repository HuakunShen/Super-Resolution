import torch
import torch.nn as nn
from torchvision import models


# input tensor shape: batch x 3 x 224 x 224
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # second and fifth pooling layer
        self.wanted_layers = [9, 36]
        
        self.vgg_feature = models.vgg19(pretrained=True).features
        for parameter in self.parameters():
            parameter.requires_grad = False
        
        self.sequence = [nn.Sequential(i) for i in self.vgg_feature]
    
    def forward(self, x):
        features = []
        
        output = self.sequence[0](x)
        for i in range(1, len(self.sequence)):
            output = self.sequence[i](output)
            if i in self.wanted_layers:
                features.append(output)
            
        return features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! loss !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# input images must be larger than 224x224
fe = FeatureExtractor().to(device)

def loss_sum(output, target, perceptual_ratio=0.005):
    mse = nn.MSELoss()
    output_features = fe(output)
    target_features = fe(target)
    
    loss = mse(output, target)
    for i in range(len(output_features)):
        loss += perceptual_ratio * mse(output_features[i], target_features[i])
    return loss