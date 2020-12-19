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


# input images must be larger than 224x224
def loss_sum(output, target, fe: FeatureExtractor, perceptual_ratio=0.005):
    mse = nn.MSELoss()
    fe.eval()
    with torch.no_grad():
        output_features = fe(output)
        target_features = fe(target)

        loss = mse(output, target)
        for i in range(len(output_features)):
            loss += perceptual_ratio * \
                mse(output_features[i], target_features[i])
        return loss


class LossSum:
    """For computing perceptual loss."""

    def __init__(self, feature_extractor: FeatureExtractor, perceptual_ratio: float = 0.005):
        self.feature_extractor = feature_extractor
        self.perceptual_ratio = perceptual_ratio

    def __call__(self, output, target):
        mse = nn.MSELoss()
        self.feature_extractor.eval()
        # with torch.no_grad():
        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)
        loss = mse(output, target)
        for i in range(len(output_features)):
            loss += self.perceptual_ratio * \
                mse(output_features[i], target_features[i])
        return loss


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img1 = torch.rand((1, 3, 300, 300)).to(device)
    img2 = torch.rand((1, 3, 300, 300)).to(device)
    print(img1)
    fe = FeatureExtractor().to(device)
    criterion = LossSum(fe)
    print(criterion(img1, img2))
