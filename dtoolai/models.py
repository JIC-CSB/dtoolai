import torch.nn as nn
import torch.nn.functional as F

import torchvision.models


class GenNet(nn.Module):
    """Simple image classification model.

    Args:
        input_channels (int): Number of channels in input images.
        input_dim (dim): Dimension of each input image, images are therefore
            required to be dim x dim.
        n_outputs (int): Number of possible classes for images.

    """

    model_name = "simpleScalingCNN"
    categorical_output = True
    
    def __init__(self, input_channels=1, input_dim=28, n_outputs=10):
        super(GenNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        dim = (input_dim - (5 - 1)) // 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        dim = (dim - (5 - 1)) // 2
        self.fcdim = dim
        self.fc1 = nn.Linear(self.fcdim*self.fcdim*16, 84)
        self.fc2 = nn.Linear(84, n_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.fcdim*self.fcdim*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ResNet18Pretrained(nn.Module):
    """Pretrained ResNet18 model.

    Args:
        n_outputs (int): Number of possible classes for images.

    """

    model_name = "resnet18pretrained"
    categorical_output = True

    def __new__(cls, n_outputs):
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_outputs)
        model.model_name = cls.model_name
        model.categorical_output = cls.categorical_output

        return model
