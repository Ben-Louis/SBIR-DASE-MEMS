import torch.nn as nn
from .se_module import SEModule


class AlexNet(nn.Module):

    def __init__(self, f_dim=512, *args, **kwargs):
        super(AlexNet, self).__init__()
        channels = [64, 192, 384, 256, 256, 4096]

        # remake feature extraction module
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=11, stride=3, padding=5),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.dase1 = SEModule(channels[0], True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.dase2 = SEModule(channels[1], True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.dase3 = SEModule(channels[2], True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.dase4 = SEModule(channels[3], True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(channels[4] * 7 * 7, channels[5]),
                                        nn.ReLU(inplace=True),
                                        nn.BatchNorm1d(channels[5], eps=2e-5, affine=False),
                                        nn.Dropout(),
                                        nn.Linear(channels[5], f_dim))


    def forward(self, x, domain=None):
        for i in range(1, 5):
            x = self.__getattr__(f"conv{i}")(x)
            x = self.__getattr__(f"dase{i}")(x, domain)

        x = self.avgpool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x
