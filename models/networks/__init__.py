from .alexnet import AlexNet
from .resnet import resnet18
from .resnext import resnext101

networks = {
    "alexnet": AlexNet,
    "resnet18": resnet18,
    "resnext101": resnext101
}