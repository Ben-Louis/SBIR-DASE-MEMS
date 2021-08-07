import torch
import torch.nn as nn

class SEModule(nn.Module):

    def __init__(self, channels:int, reduction:int=16, domain_aware:bool=False):
        super(SEModule, self).__init__()

        self.domain_aware = domain_aware
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)

        # intermediate activation layer
        self.activate = nn.Sigmoid()

        # add the control unit
        input_channel = channels // reduction + int(self.domain_aware)
        self.fc2 = nn.Conv2d(input_channel, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, *args):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.activate(x)
        x = torch.cat((x, args[0].view(-1, 1, 1, 1)), dim=1) if self.domain_aware else x
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

    def load_state_dict(self, state_dict):
        super(SEModule, self).load_state_dict(state_dict, strict=False)

