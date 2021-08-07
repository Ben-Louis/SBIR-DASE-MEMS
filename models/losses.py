import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, dim_feature, num_classes, margin, *args, **kwargs):
        super(MarginLoss, self).__init__()

        self.num_classes = num_classes
        self.dim_feature = dim_feature
        self.margin = margin
        self.centers = nn.Parameter(F.normalize(torch.randn(self.num_classes, self.dim_feature), dim=1))
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.step = kwargs.get("pretrained_model", 0)
        self.annealing_step = kwargs.get("annealing_step", 200000)

    def compute_distance(self, x):
        raise NotImplementedError

    def compute_marginal_distance(self, x, **kwargs):
        raise NotImplementedError

    def forward(self, feature, label, **kwargs):
        raise NotImplementedError

    def create_mask(self, batch_size, target):
        # one-hot --> mask
        mask = torch.zeros(batch_size, self.num_classes, device=target.device)
        mask.scatter_(1, target.view(batch_size, 1), 1)
        return mask.bool()


# Angular Distance Margin Loss
class ASoftmax(MarginLoss):
    def forward(self, feature, label, **kwargs):
        alpha = max(min((self.step / self.annealing_step), 1), 0)
        self.step += 1
        mask = self.create_mask(feature.size(0), label)

        dist = F.normalize(feature, dim=1).mm(F.normalize(self.centers, dim=1).t())
        dist_with_margin = self.compute_marginal_distance(dist[mask])
        dist[mask] = dist[mask] * (1-alpha) + dist_with_margin * alpha
        return self.loss(torch.norm(feature, p=2, dim=1).unsqueeze(1) * dist, label)

    def compute_marginal_distance(self, x, **kwargs):
        theta = x.acos()
        k = (self.margin * theta.detach() / math.pi).floor()
        n_one = k * 0.0 - 1
        phi_theta = (n_one ** k) * torch.cos(self.margin * theta) - 2 * k
        return phi_theta


class CosFace(MarginLoss):
    def __init__(self, dim_feature, num_classes, margin, scale, *args, **kwargs):
        super(CosFace, self).__init__(dim_feature, num_classes, margin, *args, **kwargs)
        self.scale = scale

    def forward(self, feature, label, **kwargs):
        alpha = max(min((self.step / self.annealing_step), 1), 0)
        self.step += 1
        mask = self.create_mask(feature.size(0), label)

        dist = F.normalize(feature, dim=1).mm(F.normalize(self.centers, dim=1).t())
        dist_with_margin = self.compute_marginal_distance(dist[mask])
        dist[mask] = dist[mask] * (1-alpha) + dist_with_margin * alpha
        return self.loss(self.scale * dist, label)

    def compute_marginal_distance(self, x, **kwargs):
        return x - self.margin


# Euclidean Distance Margin Loss
class MEMS(MarginLoss):
    def forward(self, feature, label, **kwargs):
        alpha = max(min((self.step / self.annealing_step), 1), 0)
        self.step += 1
        mask = self.create_mask(feature.size(0), label)

        dist = torch.cdist(feature, self.centers).pow(2)
        margin = (self.margin - 1) * alpha + 1
        dist[mask] = self.compute_marginal_distance(dist[mask], margin=margin)
        return self.loss(-dist, label)

    def compute_marginal_distance(self, x, **kwargs):
        return x * (kwargs["margin"] ** 2)
