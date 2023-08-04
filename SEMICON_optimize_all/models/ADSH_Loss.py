import math

import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional

class ADSH_Loss(nn.Module):
    def __init__(self, code_length, gamma, dataset_name, device):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        num_classes = 200  # cub
        if dataset_name == 'aircraft':
            num_classes = 100
        self.gamma = gamma
        self.centers = np.loadtxt('centers/center_%s_%s.txt' % (num_classes, code_length))
        self.centers = self.centers.transpose()
        self.centers = torch.from_numpy(self.centers).to(device).to(torch.float)

    def forward(self, F, B, S, omega, targets):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0]) / self.code_length * 12
        quantization_loss = ((F - B[omega, :]) ** 2).sum() / (F.shape[0] * B.shape[0]) * self.gamma / self.code_length * 12

        center_loss = self.calculate_center_loss(F, targets)

        loss = hash_loss + quantization_loss + center_loss
        return loss, hash_loss, quantization_loss

    def calculate_center_loss(self, F, targets):
        P = self.calculate_P(F, self.code_length)
        ret = (-1 / F.shape[0]) * torch.sum(targets * torch.log(P) + (1 - targets) * torch.log(1 - P))
        return ret

    def calculate_P(self, F, q):
        # 计算平方根
        sqrt_q = math.sqrt(q)

        # 归一化矩阵 B 和 H
        F_norm = torch.nn.functional.normalize(F, p=2, dim=1)
        H_norm = torch.nn.functional.normalize(self.centers, p=2, dim=1)

        C = F_norm @ torch.transpose(H_norm, 0, 1)

        # 计算指数化和求和
        S = torch.exp(sqrt_q * C).sum(dim=1)

        # 计算每个元素的值
        P = torch.exp(sqrt_q * C) / S.unsqueeze(1)

        return P

