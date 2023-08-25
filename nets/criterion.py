import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseWeightedLoss
import pdb
import math


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


class EvidenceLoss(torch.nn.Module):
    def __init__(self, num_classes, evidence='exp', uncertain=True, loss_type='log'):
        super().__init__()
        self.num_classes = num_classes
        self.evidence = evidence
        self.eps = 1e-10
        self.uncertain = uncertain
        self.loss_type = loss_type

    def edl_loss(self, func, y, alpha, fore_only=False):
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = func(S) - func(alpha)
        loss = torch.sum(y * p, dim=1, keepdim=True) 
        return loss
    
    def bce_loss(self, y, alpha, fore_only = False):
        if fore_only:
            loss = F.binary_cross_entropy(alpha[...,0:1], y[...,0:1], reduction='none')
        else:
            loss = F.binary_cross_entropy(alpha, y, reduction='none')
        return loss


    def forward(self, output, target, fore_only = False, **kwargs):
        y = target
        n_y = 1 - target
        gt_y = torch.stack((y, n_y), dim=-1)

        if self.uncertain:
            if self.evidence == 'relu':
                evidence = relu_evidence(output)
            elif self.evidence == 'exp':
                evidence = exp_evidence(output)
            elif self.evidence == 'softplus':
                evidence = softplus_evidence(output)
            else:
                raise NotImplementedError
            alpha = evidence + 1
        else:
            alpha = output.softmax(dim=-1)

        if self.loss_type == 'log':
            C = alpha.shape[-2]
            results = self.edl_loss(torch.log, gt_y.reshape(-1, 2), alpha.reshape(-1, 2), fore_only).reshape(-1, C).sum(dim=-1).mean()
        elif self.loss_type == 'bce':
            if self.uncertain:
                alpha, _ = self.get_predictions(output)
            results = self.bce_loss(gt_y, alpha, fore_only).sum(dim=-1).mean()

        else:
            raise NotImplementedError

        return results

    def get_predictions(self, x):
        evidence = exp_evidence(x)
        S = evidence + torch.ones_like(x)
        p = evidence / torch.sum(S, dim=-1, keepdim=True)
        u = self.num_classes / torch.sum(S, dim=-1, keepdim=True)
        return p, u
    
    def get_alpha(self, x):
        evidence = exp_evidence(x)
        S = evidence + torch.ones_like(x)
        return S


class MutualLearningLoss(torch.nn.Module):

    def __init__(self, weight=1.0, lamb=1.0):
        super().__init__()

        self.weight = weight
        self.lamb = lamb


    def get_positive_expectation(self, p_samples, measure='JSD', average=True):
        log_2 = math.log(2.)
        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            Ep = log_2 - F.softplus(-p_samples)
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'H2':
            Ep = torch.ones_like(p_samples) - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            raise ValueError('Unknown measurement {}'.format(measure))
        if average:
            return Ep.mean()
        else:
            return Ep


    def get_negative_expectation(self, q_samples, measure='JSD', average=True):

        log_2 = math.log(2.)
        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            Eq = F.softplus(-q_samples) + q_samples - log_2
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            raise ValueError('Unknown measurement {}'.format(measure))
        if average:
            return Eq.mean()
        else:
            return Eq

    def forward(self, X, Y, measure='JSD'):
        # X (B, T, C)
        # Y (B, T, C)

        E_pos = self.get_positive_expectation(X, measure, average=False)
        E_neg = self.get_negative_expectation(Y, measure, average=False)

        # E_neg = torch.sum(E_neg, dim=1)  # (bsz, )

        E = E_neg - E_pos  # (B, T, C)
        return torch.mean(E)
