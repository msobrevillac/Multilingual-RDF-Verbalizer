import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

# Implementation found at 
#https://medium.com/towards-artificial-intelligence/how-to-use-label-smoothing-for-regularization-aa349f7f1dbb
class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0, reduction='sum'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.padding_idx = padding_idx
    
    def forward(self, x, target):
        n = x.size()[-1]
        pred = x.clone()
        for i, tgt in enumerate(target):
          if tgt.data == self.padding_idx:
            pred[i] = 0
        loss = reduce_loss(-pred.sum(dim=-1), self.reduction)
        nll = F.nll_loss(x, target, ignore_index=self.padding_idx, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.smoothing)


class LossCompute:
    "A loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm=1):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            #self.opt.optimizer.zero_grad()
        return loss.item() * norm
