import torch
from torch import nn

class L1Loss(nn.Module):
    def __init__(self, ohem=False, topk=0.6):
        super(L1Loss, self).__init__()
        self.ohem = ohem
        self.topk = topk
        self.loss = nn.L1Loss(reduction='none')

    def forward(self, inputs, targets, mask):
        # print(inputs[mask], targets[mask]) # to check amp work or not
        loss = self.loss(inputs[mask], targets[mask])

        if self.ohem:
            num_hard_samples = int(self.topk * loss.numel())
            loss, _ = torch.topk(loss.flatten(), 
                                 num_hard_samples)

        return torch.mean(loss)

loss_dict = {'l1': L1Loss}