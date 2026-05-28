# -*- coding: utf-8 -*-
"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()  

    def forward(self, out, label):

        p_final = out["p_final"]   # logits
        p_full  = out["p_full"]    # logits
        p_remove = out["p_remove"]
        mu = out["mu"]
        logvar = out["logvar"]

        label = label.float()

        # 主损失
        L_cls = self.bce(p_final, label)

        # 辅助损失
        L_aux = self.bce(p_full, label)

        # necessity loss
        L_nec = 0
        pos_mask = label > 0.5

        if pos_mask.sum() > 0:
            for p_r in p_remove:
                diff = p_full[pos_mask] - p_r[pos_mask]
                L_nec += F.relu(MARGIN - diff).mean()
            L_nec = L_nec / len(p_remove)

        # KL
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 总损失
        loss = L_cls + LAMBDA_AUX * L_aux + LAMBDA_NEC * L_nec + KL

        return loss