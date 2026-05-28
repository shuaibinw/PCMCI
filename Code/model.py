# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


class DualSeqEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.short = nn.Sequential(
            nn.Conv1d(8, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, 7, padding=3), nn.ReLU()
        )

        self.long = nn.Sequential(
            nn.Conv1d(8, 64, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, 7, stride=4, padding=3), nn.ReLU(),
            nn.Conv1d(128, 128, 7, stride=2, padding=3), nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(TOKEN_SIZE)

        enc = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=2)

    def forward(self, s1, s2):
        x = torch.cat([s1, s2], dim=2).permute(0, 2, 1)

        if x.shape[-1] <= SEQ_SHORT_THRESHOLD:
            x = self.short(x)
        else:
            x = self.long(x)

        x = self.pool(x).permute(0, 2, 1)
        x = self.trans(x)

        return x.mean(dim=1)


class HiCEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32*4*4,64), nn.ReLU()
        )

    def forward(self,x):
        return self.net(x.unsqueeze(1))


class PriorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(PRIOR_DIM,32)

    def forward(self,x):
        return F.relu(self.fc(x))


class CausalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(160,64)
        self.mu = nn.Linear(64,64)
        self.logvar = nn.Linear(64,64)

    def forward(self,H,pr):
        x = torch.cat([H,pr],dim=1)
        x = F.relu(self.fc(x))

        mu = self.mu(x)
        logvar = self.logvar(x)

        std = torch.exp(0.5*logvar)
        z = mu + torch.randn_like(std)*std

        return z,mu,logvar


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64+128,128)

    def forward(self,z,H):
        return F.relu(self.fc(torch.cat([z,H],dim=1)))


def build_head(in_dim):
    return nn.Sequential(
        nn.Linear(in_dim,256), nn.ReLU(),
        nn.Linear(256,64), nn.ReLU(),
        nn.Linear(64,1)
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.seq = DualSeqEncoder()
        self.hic = HiCEncoder()
        self.prior = PriorEncoder()
        self.causal = CausalEncoder()
        self.diff = Diffusion()

        self.head = build_head(224)
        self.final = build_head(229)

    def forward(self,s1,s2,prior,hic):

        H = self.seq(s1,s2)

        pr_f = self.prior(prior)

        z,mu,logvar = self.causal(H,pr_f)
        CS = self.diff(z,H)

        hic_f = self.hic(hic)

        main = torch.cat([CS,hic_f,pr_f],dim=1)

        p_full = self.head(main).squeeze(-1)   

        def zero(p,a,b):
            p = p.clone()
            p[:,a:b]=0
            return p

        groups=[(0,2),(2,3),(3,17),(17,31)]

        outputs=[]
        for g in groups:
            pr0 = zero(prior,*g)
            prf0 = self.prior(pr0)

            z0,_,_ = self.causal(H,prf0)
            CS0 = self.diff(z0,H)

            outputs.append(
                self.head(torch.cat([CS0,hic_f,prf0],dim=1)).squeeze(-1)  
            )

        hic0 = torch.zeros_like(hic)
        hic_f0 = self.hic(hic0)

        outputs.append(
            self.head(torch.cat([CS,hic_f0,pr_f],dim=1)).squeeze(-1)  
        )


        nec = torch.stack([p_full - p for p in outputs], dim=1)

        final_feat = torch.cat([main,nec],dim=1)

        p_final = self.final(final_feat).squeeze(-1)  

        return {
            "p_final":p_final,
            "p_full":p_full,
            "p_remove":outputs,
            "nec":nec,
            "mu":mu,
            "logvar":logvar
        }