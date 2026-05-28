# -*- coding: utf-8 -*-
"""
dataset.py

适配你的真实数据格式：
data_train_1.npy
data_train_2.npy
causal_prior_train.npy
hic_patch_train.npy
data_train_label.npy

支持：
✔ 5kb / 450kb 自动识别
✔ 混合训练
✔ 数据安全检查
✔ NaN/Inf 清洗
✔ debug打印
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ===============================
# 工具函数：清洗异常值
# ===============================
def sanitize(arr, name=""):
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()

    if n_nan + n_inf > 0:
        print(f"[WARN] {name}: NaN={n_nan}, Inf={n_inf} → cleaned")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)

    return arr


# ===============================
# 主 Dataset
# ===============================
class ChromDataset(Dataset):
    def __init__(self, root, split, debug=False):
        """
        root: 数据目录
        split: train / val / test
        """

        self.root = root
        self.split = split

        # ===== 加载数据 =====
        self.s1 = sanitize(
            np.load(os.path.join(root, f"data_{split}_1.npy")),
            f"s1_{split}"
        ).astype(np.float32)

        self.s2 = sanitize(
            np.load(os.path.join(root, f"data_{split}_2.npy")),
            f"s2_{split}"
        ).astype(np.float32)

        self.prior = sanitize(
            np.load(os.path.join(root, f"causal_prior_{split}.npy")),
            f"prior_{split}"
        ).astype(np.float32)

        self.hic = sanitize(
            np.load(os.path.join(root, f"hic_patch_{split}.npy")),
            f"hic_{split}"
        ).astype(np.float32)

        self.label = sanitize(
            np.load(os.path.join(root, f"data_{split}_label.npy")),
            f"label_{split}"
        ).astype(np.float32)

        # ===== 基本检查 =====
        assert len(self.s1) == len(self.label), "s1 size mismatch"
        assert self.s1.shape == self.s2.shape, "s1/s2 shape mismatch"
        assert self.prior.shape[1] == 31, "prior dim must be 31"
        assert self.hic.shape[1:] == (11, 11), "hic must be 11x11"

        self.N = len(self.label)
        self.seq_len = self.s1.shape[1]

        # ===== 自动识别 short / long =====
        if self.seq_len <= 10000:
            self.mode = "SHORT"
        else:
            self.mode = "LONG"

        # ===== 打印信息 =====
        print(f"\n[{split.upper()} DATASET]")
        print(f"  Samples : {self.N}")
        print(f"  Seq_len : {self.seq_len}  ({self.mode})")
        print(f"  s1      : {self.s1.shape}")
        print(f"  prior   : {self.prior.shape}")
        print(f"  hic     : {self.hic.shape}")
        print(f"  Pos     : {int(self.label.sum())}")
        print(f"  Neg     : {int((self.label == 0).sum())}")

        if debug:
            print("\n[DEBUG SAMPLE]")
            print("  s1[0]:", self.s1[0].shape)
            print("  prior[0]:", self.prior[0].shape)
            print("  hic[0]:", self.hic[0].shape)
            print("  label[0]:", self.label[0])

    # ===============================
    def __len__(self):
        return self.N

    # ===============================
    def __getitem__(self, idx):
        s1 = torch.from_numpy(self.s1[idx])
        s2 = torch.from_numpy(self.s2[idx])
        pr = torch.from_numpy(self.prior[idx])
        hic = torch.from_numpy(self.hic[idx])
        lab = torch.tensor(self.label[idx])

        return s1, s2, pr, hic, lab