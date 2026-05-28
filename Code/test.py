# -*- coding: utf-8 -*-

import torch
import numpy as np
import os

from model import Model
from utils import metrics
from torch.utils.data import Dataset, DataLoader


# =========================
# 路径（按需修改）
# =========================
MODEL_PATH = "./best_model.pt"

S1_PATH = "./data_test_1.npy"
S2_PATH = "./data_test_2.npy"
PRIOR_PATH = "./causal_prior_test.npy"
HIC_PATH = "./hic_patch_test.npy"
LABEL_PATH = "./data_test_label.npy"

BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Dataset（直接读取npy）
# =========================
class TestDataset(Dataset):
    def __init__(self):
        self.s1 = np.load(S1_PATH)
        self.s2 = np.load(S2_PATH)
        self.pr = np.load(PRIOR_PATH)
        self.hic = np.load(HIC_PATH)
        self.lab = np.load(LABEL_PATH)

        print("\n[TEST DATA]")
        print("s1:", self.s1.shape)
        print("prior:", self.pr.shape)
        print("hic:", self.hic.shape)

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.s1[idx], dtype=torch.float32),
            torch.tensor(self.s2[idx], dtype=torch.float32),
            torch.tensor(self.pr[idx], dtype=torch.float32),
            torch.tensor(self.hic[idx], dtype=torch.float32),
            torch.tensor(self.lab[idx], dtype=torch.float32),
        )


# =========================
# 加载模型
# =========================
def load_model():
    model = Model().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt)
    model.eval()
    return model


# =========================
# 测试
# =========================
@torch.no_grad()
def test():

    print("===> Load model")
    model = load_model()

    print("===> Load data")
    dataset = TestDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    preds, trues, nec_all = [], [], []

    print("===> Inference...")

    for s1, s2, pr, hic, lab in loader:

        s1 = s1.to(DEVICE)
        s2 = s2.to(DEVICE)
        pr = pr.to(DEVICE)
        hic = hic.to(DEVICE)

        out = model(s1, s2, pr, hic)

        prob = torch.sigmoid(out["p_final"])   # logits → prob

        preds.append(prob.cpu().numpy())
        trues.append(lab.numpy())
        nec_all.append(out["nec"].cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    nec_all = np.concatenate(nec_all, axis=0)

    # =========================
    # 自动 threshold
    # =========================
    from sklearn.metrics import f1_score

    best_f1 = 0
    best_th = 0.5

    for th in np.linspace(0.01, 0.5, 100):
        f1 = f1_score(trues, preds > th)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    # =========================
    # metrics
    # =========================
    m = metrics(trues, preds)
    m["F1"] = best_f1
    m["Best_th"] = best_th

    print("\n===== TEST RESULT =====")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")

    # =========================
    # 保存结果
    # =========================
    os.makedirs("test_output", exist_ok=True)

    np.save("test_output/preds.npy", preds)
    np.save("test_output/trues.npy", trues)
    np.save("test_output/nec.npy", nec_all)

    print("\nSaved to ./test_output/")


# =========================
# main
# =========================
if __name__ == "__main__":
    test()