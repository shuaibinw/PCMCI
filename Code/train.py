# -*- coding: utf-8 -*-
"""
运行：python train.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset import ChromDataset
from model import Model
from loss import Loss
from utils import metrics
from config import *

# ==========================================================
# evaluate（val / test 通用）
# ==========================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    preds, trues, nec_all = [], [], []

    for s1, s2, pr, hic, lab in loader:
        s1 = s1.to(device, non_blocking=True)
        s2 = s2.to(device, non_blocking=True)
        pr = pr.to(device, non_blocking=True)
        hic = hic.to(device, non_blocking=True)

        out = model(s1, s2, pr, hic)

        # Logits -> Probability
        pred = torch.sigmoid(out["p_final"])

        preds.append(pred.cpu().numpy())
        trues.append(lab.numpy())
        nec_all.append(out["nec"].cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    nec_all = np.concatenate(nec_all, axis=0)

    return metrics(trues, preds), nec_all, preds, trues


# ==========================================================
# 主函数
# ==========================================================
def main():
    # 使用 config 中定义的 DEVICE
    device = DEVICE
    print(f"Using device: {device}")

    # ======================
    # 数据准备
    # ======================
    DATA_ROOT = "/public/home/shenyin_wsb_2606/Fourth/IMR90/"

    train_set = ChromDataset(DATA_ROOT, "train")
    val_set   = ChromDataset(DATA_ROOT, "val")
    test_set  = ChromDataset(DATA_ROOT, "test")

    # 单卡直接设置 shuffle=True/False，无需 Sampler
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ======================
    # 模型与组件
    # ======================
    model = Model().to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    scaler = GradScaler()
    loss_fn = Loss()

    best_auc = 0
    patience_counter = 0

    # ======================================================
    # 训练循环
    # ======================================================
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for s1, s2, pr, hic, lab in train_loader:
            s1 = s1.to(device)
            s2 = s2.to(device)
            pr = pr.to(device)
            hic = hic.to(device)
            lab = lab.to(device)

            optimizer.zero_grad()

            with autocast():
                out = model(s1, s2, pr, hic)
                loss = loss_fn(out, lab)

            scaler.scale(loss).backward()
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        # ======================
        # 验证
        # ======================
        val_metrics, _, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss:.4f} | VAL AUC: {val_metrics['AUC']:.4f}")
        print(f"  Metrics: {val_metrics}")

        scheduler.step(val_metrics["AUC"])

        # ======================
        # 保存最佳模型
        # ======================
        if val_metrics["AUC"] > best_auc:
            best_auc = val_metrics["AUC"]
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("  ✓ Best model saved")
        else:
            patience_counter += 1
            if patience_counter > 10:
                print("Early stopping triggered.")
                break

    # ======================================================
    # 测试阶段
    # ======================================================
    print("\n" + "="*25 + " FINAL TEST " + "="*25)
    
    # 加载权重 (不需要 .module)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    test_metrics, nec_matrix, preds, trues = evaluate(
        model, test_loader, device
    )

    print("Final Test Metrics:", test_metrics)

    # 保存结果
    np.save("test_preds.npy", preds)
    np.save("test_labels.npy", trues)
    np.save("test_nec_matrix.npy", nec_matrix)
    print("✓ Test results and NEC matrix saved.")


if __name__ == "__main__":
    main()