import torch

# ======================
# 数据
# ======================
PRIOR_DIM = 31
HIC_SIZE = 11

# ======================
# 模型
# ======================
SEQ_SHORT_THRESHOLD = 10000
TOKEN_SIZE = 64

# ======================
# 训练
# ======================
BATCH_SIZE = 16  # 单卡可以适当调大 batch_size
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-5

MARGIN = 0.1
LAMBDA_NEC = 0.5
LAMBDA_AUX = 0.3
LAMBDA_KL = 1e-3

# ======================
# 系统
# ======================
NUM_WORKERS = 4
# 自动检测，也可以手动指定 "cuda:0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")