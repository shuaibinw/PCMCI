from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import numpy as np

def metrics(y_true,y_prob):
    y_pred = (y_prob>0.5).astype(int)
    return {
        "AUC":roc_auc_score(y_true,y_prob),
        "AUPR":average_precision_score(y_true,y_prob),
        "ACC":accuracy_score(y_true,y_pred),
        "F1":f1_score(y_true,y_pred)
    }