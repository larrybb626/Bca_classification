import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    # tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr) / float(SR.size(0))

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1)*1 + (GT == 1)*1) == 2)*1
    FN = (((SR == 0)*1 + (GT == 1)*1) == 2)*1

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0)*1 + (GT == 0)*1) == 2)*1
    FP = (((SR == 1)*1 + (GT == 0)*1) == 2)*1

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = (((SR == 1)*1 + (GT == 1)*1) == 2)*1
    FP = (((SR == 1)*1 + (GT == 0)*1) == 2)*1

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC

def get_AUC(SR, GT):
    SR = SR.view(-1)
    GT = GT.view(-1)
    pred_value = SR.data.cpu().numpy()
    true_label = GT.data.cpu().numpy()

    if (sum(true_label) == len(pred_value)) or (sum(true_label) == 0):
        AUC = 0
        print('only one class')
    else:
        AUC = metrics.roc_auc_score(true_label, pred_value)

    return AUC

def get_roc(SR,GT,auc,path):
    SR = SR.view(-1)
    GT = GT.view(-1)
    pred_value = SR.data.cpu().numpy()
    true_label = GT.data.cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(true_label, pred_value)
    # fig = plt.figure()
    # plt.plot(fpr,tpr,'b',label='ROC %.4f'%(auc))
    # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Standard', alpha=.8)
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC_curve')
    # plt.legend(loc='lower right')
    # fig.savefig(os.path.join(path, 'roc_curve.PNG'))
    # plt.close()
    return fpr, tpr
def get_roc_inside(fpr,tpr,path):
    fpr.sort()
    tpr.sort()
    auc = metrics.auc(fpr,tpr)
    fig = plt.figure()
    plt.plot(fpr,tpr,'b',label='ROC %.4f'%(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_curve')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(path, 'roc_curve.PNG'))
    plt.close()


def get_roc_plot_extra(AUC, FPR, TPR, path):
    color_line = ['y','m','c','r','g']
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure()
    for i in range(5):
        current_color = color_line[i]
        tprs.append(interpolate(mean_fpr, FPR[i], TPR[i]))
        tprs[-1][0] = 0.0
        plt.plot(FPR[i],TPR[i],lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i+1,AUC[i]))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    auc_upper = metrics.auc(mean_fpr,tprs_upper)
    auc_lower = metrics.auc(mean_fpr,tprs_lower)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2,label = '95CI : %0.2f - %0.2f'%(auc_lower,auc_upper))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_curve')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(path, 'roc_curve_5_new_extra.PNG'))
    plt.close()

def get_best_threshold(SR, GT):
    SR = SR.view(-1)
    GT = GT.view(-1)
    pred_value = SR.data.cpu().numpy()
    true_label = GT.data.cpu().numpy()

    # 计算最佳阈值
    fpr, tpr, thresholds = metrics.roc_curve(true_label, pred_value)
    # 计算约登指数
    Youden_index = tpr + (1 - fpr)
    best_threshold = thresholds[Youden_index == np.max(Youden_index)][0]

    # have no idea about that threshold is bigger than 1 sometimes
    # maybe can find in https://github.com/scikit-learn/scikit-learn/commit/4d9a67f77787ffe9955187865f9b95e19286f069
    # or https://github.com/scikit-learn/scikit-learn/issues/3097
    if best_threshold > 1:
        best_threshold = 0.5

    return best_threshold


