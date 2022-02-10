import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, \
    precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-5: round(hfe(x,y,eps),5)


def sklearn_accuracy_score(N0=100, N1=10):
    label = np.random.randint(0, N1, size=[N0])
    predict = np.random.randint(0, N1, size=[N0])
    np1 = accuracy_score(label, predict)
    np1_ = (label==predict).mean()
    print('sklearn_accuracy_score:: np vs skl: ', hfe_r5(np1,np1_))


def sklearn_precision_score(N0=100):
    label = np.random.randint(0, 2, size=[N0])
    predict = np.random.randint(0, 2, size=[N0])
    np1 = precision_score(label, predict)
    np1_ = label[predict.astype(np.bool)].sum() / predict.sum()
    print('sklearn_precision_score:: np vs skl: ', hfe_r5(np1,np1_))


def sklearn_recall_score(N0=100):
    label = np.random.randint(0, 2, size=[N0])
    predict = np.random.randint(0, 2, size=[N0])
    np1 = recall_score(label, predict)
    np1_ = label[predict.astype(np.bool)].sum()/label.sum()
    print('sklearn_recall_score:: np vs skl: ', hfe_r5(np1,np1_))


def sklearn_f1_score(N0=100):
    label = np.random.randint(0, 2, size=[N0])
    predict = np.random.randint(0, 2, size=[N0])
    np1 = f1_score(label, predict)

    precision = label[predict.astype(np.bool)].sum() / predict.sum()
    recall = label[predict.astype(np.bool)].sum()/label.sum()
    np1_ = 2*precision*recall / (precision+recall)
    print('sklearn_f1_score:: np vs skl: ', hfe_r5(np1,np1_))


def sklearn_roc_curve(N0=100):
    label = np.random.randint(0, 2, size=[N0])
    predict_proba = np.random.rand(N0)
    fpr,tpr,thresholds = roc_curve(label, predict_proba)

    fpr_ = [None]*len(thresholds)
    tpr_ = [None]*len(thresholds)
    numT = (label==1).sum()
    numF = (label==0).sum()
    tmp1 = label.astype(np.bool)
    tmp2 = np.logical_not(tmp1)
    for ind1,x in enumerate(thresholds):
        tpr_[ind1] = (predict_proba[tmp1]>=x).sum() / numT
        fpr_[ind1] = (predict_proba[tmp2]>=x).sum() / numF
    tpr_ = np.array(tpr_)
    fpr_ = np.array(fpr_)
    print('sklearn_roc_curve TPR:: np vs skl: ', hfe_r5(tpr,tpr_))
    print('sklearn_roc_curve FPR:: np vs skl: ', hfe_r5(fpr,fpr_))


def sklearn_auc():
    N0 = 20
    x = np.sort(np.random.rand(N0))
    y = np.random.rand(N0)
    np1 = auc(x, y)
    np1_ = np.trapz(y, x)
    print('sklearn_auc:: np vs skl: ', hfe_r5(np1, np1_))


def sklearn_roc_auc_score(N0=100):
    label = np.random.randint(0, 2, size=[N0])
    predict_proba = np.random.rand(N0)
    np1 = roc_auc_score(label, predict_proba)

    fpr,tpr,_ = roc_curve(label, predict_proba)
    np1_ = auc(fpr, tpr)
    print('sklearn_roc_auc_score:: np vs skl: ', hfe_r5(np1, np1_))


def sklearn_precision_recall_curve(N0=100):
    label = np.random.randint(0, 2, size=[N0])
    predict_proba = np.random.rand(N0)
    precision,recall,thresholds = precision_recall_curve(label, predict_proba)

    precision_ = [None]*len(thresholds)
    recall_ = [None]*len(thresholds)
    numT = (label==1).sum()
    numF = (label==0).sum()
    tmp1 = label.astype(np.bool)
    tmp2 = np.logical_not(tmp1)
    for ind1,x in enumerate(thresholds):
        precision_[ind1] = (predict_proba[tmp1]>=x).sum() / (predict_proba>=x).sum()
        recall_[ind1] = (predict_proba[tmp1]>=x).sum() / numT #same as tpr
    precision_ = np.array(precision_+[1])
    recall_ = np.array(recall_+[0])
    print('sklearn_roc_curve precision:: np vs skl: ', hfe_r5(precision,precision_))
    print('sklearn_roc_curve recall:: np vs skl: ', hfe_r5(recall,recall_))


def sklearn_average_precision_score(N0=100):
    label = np.random.randint(0, 2, size=[N0])
    predict_proba = np.random.rand(N0)
    np1 = average_precision_score(label, predict_proba)

    precision,recall,_ = precision_recall_curve(label, predict_proba)
    np1_ = -np.sum(np.diff(recall) * precision[:-1])
    print('sklearn_average_precision_score:: np vs skl: ', hfe_r5(np1, np1_))

if __name__=='__main__':
    sklearn_accuracy_score()
    print()
    sklearn_precision_score()
    print()
    sklearn_recall_score()
    print()
    sklearn_f1_score()
    print()
    sklearn_roc_curve()
    print()
    sklearn_auc()
    print()
    sklearn_roc_auc_score()
    print()
    sklearn_precision_recall_curve()
    print()
    sklearn_average_precision_score()
    print()
