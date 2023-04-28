import numpy as np

from Params import args


def metrics_recall(PredLocs,TstLocs):
    fp=0
    tstNum = len(TstLocs)
    for val in TstLocs:  # 在测试集中
        if val in PredLocs:
            fp += 1
    recall = fp / tstNum

    return recall


def metrics_ndcg(PredLocs,TstLocs,topK):
    dcg=0
    tstNum = len(TstLocs)
    maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topK))])  # 计算dcg

    for val in TstLocs:  # 在测试集中
        if val in PredLocs:
            dcg += np.reciprocal(np.log2(PredLocs.index(val) + 2))
    ndcg = dcg / maxDcg

    return ndcg

