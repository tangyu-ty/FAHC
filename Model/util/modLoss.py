import torch.nn.functional as F
import torch


def calcRegLoss(model):  # L2 Loss
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()

    return ret


def contrastLoss(embeds1, embeds2, nodes, temp):  # 对比loss
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]

    nume = torch.exp(torch.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = torch.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8  # 分母
    # 这一部分是对比loss的核心，temp是tao，平滑参数
    ret = -torch.log(nume / deno).mean()
    return ret


def BPRLoss(ancEmbeds, posEmbeds, negEmbeds):
    scoreDiff = torch.sum(ancEmbeds * posEmbeds, dim=-1) - torch.sum(ancEmbeds * negEmbeds, dim=-1)
    ret=- torch.nn.functional.logsigmoid(scoreDiff).mean()
    #bpr=- (scoreDiff).sigmoid().log().sum()
    return ret