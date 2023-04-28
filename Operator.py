import logging

import torch

from Model.util.metrics import metrics_recall, metrics_ndcg
from Params import args
from Model.util.modLoss import calcRegLoss

class Operator:

    def __init__(self,earlystopping):
        self.logger = logging.getLogger()

        self.earlystopping = earlystopping

    def registerHandler(self, handler):
        self.handler = handler
        self.logger.info(f"Registered handler")

    def registerModel(self, model, checkpoint = None):
        self.model = model.to(args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        if checkpoint!=None :
            self.model.load_state_dict(checkpoint['model'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            self.stloc = checkpoint['epoch']+1
            self.logger.info(f"Registered trained model")
        else:
            self.stloc = 0
            self.logger.info(f"Registered model")



    def train(self):
        for ep in range(self.stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            self.logger.info(f"Epoch:{ep} Loss:{reses['Loss']:{args.outAcc}} BPRLoss:{reses['BPRLoss']:{args.outAcc}} Loss2:{reses['Loss2']:{args.outAcc}}")
            if tstFlag or ep==args.epoch :
                reses2 = self.testEpoch(args.topk2,val=True)
                self.logger.info(f"Test:{ep // args.tstEpoch} Recall@{args.topk2}:{reses2['Recall']:{args.outAcc}} NDCG@{args.topk2}:{reses2['NDCG']:{args.outAcc}}")
                state = {'model': self.model.state_dict(), 'optimizer': self.opt.state_dict(), 'epoch': ep}
                self.earlystopping(reses2['Recall'], state)
                if self.earlystopping.early_stop:
                    self.logger.info("earlystopp-->break")
                    break

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader  # 加载训练集
        trnLoader.dataset.negSampling()  # 得到负样本
        epLoss, epBPRLoss,epLoss2 = 0, 0, 0  # 损失
        steps = trnLoader.dataset.__len__() // args.batch  # 图的batch 训练
        for i, tem in enumerate(trnLoader):  # loader会加载一个batch的数据
            ancs, poss, negs = tem  # 每个样本
            ancs = ancs.long().to(args.device)
            poss = poss.long().to(args.device)
            negs = negs.long().to(args.device)
            bprLoss, Loss2 = self.model.calcLosses(ancs, poss, negs, self.handler.torchBiAdj, args.keepRate)
            # 模型计算loss，相当于前向传播了的结果并计算其loss
            regLoss = calcRegLoss(self.model) * args.reg  # L2正则
            loss = bprLoss +Loss2*args.reg_loss2 + regLoss # 总的loss
            epLoss += loss.item()  # epoch的loss
            epBPRLoss += bprLoss.item()  # bpr的loss
            epLoss2+=Loss2.item()
            self.opt.zero_grad()  # 梯度置
            loss.backward()  # 反向传播
            self.opt.step()  # 更新参数
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['BPRLoss'] = epBPRLoss / steps
        ret['Loss2'] = epLoss2 /steps
        return ret

    def testEpoch(self,topK,val=False):
        if val==False:
            Loader = self.handler.tstLoader
        else:
            Loader = self.handler.valLoader
        epRecall, epNdcg = [0] * 2
        i = 0
        num = Loader.dataset.__len__()  # 测试集数量
        for usr, trnMask in Loader:
            i += 1
            usr = usr.long().to(args.device)
            trnMask = trnMask.to(args.device)
            usrEmbeds, itmEmbeds = self.model.predict(self.handler.torchBiAdj)  # 预测，前向传播，得到embed
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (
                    1 - trnMask) - trnMask * 1e8  # 计算边存在不
            _, topLocs = torch.topk(allPreds, topK)  # 按照维度1，进行topk排序
            # topLocs是（user，args.topk）的维度
            if val==False:
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr,topK)
            else:
                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.valLoader.dataset.valLocs, usr,topK)
            epRecall += recall
            epNdcg += ndcg
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds,topK):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):#对于每一个id
            temTopLocs = list(topLocs[i])#将用户的item的list打印出来
            temTstLocs = tstLocs[batIds[i]]#找到测试集的对应值
            recall=metrics_recall(temTopLocs,temTstLocs)
            ndcg = metrics_ndcg(temTopLocs, temTstLocs,topK)
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg