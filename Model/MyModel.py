import torch.nn
from torch import nn
import torch.nn.functional as F

from Model.util.modLoss import BPRLoss
from Params import args

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class OrgModel(nn.Module):
    def __init__(self):
        super(OrgModel, self).__init__()

        self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
        self.gcnLayer = GCNLayer()
        self.hgnnLayer = HGNNLayer()
        self.Hyper = nn.Parameter(init(torch.empty(args.latdim, args.hyperNum)))

        self.edgeDropper = SpAdjDropEdge()

        self.zishiying = nn.Parameter(init(torch.empty(args.item + args.user, args.latdim)))
    def forward(self, adj, keepRate):
        embeds = torch.cat([self.uEmbeds, self.iEmbeds], dim=0)
        lats = [embeds]
        HyperLats =[]
        allHyper=embeds @ self.Hyper
        #gcnLat=[]
        temEmbeds=0
        for i in range(args.gnn_layer):
            temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1], self.zishiying)
            #gcnLat.append(temEmbeds)
            hyperLat = self.hgnnLayer(F.dropout(allHyper, p=1 - keepRate),lats[-1])
            HyperLats.append(hyperLat)
            lats.append(HyperLats[-1]+temEmbeds)
        embeds = temEmbeds
        return embeds, HyperLats

    def calcLosses(self, ancs, poss, negs, adj, keepRate):
        embeds, hyperEmbedsLats = self.forward(adj, keepRate)
        uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]
        ancEmbeds = uEmbeds[ancs]
        posEmbeds = iEmbeds[poss]
        negEmbeds = iEmbeds[negs]
        bprLoss = BPRLoss(ancEmbeds, posEmbeds, negEmbeds)
        logsigmoidLoss = 0
        for i in range(args.gnn_layer):
            logsigmoidLoss +=-nn.functional.logsigmoid(hyperEmbedsLats[i]).mean()
            #logsigmoidLoss+=torch.nn.ELU()(hyperEmbedsLats[i]).mean()
        return bprLoss, logsigmoidLoss

    def predict(self, adj):
        embeds, _ = self.forward(adj, 1.0)
        return embeds[:args.user], embeds[args.user:]
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=args.leaky)
    def forward(self, adj, embeds, zishiying):
        #aij = torch.nn.Softsign()(zishiying)
        #aij = torch.tanh(zishiying)
        aij = torch.nn.functional.sigmoid(zishiying)*2-1
        ret = self.act(torch.spmm(adj, embeds*aij))
        return ret


class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

    def forward(self, adj, embeds):

        lat = self.act(adj.T @embeds)#(H,d)
        ret = self.act(adj@lat)#zibianma
        return ret


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
