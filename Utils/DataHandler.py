import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp

import torch
import torch.utils.data as data
import torch.utils.data as dataloader


class DataHandler:
    def __init__(self, data, val=False):
        predir = f'Data/{data}/'
        self.predir = predir
        self.trnfile = predir + 'trnMat.pkl'
        self.valfile = predir + 'valMat.pkl'
        self.tstfile = predir + 'tstMat.pkl'

    def loadFile(self, filename):  # 加载文件并转为coo格式和float32
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        a = sp.csr_matrix((args.user, args.user))  # Empty CSR Matrix
        b = sp.csr_matrix((args.item, args.item))
        #Idegree = np.array(mat.sum(axis=0)).tolist()[0]
        #Udegree = np.array(mat.sum(axis=1)).tolist()[0]
        #mat = mat.todense()
        #mat[np.isinf(mat)] = 0.0
        #for i in range(len(Idegree)):
         #   if Idegree[i] < 30 or Idegree[i]>45:
          #      mat[:][i] = 0
        #for j in range(len(Udegree)):
         #   if Udegree[j] < 30 or Udegree[j]>45:
          #      mat[j][:] = 0
        #mat = sp.coo_matrix(mat)
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])  # mat dimension is (u+i,i+u)
        # mat = (mat != 0) * 1.0
        mat = (mat) * 1.0
        mat = self.normalizeAdj(mat)

        # transfer tensor to device(cuda)
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).to(args.device)

    def LoadData(self):
        trnMat = self.loadFile(self.trnfile)
        tstMat = self.loadFile(self.tstfile)
        valMat = self.loadFile(self.valfile)
        self.x = trnMat
        args.user, args.item = trnMat.shape  # 得到训练数据
        self.torchBiAdj = self.makeTorchAdj(trnMat)
        trnData = TrnData(trnMat)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
        tstData = TstData(tstMat, trnMat)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)
        valData = ValData(valMat, trnMat)
        self.valLoader = dataloader.DataLoader(valData, batch_size=args.tstBat, shuffle=False, num_workers=0)
                                               #pin_memory=True)
        args.interaction = trnData.__len__()


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()  # 返回稀疏矩阵的dok_matrix形式，以字典kv记录不为0的值，v是具体值
        self.negs = np.zeros(len(self.rows)).astype(np.int32)  # 负样本，一个user一个负样本

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]  # item
            while True:
                iNeg = np.random.randint(args.item)  # [0,args.item)左闭右开
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)  # 交互数量

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]  # coo矩阵的存储格式[row,col,data]#由于这里是1,就不返回具体数值来了


class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]  # 生成user长度的tstLocs
        tstUsrs = set()
        for i in range(len(coomat.data)):  # 遍历测试数据
            row = coomat.row[i]  # 得到第i条数据的row和col，也就是user和item
            col = coomat.col[i]
            if tstLocs[row] is None:  # 如果该row（也就是user）没有则生成一个list（list存交互项目）
                tstLocs[row] = list()
            tstLocs[row].append(col)  # 一个嵌套list
            tstUsrs.add(row)  # 是一个set，存放user名字的item列表。
        tstUsrs = np.array(list(tstUsrs))  # list列表
        self.tstUsrs = tstUsrs  # 得到user的set
        self.tstLocs = tstLocs  # 得到item的list,后面需要取出来对比

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])  # 返回user和？
        # 不太能理解这个mask，这里应该是idx用户在训练集的所有item


class ValData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        valLocs = [None] * coomat.shape[0]  # 生成user长度的valLocs
        valUsrs = set()
        for i in range(len(coomat.data)):  # 遍历测试数据
            row = coomat.row[i]  # 得到第i条数据的row和col，也就是user和item
            col = coomat.col[i]
            if valLocs[row] is None:  # 如果该row（也就是user）没有则生成一个list（list存交互项目）
                valLocs[row] = list()
            valLocs[row].append(col)  # 一个嵌套list
            valUsrs.add(row)  # 是一个set，存放user名字的item列表。
        valUsrs = np.array(list(valUsrs))  # list列表
        self.valUsrs = valUsrs  # 得到user的set
        self.valLocs = valLocs  # 得到item的list,后面需要取出来对比

    def __len__(self):
        return len(self.valUsrs)

    def __getitem__(self, idx):
        return self.valUsrs[idx], np.reshape(self.csrmat[self.valUsrs[idx]].toarray(), [-1])  # 返回user和？
        # 不太能理解这个mask，这里应该是idx用户在训练集的所有item