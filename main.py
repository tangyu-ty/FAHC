# 这是一个示例 Python 脚本。
import datetime
import os
import random

import numpy as np
import torch

from Model.MyModel import OrgModel
from Utils.DataHandler import DataHandler

from Params import args
from Utils.Logging import get_logger
from Operator import Operator
from Utils.setSeed import setup_seed
from Utils.torchtools import EarlyStopping
from Utils.DIR import mkdir
# 1.设置随机种子

mkdir()
# 2.设置log存储的位置
fileName = datetime.datetime.now().strftime(f'%Y-%m-%d--%H-%M-%S')
fileName = f"{args.hyperParamIndex}-"+fileName
logger = get_logger(f"./Log/{fileName}.log")
setup_seed(args.seed)
logger.info(f"log build on {fileName}")
args_items = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
logger.info('\n'+args_items)
# 3.加载数据和设备
args.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"device:{args.device}")
handler = DataHandler(args.data)

handler.LoadData()
logger.info(f"user={args.user},item={args.item}")
logger.info(f"interaction={args.interaction}")
# 4.加载模型和其他工具
earlystopping = EarlyStopping(patience=args.patience, verbose=True, trace_func=logger, filename=fileName)
op = Operator(earlystopping)
model = OrgModel()
# 5.是否开始训练或继续训练
pre_training = False
if pre_training:
    checkpoint = torch.load("./Saved/checkpoint.pth")
else:
    checkpoint = None
op.registerHandler(handler)
op.registerModel(model, checkpoint)  # checkpoint只保留模型的参数。
op.train()
logger.info(f"********************end_train********************")
# 6.找到最佳的model,加载验证集(可选)
logger.info(f"Verify optimal model performance")
Bestcheckpoint = torch.load("./Saved/" + fileName + "-checkpoint.pth")
logger.info(f"loading Bestcheckpoint {fileName}-checkpoint.pth")
op.registerModel(model, Bestcheckpoint)
logger.info(f"loading model with  Bestcheckpoint")
ret1 = op.testEpoch(args.topk1)
ret2 = op.testEpoch(args.topk2)
ret3 = op.testEpoch(args.topk3)
logger.info(f"testdate")
logger.info(f"best_model_recall@{args.topk2}::{ret2['Recall']:{args.outAcc}}")
logger.info(f"best_model_ndcg@{args.topk2}::{ret2['NDCG']:{args.outAcc}}")
logger.info(f"********************result********************")
logger.info("Hyperparameter")
args_items = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
logger.info('\n'+args_items)
logger.info('Test-meter')
logger.info(f"recall@{args.topk1}::{ret1['Recall']:{args.outAcc}}   ndcg@{args.topk1}::{ret1['NDCG']:{args.outAcc}}")
logger.info(f"recall@{args.topk2}::{ret2['Recall']:{args.outAcc}}   ndcg@{args.topk2}::{ret2['NDCG']:{args.outAcc}}")
logger.info(f"recall@{args.topk3}::{ret3['Recall']:{args.outAcc}}   ndcg@{args.topk3}::{ret3['NDCG']:{args.outAcc}}")