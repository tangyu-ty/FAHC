import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu')
    parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
    parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')
    parser.add_argument('--reg_loss2', default=1, type=float, help='weight decay Loss2')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=32, type=int, help='embedding size')
    parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--topk1', default=10, type=int, help='K of top K')
    parser.add_argument('--topk2', default=20, type=int, help='K of top K')
    parser.add_argument('--topk3', default=40, type=int, help='K of top K')
    parser.add_argument('--keepRate', default=0.5, type=float, help='ratio of edges to keep')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--tstEpoch', default=5, type=int, help='number of epoch to test while training')
    parser.add_argument('--seed', default=2022, type=int, help='Fixed random seeds')
    parser.add_argument('--outAcc', default=".6f", type=str, help='Output accuracy')
    parser.add_argument('--patience', default=3, type=int, help='early stop patience')
    parser.add_argument('--hyperParamIndex', default=0, type=int, help='hyperParam_index exec commend from queue index')
    return parser.parse_args()


args = ParseArgs()
