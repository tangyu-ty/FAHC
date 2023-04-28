import itertools
import os
import yaml
import queue
import threading
from multiprocessing.pool import ThreadPool
import time
from Utils.Logging import get_logger

file_path = "hyperParam.yaml"
with open(file_path, 'r', encoding='utf-8') as f:
    f1 = yaml.safe_load(f)
keys = list(dict(f1).keys())
values = dict(f1).values()
ret = itertools.product(*values)
logger = get_logger("hpyerParam.log", verbosity=1, name="hpyerParam")
hpyerkey = list()
hpyerval = list()

for i in ret:
    hpyerkey.append(keys)
    hpyerval.append(i)

index = 0
q = queue.Queue(maxsize=10000000)
for k, v in zip(hpyerkey, hpyerval):
    command = "python main.py "
    for x, y in zip(k, v):
        command = command + f"--{x}={y} "
    command = command + f"--hyperParamIndex={index} "
    q.put(command)
    index = index + 1


# print(index)
def Execute(q):
    while not q.empty():
        command = q.get()
        os.system(command)
        logger.info(command)

if __name__=="__main__":
    t = ThreadPool(1)
    for i in range(1):
        t.apply_async(Execute, args=(q,))
    t.close()
    t.join()