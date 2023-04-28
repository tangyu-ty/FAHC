# 生成requestment.txt相关依赖
```bash
pip install pipreqs
pipreqs . --encoding=utf8 --force
```
# 安装相关依赖
```bash
pip install -r requirements.txt
```
里面的torch版本如下安装
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

# 运行
```bash
python main.py --data=yelp
```
具体超参数请查看`Params.py`文件

# 超参数调优
1、写入`hyperParam.yaml`中，通过排列组合进行训练。(为了充分利用资源我们采用多线程异步训练)

运行

```bash
python HyperParamMain.py
```



2、在`Hyper.py`中写入调试命令顺序执行

运行

```bash
python Hyper.py
```

# 其他

日志文件生成在`/Log`中、模型对应日志文件名存在`Saved`中。

