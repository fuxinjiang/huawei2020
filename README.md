# 华为2020全球校园AI算法精英大赛 赛道b图像检索
## 比赛主页
[链接](https://developer.huawei.com/consumer/en/activity/devStarAI/algo/competition.html#/preliminary/info/digix-trail-04/introduction)
## 模型采用IBNnet以及EfficientNet 
python环境：Python3.6.5  
python依赖环境： pytorch 1.6.0+cu101 torchvision 0.7.0+cu101  
训练数据位置../train_data/  
测试数据位置../test_data_A/ 
## 文件说明
config文件夹里面是定义的初始参数
configs文件夹里面是不同模型的训练参数，对config文件的参数进行更新
datasets包含数据预处理过程以及dataloader创建过程
loss里面包含triplet loss以及acrface loss，本次比赛使用的tripletloss+smoothingsoftmax
model里面是定义的resnet_ibn_a, resnet_ibn_bi 以及seresnet_ibn_a模型
processor包含训练和预测代码
solver里面包含模型训练求解器，本次比赛采用的是SGD+Warmup
utils里面主要是对图片向量化之后的特征，进行reranking计算距离矩阵
## 运行顺序   
模型训练 python train.py --config_file configs/naic_round2_model_b.yml  
模型推理 python test.py --config_file configs/naic_round2_model_b.yml  
模型融合 python ronghe.py   
## 成绩
初赛排名 rank6 决赛排名 rank9
