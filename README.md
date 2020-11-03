# 华为2020全球校园AI算法精英大赛
## 模型采用IBNnet以及EfficientNet 
python环境：Python3.6.5  
python依赖环境： pytorch 1.6.0+cu101 torchvision 0.7.0+cu101  
训练数据位置../train_data/  
测试数据位置../test_data_A/    
模型训练 python train.py --config_file configs/naic_round2_model_b.yml  
模型推理 python test.py --config_file configs/naic_round2_model_b.yml  
模型融合 python final_result.py   
## 成绩
初赛排名 rank6 决赛排名 rank9
